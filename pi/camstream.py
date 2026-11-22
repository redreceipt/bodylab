#!/usr/bin/env python3
"""BodyLab camera: MJPEG at http://<tailscale-ip>:8000/stream.mjpg, bound to the tailnet address only.
A Raspberry Pi camera (picamera2), or any Linux webcam via ffmpeg (DEVICE=/dev/video0, SIZE=1280x720, FPS=15)."""
import io
import os
import socketserver
import subprocess
import threading
import time
from http import server
from threading import Condition

BIND = os.environ.get('BIND') or subprocess.check_output(['tailscale', 'ip', '-4'], text=True).split()[0]
SIZE = tuple(map(int, os.environ.get('SIZE', '1536x864').split('x')))
FPS = int(os.environ.get('FPS', 15))


class Output(io.BufferedIOBase):
    def __init__(self):
        self.frame, self.cond = None, Condition()

    def write(self, buf):
        with self.cond:
            self.frame = buf
            self.cond.notify_all()


class Handler(server.BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def do_GET(self):
        if self.path != '/stream.mjpg':
            return self.send_error(404)
        self.send_response(200)
        self.send_header('Content-Type', 'multipart/x-mixed-replace; boundary=FRAME')
        self.send_header('Cache-Control', 'no-store')
        self.end_headers()
        try:
            while True:
                with out.cond:
                    out.cond.wait()
                    frame = out.frame
                self.wfile.write(b'--FRAME\r\nContent-Type: image/jpeg\r\nContent-Length: %d\r\n\r\n%s\r\n' % (len(frame), frame))
        except (BrokenPipeError, ConnectionResetError):
            pass


class Server(socketserver.ThreadingMixIn, server.HTTPServer):
    allow_reuse_address = daemon_threads = True


def webcam():
    """The webcam's own MJPEG frames, passed through by ffmpeg without re-encoding (restarted if it stops)."""
    while True:
        p = subprocess.Popen(['ffmpeg', '-loglevel', 'error', '-f', 'v4l2', '-input_format', 'mjpeg', '-video_size', '%dx%d' % SIZE,
                              '-framerate', str(FPS), '-i', os.environ.get('DEVICE', '/dev/video0'), '-c', 'copy', '-f', 'mjpeg', '-'],
                             stdout=subprocess.PIPE)
        buf = b''
        while chunk := p.stdout.read(1 << 16):
            buf += chunk
            while (start := buf.find(b'\xff\xd8')) != -1 and (end := buf.find(b'\xff\xd9', start)) != -1:
                out.write(buf[start:end + 2])  # one JPEG (0xFFD9 can't occur inside one: 0xFF bytes are stuffed)
                buf = buf[end + 2:]
        p.wait()
        time.sleep(2)


out = Output()
try:
    from picamera2 import Picamera2
    from picamera2.encoders import MJPEGEncoder
    from picamera2.outputs import FileOutput
except ImportError:
    threading.Thread(target=webcam, daemon=True).start()
    Server((BIND, 8000), Handler).serve_forever()
cam = Picamera2()
full = max(cam.sensor_modes, key=lambda m: (m['crop_limits'][2] * m['crop_limits'][3], m['fps']))  # widest field of view
cam.configure(cam.create_video_configuration(main={'size': SIZE}, sensor={'output_size': full['size']},
                                             controls={'FrameRate': FPS}))
if 'AfMode' in cam.camera_controls:
    cam.set_controls({'AfMode': 2})  # continuous autofocus (Camera Module 3)
cam.start_recording(MJPEGEncoder(), FileOutput(out))
try:
    Server((BIND, 8000), Handler).serve_forever()
finally:
    cam.stop_recording()
