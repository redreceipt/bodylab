#!/usr/bin/env python3
"""BodyLab camera: MJPEG at http://<tailscale-ip>:8000/stream.mjpg, bound to the tailnet address only."""
import io
import os
import socketserver
import subprocess
from http import server
from threading import Condition

from picamera2 import Picamera2
from picamera2.encoders import MJPEGEncoder
from picamera2.outputs import FileOutput

BIND = os.environ.get('BIND') or subprocess.check_output(['tailscale', 'ip', '-4'], text=True).split()[0]
SIZE = tuple(map(int, os.environ.get('SIZE', '1536x864').split('x')))


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


cam = Picamera2()
full = max(cam.sensor_modes, key=lambda m: (m['crop_limits'][2] * m['crop_limits'][3], m['fps']))  # widest field of view
cam.configure(cam.create_video_configuration(main={'size': SIZE}, sensor={'output_size': full['size']},
                                             controls={'FrameRate': 15}))
if 'AfMode' in cam.camera_controls:
    cam.set_controls({'AfMode': 2})  # continuous autofocus (Camera Module 3)
out = Output()
cam.start_recording(MJPEGEncoder(), FileOutput(out))
try:
    Server((BIND, 8000), Handler).serve_forever()
finally:
    cam.stop_recording()
