"""BodyLab: local body-composition tracker fed by a network camera. Run: uv run app.py"""
import hmac
import json
import mimetypes
import os
import re
import secrets
import sqlite3
import statistics
import subprocess
import threading
import time
import urllib.request
from collections import deque
from datetime import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks.python import BaseOptions, vision

import measure

ROOT = Path(__file__).parent
MODEL = ROOT / 'models' / 'pose_landmarker_heavy.task'
MODEL_URL = ('https://storage.googleapis.com/mediapipe-models/pose_landmarker/'
             'pose_landmarker_heavy/float16/latest/pose_landmarker_heavy.task')
ROTATE = {90: cv2.ROTATE_90_CLOCKWISE, 180: cv2.ROTATE_180, 270: cv2.ROTATE_90_COUNTERCLOCKWISE}
PORT = int(os.environ.get('PORT', '8765'))
SYNC_PORT = 8443  # `tailscale serve --bg --https=8443 8765` exposes /api/sync to your tailnet
COOLDOWN_S = float(os.environ.get('COOLDOWN_MIN', '10')) * 60
FRONT_FRAMES, SIDE_FRAMES, LIVE_FRAMES = 40, 20, 12
DB_PATH = Path(os.environ.get('BODYLAB_DB') or ROOT / 'data' / 'bodylab.db')
DAILY = {'weight': (20, 400), 'kcal': (0, 15000), 'protein': (0, 1000), 'sleep': (0, 24), 'active': (0, 20000)}


def db():
    c = sqlite3.connect(DB_PATH, timeout=10)
    c.row_factory = sqlite3.Row
    return c


def init_db():
    DB_PATH.parent.mkdir(exist_ok=True)
    with db() as c:
        c.executescript('''
            create table if not exists scans(id integer primary key, ts real, raw text);
            create table if not exists logs(id integer primary key, ts real, kind text, data text, ext_id text unique);
            create table if not exists goals(metric text primary key, target real, start real, ts real);
            create table if not exists settings(k text primary key, v text);''')


def get_profile():
    r = db().execute("select v from settings where k = 'profile'").fetchone()
    return json.loads(r[0]) if r else None


def set_profile(p):
    with db() as c:
        c.execute("insert or replace into settings values ('profile', ?)", (json.dumps(p),))


def sync_token():
    with db() as c:
        c.execute("insert or ignore into settings values ('sync_token', ?)", (secrets.token_urlsafe(24),))
        return c.execute("select v from settings where k = 'sync_token'").fetchone()[0]


def sync_url():
    try:
        st = json.loads(subprocess.run(['tailscale', 'status', '--json'], capture_output=True, timeout=5).stdout)
        return f"https://{st['Self']['DNSName'].rstrip('.')}:{SYNC_PORT}"
    except Exception:  # no Tailscale CLI: the Profile panel just shows the token
        return None


def ensure_model():
    if not MODEL.exists():
        MODEL.parent.mkdir(exist_ok=True)
        print('Downloading pose model (30 MB)…', flush=True)
        urllib.request.urlretrieve(MODEL_URL, MODEL.with_suffix('.part'))
        MODEL.with_suffix('.part').rename(MODEL)


def get_logs():
    return [{'id': r['id'], 'ts': r['ts'], 'kind': r['kind'], **json.loads(r['data'])}
            for r in db().execute('select * from logs order by ts')]


def weights(logs):
    return [(l['ts'], l['weight']) for l in logs if l.get('weight')]


def median(frames):
    """Per-key median across frames; keys missing in over half the frames are dropped."""
    out = {}
    for k in {k for f in frames for k in f}:
        vals = [f[k] for f in frames if f.get(k) is not None]
        if len(vals) >= len(frames) / 2:
            out[k] = float(statistics.median(vals))
    return out


def app_state():
    prof, logs, c = get_profile(), get_logs(), db()
    goals = {r['metric']: {'target': r['target'], 'start': r['start'], 'ts': r['ts']} for r in c.execute('select * from goals')}
    scans = []
    if prof and prof.get('height_cm'):
        w = weights(logs)
        for r in c.execute('select * from scans order by ts'):
            raw = json.loads(r['raw'])
            m, dims = measure.derive(raw, prof, measure.nearest(w, r['ts']))
            scans.append({'id': r['id'], 'ts': r['ts'], 'side': 'neck_d' in raw, 'm': m, 'dims': dims})
    return {'profile': prof, 'scans': scans, 'logs': logs, 'goals': goals, 'sync': {'token': sync_token(), 'url': SYNC_URL},
            'insights': measure.insights(scans, logs, goals)}


class Capture:
    """Reads the camera, runs pose + segmentation, and turns good frames into saved scans."""

    def __init__(self):
        self.frame = self.jpeg = self.url = None
        self.frame_ts = 0.0
        self.status = {'state': 'starting', 'msg': 'Connecting to camera…'}
        self.front, self.side = deque(maxlen=FRONT_FRAMES), deque(maxlen=SIDE_FRAMES)
        self.scan_id, self.side_saved, self.last_seen, self.last_scan = None, False, 0.0, 0.0
        self.want_rescan = False
        self.prof, self.weight, self.prof_ts = {}, None, 0.0
        threading.Thread(target=self.read_loop, daemon=True).start()
        threading.Thread(target=self.process_loop, daemon=True).start()

    def rescan(self):
        self.want_rescan = True  # applied on the processing thread, which owns the frame buffers

    def reset(self):
        self.front.clear()
        self.side.clear()
        self.scan_id, self.side_saved = None, False

    def read_loop(self):
        while True:
            self.url = (get_profile() or {}).get('camera_url')
            vc = cv2.VideoCapture(self.url) if self.url else None
            while vc and vc.isOpened():
                ok, f = vc.read()
                if not ok:
                    break
                self.frame, self.frame_ts = f, time.time()
            if vc:
                vc.release()
            time.sleep(2)

    def process_loop(self):
        landmarker = vision.PoseLandmarker.create_from_options(vision.PoseLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(MODEL)),
            running_mode=vision.RunningMode.VIDEO, output_segmentation_masks=True))
        last, ts_ms = None, 0
        while True:
            f = self.frame
            if time.time() - self.frame_ts > 5:
                self.status = {'state': 'offline', 'msg': f'Camera offline ({self.url}) — retrying…' if self.url
                               else 'Set your camera URL in Profile'}
            if f is None or f is last:
                time.sleep(.01)
                continue
            last = f
            if time.time() - self.prof_ts > 2:
                self.prof, self.weight, self.prof_ts = get_profile() or {}, (weights(get_logs()) or [(0, None)])[-1][1], time.time()
            if self.prof.get('rotate') in ROTATE:  # camera mounted sideways / upside down
                f = cv2.rotate(f, ROTATE[self.prof['rotate']])
            ts_ms = max(ts_ms + 1, int(time.monotonic() * 1000))
            res = landmarker.detect_for_video(mp.Image(image_format=mp.ImageFormat.SRGB,
                                                       data=cv2.cvtColor(f, cv2.COLOR_BGR2RGB)), ts_ms)
            try:
                self.step(f.copy(), res)
            except Exception as e:  # keep the camera loop alive; surface the error in the UI
                self.status = {'state': 'error', 'msg': f'Processing error: {e}'}

    def step(self, f, res):
        now = time.time()
        if self.want_rescan:
            self.reset()
            self.want_rescan, self.last_scan = False, 0.0
        r, mask = None, None
        if res.pose_landmarks:
            self.last_seen = now
            h, w = f.shape[:2]
            lms = res.pose_landmarks[0]
            pts = np.array([[l.x * w, l.y * h] for l in lms])
            vis = np.array([l.visibility or 0 for l in lms])
            mask = res.segmentation_masks[0].numpy_view().squeeze()
            r = measure.measure(pts, vis, mask, female=self.prof.get('sex') == 'female')
            if not r['problem']:
                (self.front if r['view'] == 'front' else self.side).append(r['raw'])
        elif now - self.last_seen > 8:
            self.reset()
        self.status = self.next_status(now, r)
        self.publish(f, res, mask, r)

    def next_status(self, now, r):
        st = {'state': 'idle', 'front': len(self.front) / FRONT_FRAMES, 'side': len(self.side) / SIDE_FRAMES}
        if not self.prof.get('height_cm'):
            return {**st, 'msg': 'Set your height in Profile to start scanning'}
        if r is None:
            return {**st, 'msg': 'Step in front of the camera — full body, fitted clothes, arms slightly out'}
        if len(self.front) >= LIVE_FRAMES:
            st['live'] = dict(zip(('m', 'dims'), measure.derive(median(list(self.front)[-LIVE_FRAMES:]), self.prof, self.weight)))
        wait = self.last_scan + COOLDOWN_S - now
        if self.scan_id is None and len(self.front) >= FRONT_FRAMES and wait <= 0:
            raw = {**median(list(self.front)), 'n_front': len(self.front)}
            with db() as c:
                self.scan_id = c.execute('insert into scans(ts, raw) values (?, ?)', (now, json.dumps(raw))).lastrowid
            self.last_scan = now
        if self.scan_id and not self.side_saved and len(self.side) >= SIDE_FRAMES:
            with db() as c:
                raw = json.loads(c.execute('select raw from scans where id = ?', (self.scan_id,)).fetchone()[0])
                raw.update({k: v for k, v in median(list(self.side)).items() if k.endswith('_d')}, n_side=len(self.side))
                c.execute('update scans set raw = ? where id = ?', (json.dumps(raw), self.scan_id))
            self.side_saved = True
        st['scan_id'] = self.scan_id
        if r['problem']:
            return {**st, 'state': 'adjust', 'msg': r['problem']}
        if self.side_saved:
            return {**st, 'state': 'done', 'msg': 'Scan complete ✓ (front + side)'}
        if self.scan_id:
            return {**st, 'state': 'saved', 'msg': 'Front scan saved ✓ — turn 90° sideways for a more accurate scan'}
        if wait > 0:
            return {**st, 'state': 'live', 'msg': f'Tracking live · next saved scan in {wait / 60:.0f} min (or press Rescan)'}
        return {**st, 'state': 'scanning', 'msg': f'Hold still… scanning {r["view"]}'}

    def publish(self, f, res, mask, r):
        if mask is not None:
            sel = mask > .5
            f[sel] = (f[sel] * .55 + np.array([255, 190, 60]) * .45).astype(np.uint8)
            h, w = f.shape[:2]
            lms = res.pose_landmarks[0]
            for c in vision.PoseLandmarksConnections.POSE_LANDMARKS:
                a, b = lms[c.start], lms[c.end]
                cv2.line(f, (int(a.x * w), int(a.y * h)), (int(b.x * w), int(b.y * h)), (255, 255, 255), 2)
        for a, b in (r or {}).get('lines', []):
            cv2.line(f, tuple(map(int, a)), tuple(map(int, b)), (60, 255, 120), 3)
        self.jpeg = cv2.imencode('.jpg', f, [cv2.IMWRITE_JPEG_QUALITY, 70])[1].tobytes()


def num(v, lo, hi, name):
    v = float(v)
    if not lo <= v <= hi:
        raise ValueError(f'{name} must be between {lo} and {hi}')
    return v


def ts_of(b):
    d = b.get('date')
    return datetime.strptime(d, '%Y-%m-%d').replace(hour=12).timestamp() if d else time.time()


def post_profile(b):
    """Partial update: only the fields sent are changed."""
    prof = get_profile() or {}
    if b.get('height_cm') not in (None, ''):
        prof['height_cm'] = num(b['height_cm'], 100, 250, 'Height (cm)')
    if 'sex' in b:
        prof['sex'] = 'female' if b['sex'] == 'female' else 'male'
    if 'units' in b:
        prof['units'] = 'imperial' if b['units'] == 'imperial' else 'metric'
    if 'camera_url' in b:
        url = str(b['camera_url']).strip()
        if url and urlparse(url).scheme not in ('http', 'https', 'rtsp'):
            raise ValueError('Camera URL must start with http://, https:// or rtsp://')
        prof['camera_url'] = url
    if 'skin' in b:
        if not re.fullmatch(r'#[0-9a-fA-F]{6}', str(b['skin'])):
            raise ValueError('Skin tone must be a #rrggbb colour')
        prof['skin'] = b['skin']
    if 'rotate' in b:
        prof['rotate'] = int(b['rotate']) if int(b['rotate']) in ROTATE else 0
    set_profile(prof)


def workout_data(b):
    return {'type': str(b.get('type') or 'Workout')[:40], 'minutes': num(b.get('minutes'), 1, 600, 'Minutes'),
            'rpe': num(b.get('rpe') or 5, 1, 10, 'RPE'),
            'focus': [f for f in b.get('focus') or [] if f in measure.FOCUS], 'notes': str(b.get('notes') or '')[:500]}


def daily_data(b):
    return {k: num(b[k], lo, hi, k) for k, (lo, hi) in DAILY.items() if b.get(k) not in (None, '')}


def post_log(b):
    kind = b.get('kind')
    if kind == 'workout':
        data = workout_data(b)
    elif kind == 'daily':
        data = daily_data(b)
        if not data:
            raise ValueError('Enter at least one value')
    else:
        raise ValueError('Unknown log kind')
    with db() as c:
        c.execute('insert into logs(ts, kind, data) values (?, ?, ?)', (ts_of(b), kind, json.dumps(data)))


def post_sync(b):
    """Apple Health batch from the iOS app. Upserts by Health ID/day, so re-syncing never duplicates."""
    rows = [(ts_of(d), 'daily', data, f"health:day:{d['date']}") for d in b.get('days') or [] if (data := daily_data(d))]
    rows += [(num(w['start'], 0, 1e10, 'start'), 'workout', workout_data(w), f"health:workout:{str(w['id'])[:64]}")
             for w in b.get('workouts') or []]
    with db() as c:
        c.executemany('insert into logs(ts, kind, data, ext_id) values (?, ?, ?, ?) '
                      'on conflict(ext_id) do update set ts = excluded.ts, data = excluded.data',
                      [(ts, kind, json.dumps(data), ext) for ts, kind, data, ext in rows])
    return {'ok': True, 'saved': len(rows)}


def post_goal(b):
    metric = b.get('metric')
    if metric not in measure.METRICS:
        raise ValueError('Unknown metric')
    st = app_state()
    pts = measure.series(metric, st['scans'], st['logs'])
    with db() as c:
        c.execute('insert or replace into goals values (?, ?, ?, ?)',
                  (metric, num(b.get('target'), -1e4, 1e4, 'Target'), pts[-1][1] if pts else None, time.time()))


def post_calibrate(b):
    metric, prof = b.get('metric'), get_profile()
    if metric not in measure.CALIBRATABLE or not prof:
        raise ValueError('Cannot calibrate that')
    cal = prof.setdefault('calib', {})
    if b.get('value') in (None, ''):
        cal.pop(metric, None)
    else:
        row = db().execute('select raw from scans order by ts desc limit 1').fetchone()
        estimate = row and measure.derive(json.loads(row[0]), {**prof, 'calib': {}})[0].get(metric)
        if not estimate:
            raise ValueError('Need a scan with this measurement first')
        factor = num(b['value'], 1, 300, 'Measurement') / estimate
        if not .6 <= factor <= 1.6:
            raise ValueError(f'That is {factor:.2f}× the camera estimate — check the units or rescan')
        cal[metric] = factor
    set_profile(prof)


POST = {'/api/profile': post_profile, '/api/logs': post_log, '/api/goals': post_goal,
        '/api/calibrate': post_calibrate, '/api/rescan': lambda b: cap.rescan(), '/api/sync': post_sync}
LOCAL = ('localhost', '127.0.0.1')


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def send(self, code, body=b'', ctype='application/json'):
        self.send_response(code)
        self.send_header('Content-Type', ctype)
        self.send_header('Content-Length', str(len(body)))
        self.send_header('Cache-Control', 'no-store')
        self.end_headers()
        self.wfile.write(body)

    def reply(self, obj, code=200):
        self.send(code, json.dumps(obj).encode())

    def allowed(self):
        # Blocks DNS rebinding and cross-site requests: the camera feed and data stay with this origin.
        origin = self.headers.get('Origin')
        if urlparse(f'//{self.headers.get("Host", "")}').hostname not in LOCAL or (origin and urlparse(origin).hostname not in LOCAL):
            self.send(403, b'forbidden', 'text/plain')
            return False
        return True

    def do_GET(self):
        if not self.allowed():
            return
        path = urlparse(self.path).path
        if path == '/':
            path = '/static/index.html'
        if path == '/api/state':
            return self.reply(app_state())
        if path == '/api/status':
            return self.reply(cap.status)
        if path == '/api/live.mjpg':
            return self.stream()
        f = (ROOT / path.lstrip('/')).resolve()
        if path.startswith('/static/') and (ROOT / 'static') in f.parents and f.is_file():
            return self.send(200, f.read_bytes(), mimetypes.guess_type(f.name)[0] or 'application/octet-stream')
        self.send(404, b'not found', 'text/plain')

    def do_POST(self):
        path = urlparse(self.path).path
        if path == '/api/sync':  # reachable from the tailnet via `tailscale serve`, so it needs the token
            if not hmac.compare_digest(self.headers.get('Authorization', '').encode(), f'Bearer {sync_token()}'.encode()):
                return self.reply({'error': 'bad sync token'}, 401)
        elif not self.allowed():
            return
        if path not in POST:
            return self.reply({'error': 'not found'}, 404)
        size = int(self.headers.get('Content-Length') or 0)
        if size > 5_000_000:
            return self.reply({'error': 'too large'}, 413)
        try:
            result = POST[path](json.loads(self.rfile.read(size) or b'{}'))
        except (ValueError, TypeError, KeyError) as e:
            return self.reply({'error': str(e)}, 400)
        self.reply(result or {'ok': True})

    def do_DELETE(self):
        if not self.allowed():
            return
        parts = urlparse(self.path).path.strip('/').split('/')
        if len(parts) != 3 or parts[0] != 'api' or parts[1] not in ('logs', 'scans', 'goals'):
            return self.reply({'error': 'not found'}, 404)
        with db() as c:
            c.execute(f"delete from {parts[1]} where {'metric' if parts[1] == 'goals' else 'id'} = ?", (parts[2],))
        self.reply({'ok': True})

    def stream(self):
        self.send_response(200)
        self.send_header('Content-Type', 'multipart/x-mixed-replace; boundary=frame')
        self.send_header('Cache-Control', 'no-store')
        self.end_headers()
        last = None
        try:
            while True:
                j = cap.jpeg
                if j is None or j is last:
                    time.sleep(.03)
                    continue
                last = j
                self.wfile.write(b'--frame\r\nContent-Type: image/jpeg\r\nContent-Length: %d\r\n\r\n%s\r\n' % (len(j), j))
        except (BrokenPipeError, ConnectionResetError):
            pass


if __name__ == '__main__':
    init_db()
    ensure_model()
    SYNC_URL = sync_url()
    cap = Capture()
    print(f'BodyLab → http://localhost:{PORT}', flush=True)
    ThreadingHTTPServer(('127.0.0.1', PORT), Handler).serve_forever()
