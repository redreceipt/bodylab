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
FACE_MODEL = ROOT / 'models' / 'face_landmarker.task'
FACE_MODEL_URL = 'https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/latest/face_landmarker.task'
ZOO = 'https://github.com/opencv/opencv_zoo/raw/48b10fa505aeb51fbd127fc60a5c660040e823b7/models/'
FACE_DET = (ROOT / 'models' / 'face_detection_yunet_2023mar.onnx', ZOO + 'face_detection_yunet/face_detection_yunet_2023mar.onnx')
FACE_REC = (ROOT / 'models' / 'face_recognition_sface_2021dec.onnx', ZOO + 'face_recognition_sface/face_recognition_sface_2021dec.onnx')
SAME_PERSON = .363  # SFace cosine-similarity threshold from its model card
SKIN_TOLERANCE = 14  # Cr/Cb distance (0-255) from your face's skin that still counts as skin
ROTATE = {90: cv2.ROTATE_90_CLOCKWISE, 180: cv2.ROTATE_180, 270: cv2.ROTATE_90_COUNTERCLOCKWISE}
PORT = int(os.environ.get('PORT', '8765'))
SYNC_PORT = 8443  # `tailscale serve --bg --https=8443 http://127.0.0.1:8765` puts BodyLab on your tailnet
TS_HOST = TS_OWNER = None  # set at startup from `tailscale status`
COOLDOWN_S = float(os.environ.get('COOLDOWN_MIN', '10')) * 60
FRONT_FRAMES, SIDE_FRAMES, LIVE_FRAMES = 40, 20, 12
DB_PATH = Path(os.environ.get('BODYLAB_DB') or ROOT / 'data' / 'bodylab.db')
# Daily inputs (manual or Apple Health) and their sane ranges.
DAILY = {'weight': (20, 400), 'scale_bf': (2, 75), 'scale_lean': (10, 300), 'tape_waist': (30, 300),
         'kcal': (0, 15000), 'protein': (0, 1000), 'carbs': (0, 2000), 'fat': (0, 1000), 'fiber': (0, 300),
         'sugar': (0, 1000), 'water': (0, 20), 'alcohol': (0, 50),
         'sleep': (0, 24), 'sleep_deep': (0, 24), 'sleep_rem': (0, 24),
         'active': (0, 20000), 'basal': (0, 10000), 'steps': (0, 200000), 'exercise': (0, 1440),
         'rhr': (20, 250), 'hrv': (1, 500), 'vo2max': (5, 100), 'walk_hr': (20, 250), 'resp': (2, 80)}


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


def tailnet():
    """(this Mac's tailnet hostname, owner's login) or (None, None) without Tailscale."""
    try:
        st = json.loads(subprocess.run(['tailscale', 'status', '--json'], capture_output=True, timeout=5).stdout)
        return st['Self']['DNSName'].rstrip('.'), st['User'][str(st['Self']['UserID'])]['LoginName']
    except Exception:
        return None, None


def ensure_model(path=MODEL, url=MODEL_URL):
    if not path.exists():
        path.parent.mkdir(exist_ok=True)
        print(f'Downloading {path.name}…', flush=True)
        urllib.request.urlretrieve(url, path.with_suffix('.part'))
        path.with_suffix('.part').rename(path)


REV = threading.Lock()


def data_rev():
    """Changes whenever another connection commits (iPhone sync, edits) or the face photo changes."""
    global _rev_db
    with REV:
        if '_rev_db' not in globals():
            _rev_db = sqlite3.connect(DB_PATH, check_same_thread=False)
        version = _rev_db.execute('pragma data_version').fetchone()[0]
    face = face_files()[1]
    return f'{version}-{face.stat().st_mtime if face.exists() else 0}'


def face_files():
    return DB_PATH.parent / 'face.jpg', DB_PATH.parent / 'face.json'


class Faces:
    """OpenCV YuNet + SFace: one embedding per face, to check the person being scanned is you."""

    def __init__(self):
        for path, url in (FACE_DET, FACE_REC):
            ensure_model(path, url)
        self.det = cv2.FaceDetectorYN.create(str(FACE_DET[0]), '', (320, 320), .7)
        self.rec = cv2.FaceRecognizerSF.create(str(FACE_REC[0]), '')

    def embed(self, img):
        """Unit-length embedding of the largest face in a BGR image, or None."""
        self.det.setInputSize((img.shape[1], img.shape[0]))
        faces = self.det.detect(img)[1]
        if faces is None:
            return None
        e = self.rec.feature(self.rec.alignCrop(img, max(faces, key=lambda f: f[2] * f[3]))).flatten()
        return e / np.linalg.norm(e)


def post_face(raw):
    """Face photo for the avatar: stored with its 478 landmarks next to the database (never leaves this Mac)."""
    img = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError('That is not an image')
    if (k := 1600 / max(img.shape[:2])) < 1:
        img = cv2.resize(img, None, fx=k, fy=k, interpolation=cv2.INTER_AREA)
    ensure_model(FACE_MODEL, FACE_MODEL_URL)
    landmarker = vision.FaceLandmarker.create_from_options(
        vision.FaceLandmarkerOptions(base_options=BaseOptions(model_asset_path=str(FACE_MODEL)), num_faces=2))
    try:
        faces = landmarker.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(img, cv2.COLOR_BGR2RGB))).face_landmarks
    finally:
        landmarker.close()
    if len(faces) != 1:
        raise ValueError('Need exactly one face in the photo' if faces else 'No face found — try a brighter, closer photo')
    h, w = img.shape[:2]
    pts = np.array([[l.x * w, l.y * h] for l in faces[0]])
    if np.ptp(pts[:, 0]) < 150:
        raise ValueError('Face is too small — hold the camera closer')
    left, right = np.linalg.norm(pts[1] - pts[263]), np.linalg.norm(pts[1] - pts[33])  # nose tip to outer eye corners
    if abs(left - right) / (left + right) > .2:
        raise ValueError('Look straight at the camera')
    if np.linalg.norm(pts[13] - pts[14]) > .03 * np.ptp(pts[:, 1]):  # inner lips apart
        raise ValueError('Close your mouth (neutral expression) — teeth don\'t map onto the 3D face')
    jpg, meta = face_files()
    cv2.imwrite(str(jpg), img, [cv2.IMWRITE_JPEG_QUALITY, 92])
    emb = Faces().embed(img)  # enrols you, so scans are only saved for you
    meta.write_text(json.dumps({'w': w, 'h': h, 'ts': time.time(), 'landmarks': (pts / [w, h]).round(5).tolist(),
                                'embedding': None if emb is None else emb.round(5).tolist()}))


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
    scans, hidden = [], 0
    if prof and prof.get('height_cm'):
        w = weights(logs)
        for r in c.execute('select * from scans order by ts'):
            raw = json.loads(r['raw'])
            if measure.implausible(raw):  # misread poses saved before the plausibility check; kept, not shown
                hidden += 1
                continue
            m, dims = measure.derive(raw, prof, measure.nearest(w, r['ts']))
            scans.append({'id': r['id'], 'ts': r['ts'], 'side': 'neck_d' in raw, 'm': m, 'dims': dims})
    face = face_files()[1]
    face = json.loads(face.read_text()) if face.exists() else None
    if face:
        face.pop('embedding', None)  # stays server-side
    sync = f'https://{TS_HOST}:{SYNC_PORT}' if TS_HOST else None
    return {'profile': prof, 'scans': scans, 'logs': logs, 'goals': goals, 'sync': {'token': sync_token(), 'url': sync},
            'face': face, 'hidden_scans': hidden,
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
        self.prof, self.weight, self.prof_ts, self.me = {}, None, 0.0, None
        self.who, self.misses, self.skin = None, 0, None  # per session: 'me' / 'other', and your skin chroma
        threading.Thread(target=self.read_loop, daemon=True).start()
        threading.Thread(target=self.process_loop, daemon=True).start()

    def rescan(self):
        self.want_rescan = True  # applied on the processing thread, which owns the frame buffers

    def reset(self):
        self.front.clear()
        self.side.clear()
        self.scan_id, self.side_saved = None, False
        self.who, self.misses, self.skin = None, 0, None

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
            base_options=BaseOptions(model_asset_path=str(MODEL)), num_poses=2,
            running_mode=vision.RunningMode.VIDEO, output_segmentation_masks=True))
        self.faces = Faces()
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
                face = face_files()[1]
                emb = json.loads(face.read_text()).get('embedding') if face.exists() else None
                self.me = np.array(emb) if emb else None
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
        people = [i for i, m in enumerate(res.segmentation_masks or []) if (m.numpy_view() > .5).mean() > .01]
        if len(people) > 1:  # someone else walked in: whose frames are whose is no longer certain
            self.last_seen = now
            self.reset()
            r = {'problem': 'Only one person in the frame, please', 'lines': [], 'view': None}
            mask = res.segmentation_masks[0].numpy_view().squeeze()
        elif res.pose_landmarks:
            self.last_seen = now
            h, w = f.shape[:2]
            lms = res.pose_landmarks[0]
            pts = np.array([[l.x * w, l.y * h] for l in lms])
            vis = np.array([l.visibility or 0 for l in lms])
            mask = res.segmentation_masks[0].numpy_view().squeeze()
            r = measure.measure(pts, vis, mask, female=self.prof.get('sex') == 'female')
            if not r['problem']:
                r['problem'] = self.check_person(f, pts, vis, r)
            if not r['problem']:
                (self.front if r['view'] == 'front' else self.side).append(r['raw'])
        elif now - self.last_seen > 8:
            self.reset()
        self.status = self.next_status(now, r)
        self.publish(f, res, mask, r)

    def check_person(self, f, pts, vis, r):
        """Is this you (face matches your photo) and in minimal clothing? Returns a problem or None."""
        if self.who == 'other':
            return 'Not you — BodyLab only saves scans for the profile owner'
        if self.me is not None and self.who != 'me':
            if r['view'] == 'side':
                return 'Face the camera first so BodyLab can check it is you'
            span = max(np.linalg.norm(pts[7] - pts[8]), 20)  # ear to ear
            (x0, y0), (x1, y1) = (pts[0] - 1.6 * span).astype(int), (pts[0] + 1.6 * span).astype(int)
            crop = f[max(y0, 0):max(y1, 0), max(x0, 0):max(x1, 0)]
            emb = self.faces.embed(cv2.resize(crop, (320, 320), interpolation=cv2.INTER_CUBIC)) if crop.size else None
            if emb is None:
                return 'Face the camera so BodyLab can check it is you'
            if float(emb @ self.me) >= SAME_PERSON:
                self.who = 'me'
            else:
                self.misses += 1
                if self.misses >= 8:
                    self.who = 'other'
                return 'Checking it is you…'
        if self.prof.get('bare_only', True):
            ycc = cv2.cvtColor(f, cv2.COLOR_BGR2YCrCb)
            if vis[0] > .5 and r['view'] == 'front':  # skin chroma from your cheeks, in this frame's lighting
                eye = np.linalg.norm(pts[2] - pts[5])
                patches = [ycc[int(y - eye * .15):int(y + eye * .15) + 1, int(x - eye * .15):int(x + eye * .15) + 1, 1:].reshape(-1, 2)
                           for x, y in ((pts[2] + pts[9]) / 2, (pts[5] + pts[10]) / 2)]
                self.skin = np.median(np.vstack(patches), axis=0)
            if self.skin is not None:
                for name in ('waist',) if self.prof.get('sex') == 'female' or r['view'] == 'side' else ('chest', 'waist'):
                    if measure.skin_share(ycc, self.skin, *r['cuts'][name], SKIN_TOLERANCE) < .5:
                        return f'Clothing on your {name} — scans only count in minimal clothing (underwear is fine)'
        return None

    def next_status(self, now, r):
        st = {'state': 'idle', 'front': len(self.front) / FRONT_FRAMES, 'side': len(self.side) / SIDE_FRAMES}
        if not self.prof.get('height_cm'):
            return {**st, 'msg': 'Set your height in Profile to start scanning'}
        if r is None:
            return {**st, 'msg': 'Step in front of the camera — whole body, minimal clothing, arms slightly out'}
        if len(self.front) >= LIVE_FRAMES and self.who != 'other':
            st['live'] = dict(zip(('m', 'dims'), measure.derive(median(list(self.front)[-LIVE_FRAMES:]), self.prof, self.weight)))
        wait = self.last_scan + COOLDOWN_S - now
        if self.scan_id is None and len(self.front) >= FRONT_FRAMES and wait <= 0 and (self.me is None or self.who == 'me'):
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
        if self.prof.get('preview', 'silhouette') == 'silhouette':  # private by default: no camera pixels leave the loop
            f[:] = (40, 26, 20)
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
    for k, pattern in (('hair', r'[a-z0-9]{1,20}'), ('hair_color', r'#[0-9a-fA-F]{6}'), ('eye_color', r'[a-z]{1,20}')):
        if k in b:
            if not re.fullmatch(pattern, str(b[k])):
                raise ValueError(f'Invalid {k}')
            prof[k] = b[k]
    if 'rotate' in b:
        prof['rotate'] = int(b['rotate']) if int(b['rotate']) in ROTATE else 0
    if 'bare_only' in b:
        prof['bare_only'] = bool(b['bare_only'])
    if 'preview' in b:
        prof['preview'] = 'camera' if b['preview'] == 'camera' else 'silhouette'
    set_profile(prof)


def workout_data(b):
    data = {'type': str(b.get('type') or 'Workout')[:40], 'minutes': num(b.get('minutes'), 1, 600, 'Minutes'),
            'rpe': num(b.get('rpe') or 5, 1, 10, 'RPE'),
            'focus': [f for f in b.get('focus') or [] if f in measure.FOCUS], 'notes': str(b.get('notes') or '')[:500]}
    for k, lo, hi in (('kcal', 0, 10000), ('hr', 20, 250)):  # from Apple Health
        if b.get(k) is not None:
            data[k] = num(b[k], lo, hi, k)
    return data


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
    """Apple Health batch from the iOS app. Upserts by Health ID/day, so re-syncing never duplicates.
    Health's height/sex/birth date fill profile fields the user hasn't set (that's the whole onboarding)."""
    hp, prof = b.get('profile') or {}, get_profile() or {}
    fill = {'height_cm': hp.get('height_cm') and num(hp['height_cm'], 100, 250, 'height'),
            'sex': hp.get('sex') if hp.get('sex') in ('male', 'female') else None,
            'birth_date': hp.get('birth_date') and datetime.strptime(hp['birth_date'], '%Y-%m-%d').date().isoformat()}
    if any(v and k not in prof for k, v in fill.items()):
        set_profile({**{k: v for k, v in fill.items() if v}, **prof})
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
        # Only this Mac, or you on your tailnet via `tailscale serve` (which sets a verified Tailscale-User-Login).
        # The Host/Origin checks also block DNS rebinding and cross-site requests.
        host, origin = urlparse(f'//{self.headers.get("Host", "")}').hostname, self.headers.get('Origin')
        ok = host in LOCAL or (TS_HOST and host == TS_HOST and self.headers.get('Tailscale-User-Login') == TS_OWNER)
        if not ok or (origin and urlparse(origin).hostname not in (*LOCAL, TS_HOST)):
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
            return self.reply({**cap.status, 'rev': data_rev()})
        if path == '/api/live.mjpg':
            return self.stream()
        if path == '/api/face.jpg' and face_files()[0].exists():
            return self.send(200, face_files()[0].read_bytes(), 'image/jpeg')
        f = (ROOT / path.lstrip('/')).resolve()
        if path.startswith('/static/') and (ROOT / 'static') in f.parents and f.is_file():
            return self.send(200, f.read_bytes(), mimetypes.guess_type(f.name)[0] or 'application/octet-stream')
        self.send(404, b'not found', 'text/plain')

    def do_POST(self):
        path = urlparse(self.path).path
        auth = self.headers.get('Authorization')
        if auth is not None and path in ('/api/sync', '/api/face'):  # the iPhone, via `tailscale serve`
            if not hmac.compare_digest(auth.encode(), f'Bearer {sync_token()}'.encode()):
                return self.reply({'error': 'bad sync token'}, 401)
        elif not self.allowed():
            return
        if path not in POST and path != '/api/face':
            return self.reply({'error': 'not found'}, 404)
        size = int(self.headers.get('Content-Length') or 0)
        if size > 15_000_000:
            return self.reply({'error': 'too large'}, 413)
        try:
            body = self.rfile.read(size)
            result = post_face(body) if path == '/api/face' else POST[path](json.loads(body or b'{}'))
        except (ValueError, TypeError, KeyError) as e:
            return self.reply({'error': str(e)}, 400)
        self.reply(result or {'ok': True})

    def do_DELETE(self):
        if not self.allowed():
            return
        parts = urlparse(self.path).path.strip('/').split('/')
        if parts == ['api', 'face']:
            for f in face_files():
                f.unlink(missing_ok=True)
            return self.reply({'ok': True})
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
    jpg, meta = face_files()
    if meta.exists() and 'embedding' not in json.loads(meta.read_text()):  # photo added before identity checks existed
        try:
            post_face(jpg.read_bytes())
        except ValueError as e:
            print(f'Face photo needs replacing: {e}', flush=True)
    TS_HOST, TS_OWNER = tailnet()
    cap = Capture()
    print(f'BodyLab → http://localhost:{PORT}' + (f'  ·  tailnet: https://{TS_HOST}:{SYNC_PORT}' if TS_HOST else ''), flush=True)
    ThreadingHTTPServer(('127.0.0.1', PORT), Handler).serve_forever()
