"""BodyLab: local body-composition tracker fed by a network camera. Run: uv run app.py"""
import hmac
import json
import os

# Quiet the native libraries (MediaPipe, OpenCV, ffmpeg reconnect spam) so the log can't balloon.
os.environ.setdefault('GLOG_minloglevel', '2')
os.environ.setdefault('OPENCV_LOG_LEVEL', 'ERROR')
os.environ.setdefault('OPENCV_FFMPEG_LOGLEVEL', '-8')
import mimetypes
import re
import stat
import secrets
import sqlite3
import statistics
import subprocess
import sys
import threading
import time
import urllib.request
from collections import deque
from datetime import date, datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

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
CLOTHES_MODEL = (ROOT / 'models' / 'selfie_multiclass_256x256.tflite',  # skin vs clothes, per pixel
                 'https://storage.googleapis.com/mediapipe-models/image_segmenter/selfie_multiclass_256x256/float32/latest/selfie_multiclass_256x256.tflite')
HAIR_MODEL = (ROOT / 'models' / 'hair_segmenter.tflite',
              'https://storage.googleapis.com/mediapipe-models/image_segmenter/hair_segmenter/float32/latest/hair_segmenter.tflite')
ZOO = 'https://github.com/opencv/opencv_zoo/raw/48b10fa505aeb51fbd127fc60a5c660040e823b7/models/'
FACE_DET = (ROOT / 'models' / 'face_detection_yunet_2023mar.onnx', ZOO + 'face_detection_yunet/face_detection_yunet_2023mar.onnx')
FACE_REC = (ROOT / 'models' / 'face_recognition_sface_2021dec.onnx', ZOO + 'face_recognition_sface/face_recognition_sface_2021dec.onnx')
SAME_PERSON = .363  # SFace cosine-similarity threshold from its model card
SKIN_TOLERANCE = 14  # Cr/Cb distance (0-255) from your face's skin that still counts as skin
ROTATE = {90: cv2.ROTATE_90_CLOCKWISE, 180: cv2.ROTATE_180, 270: cv2.ROTATE_90_COUNTERCLOCKWISE}
PORT = int(os.environ.get('PORT', '8765'))
SYNC_PORT = 8443  # `tailscale serve --bg --https=8443 http://127.0.0.1:8765` puts BodyLab on your tailnet
TS_HOST = TS_OWNER = None  # set at startup from `tailscale status`
DB_PATH = Path(os.environ.get('BODYLAB_DB') or ROOT / 'data' / 'bodylab.db')
# Daily inputs (manual or Apple Health) and their sane ranges.
FRAMES_PER_DAY, KEEP_FRAMES_DAYS = 400, 45  # raw evidence for refits; daily body estimates are kept forever
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
            create table if not exists frames(id integer primary key, ts real, data text);
            create index if not exists frames_ts on frames(ts);
            create table if not exists logs(id integer primary key, ts real, kind text, data text, ext_id text unique);
            create table if not exists goals(metric text primary key, target real, start real, ts real);
            create table if not exists settings(k text primary key, v text);
            create table if not exists fits(day text primary key, ts real, params text, mesh blob);''')
        if 'day' not in [r[1] for r in c.execute('pragma table_info(scans)')]:  # one fused estimate per day
            c.execute('alter table scans add column day text')
        c.execute('create unique index if not exists scans_day on scans(day)')


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
    ensure_model(*HAIR_MODEL)
    seg = vision.ImageSegmenter.create_from_options(vision.ImageSegmenterOptions(base_options=BaseOptions(model_asset_path=str(HAIR_MODEL[0]))))
    try:
        masks = seg.segment(mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(img, cv2.COLOR_BGR2RGB))).confidence_masks
    finally:
        seg.close()
    hair = measure.hair_style(masks[-1].numpy_view().squeeze(), img, pts)  # last mask = hair
    meta.write_text(json.dumps({'w': w, 'h': h, 'ts': time.time(), 'landmarks': (pts / [w, h]).round(5).tolist(), 'hair': hair,
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


def console_state():
    """What the developer console shows beyond the app's state: the latest fit, today's frames, reference images."""
    c, refs = db(), sorted(reference_dir().glob('*.jpg'), key=lambda p: p.stat().st_mtime) if reference_dir().exists() else []
    history = [{'day': r['day'], **{k: json.loads(r['params']).get(k) for k in ('frames', 'dropped', 'fit', 'fit_cams')}}
               for r in c.execute('select day, params from fits order by day desc limit 7')]
    f = c.execute('select day, params from fits order by day desc limit 1').fetchone()
    info = json.loads(f['params']) if f else None
    frames = c.execute("select coalesce(json_extract(data, '$.cam'), 0), count(*) from frames where ts >= ? group by 1",
                       (day_start(time.time()),)).fetchall()
    return {'fit_info': info and {'day': f['day'], **{k: info.get(k) for k in ('frames', 'fit', 'phenotypes', 'tape')}}, 'fit_history': history,
            'frames_today': {r[0]: r[1] for r in frames}, 'weight': (weights(get_logs()) or [(0, None)])[-1][1],
            'references': {'count': len(refs), 'mb': sum(p.stat().st_size for p in refs) / 2 ** 20, 'latest': [p.name for p in refs[-24:][::-1]]}}


def app_state():
    prof, logs, c = get_profile(), get_logs(), db()
    goals = {r['metric']: {'target': r['target'], 'start': r['start'], 'ts': r['ts']} for r in c.execute('select * from goals')}
    scans, hidden = [], 0
    if prof and prof.get('height_cm'):
        w, rows = weights(logs), []
        fits = {r['day']: json.loads(r['params'])['tape'] for r in c.execute('select day, params from fits')}
        for r in c.execute('select * from scans order by ts'):
            raw = json.loads(r['raw']) | ({'tape': fits[r['day']]} if r['day'] in fits else {})
            if measure.implausible(raw):  # misread poses saved before the plausibility check; kept, not shown
                hidden += 1
            else:
                rows.append((r, raw))
        for r, raw in rows:
            m, dims = measure.derive(raw, prof, measure.nearest(w, r['ts']))
            scans.append({'id': r['id'], 'ts': r['ts'], 'day': r['day'], 'fit': 'tape' in raw, 'side': raw.get('n_angle', 0) >= 30,
                          'frames': raw.get('n_frames'), 'm': m, 'dims': dims})
    face = face_files()[1]
    face = json.loads(face.read_text()) if face.exists() else None
    if face:
        face.pop('embedding', None)  # stays server-side
    sync = f'https://{TS_HOST}:{SYNC_PORT}' if TS_HOST else None
    return {'profile': prof, 'scans': scans, 'logs': logs, 'goals': goals, 'sync': {'token': sync_token(), 'url': sync},
            'face': face, 'hidden_scans': hidden, 'recommended_bf': measure.RECOMMENDED_BF[(prof or {}).get('sex') or 'male'],
            **console_state(), 'reference_images': len(list(reference_dir().glob('*.jpg'))),
            'skin_baked': (DB_PATH.parent / 'skin.jpg').stat().st_mtime if (DB_PATH.parent / 'skin.jpg').exists() else None,
            'skin': json.loads((DB_PATH.parent / 'skin.json').read_text()) if (DB_PATH.parent / 'skin.json').exists() else None,
            'insights': measure.insights(scans, logs, goals)}


class Gate:
    """Decides which frames are you. Nothing counts until your face matches your photo; your face is re-checked
    whenever it's visible; a face that clearly isn't you throws away everything since your last match. A session
    (one person continuously in view) ends when nobody's in view, and only confirmed frames survive it. Frames
    count only within a minute of a match, so someone who never faces the camera can't slip in for long."""
    MISSES = 3  # consecutive clear mismatches before deciding it's someone else
    MATCHES = 2  # separate matches before trusting it's you (one lucky score in bad light isn't enough)
    WINDOW = 60  # seconds either side of a match

    def __init__(self, enrolled=True):
        self.enrolled = enrolled
        self.reset()

    def reset(self):
        self.me = self.other = False
        self.misses, self.matches, self.waiting, self.last = 0, 0, [], -np.inf

    def frame(self, rec):
        """A measured frame. Returns the frames that count now."""
        if not self.enrolled:  # no face photo yet: anyone alone in view counts
            return [rec]
        if not self.other:
            self.waiting = (self.waiting + [rec])[-600:]
        return []

    def face(self, score, ts, frontal=True):
        """Your face's similarity to your photo (SFace cosine) at time ts. Returns frames now confirmed as you."""
        if score >= SAME_PERSON:
            self.matches, self.misses = self.matches + 1, 0
            if self.matches < self.MATCHES:
                return []
            ready = [r for r in self.waiting if r['ts'] - self.last <= self.WINDOW or ts - r['ts'] <= self.WINDOW]
            self.me, self.other, self.last, self.waiting = True, False, ts, []  # clear matches end a false alarm (from now on)
            return ready
        if score < .2 and frontal:  # clearly someone else; turned faces, distance and dim light decide nothing
            self.misses += 1
            if self.misses >= self.MISSES:
                self.me, self.other, self.waiting = False, True, []
        return []

    def end(self):
        """The person left. Frames since your last match count only if it was you and nobody else showed up."""
        ready = [r for r in self.waiting if r['ts'] - self.last <= self.WINDOW] if self.me and not self.other else []
        self.reset()
        return ready


class Capture:
    """Watches the camera all the time and turns every useful frame into evidence about your body.

    No special pose: each frame contributes whatever it shows (a waist width at some angle, a calf, your
    torso length). Frames are stored (twice a second at most) and fused into one body estimate per day."""

    def __init__(self, cam=0):
        self.cam, self.key = cam, '' if cam == 0 else str(cam + 1)  # profile keys camera_url/rotate, camera2_url/rotate2
        self.frame = self.jpeg = self.url = None
        self.frame_ts = self.last_seen = self.last_store = self.last_fit = 0.0
        self.status = {'state': 'starting', 'msg': 'Connecting to camera…'}
        self.want_clear, self.dirty = False, True
        self.prof, self.weight, self.prof_ts, self.me = {}, None, 0.0, None
        self.gate, self.skin, self.match, self.last_id, self.stored, self.last_ref = Gate(), None, None, 0.0, [None, 0], 0.0
        threading.Thread(target=self.read_loop, daemon=True).start()
        threading.Thread(target=self.process_loop, daemon=True).start()

    def end_session(self, keep=True):
        """A session = one person continuously in view. Identity and skin colour are per session."""
        for rec in self.gate.end() if keep else self.gate.reset() or []:
            self.keep(rec)
        self.skin, self.match = None, None

    def clear_today(self):
        self.want_clear = True  # applied on the processing thread

    def read_loop(self):
        while True:
            self.url = (get_profile() or {}).get(f'camera{self.key}_url')
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
        self.clothes = vision.ImageSegmenter.create_from_options(vision.ImageSegmenterOptions(
            base_options=BaseOptions(model_asset_path=str(CLOTHES_MODEL[0])), output_category_mask=True))
        last, ts_ms = None, 0
        while True:
            f = self.frame
            if time.time() - self.prof_ts > 2:
                self.prof, self.weight, self.prof_ts = get_profile() or {}, (weights(get_logs()) or [(0, None)])[-1][1], time.time()
                face = face_files()[1]
                emb = json.loads(face.read_text()).get('embedding') if face.exists() else None
                self.me = np.array(emb) if emb else None
                self.gate.enrolled = self.me is not None
            if time.time() - self.frame_ts > 5:
                self.status = {'state': 'offline', 'msg': f'Camera offline ({self.url}) — retrying…' if self.url
                               else 'Set your camera URL in Settings → Camera' if self.cam == 0 else 'No second camera'}
            if self.cam == 0 and self.dirty and time.time() - self.last_fit > 5 and self.prof.get('height_cm'):  # the first camera fuses all cameras' frames
                self.fit(time.time())
            if f is None or f is last:
                time.sleep(.01)
                continue
            last = f
            if self.prof.get(f'rotate{self.key}') in ROTATE:  # camera mounted sideways / upside down
                f = cv2.rotate(f, ROTATE[self.prof[f'rotate{self.key}']])
            ts_ms = max(ts_ms + 1, int(time.monotonic() * 1000))
            res = landmarker.detect_for_video(mp.Image(image_format=mp.ImageFormat.SRGB,
                                                       data=cv2.cvtColor(f, cv2.COLOR_BGR2RGB)), ts_ms)
            try:
                self.step(f.copy(), res)
            except Exception as e:  # keep the camera loop alive; surface the error in the UI
                self.status = {'state': 'error', 'msg': f'Processing error: {e}'}

    def step(self, f, res):
        now = time.time()
        if self.want_clear:
            with db() as c:
                c.execute('delete from frames where ts >= ?', (day_start(now),))
                c.execute('delete from scans where day = ?', (date.today().isoformat(),))
            self.want_clear, self.dirty = False, False
        r, mask, msg = None, None, None
        people = [i for i, m in enumerate(res.segmentation_masks or []) if (m.numpy_view() > .5).mean() > .01]
        if len(people) > 1:  # whose frames are whose is no longer certain
            self.last_seen = now
            self.end_session(keep=False)  # anything not yet confirmed is dropped
            msg = 'Two people in view — not measuring until one leaves'
            mask = res.segmentation_masks[0].numpy_view().squeeze()
        elif res.pose_landmarks:
            self.last_seen = now
            h, w = f.shape[:2]
            lms = res.pose_landmarks[0]
            pts = np.array([[l.x * w, l.y * h] for l in lms])
            vis = np.array([l.visibility or 0 for l in lms])
            world = np.array([[l.x, l.y, l.z] for l in res.pose_world_landmarks[0]])
            mask = res.segmentation_masks[0].numpy_view().squeeze()
            bare = self.prof.get('bare_only', True)
            ycc = cv2.cvtColor(f, cv2.COLOR_BGR2YCrCb) if bare else None
            facing = vis[[0, 2, 5, 9, 10]].min() > .5
            if facing and bare:  # your skin chroma, from your cheeks in this frame's light
                eye = np.linalg.norm(pts[2] - pts[5])
                patches = [ycc[int(y - eye * .15):int(y + eye * .15) + 1, int(x - eye * .15):int(x + eye * .15) + 1, 1:].reshape(-1, 2)
                           for x, y in ((pts[2] + pts[9]) / 2, (pts[5] + pts[10]) / 2)]
                self.skin = np.median(np.vstack(patches), axis=0)
            if facing and self.me is not None and (not self.gate.me or now - self.last_id > 2):  # re-checked every 2 s
                self.last_id = now
                eyes = np.linalg.norm(pts[2] - pts[5])  # a face seen near head-on: both ears, nose between the eyes
                self.identify(f, pts, vis[[7, 8]].min() > .5 and abs(pts[0][0] - (pts[2][0] + pts[5][0]) / 2) < .25 * eyes)
            r = measure.observe(pts, vis, world, mask, female=self.prof.get('sex') == 'female',
                                torso_ratio=self.prof.get('torso_ratio', .29), ycc=ycc, skin=self.skin)
            msg = r['problem']
            if f[mask > .5].mean() < 40:  # in the dark the person mask (and every width) is unreliable
                msg, r['obs'] = 'Too dark to measure', {}
            if self.gate.other:
                msg, r['obs'] = 'Not you — BodyLab only measures the profile owner', {}
            elif bare and self.skin is None:
                msg, r['obs'] = 'Face the camera for a moment so BodyLab can see your skin tone', {}
            skin = cat = box = None
            if r['obs'] and bare:  # clothes inflate every outline: the torso must be bare (underwear is fine)
                skin, cat, box = bare_skin(self.clothes, f, mask, pts)
                if (skin['torso'] or 0) < .6:  # unseen isn't bare
                    msg, r['obs'] = 'Clothes on — BodyLab measures bare skin (underwear is fine)', {}
                clothed = lambda k: skin[k] is not None and skin[k] < .5  # noqa: E731  legs: only when clothes are clearly seen
                covered = (['hips', 'thigh_l_w', 'thigh_r_w'] if clothed('thighs') else []) + (['calf_l_w', 'calf_r_w'] if clothed('shins') else [])
                for k in covered:  # shorts or trousers: those widths would be the clothes'
                    r['obs'].pop(k, None)
            if r['obs']:
                rec = {'ts': now, 'cam': self.cam, 'theta': round(r['theta'], 1), 'full': r['full'], 'obs': {k: round(v, 5) for k, v in r['obs'].items()},
                       'sil': silhouette(mask, pts, world, r['origin'], r['H'], cat=cat, box=box)} | ({'skin': skin} if skin else {})
                if self.prof.get('reference') and now - self.last_ref > 1:
                    try:  # a developer extra: never costs the frame
                        self.last_ref, rec['jpg'] = now, reference_crop(f, mask, r, skin)  # saved only if it's you
                    except Exception as e:
                        print('reference image failed:', e, flush=True)
                for x in self.gate.frame(rec):
                    self.keep(x)
                if self.gate.enrolled and not self.gate.me:
                    msg = msg or 'Face the camera for a moment so BodyLab knows it is you'
        elif now - self.last_seen > 1.5:  # nobody in view: the next person starts a new session
            self.end_session()
        self.status = self.next_status(r, msg)
        self.publish(f, res, mask, r)

    def identify(self, f, pts, frontal=True):
        span = max(np.linalg.norm(pts[7] - pts[8]), 20)  # ear to ear
        (x0, y0), (x1, y1) = (pts[0] - 1.6 * span).astype(int), (pts[0] + 1.6 * span).astype(int)
        crop = f[max(y0, 0):max(y1, 0), max(x0, 0):max(x1, 0)]
        if not crop.size or crop.mean() < 45:  # too dark to tell faces apart: decide nothing
            return
        emb = self.faces.embed(cv2.resize(crop, (320, 320), interpolation=cv2.INTER_CUBIC))
        if emb is None:
            return
        self.match = round(float(emb @ self.me), 2)
        for rec in self.gate.face(self.match, time.time(), frontal):
            self.keep(rec)

    def keep(self, rec):
        if 'jpg' in rec:
            save_reference(rec.pop('jpg'), rec)
        day = date.fromtimestamp(rec['ts'])
        if self.stored[0] != day:
            self.stored = [day, db().execute("select count(*) from frames where ts >= ? and coalesce(json_extract(data, '$.cam'), 0) = ?",
                                             (day_start(rec['ts']), self.cam)).fetchone()[0]]  # 400 a day per camera
            if day != date.today():
                return
            tidy()  # once a day
        if rec['ts'] - self.last_store < .5 or self.stored[1] >= FRAMES_PER_DAY:  # 2/s at most, 400/day is plenty
            return
        self.stored[1] += 1
        self.last_store, cap.dirty = rec['ts'], True  # the first camera fuses
        with db() as c:
            c.execute('insert into frames(ts, data) values (?, ?)', (rec['ts'], json.dumps(rec)))

    def fit(self, now):
        """Fuse today's frames into today's body estimate (one row in `scans` per day)."""
        self.last_fit, self.dirty = now, False
        with db() as c:
            frames = [json.loads(r[0]) for r in c.execute('select data from frames where ts >= ? order by ts', (day_start(now),))]
        raw = measure.estimate(frames) if frames else None
        self.today = {'frames': len(frames), 'angle': max((f['theta'] for f in frames), default=0)}
        if not raw or measure.implausible(raw):
            return
        with db() as c:
            c.execute('insert into scans(ts, raw, day) values (?, ?, ?) on conflict(day) do update set ts = excluded.ts, raw = excluded.raw',
                      (now, json.dumps(raw), date.fromtimestamp(now).isoformat()))
        if raw.get('torso_ratio') and abs(raw['torso_ratio'] - self.prof.get('torso_ratio', 0)) > .002:
            set_profile({**(get_profile() or {}), 'torso_ratio': round(raw['torso_ratio'], 4)})  # scales partial views

    def next_status(self, r, msg):
        today = getattr(self, 'today', {'frames': 0, 'angle': 0})
        st = {'state': 'idle', 'frames': today['frames'], 'angle': round(today['angle']), 'match': self.match}
        if not self.prof.get('height_cm'):
            return {**st, 'msg': 'Set your height in Settings to start'}
        if r is None:
            return {**st, 'state': 'adjust' if msg else 'idle', 'msg': msg or 'Watching — walk past any time; every view helps'}
        if r['obs'] and not msg:  # msg here: waiting for your face, not you, or too dark
            return {**st, 'state': 'scanning', 'msg': f"Measuring — {today['frames']} views today, turned up to {round(today['angle'])}°"}
        return {**st, 'state': 'adjust', 'msg': msg or 'Watching'}

    def publish(self, f, res, mask, r):
        if self.prof.get('preview', 'silhouette') == 'silhouette':
            # Private by default: a coarse mosaic shows the room is live; you're only a solid silhouette.
            h, w = f.shape[:2]
            f[:] = cv2.resize(cv2.resize(f, (max(w // 40, 1), max(h // 40, 1)), interpolation=cv2.INTER_AREA), (w, h),
                              interpolation=cv2.INTER_NEAREST) // 3
            if mask is not None:
                f[cv2.dilate((mask > .3).astype(np.uint8), np.ones((25, 25), np.uint8)) > 0] = (40, 26, 20)
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


def silhouette(mask, pts, world, origin, H, n=160, cat=None, box=None):
    """A scale-free outline of the body (n points along the mask's outer contour) plus the 2D and 3D landmarks,
    relative to the hip centre in units of standing height: what fitting a full body shape needs. No image. With
    the clothes segmenter's categories (cat, over the crop box): which outline points lie on clothes or hair ('off'),
    where the outline isn't the body's."""
    contours = cv2.findContours((mask > .5).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)[0]
    c = max(contours, key=cv2.contourArea)[:, 0].astype(float) if contours else np.zeros((1, 2))
    seg = np.r_[0, np.cumsum(np.linalg.norm(np.diff(c, axis=0, append=c[:1]), axis=1))]
    t = np.linspace(0, seg[-1], n, endpoint=False)
    outline = np.stack([np.interp(t, seg, np.r_[c[:, k], c[0, k]]) for k in (0, 1)], 1)
    out = {'c': ((outline - origin) / H).round(4).tolist(), 'l': ((pts - origin) / H).round(4).tolist(), 'w': world.round(3).tolist()}
    if cat is not None:  # a few px inside the outline: clothes (4) or hair (1) there means it's not the body's edge
        centre = np.array(c.mean(0))
        inner = outline + (centre - outline) / np.maximum(np.linalg.norm(centre - outline, axis=1, keepdims=True), 1) * 6
        q = np.clip(np.round(inner - [box[2], box[0]]).astype(int), 0, [cat.shape[1] - 1, cat.shape[0] - 1])
        out['off'] = np.isin(cat[q[:, 1], q[:, 0]], (1, 4)).astype(int).tolist()
    return out


def refit_loop():
    """Refit today's body shape (fit.py, PyTorch) as frames arrive: once there are enough, then whenever they've
    grown by a quarter. Skipped if the optional `fit` dependencies aren't installed (uv sync --group fit)."""
    while True:
        time.sleep(600)
        day = date.today().isoformat()
        with db() as c:
            n = c.execute('select count(*) from frames where ts >= ?', (day_start(time.time()),)).fetchone()[0]
            r = c.execute('select params from fits where day = ?', (day,)).fetchone()
        if n < 30 or (r and n < 1.25 * json.loads(r[0])['frames']):
            continue
        p = subprocess.run(['nice', sys.executable, str(ROOT / 'fit.py'), '--db', str(DB_PATH), '--day', day, '--store'],
                           capture_output=True, text=True, timeout=3600)
        if p.returncode:
            print('body fit failed:', (p.stderr.strip().splitlines() or ['?'])[-1], flush=True)
            if 'ModuleNotFoundError' in p.stderr:
                return


REF_CAP = int(os.environ.get('BODYLAB_REFERENCE_MB', 1000)) * 2 ** 20  # disk for reference images; the weakest go first


def reference_dir():
    return DB_PATH.parent / 'reference'


def person_box(mask, margin=.1):
    """(y0, y1, x0, x1) around the person mask, with a margin."""
    ys, xs = np.nonzero(mask > .5)
    if not len(ys):
        return None
    m, (h, w) = margin * (ys.max() - ys.min()), mask.shape[:2]
    return max(int(ys.min() - m), 0), min(int(ys.max() + m), h), max(int(xs.min() - m), 0), min(int(xs.max() + m), w)


def bare_skin(segmenter, f, mask, pts):
    """Share of skin (vs clothes) over the torso (shoulders to hips), thighs (hips to knees) and shins (knees to ankles),
    from MediaPipe's multiclass person segmenter on the person's crop. Also returns its category mask and the crop box."""
    box = person_box(mask)
    y0, y1, x0, x1 = box
    crop = np.ascontiguousarray(cv2.cvtColor(f[y0:y1, x0:x1], cv2.COLOR_BGR2RGB))
    cat = segmenter.segment(mp.Image(image_format=mp.ImageFormat.SRGB, data=crop)).category_mask.numpy_view().reshape(crop.shape[:2])
    p = pts - [x0, y0]

    def share(quad):
        region = np.zeros(cat.shape, np.uint8)
        cv2.fillConvexPoly(region, np.round(quad).astype(np.int32), 1)
        c = cat[region > 0]
        skin, cloth = (c == 2).sum(), (c == 4).sum()  # 2 = body skin, 4 = clothes
        return round(float(skin / (skin + cloth)), 2) if skin + cloth > .3 * len(c) else None  # too little seen to say
    def limb(a, b, half):  # a band along a bone, `half` its length to each side
        d = p[b] - p[a]
        n = np.array([-d[1], d[0]]) * half
        return share([p[a] + n, p[b] + n, p[b] - n, p[a] - n])
    legs = lambda a: min((x for x in a if x is not None), default=None)  # noqa: E731
    return {'torso': share(p[[11, 12, 24, 23]]), 'thighs': legs([limb(23, 25, .15), limb(24, 26, .15)]),
            'shins': legs([limb(25, 27, .08), limb(26, 28, .08)])}, cat, box


def reference_crop(f, mask, r, skin):
    """The person (10 % margin), background greyed out, at most 1600 px tall, as JPEG: you, not the room. With where
    the crop sits (box, scale, hip centre, px per standing height: maps the stored outline onto it) and a score:
    bare skin, sharpness and size, so better captures replace weaker ones."""
    box = person_box(mask)
    if not box:
        return None
    y0, y1, x0, x1 = box
    crop = f[y0:y1, x0:x1].copy()
    person = cv2.dilate((mask[y0:y1, x0:x1] > .5).astype(np.uint8), np.ones((9, 9), np.uint8)) > 0
    crop[~person] = 128
    k = min(1, 1600 / crop.shape[0])
    sharp = cv2.Laplacian(cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY), cv2.CV_64F)[person].var()
    bare = sum(.5 if skin.get(p) is None else skin[p] for p in ('torso', 'thighs', 'shins')) / 3 if skin else .5
    score = round(bare + .5 * min(1, sharp / 300) + .5 * min(1, crop.shape[0] * k / 1600), 3)
    jpg = cv2.imencode('.jpg', cv2.resize(crop, None, fx=k, fy=k, interpolation=cv2.INTER_AREA), [cv2.IMWRITE_JPEG_QUALITY, 88])[1].tobytes()
    return jpg, {'box': [y0, x0], 'k': k, 'origin': [round(float(v), 1) for v in r['origin']], 'H': round(float(r['H']), 1), 'score': score}


def save_reference(ref, rec):
    """Keep reference images, each with its frame's record, up to REF_CAP on disk: beyond it the lowest-scoring capture
    of the most-represented turn and camera goes, so every angle stays covered. Local only (data/, gitignored)."""
    if not ref:
        return
    jpg, info = ref
    d = reference_dir()
    d.mkdir(exist_ok=True)
    name = f"{date.fromtimestamp(rec['ts'])}_{int(abs(rec['theta']) // 10) * 10:02d}_{int(rec['ts'])}_cam{rec.get('cam', 0) + 1}"
    (d / f'{name}.jpg').write_bytes(jpg)
    (d / f'{name}.json').write_text(json.dumps(rec | {'ref': info}))
    files = list(d.glob('*.jpg'))
    total = sum(p.stat().st_size + p.with_suffix('.json').stat().st_size for p in files if p.with_suffix('.json').exists())
    if total <= REF_CAP:
        return
    buckets = {}
    for p in files:  # bucket: turn and camera, from the name (day_turn_ts_camN)
        buckets.setdefault(p.stem.split('_')[1] + p.stem[-4:], []).append(p)
    score = lambda p: json.loads(p.with_suffix('.json').read_text()).get('ref', {}).get('score', 0) if p.with_suffix('.json').exists() else 0  # noqa: E731
    while total > REF_CAP:
        old = min(max(buckets.values(), key=len), key=score)
        buckets = {k: [p for p in v if p != old] for k, v in buckets.items()}
        total -= old.stat().st_size + (old.with_suffix('.json').stat().st_size if old.with_suffix('.json').exists() else 0)
        old.unlink(missing_ok=True)
        old.with_suffix('.json').unlink(missing_ok=True)


def tidy():
    """Keep BodyLab's disk use bounded: raw frames for 45 days, and its own log under 5 MB."""
    with db() as c:  # freed pages are reused, so the file stops growing once it holds 45 days
        c.execute('delete from frames where ts < ?', (time.time() - KEEP_FRAMES_DAYS * 86400,))
    out = os.fstat(1)
    if stat.S_ISREG(out.st_mode) and out.st_size > 5_000_000:
        os.ftruncate(1, 0)


def day_start(ts):
    return datetime.combine(date.fromtimestamp(ts), datetime.min.time()).timestamp()


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
    for key in ('camera_url', 'camera2_url'):
        if key not in b:
            continue
        url = str(b[key]).strip()
        if url and urlparse(url).scheme not in ('http', 'https', 'rtsp'):
            raise ValueError('Camera URL must start with http://, https:// or rtsp://')
        prof[key] = url
    if 'skin' in b:
        if not re.fullmatch(r'#[0-9a-fA-F]{6}', str(b['skin'])):
            raise ValueError('Skin tone must be a #rrggbb colour')
        prof['skin'] = b['skin']
    for k, pattern in (('hair', r'[a-z0-9]{1,20}'), ('hair_color', r'#[0-9a-fA-F]{6}'), ('eye_color', r'[a-z]{1,20}')):
        if k in b:
            if not re.fullmatch(pattern, str(b[k])):
                raise ValueError(f'Invalid {k}')
            prof[k] = b[k]
    for key in ('rotate', 'rotate2'):
        if key in b:
            prof[key] = int(b[key]) if int(b[key]) in ROTATE else 0
    for k in ('bare_only', 'clothes', 'reference'):
        if k in b:
            prof[k] = bool(b[k])
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
    elif kind in ('daily', 'reference'):
        data = daily_data(b) if kind == 'daily' else {k: num(b[k], lo, hi, k) for k, (lo, hi) in measure.REFERENCE.items()
                                                       if b.get(k) not in (None, '')}
        if not data:
            raise ValueError('Enter at least one value')
    else:
        raise ValueError('Unknown log kind')
    with db() as c:
        c.execute('insert into logs(ts, kind, data) values (?, ?, ?)', (ts_of(b), kind, json.dumps(data)))


def post_sync(b):
    """Apple Health batch from the iOS app. Upserts by Health ID/day, so re-syncing never duplicates.
    Height, sex, birth date (Apple Health) and units (the phone) are the source of truth for the profile."""
    hp, prof = b.get('profile') or {}, get_profile() or {}
    health = {'height_cm': hp.get('height_cm') and num(hp['height_cm'], 100, 250, 'height'),
              'sex': hp.get('sex') if hp.get('sex') in ('male', 'female') else None,
              'birth_date': hp.get('birth_date') and datetime.strptime(hp['birth_date'], '%Y-%m-%d').date().isoformat(),
              'units': hp.get('units') if hp.get('units') in ('metric', 'imperial') else None}
    if any(v and prof.get(k) != v for k, v in health.items()):
        set_profile({**prof, **{k: v for k, v in health.items() if v}})
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


def post_predict(b):
    """What you'd look like after `weeks` of a plan, from your latest body estimate and weigh-in."""
    st = app_state()
    prof, scan = st['profile'] or {}, (st['scans'] or [None])[-1]
    if not scan or 'bf' not in scan['m']:
        raise ValueError('Needs a body estimate first: walk past the camera')
    m = dict(scan['m'])
    m['weight'] = measure.nearest(weights(st['logs']), time.time(), 90) or m.get('weight')
    if not m['weight']:
        raise ValueError('Needs a recent weigh-in (log one, or sync a smart scale through Apple Health)')
    sex = prof.get('sex', 'male')
    if 'goal_weight' in b:  # your goal: that weight at that body fat
        w, bf = num(b['goal_weight'], 30, 300, 'goal weight'), num(b.get('goal_bf', measure.RECOMMENDED_BF[sex]), 3, 60, 'goal body fat')
        out = {'m': measure.at_composition(m, sex, w * bf / 100, w * (1 - bf / 100), prof['height_cm'],
                                           measure.personal_slopes(st['scans'], sex)), 'warning': None}
    else:
        out = measure.predict(m, sex, int(num(b.get('weeks', 12), 1, 104, 'weeks')), num(b.get('kcal', 0), -1500, 1500, 'kcal'),
                              num(b.get('strength', 0), 0, 7, 'strength'), num(b.get('protein', 1.2), .5, 3.5, 'protein'),
                              prof['height_cm'], measure.personal_slopes(st['scans'], sex))
    dims = dict(scan['dims'])  # the predicted body: each part's width and depth scaled by its circumference change
    for part in [*measure.TORSO, *measure.LIMBS]:
        if part in m and part in out['m'] and m[part]:
            for k in (f'{part}_w', f'{part}_d'):
                if k in dims:
                    dims[k] = round(dims[k] * out['m'][part] / m[part], 2)
    if 'shoulders' in m and 'shoulders_w' in dims:
        dims['shoulders_w'] = round(dims['shoulders_w'] * out['m']['shoulders'] / m['shoulders'], 2)
    kcal, source = measure.maintenance(st['logs'], st['scans'])
    return {**out, 'dims': dims, 'start': {k: m[k] for k in ('weight', 'bf') if k in m},
            'maintenance': kcal and round(kcal), 'maintenance_source': source}


POST = {'/api/profile': post_profile, '/api/logs': post_log, '/api/goals': post_goal,
        '/api/clear-today': lambda b: cap.clear_today(), '/api/sync': post_sync,
        '/api/predict': post_predict}
LOCAL = ('localhost', '127.0.0.1')
TOKEN_PATHS = ('/api/sync', '/api/face', '/api/logs')  # what the paired iPhone may call


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

    def token_ok(self):
        auth = self.headers.get('Authorization')
        return auth is not None and hmac.compare_digest(auth.encode(), f'Bearer {sync_token()}'.encode())

    def do_GET(self):
        if not (self.token_ok() or self.allowed()):  # the paired iPhone, or you in the browser
            return
        path = urlparse(self.path).path
        if path == '/':  # this Mac's developer console; the app (iPhone) asks for ?app=1
            path = '/static/index.html' if 'app' in parse_qs(urlparse(self.path).query) else '/static/console.html'
        if re.fullmatch(r'/api/reference/[\w.-]+\.jpg', path) and (reference_dir() / path.rsplit('/', 1)[1]).exists():
            return self.send(200, (reference_dir() / path.rsplit('/', 1)[1]).read_bytes(), 'image/jpeg')
        if path == '/api/state':
            return self.reply(app_state())
        if path == '/api/status':
            return self.reply({**cap.status, 'rev': data_rev()} | ({'cam2': cap2.status} if cap2.url else {}))
        if path == '/api/live.mjpg':
            return self.stream(cap2 if parse_qs(urlparse(self.path).query).get('cam') == ['2'] else cap)
        if path == '/api/skin.jpg' and (DB_PATH.parent / 'skin.jpg').exists():  # your skin, baked from the cameras (fit.py)
            return self.send(200, (DB_PATH.parent / 'skin.jpg').read_bytes(), 'image/jpeg')
        if path == '/api/skin' and (DB_PATH.parent / 'skin.bin').exists():  # bone weights for posing fitted bodies (fit.py)
            return self.send(200, (DB_PATH.parent / 'skin.bin').read_bytes(), 'application/octet-stream')
        if path.startswith('/api/fit/'):  # a day's fitted body: rest mesh (int16 mm) + stance bone transforms (fit.py)
            r = db().execute('select mesh from fits where day = ?', (path.rsplit('/', 1)[1],)).fetchone()
            return self.send(200, r[0], 'application/octet-stream') if r else self.send(404, b'no fit', 'text/plain')
        if path == '/api/face.jpg' and face_files()[0].exists():
            return self.send(200, face_files()[0].read_bytes(), 'image/jpeg')
        f = (ROOT / path.lstrip('/')).resolve()
        if path.startswith('/static/') and (ROOT / 'static') in f.parents and f.is_file():
            return self.send(200, f.read_bytes(), mimetypes.guess_type(f.name)[0] or 'application/octet-stream')
        self.send(404, b'not found', 'text/plain')

    def do_POST(self):
        path = urlparse(self.path).path
        auth = self.headers.get('Authorization')
        if auth is not None and path in TOKEN_PATHS:  # the iPhone, via `tailscale serve`
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

    def stream(self, c):
        self.send_response(200)
        self.send_header('Content-Type', 'multipart/x-mixed-replace; boundary=frame')
        self.send_header('Cache-Control', 'no-store')
        self.end_headers()
        last = None
        try:
            while True:
                j = c.jpeg
                if j is None or j is last:
                    time.sleep(.03)
                    continue
                last = j
                self.wfile.write(b'--frame\r\nContent-Type: image/jpeg\r\nContent-Length: %d\r\n\r\n%s\r\n' % (len(j), j))
        except (BrokenPipeError, ConnectionResetError):
            pass


if __name__ == '__main__':
    init_db()
    tidy()
    ensure_model()
    ensure_model(*CLOTHES_MODEL)
    jpg, meta = face_files()
    if meta.exists() and 'fringe' not in json.loads(meta.read_text()).get('hair', {}):  # photo analysed by an older version
        try:
            post_face(jpg.read_bytes())
        except ValueError as e:
            print(f'Face photo needs replacing: {e}', flush=True)
    TS_HOST, TS_OWNER = tailnet()
    cap, cap2 = Capture(), Capture(1)
    threading.Thread(target=refit_loop, daemon=True).start()
    print(f'BodyLab → http://localhost:{PORT}' + (f'  ·  tailnet: https://{TS_HOST}:{SYNC_PORT}' if TS_HOST else ''), flush=True)
    ThreadingHTTPServer(('127.0.0.1', PORT), Handler).serve_forever()
