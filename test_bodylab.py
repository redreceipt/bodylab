"""Self-check: synthetic 175 cm figure drawn at 4 px/cm, plus fusion, insights and sync. Run: uv run python test_bodylab.py"""
import math
import time

import cv2
import numpy as np

import measure as M

S = 4  # px per cm; figure spans y=250 (head top) .. 950 (floor) = 700 px = 175 cm
img = np.zeros((1000, 800), np.uint8)
cv2.ellipse(img, (400, 290), (36, 40), 0, 0, 360, 255, -1)          # head
cv2.line(img, (400, 320), (400, 390), 255, 12 * S)                    # neck 12 cm
cv2.rectangle(img, (330, 380), (470, 560), 255, -1)                   # chest/waist 35 cm
cv2.rectangle(img, (320, 560), (480, 650), 255, -1)                   # hips 40 cm
pts = np.zeros((33, 2))
pts[[7, 8]] = (380, 290), (420, 290)
pts[[11, 12]] = (460, 395), (340, 395)                                # person's left = image right
pts[[23, 24]] = (435, 600), (365, 600)
pts[[25, 26]] = (440, 780), (360, 780)
pts[[27, 28]] = (440, 930), (360, 930)
for s, sign in ((11, 1), (12, -1)):                                   # arms 40 deg out: 10 cm upper, 7.5 cm forearm
    sh = pts[s]
    el = sh + (sign * 120 * math.sin(math.radians(40)), 120 * math.cos(math.radians(40)))
    wr = el + (sign * 110 * math.sin(math.radians(40)), 110 * math.cos(math.radians(40)))
    pts[s + 2], pts[s + 4] = el, wr
    cv2.line(img, tuple(sh.astype(int)), tuple(el.astype(int)), 255, 10 * S)
    cv2.line(img, tuple(el.astype(int)), tuple(wr.astype(int)), 255, 30)
for h, k, a in ((23, 25, 27), (24, 26, 28)):
    cv2.line(img, tuple(pts[h].astype(int)), tuple(pts[k].astype(int)), 255, 70)   # thighs 17.5 cm (touching)
    cv2.line(img, tuple(pts[k].astype(int)), tuple(pts[a].astype(int)), 255, 48)   # calves 12 cm
    cv2.rectangle(img, (int(pts[a][0]) - 20, 930), (int(pts[a][0]) + 20, 950), 255, -1)  # feet to the floor

world = np.zeros((33, 3))
world[[11, 12, 23, 24], 0] = .2, -.2, .1, -.1   # shoulders and hips square to the camera: yaw 0
r = M.observe(pts, np.ones(33), world, img / 255.0)
assert not r['problem'] and r['full'] and r['theta'] == 0, r['problem']
obs = r['obs']
cm = {k: v * 175 for k, v in obs.items()}
expect = {'neck': 12, 'chest': 35, 'waist': 35, 'hips': 40, 'arm_l_w': 10, 'arm_r_w': 10,
          'thigh_l_w': 17.5, 'thigh_r_w': 17.5, 'calf_l_w': 12, 'calf_r_w': 12, 'lv_shoulder': 138.75}
for k, v in expect.items():
    assert abs(cm[k] - v) < 1.0, (k, cm[k], v)
assert cm['shoulders_w'] > cm['chest']

# Same figure, but only from the hips up (feet out of frame): scaled by the torso length learned above.
crop = M.observe(pts, np.ones(33), world, img[:700] / 255.0, torso_ratio=obs['torso_ratio'])
assert not crop['full'] and abs(crop['obs']['waist'] * 175 - 35) < 1.0

# Arms down against the body: the slices they'd widen are skipped; the rest of the frame still counts.
bad, bpts = img.copy(), pts.copy()
bpts[[13, 14, 15, 16]] = (485, 500), (315, 500), (490, 600), (310, 600)
for s, e, w in ((11, 13, 15), (12, 14, 16)):
    cv2.line(bad, tuple(bpts[s].astype(int)), tuple(bpts[e].astype(int)), 255, 40)
    cv2.line(bad, tuple(bpts[e].astype(int)), tuple(bpts[w].astype(int)), 255, 30)
arms_down = M.observe(bpts, np.ones(33), world, bad / 255.0)['obs']
assert 'waist' not in arms_down and 'chest' not in arms_down and 'neck' in arms_down

# Many frames from random angles: the ellipse fit recovers width AND depth that no single view shows.
rng = np.random.default_rng(1)
frames = []
for t in rng.uniform(0, 90, 60):
    th = math.radians(t)
    width = 2 * math.sqrt((.1 * math.cos(th)) ** 2 + (.075 * math.sin(th)) ** 2) * rng.normal(1, .02)  # waist 35x26 cm (H=175)
    frames.append({'theta': t, 'full': True, 'obs': {**obs, 'waist': width}})
frames[0]['obs'] = {**obs, 'waist': .3}  # one misread frame
est = M.estimate(frames)
assert abs(est['waist_w'] * 175 - 35) < .7 and abs(est['waist_d'] * 175 - 26.25) < .7, (est['waist_w'] * 175, est['waist_d'] * 175)
assert abs(est['neck_d'] / est['neck_w'] - 1) < 1e-9  # front-on only: depth falls back to the typical ratio
m, dims = M.derive(est, {'height_cm': 175, 'sex': 'male'})
assert M.implausible(est) is None and 'bf' in m
assert M.implausible({**est, 'neck_w': .13, 'hips_w': .12}) == ['neck_w', 'hips_w']  # e.g. the 70 cm 'neck' misread

# Clothing check: a skin-coloured torso line passes, a blue-shirt line doesn't.
ycc = np.zeros((100, 100, 3), np.uint8)
ycc[:50] = (150, 150, 110)   # skin chroma (Cr 150, Cb 110)
ycc[50:] = (90, 110, 160)    # blue shirt
assert M.skin_share(ycc, (150, 110), (10, 20), (90, 20)) == 1 and M.skin_share(ycc, (150, 110), (10, 80), (90, 80)) == 0

# Hair: a face with dark hair down to the shoulders reads as long; a thin cap of hair as a buzz cut.
face = np.zeros((478, 2))
face[:, 0], face[:, 1] = np.linspace(100, 200, 478), 150
face[10], face[152] = (150, 100), (150, 220)
photo = np.full((400, 300, 3), 200, np.uint8)
long_hair = np.zeros((400, 300))
long_hair[60:330, 70:230] = 1
long_hair[100:220, 100:200] = 0  # the face itself
photo[long_hair > 0] = (20, 40, 60)  # BGR dark brown
h = M.hair_style(long_hair, photo, face)
assert h == {'style': 'long01', 'color': '#3c2814'}, h  # uniform hair: lit-strand colour = the colour
buzz = np.zeros((400, 300))
buzz[92:100, 100:200] = 1
assert M.hair_style(buzz, photo, face)['style'] == 'short04'

# Formulas: circle perimeter, published Navy example (~17.2% for 178/38/86 cm male).
assert abs(M.ellipse(10, 10) - math.pi * 10) < 1e-9
assert abs(M.navy_bf('male', 178, 38, 86, 0) - 17.2) < .1
assert 'whtr' in m

# Insights: steady 0.1 %/day drop toward a goal projects an ETA; weekly table has 8 rows.
now = time.time()
scans = [{'ts': now - d * 86400, 'm': {'bf': 20 - (30 - d) * .1}} for d in range(30, -1, -3)]
logs = [{'kind': 'workout', 'ts': now - d * 86400, 'minutes': 45, 'rpe': 7, 'focus': ['arms']} for d in range(0, 28, 2)]
ins = M.insights(scans, logs, {'bf': {'target': 15, 'start': 20, 'ts': now}}, now)
g = ins['goals']['bf']
assert len(ins['weeks']) == 8 and abs(g['rate_per_week'] + .7) < .01 and g['eta'] and .5 < g['progress'] < .7, g
assert ins['parts']['arm_l']['workouts'] == 14 and ins['parts']['waist']['workouts'] == 0
print('ok', {k: round(cm[k], 1) for k in expect})

# Apple Health sync: resending the same window updates rows instead of duplicating them.
import tempfile  # noqa: E402
from pathlib import Path  # noqa: E402

import app  # noqa: E402

app.DB_PATH = Path(tempfile.mkdtemp()) / 'test.db'
app.init_db()
batch = {'days': [{'date': '2026-11-20', 'weight': 80.5, 'sleep': 7.2}, {'date': '2026-11-21'}],
         'workouts': [{'id': 'W1', 'start': now, 'minutes': 45, 'type': 'Strength', 'focus': ['full body'], 'rpe': 7.4}]}
assert app.post_sync(batch)['saved'] == 2  # empty day skipped
batch['days'][0]['weight'] = 80.1
app.post_sync(batch)
logs = app.get_logs()
assert len(logs) == 2 and [l['weight'] for l in logs if l['kind'] == 'daily'] == [80.1], logs

# Health onboarding fills missing profile fields but never overwrites what the user set.
app.post_sync({'profile': {'height_cm': 180.3, 'sex': 'male', 'birth_date': '1990-05-02'}})
assert app.get_profile() == {'height_cm': 180.3, 'sex': 'male', 'birth_date': '1990-05-02'}
app.post_profile({'height_cm': 179})
app.post_sync({'profile': {'height_cm': 182}})
assert app.get_profile()['height_cm'] == 179

# Energy balance: 500 kcal/day deficit for 28 days predicts ~1.8 kg lost.
eb = [{'kind': 'daily', 'ts': now - d * 86400, 'kcal': 2000, 'active': 600, 'basal': 1900} for d in range(1, 28)]
s = M.insights([], eb, {}, now)['summary']
assert s['balance'] == -500 and abs(s['predicted_kg'] + 1.82) < .01, s

# Gaps don't mislead: a half-logged day and today's running totals are dropped, and too few days means no balance.
eb[3]['kcal'] = 700  # forgot to log dinner
eb.append({'kind': 'daily', 'ts': now, 'kcal': 400, 'active': 100, 'basal': 600, 'steps': 900})  # today, not over yet
i = M.insights([], eb, {}, now)
assert i['summary']['balance'] == -500 and i['summary']['balance_days'] == 26, i['summary']
assert any('partly logged' in g for g in i['gaps']) and i['health']['steps']['days'] == 0
assert M.insights([], eb[:10], {}, now)['summary']['balance'] is None  # 9 logged days of 28
# Weight change uses the trend, not the last noisy reading.
wl = [{'kind': 'daily', 'ts': now - d * 86400, 'weight': 80 - (27 - d) * .05 + (1.5 if d == 0 else 0)} for d in range(27, -1, -3)]
assert abs(M.insights([], wl, {}, now)['summary']['changes']['weight'] + 1.35) < .1  # true -1.35; last-minus-first says +0.15
print('sync ok')
