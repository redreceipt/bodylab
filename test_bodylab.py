"""Self-check: synthetic 175 cm figure drawn at 4 px/cm, plus fusion, insights and sync. Run: uv run python test_bodylab.py"""
import json
import math
import time

import cv2
import numpy as np

import measure as M

S = 4  # px per cm; figure spans y=250 (head top) .. 950 (floor) = 700 px = 175 cm
img = np.zeros((1000, 800), np.uint8)
cv2.ellipse(img, (400, 290), (36, 40), 0, 0, 360, 255, -1)          # head
cv2.line(img, (400, 320), (400, 390), 255, 12 * S)                    # neck 12 cm
cv2.rectangle(img, (330, 380), (470, 575), 255, -1)                   # chest/waist 35 cm
cv2.rectangle(img, (320, 575), (480, 650), 255, -1)                   # hips 40 cm
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
world[[11, 12], 1], world[[25, 26], 1], world[[27, 28], 1] = -.5, .45, .9  # standing: shoulders up (y down), knees, ankles
world[[25, 27], 0], world[[26, 28], 0] = .1, -.1
seated = world.copy()
seated[[25, 26], 1], seated[[25, 26], 2] = 0, -.45  # thighs forward, level with the hips
seated[[27, 28], 1], seated[[27, 28], 2] = .45, -.45  # shins down
assert not M.bent(world) and M.bent(seated)
facing_seat = pts.copy()
facing_seat[[25, 26], 1] = pts[[23, 24], 1] + 40  # knees only a little below the hips: thighs toward the camera
assert not M.foreshortened(pts) and M.foreshortened(facing_seat)
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
    frames.append({'theta': t, 'full': True, 'obs': {**obs, 'waist': width, 'neck': obs['neck'] * (1.3 if t > 30 else 1)}})  # jaw in side views
frames[0]['obs'] = {**obs, 'waist': .3}  # one misread frame
est = M.estimate(frames)
assert abs(est['waist_w'] * 175 - 35) < .7 and abs(est['waist_d'] * 175 - 26.25) < .7, (est['waist_w'] * 175, est['waist_d'] * 175)
assert abs(est['neck_w'] / obs['neck'] - 1) < .02 and abs(est['neck_d'] / est['neck_w'] - M.TORSO['neck']) < 1e-9  # front views only
m, dims = M.derive(est, {'height_cm': 175, 'sex': 'male'})
assert M.implausible(est) is None and 'bf' in m
assert M.implausible({**est, 'neck_w': .13, 'hips_w': .12}) == ['neck_w', 'hips_w']  # e.g. the 70 cm 'neck' misread
# A body-shape fit's tape measurements win over ellipses, and give body fat before the camera has a waist.
m2, _ = M.derive({'neck_w': .07, 'hips_w': .2, 'tape': {'neck': 39.6, 'waist': 88.5, 'hips': 105.8}}, {'height_cm': 182.9, 'sex': 'male'})
assert m2['waist'] == 88.5 and abs(m2['bf'] - M.navy_bf('male', 182.9, 39.6, 88.5, 105.8)) < .01, m2
# With a smart scale's reading that week, body fat is the mean of the two estimates; each stays visible.
m3, _ = M.derive({'neck_w': .07, 'hips_w': .2, 'tape': {'neck': 39.6, 'waist': 88.5, 'hips': 105.8}}, {'height_cm': 182.9, 'sex': 'male'}, 80, 20.0)
assert m3['bf_cam'] == m2['bf'] and m3['bf_scale'] == 20 and abs(m3['bf'] - (m2['bf'] + 20) / 2) < .01 and abs(m3['lean_kg'] - 80 * (1 - m3['bf'] / 100)) < .01, m3

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
assert h == {'style': 'long01', 'color': '#34281c', 'fringe': False}, h  # uniform #3c2814 hair, 40 % less saturated
bangs = long_hair.copy()
bangs[100:150, 100:200] = 1  # hair down over the forehead
assert M.hair_style(bangs, photo, face)['fringe']
buzz = np.zeros((400, 300))
buzz[92:100, 100:200] = 1
assert M.hair_style(buzz, photo, face)['style'] == 'short04'

# Formulas: circle perimeter, published Navy example (~17.2% for 178/38/86 cm male).
assert abs(M.ellipse(10, 10) - math.pi * 10) < 1e-9
assert abs(M.navy_bf('male', 178, 38, 86, 0) - 17.2) < .1
assert 'whtr' in m
# A goal: 84 kg at 12 % from 80 kg at 20 %: 6 kg less fat, 10 kg more lean; parts follow their cm/kg.
g = M.at_composition({'weight': 80, 'bf': 20, 'chest': 100, 'waist': 90}, 'male', 84 * .12, 84 * .88, 180)
assert abs(g['fat_kg'] - 10.08) < .01 and abs(g['chest'] - (100 + .5 * -5.92 + .8 * 9.92)) < .01 and abs(g['waist'] - (90 - 1.1 * 5.92 + .1 * 9.92)) < .01, g

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

# Apple Health and the phone are the source of truth for the profile.
app.post_sync({'profile': {'height_cm': 180.3, 'sex': 'male', 'birth_date': '1990-05-02', 'units': 'imperial'}})
assert app.get_profile() == {'height_cm': 180.3, 'sex': 'male', 'birth_date': '1990-05-02', 'units': 'imperial'}
app.post_sync({'profile': {'height_cm': 182}})
assert app.get_profile()['height_cm'] == 182

# Only your frames count: waiting until your face matches, dropped when someone else's face shows up.
g, R = app.Gate(), lambda ts: {'ts': ts}
a, b = R(0), R(5)
assert g.frame(a) == [] and g.face(.8, 1) == [] and g.face(.8, 2) == [a] and g.frame(b) == [] and g.end() == [b]  # you, then you left
g.frame(R(0))
assert g.face(.1, 1) == [] and g.face(.1, 2) == [] and g.face(.1, 3) == [] and g.other and g.end() == []  # her: nothing kept
d = R(0)
g.frame(d)
assert g.face(.9, 1) == [] and g.face(.9, 2) == [d] and g.frame(R(5)) == [] and [g.face(.05, 6 + i) for i in range(3)] and g.end() == []  # you, then her
g.frame(R(0))
assert g.face(.3, 1) == [] and g.end() == []  # never clearly you: nothing kept
g.frame(R(0))
assert g.face(.5, 1) == [] and g.end() == []  # one lucky match isn't enough
old, new, late = R(0), R(100), R(200)
assert [g.frame(x) for x in (old, new)] and g.face(.8, 120) == [] and g.face(.8, 130) == [new] and g.frame(late) == [] and g.end() == []
# ^ frames over a minute from any match (her, back to the camera, just before or after you) never count
g.frame(R(0))
assert [g.face(.05, 1 + i, frontal=False) for i in range(5)] and not g.other  # a turned face never counts against you
g.end()
g.frame(R(0))
[g.face(.05, 1 + i) for i in range(3)]
assert g.other and g.frame(R(5)) == [] and g.face(.8, 6) == [] and g.face(.8, 7) == [] and not g.other  # looking down, then clearly you
assert g.frame(R(8)) == [] and g.end() == [R(8)]  # counts from then on
assert app.Gate(enrolled=False).frame('x') == ['x']  # no face photo yet: anyone alone counts

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
# Today's checklist: a weigh-in and a short food log show as done / incomplete.
d = [{'kind': 'daily', 'ts': now - k * 86400, 'kcal': 2200, 'steps': 8000} for k in range(1, 8)]
d += [{'kind': 'daily', 'ts': now, 'kcal': 600, 'weight': 80}]
day = M.insights([], d, {}, now)['days'][-1]
assert day['weight'] and not day['food'] and not day['sleep'] and not day['scan'] and day['kcal'] == 600, day
# Predictions: a 500 kcal/day deficit for 12 weeks loses ~5 kg, mostly fat; lifting + protein keeps more lean mass.
start = {'weight': 90, 'bf': 25, 'waist': 95, 'hips': 102, 'chest': 104, 'neck': 40, 'arm_l': 35}
rest = M.predict(start, 'male', 12, -500, 0, 1.0, 180)
lift = M.predict(start, 'male', 12, -500, 4, 2.0, 180)
lost = 90 - rest['m']['weight']
assert 4 < lost < 6.5 and rest['m']['bf'] < 25 and rest['m']['waist'] < 95, rest['m']
assert lift['m']['lean_kg'] > rest['m']['lean_kg'] and lift['m']['bf'] < rest['m']['bf'] and len(rest['path']) == 13
crash = M.predict({'weight': 63, 'bf': 28}, 'female', 30, -800, 3, 1.8, 165)
assert crash['warning'] and crash['m']['bf'] >= 11.5 and M.predict(start, 'male', 12, -500, 4, 2.0, 180)['warning'] is None
# Skin vs clothes on the person's crop (segmenter stubbed: MediaPipe returns an (h, w, 1) category mask).
class Seg:
    def segment(self, image):
        h, w = image.numpy_view().shape[:2]
        cat = np.full((h, w, 1), 4, np.uint8)  # clothes everywhere...
        cat[:h // 2] = 2  # ...but bare skin above the middle
        return type('R', (), {'category_mask': type('M', (), {'numpy_view': lambda self: cat})()})()
frame = np.zeros((1000, 800, 3), np.uint8)
skin, cat, box = app.bare_skin(Seg(), frame, img / 255.0, pts)
assert skin['thighs'] == 0 and skin['shins'] in (0, None) and 0 < skin['torso'] < 1 and cat.shape == (box[1] - box[0], box[3] - box[2]), (skin, cat.shape, box)
ref_jpg, ref_info = app.reference_crop(frame, img / 255.0, {'origin': pts[23:25].mean(0), 'H': 700.0}, skin)
assert cv2.imdecode(np.frombuffer(ref_jpg, np.uint8), cv2.IMREAD_COLOR) is not None and ref_info['k'] == 1 and 0 < ref_info['score'] < 2
app.REF_CAP = 3 * (len(ref_jpg) + 400)  # room for about three
for i, (sc, th) in enumerate(((.5, 12.), (1.5, 14.), (.9, 16.), (1.2, 50.))):  # over the cap: the weakest of the fullest turn goes
    app.save_reference((ref_jpg, ref_info | {'score': sc}), {'ts': now + i, 'theta': th, 'cam': 0})
kept = sorted(json.loads(p.read_text())['ref']['score'] for p in app.reference_dir().glob('*.json'))
assert kept == [.9, 1.2, 1.5], kept
bent_over = world.copy()
bent_over[[11, 12], 2] = -.45  # shoulders forward over the hips
assert M.bent(bent_over)
# Per outline point: on clothes or hair, it isn't the body's edge (the segmenter's categories over the crop box).
cat_full = np.full((1000, 800), 2, np.uint8)
cat_full[560:] = 4  # clothes below the waist
sil_c = app.silhouette(img / 255.0, pts, world, pts[23:25].mean(0), 700.0, cat=cat_full, box=(0, 1000, 0, 800))
c_y = np.array(sil_c['c'])[:, 1] * 700 + pts[23:25].mean(0)[1]
assert len(sil_c['off']) == 160 and all(o == (y > 575) for o, y in zip(sil_c['off'], c_y) if abs(y - 560) > 15)
assert all(app.silhouette(img / 255.0, pts, world, pts[23:25].mean(0), 700.0, cat=np.zeros((1000, 800), np.uint8), box=(0, 1000, 0, 800))['off'])  # the mask took in background
# The stored silhouette is scale-free: same body twice the size gives the same outline.
sil = app.silhouette(img / 255.0, pts, world, pts[23:25].mean(0), 700.0)
big = app.silhouette(cv2.resize(img, None, fx=2, fy=2) / 255.0, pts * 2, world, pts[23:25].mean(0) * 2, 1400.0)
assert len(sil['c']) == 160 and abs(np.array(sil['c']).min(0) - np.array(big['c']).min(0)).max() < .01
# One impossible Apple Health value (another app's) is skipped, not the whole sync; typed in, it's an error.
assert app.daily_data({'water': 30, 'kcal': 2000}, strict=False) == {'kcal': 2000.0}
try:
    app.daily_data({'water': 30})
    raise AssertionError('typed water of 30 L accepted')
except ValueError:
    pass
print('sync ok')

# Body-shape fit helpers (fit.py; the fit itself needs the optional PyTorch group).

import fit as FIT  # noqa: E402
a = np.linspace(-math.pi, 0, 30)
cut = np.r_[np.c_[.2 * np.cos(a), .4 * np.sin(a)], np.c_[np.linspace(.2, .15, 6), np.linspace(0, .3, 6)],  # round top, slanted sides
            np.c_[np.linspace(.15, -.15, 9), np.full(9, .3)], np.c_[np.linspace(-.15, -.2, 6), np.linspace(.3, 0, 6)]]
assert FIT.borders(cut) == [-np.inf, np.inf, -np.inf, .3]  # only the flat bottom is the image's edge
rz = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1.]])
pts = np.random.default_rng(2).normal(size=(6, 3))
assert np.allclose(FIT.kabsch(pts, pts @ rz.T), rz)
meta = json.loads((FIT.ROOT / 'static/assets/body.json').read_text())
t, off, n = meta['arrays']['male']
tm = np.frombuffer((FIT.ROOT / 'static/assets/body.bin').read_bytes(), np.float32, n, off).reshape(-1, 3)
J = {k: np.array([v[0], -v[2], v[1]]) / 100 for k, v in meta['joints']['male'].items()}  # Anny frame, metres
limbs = {f'{k}_{s}': (J[f'{a}_{s}'], J[f'{b}_{s}']) for s in 'lr' for k, a, b in (('arm', 'shoulder', 'elbow'), ('thigh', 'hip', 'knee'), ('calf', 'knee', 'ankle'))}
tp = FIT.tape(np.stack([tm[:, 0], -tm[:, 2], tm[:, 1]], 1) / 100, limbs=limbs)  # template in Anny's frame: tape like a person
assert 22 < tp['arm_l'] < 34 and 42 < tp['thigh_l'] < 60 and 30 < tp['calf_l'] < 42 and abs(tp['arm_l'] - tp['arm_r']) < 1, tp
assert 34 < tp['neck'] < 42 and 85 < tp['chest'] < 100 and 70 < tp['waist'] < 85 and 85 < tp['hips'] < 100 and .58 < tp['navel_h'] < .65, tp
t, off, n = meta['arrays']['female']
tf = np.frombuffer((FIT.ROOT / 'static/assets/body.bin').read_bytes(), np.float32, n, off).reshape(-1, 3)
tpf = FIT.tape(np.stack([tf[:, 0], -tf[:, 2], tf[:, 1]], 1) / 100, female=True)
assert 65 < tpf['underbust'] < tpf['chest'] - 5, tpf  # under the bust, clearly smaller than round it
print('fit helpers ok', tp, '| female', {k: tpf[k] for k in ('chest', 'underbust')})
# Shoulder slope from front outlines: a neck, shoulders falling 20° outward to the arms, then down (image y runs down).
t20 = np.tan(np.radians(20))
outline = [(-.03, -.45), (.03, -.45), (.03, -.36), (.13, -.36 + .1 * t20), (.13, .4), (-.13, .4), (-.13, -.36 + .1 * t20), (-.03, -.36)]
lm = np.zeros((33, 2))
lm[11], lm[12] = (.09, -.3), (-.09, -.3)
sl = FIT.shoulder_slope([{'theta': 0, 'sil': {'l': lm.tolist(), 'c': outline}}] * 20)
assert abs(sl - 20) < .5 and FIT.shoulder_slope([{'theta': 90, 'sil': {'l': lm.tolist(), 'c': outline}}] * 20) is None, sl
# Arm against the chest: the image-right arm is part of the torso's outline at chest height; the left one hangs clear
# (the outline stops before its centre line): only the right side touches, three heights, and never in a side view.
kp = np.zeros((15, 2))
kp[3], kp[4], kp[5], kp[6], kp[9], kp[10] = (.18, -.5), (-.18, -.5), (.2, -.2), (-.25, -.2), (.1, 0), (-.1, 0)  # KP order
C2 = np.array([[(-.15, -.6), (.24, -.6), (.24, .1), (-.15, .1)]])  # torso and right arm, one shape
hits = FIT.arm_contacts(C2, kp[None], np.array([[-1, 1, -1, 1.]]), [0], [((.5, .62, .72), ((3, 5), (4, 6)), None, None)])
assert hits and all(h[3] == 1 for h in hits) and len(hits) == 3, hits
assert not FIT.arm_contacts(C2, kp[None], np.array([[-1, 1, -1, 1.]]), [90], [((.5, .62, .72), ((3, 5), (4, 6)), None, None)])  # side view
# The stance (with the optional fit group): palms toward the thighs (thumb in front of the little finger, not inside
# it), fingers together in order front to back, none crossing.
try:
    import anny
    import torch
except ImportError:
    anny = None
if anny:
    am = anny.Anny(topology='makehuman', rig='makehuman', skinning_method='lbs', local_changes='default').to(dtype=torch.float32)
    sh = {'phenotype_kwargs': {k: torch.tensor([.5]) for k in ('muscle', 'weight', 'height', 'proportions', 'age')} | {'gender': torch.tensor([0.])},
          'local_changes_kwargs': torch.zeros(1, len(am.local_change_labels))}
    G = FIT.stance(am, sh, anny.KeypointsRegressor.coco(am, labels=list(FIT.KP)))
    rv = am(pose_parameters=torch.eye(4)[None, None].repeat(1, am.bone_count, 1, 1), **sh)['rest_vertices'][0].numpy()
    bw, bi = am.vertex_bone_weights.numpy(), am.vertex_bone_indices.numpy()
    bw, bi = (bw[0], bi[0]) if bw.ndim == 3 else (bw, bi)
    Pm = np.einsum('vk,vkij->vij', bw, G[bi])
    pv = np.einsum('vij,vj->vi', Pm[:, :3, :3], rv) + Pm[:, :3, 3]  # x out (left), y back, z up
    top = bi[np.arange(len(pv)), bw.argmax(1)]
    for s_ in 'LR':
        tip = lambda f: pv[top == am.bone_labels.index(f'finger{f}-3.{s_}')].mean(0) * 100  # noqa: E731
        d = tip(1) - tip(5)
        assert d[1] < -5 and abs(d[0]) < 4, (s_, d)
        ys = [tip(f)[1] for f in (2, 3, 4, 5)]
        assert ys == sorted(ys) and all(np.linalg.norm(tip(f) - tip(f + 1)) < 3.6 for f in (2, 3, 4)), (s_, ys)  # side by side: ~3 cm apart
    print('stance ok: palms to the thighs, fingers together')

import bake  # noqa: E402
import judge  # noqa: E402
judge._demo()
src, uv, tris, _ = bake.assets()
bary = bake.texels(uv, tris, 512)[3]
assert np.allclose(bary.sum(1), 1) and bary.min() >= 0  # every texel a point on its triangle
print('vision-check rules and texel weights ok')
