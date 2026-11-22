"""Self-check: synthetic 175 cm figure drawn at 4 px/cm. Run: uv run python test_measure.py"""
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

r = M.measure(pts, np.ones(33), img / 255.0)
assert r['view'] == 'front' and not r['problem'], r['problem']
m, dims = M.derive(r['raw'], {'height_cm': 175, 'sex': 'male'})
expect = {'neck_w': 12, 'chest_w': 35, 'waist_w': 35, 'hips_w': 40, 'arm_l_w': 10, 'arm_r_w': 10,
          'thigh_l_w': 17.5, 'thigh_r_w': 17.5, 'calf_l_w': 12, 'calf_r_w': 12, 'lv_shoulder': 138.75}
for k, v in expect.items():
    assert abs(dims[k] - v) < 1.0, (k, dims[k], v)
assert dims['shoulders_w'] > dims['chest_w']

# Arms down against the body must be rejected, not measured.
bad, bpts = img.copy(), pts.copy()
bpts[[13, 14, 15, 16]] = (485, 500), (315, 500), (490, 600), (310, 600)
for s, e, w in ((11, 13, 15), (12, 14, 16)):
    cv2.line(bad, tuple(bpts[s].astype(int)), tuple(bpts[e].astype(int)), 255, 40)
    cv2.line(bad, tuple(bpts[e].astype(int)), tuple(bpts[w].astype(int)), 255, 30)
assert 'arms' in (M.measure(bpts, np.ones(33), bad / 255.0)['problem'] or '')

# Formulas: circle perimeter, published Navy example (~17.2% for 178/38/86 cm male).
assert abs(M.ellipse(10, 10) - math.pi * 10) < 1e-9
assert abs(M.navy_bf('male', 178, 38, 86, 0) - 17.2) < .1
assert 'bf' in m and 'whtr' in m

# Insights: steady 0.1 %/day drop toward a goal projects an ETA; weekly table has 8 rows.
now = time.time()
scans = [{'ts': now - d * 86400, 'm': {'bf': 20 - (30 - d) * .1}} for d in range(30, -1, -3)]
logs = [{'kind': 'workout', 'ts': now - d * 86400, 'minutes': 45, 'rpe': 7, 'focus': ['arms']} for d in range(0, 28, 2)]
ins = M.insights(scans, logs, {'bf': {'target': 15, 'start': 20, 'ts': now}}, now)
g = ins['goals']['bf']
assert len(ins['weeks']) == 8 and abs(g['rate_per_week'] + .7) < .01 and g['eta'] and .5 < g['progress'] < .7, g
assert ins['parts']['arm_l']['workouts'] == 14 and ins['parts']['waist']['workouts'] == 0
print('ok', {k: dims[k] for k in expect})
