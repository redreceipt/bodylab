"""Turn one MediaPipe pose (pixel landmarks + person mask) into body measurements,
then derive circumferences, body-fat %, and progress insights.

Raw measurements are stored as fractions of standing height, so correcting the
profile height later rescales every past scan."""
import math
import statistics
import time
from datetime import date, datetime, timedelta

import numpy as np

EAR_L, EAR_R = 7, 8
SH_L, SH_R, EL_L, EL_R, WR_L, WR_R = 11, 12, 13, 14, 15, 16
HIP_L, HIP_R, KN_L, KN_R, AN_L, AN_R = 23, 24, 25, 26, 27, 28
FRONT_NEEDS = [SH_L, SH_R, EL_L, EL_R, WR_L, WR_R, HIP_L, HIP_R, KN_L, KN_R, AN_L, AN_R]

TORSO = {'neck': 1.0, 'chest': .72, 'waist': .78, 'hips': .75}  # typical depth/width, used until a side view exists
LIMBS = ['arm_l', 'arm_r', 'thigh_l', 'thigh_r', 'calf_l', 'calf_r']
CALIBRATABLE = [*TORSO, 'shoulders', *LIMBS]
METRICS = ['bf', 'weight', 'lean_kg', 'fat_kg', 'ffmi', 'whtr', 'whr', *CALIBRATABLE]
# Workout focus -> body parts it trains (cardio only counts toward overall load).
FOCUS = {'full body': CALIBRATABLE, 'chest': ['chest'], 'back': ['chest', 'shoulders'], 'shoulders': ['shoulders'],
         'arms': ['arm_l', 'arm_r'], 'core': ['waist'], 'legs': ['thigh_l', 'thigh_r', 'calf_l', 'calf_r'],
         'glutes': ['hips'], 'cardio': []}


def _run(m, p, d, cap):
    """Pixels walked from p along unit vector d before leaving mask m. Returns (dist, hit_edge)."""
    n = max(int(cap), 1)
    t = np.arange(n)
    x, y = np.rint(p[0] + t * d[0]).astype(int), np.rint(p[1] + t * d[1]).astype(int)
    h, w = m.shape
    ok = (x >= 0) & (x < w) & (y >= 0) & (y < h)
    inside = np.zeros(n, bool)
    inside[ok] = m[y[ok], x[ok]]
    out = np.flatnonzero(~inside)
    return (float(out[0]), True) if len(out) else (float(n), False)


def measure(pts, vis, mask, female=False):
    """pts: (33,2) pixel landmarks, vis: (33,) visibility, mask: HxW person probability.
    Returns {'view': 'front'|'side'|None, 'raw': {...}, 'lines': [(p, q)], 'problem': str|None}."""
    out = {'view': None, 'raw': {}, 'lines': [], 'problem': None}

    def fail(msg):
        out['problem'] = msg
        return out

    m = mask > .5
    h, w = m.shape
    sh, hp = (pts[SH_L] + pts[SH_R]) / 2, (pts[HIP_L] + pts[HIP_R]) / 2
    axis = hp - sh
    torso = np.linalg.norm(axis)
    if torso < 40:
        return fail('Step closer to the camera')
    down = axis / torso
    perp = np.array([-down[1], down[0]])
    if abs(down[0]) > .15:
        return fail('Stand up straight')
    ratio = np.linalg.norm(pts[SH_L] - pts[SH_R]) / torso
    view = out['view'] = 'front' if ratio > .55 else 'side' if ratio < .3 else None
    if not view:
        return fail('Face the camera, or turn fully sideways')
    hidden = {j for j in FRONT_NEEDS if vis[j] < .5}
    if view == 'front' and hidden:
        return fail('Hold your arms out where the camera can see your hands' if hidden <= {EL_L, EL_R, WR_L, WR_R}
                    else 'Keep your whole body visible')
    if view == 'side' and min(vis[[SH_L, SH_R]].max(), vis[[HIP_L, HIP_R]].max(), vis[[AN_L, AN_R]].max()) < .5:
        return fail('Keep your whole body visible')

    def rows_in(x0, x1):
        return np.flatnonzero(m[:, max(int(x0), 0):min(int(x1) + 1, w)].any(1))

    head = (pts[EAR_L] + pts[EAR_R]) / 2
    ax = pts[[AN_L, AN_R], 0]
    head_rows, foot_rows = rows_in(head[0] - .15 * torso, head[0] + .15 * torso), rows_in(ax.min() - .3 * torso, ax.max() + .3 * torso)
    if not len(head_rows) or not len(foot_rows):
        return fail('Keep your whole body visible')
    top, bottom = head_rows[0], foot_rows[-1]
    if top <= 2 or bottom >= h - 3:
        return fail('Step back so your head and feet are in view')
    H = float(bottom - top)  # standing height in px; everything is stored relative to it

    def lv(p):
        return (bottom - p[1]) / H

    def at(f):
        return sh + axis * f

    def slice_(points, pick=min, cap=torso):
        cuts = [(p, _run(m, p, perp, cap)[0], _run(m, p, -perp, cap)[0]) for p in points]
        p, a, b = pick(cuts, key=lambda c: c[1] + c[2])
        out['lines'].append((p + perp * a, p - perp * b))
        return p, a, b

    sfx = '_w' if view == 'front' else '_d'
    raw = out['raw']
    p, a, b = slice_([sh + (head - sh) * t for t in np.linspace(.2, .55, 8)], cap=.5 * torso)
    raw['neck' + sfx], lv_neck = (a + b) / H, lv(p)
    p, a, b = slice_([at(.30)])
    raw['chest' + sfx], lv_chest = (a + b) / H, lv(p)
    # Navy method: men at the navel, women at the narrowest point.
    p, wa, wb = slice_([at(f) for f in (np.linspace(.5, .8, 7) if female else [.73])])
    raw['waist' + sfx], lv_waist = (wa + wb) / H, lv(p)
    p, ha, hb = slice_([at(f) for f in np.linspace(.95, 1.15, 6)], pick=max)
    raw['hips' + sfx], lv_hips = (ha + hb) / H, lv(p)
    if view == 'side':
        return out

    # Arms touching the torso would inflate waist/hip widths.
    for j, (ra, rb) in ((EL_L, (wa, wb)), (EL_R, (wa, wb)), (WR_L, (ha, hb)), (WR_R, (ha, hb))):
        off = np.dot(pts[j] - sh, perp)
        if abs(off) < (ra if off > 0 else rb) + .03 * H:
            out['lines'].clear()
            return fail('Hold your arms slightly away from your body')

    _, a, b = slice_([at(0)])
    raw['shoulders_w'] = (a + b) / H

    def limb(j0, j1, ts, cap, inner_ok):
        seg = pts[j1] - pts[j0]
        L = np.linalg.norm(seg)
        d = seg / L
        best = None
        for t in ts:
            p = pts[j0] + seg * t
            n = np.array([-d[1], d[0]])
            if np.dot(n, perp) * np.dot(p - sh, perp) < 0:
                n = -n  # n points away from the body midline
            a, a_edge = _run(m, p, n, cap * L)
            midline = abs(np.dot(p - sh, perp))  # legs may touch: stop at the midline
            b, b_edge = _run(m, p, -n, min(cap * L, midline) if inner_ok else cap * L)
            if a_edge and (b_edge or inner_ok) and (best is None or a + b > best[1] + best[2]):
                best = (p, a, b, n)
        if best:
            p, a, b, n = best
            out['lines'].append((p + n * a, p - n * b))
            return (a + b) / H

    for s, (S, E, HIP, K, A) in {'l': (SH_L, EL_L, HIP_L, KN_L, AN_L), 'r': (SH_R, EL_R, HIP_R, KN_R, AN_R)}.items():
        raw[f'arm_{s}_w'] = limb(S, E, [.5], .3, False)
        raw[f'thigh_{s}_w'] = limb(HIP, K, [.4], .35, True)
        raw[f'calf_{s}_w'] = limb(K, A, np.linspace(.2, .45, 6), .3, True)

    seg = lambda i, j: np.linalg.norm(pts[i] - pts[j]) / H  # noqa: E731
    raw.update(lv_neck=lv_neck, lv_shoulder=lv(sh), lv_chest=lv_chest, lv_waist=lv_waist,
               lv_hips=lv_hips, lv_hip=lv(hp), lv_knee=(lv(pts[KN_L]) + lv(pts[KN_R])) / 2,
               lv_ankle=(lv(pts[AN_L]) + lv(pts[AN_R])) / 2,
               upperarm=(seg(SH_L, EL_L) + seg(SH_R, EL_R)) / 2, forearm=(seg(EL_L, WR_L) + seg(EL_R, WR_R)) / 2)
    return out


def ellipse(w, d):
    """Ramanujan's ellipse perimeter from full width and depth."""
    a, b = w / 2, d / 2
    return math.pi * (3 * (a + b) - math.sqrt((3 * a + b) * (a + 3 * b)))


def navy_bf(sex, height, neck, waist, hips):
    """US Navy circumference body-fat formula (all cm)."""
    if sex == 'female':
        return 495 / (1.29579 - .35004 * math.log10(waist + hips - neck) + .221 * math.log10(height)) - 450
    return 495 / (1.0324 - .19077 * math.log10(waist - neck) + .15456 * math.log10(height)) - 450


def derive(raw, profile, weight=None):
    """Raw height fractions -> (metrics, dims). metrics: circumferences/breadth in cm, bf %, ratios, kg.
    dims: sizes in cm for drawing the avatar."""
    H, cal = profile['height_cm'], profile.get('calib', {})
    dims = {k: v * H for k, v in raw.items() if v is not None and not k.startswith('n_')}
    m = {}
    for p, ratio in TORSO.items():
        if p + '_w' in dims:
            dims.setdefault(p + '_d', dims[p + '_w'] * ratio)
            m[p] = ellipse(dims[p + '_w'], dims[p + '_d']) * cal.get(p, 1)
    if 'shoulders_w' in dims:
        m['shoulders'] = dims['shoulders_w'] * cal.get('shoulders', 1)
    for p in LIMBS:
        if p + '_w' in dims:
            m[p] = math.pi * dims[p + '_w'] * cal.get(p, 1)
    if 'waist' in m:
        m['whtr'] = m['waist'] / H
        if 'hips' in m:
            m['whr'] = m['waist'] / m['hips']
    if {'neck', 'waist', 'hips'} <= m.keys() and m['waist'] > m['neck']:
        m['bf'] = navy_bf(profile.get('sex'), H, m['neck'], m['waist'], m['hips'])
    if weight:
        m['weight'] = weight
        if 'bf' in m:
            m['fat_kg'] = weight * m['bf'] / 100
            m['lean_kg'] = weight - m['fat_kg']
            m['ffmi'] = m['lean_kg'] / (H / 100) ** 2
    return {k: round(v, 3) for k, v in m.items()}, {k: round(v, 1) for k, v in dims.items()}


def nearest(points, ts, max_days=7):
    """Value of the (ts, value) point closest to ts within max_days, else None."""
    best = min(points, key=lambda p: abs(p[0] - ts), default=None)
    return best[1] if best and abs(best[0] - ts) <= max_days * 86400 else None


def series(metric, scans, logs):
    if metric == 'weight':
        return [(l['ts'], l['weight']) for l in logs if l.get('weight')]
    return [(s['ts'], s['m'][metric]) for s in scans if metric in s['m']]


def _delta(pts, since):
    pts = [p for p in pts if p[0] >= since]
    return round(pts[-1][1] - pts[0][1], 3) if len(pts) >= 2 else None


def insights(scans, logs, goals, now=None):
    """Inputs (workouts, food, sleep) vs outputs (body metrics): weekly table, 28-day summary,
    per-part training vs change, goal projections, and correlation notes."""
    now, day = now or time.time(), 86400
    workouts = [l for l in logs if l['kind'] == 'workout']
    daily = [l for l in logs if l['kind'] == 'daily']

    def load(ws):
        return sum(w['minutes'] * w['rpe'] for w in ws)

    def avg(xs):
        return round(statistics.fmean(xs), 2) if xs else None

    def balance(ds):  # energy in - out, only on days Health has intake, active and resting energy
        return avg([d['kcal'] - d['active'] - d['basal'] for d in ds if {'kcal', 'active', 'basal'} <= d.keys()])

    today = date.fromtimestamp(now)
    monday = datetime.combine(today - timedelta(days=today.weekday()), datetime.min.time())
    weeks = []
    for i in range(7, -1, -1):
        start = (monday - timedelta(weeks=i)).timestamp()
        wk = [w for w in workouts if start <= w['ts'] < start + 7 * day]
        dl = [d for d in daily if start <= d['ts'] < start + 7 * day]
        sc = [s['m'] for s in scans if start <= s['ts'] < start + 7 * day]
        weeks.append({'week': date.fromtimestamp(start).isoformat(), 'workouts': len(wk),
                      'minutes': sum(w['minutes'] for w in wk), 'load': load(wk),
                      **{k: avg([d[k] for d in dl if k in d]) for k in ('kcal', 'protein', 'sleep', 'weight', 'active', 'steps')},
                      'balance': balance(dl),
                      **{k: avg([m[k] for m in sc if k in m]) for k in ('bf', 'waist')}})

    notes = []
    for x, y, xl, yl in (('load', 'bf', 'training-load', 'body fat'), ('kcal', 'weight', 'calorie', 'weight'),
                         ('protein', 'bf', 'protein', 'body fat'), ('sleep', 'bf', 'sleep', 'body fat'),
                         ('active', 'bf', 'activity', 'body fat'), ('steps', 'bf', 'step-count', 'body fat'),
                         ('balance', 'weight', 'energy-balance', 'weight')):
        pairs = [(a[x], b[y] - a[y]) for a, b in zip(weeks, weeks[1:]) if None not in (a[x], a[y], b[y])]
        if len(pairs) >= 4:
            try:
                r = statistics.correlation(*zip(*pairs))
            except statistics.StatisticsError:
                continue
            if abs(r) >= .4:
                notes.append(f"Higher-{xl} weeks were followed by {'falling' if r < 0 else 'rising'} {yl} (r = {r:+.2f}).")

    since = now - 28 * day
    recent = [w for w in workouts if w['ts'] >= since]
    bal = balance([d for d in daily if d['ts'] >= since])
    summary = {'workouts': len(recent), 'load': load(recent), 'balance': bal,
               'predicted_kg': round(bal * 28 / 7700, 2) if bal is not None else None,  # ~7700 kcal per kg of fat
               'kcal': avg([d['kcal'] for d in daily if d['ts'] >= since and 'kcal' in d]),
               'changes': {k: _delta(series(k, scans, logs), since) for k in ('bf', 'waist', 'weight', 'lean_kg')}}
    parts = {}
    for p in CALIBRATABLE:
        ws = [w for w in recent if any(p in FOCUS.get(f, []) for f in w.get('focus', []))]
        parts[p] = {'workouts': len(ws), 'load': load(ws), 'change': _delta(series(p, scans, logs), since)}

    goal_out = {}
    for metric, g in goals.items():
        pts = series(metric, scans, logs)
        cur = pts[-1][1] if pts else None
        start = g['start'] if g['start'] is not None else (pts[0][1] if pts else None)
        progress = rate = eta = None
        if cur is not None and start is not None:
            progress = 1.0 if g['target'] == start else max(0.0, min(1.0, (cur - start) / (g['target'] - start)))
        window = [p for p in pts if p[0] >= now - 45 * day]
        if len(window) >= 3 and window[-1][0] - window[0][0] >= 3 * day:
            slope = statistics.linear_regression([p[0] / day for p in window], [p[1] for p in window]).slope
            rate = slope * 7
            if slope and progress != 1 and (g['target'] - cur) / slope > 0:
                eta = date.fromtimestamp(now + (g['target'] - cur) / slope * day).isoformat()
        goal_out[metric] = {**g, 'start': start, 'current': cur, 'progress': progress, 'rate_per_week': rate, 'eta': eta}

    # Every Health metric: last 7 days vs the 7 before.
    health = {k: {'avg': avg([d[k] for d in daily if d['ts'] >= now - 7 * day and k in d]),
                  'prev': avg([d[k] for d in daily if now - 14 * day <= d['ts'] < now - 7 * day and k in d])}
              for k in sorted({k for d in daily for k in d} - {'id', 'ts', 'kind'})}
    return {'weeks': weeks, 'summary': summary, 'parts': parts, 'goals': goal_out, 'notes': notes, 'health': health}
