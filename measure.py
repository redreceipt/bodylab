"""Turn camera frames (MediaPipe landmarks + person mask) into body measurements — any frame, any angle —
fuse them into one body per day, then derive circumferences, body-fat %, and progress insights.

Raw measurements are stored as fractions of standing height, so correcting the
profile height later rescales every past scan."""
import math
import statistics
import time
from datetime import date, datetime, timedelta

import cv2
import numpy as np

EAR_L, EAR_R = 7, 8
SH_L, SH_R, EL_L, EL_R, WR_L, WR_R = 11, 12, 13, 14, 15, 16
HIP_L, HIP_R, KN_L, KN_R, AN_L, AN_R = 23, 24, 25, 26, 27, 28
# Plausible adult sizes as fractions of standing height. Frames and saved scans outside these are rejected:
# they come from a misread pose (sitting, partly hidden, someone else's limbs) rather than a real body.
PLAUSIBLE = {'neck_w': (.04, .1), 'shoulders_w': (.17, .36), 'chest_w': (.12, .3), 'waist_w': (.1, .32), 'hips_w': (.13, .32),
             'neck_d': (.04, .1), 'chest_d': (.08, .25), 'waist_d': (.07, .28), 'hips_d': (.08, .26),
             'arm_l_w': (.035, .1), 'arm_r_w': (.035, .1), 'thigh_l_w': (.06, .16), 'thigh_r_w': (.06, .16),
             'calf_l_w': (.045, .11), 'calf_r_w': (.045, .11), 'lv_shoulder': (.75, .87), 'lv_hip': (.43, .58), 'lv_knee': (.22, .33)}
SEEN = {'neck': (.035, .11), 'chest': (.08, .31), 'underbust': (.07, .3), 'waist': (.07, .33), 'hips': (.08, .33)}  # any angle


def implausible(raw):
    """Measurements outside human proportions, or None if the body looks plausible."""
    bad = [k for k, (lo, hi) in PLAUSIBLE.items() if raw.get(k) is not None and not lo <= raw[k] <= hi]
    if raw.get('hips_w') and raw.get('waist_w') and raw['hips_w'] < .8 * raw['waist_w']:
        bad.append('hips_w')
    return list(dict.fromkeys(bad)) or None

TORSO = {'neck': .9, 'chest': .72, 'underbust': .72, 'waist': .78, 'hips': .75}  # typical depth/width until turned views exist
# Tape ÷ ellipse perimeter for the same width and depth: torsos are boxier than ellipses (measured on MakeHuman's templates).
SHAPE = {'neck': 1.0, 'chest': 1.07, 'underbust': 1.04, 'waist': 1.03, 'hips': 1.04}
LIMBS = ['arm_l', 'arm_r', 'thigh_l', 'thigh_r', 'calf_l', 'calf_r']
PARTS = [*TORSO, 'shoulders', *LIMBS]
METRICS = ['bf', 'weight', 'lean_kg', 'fat_kg', 'ffmi', 'whtr', 'whr', *PARTS]
# Workout focus -> body parts it trains (cardio only counts toward overall load).
FOCUS = {'full body': PARTS, 'chest': ['chest'], 'back': ['chest', 'shoulders'], 'shoulders': ['shoulders'],
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


def yaw(world, a, b):
    """How far a left/right landmark pair is turned from the camera, in degrees: 0 = facing it, 90 = side on."""
    v = world[a] - world[b]
    return math.degrees(math.atan2(abs(v[2]), abs(v[0])))


def observe(pts, vis, world, mask, female=False, torso_ratio=.29, ycc=None, skin=None):
    """Everything one frame can tell about the body, from any angle or distance — no special pose needed.

    pts: (33,2) pixel landmarks; vis: visibility; world: (33,3) MediaPipe world landmarks; mask: person
    probability; ycc + skin (Cr, Cb of your face): when given, slices that aren't bare skin are skipped.
    Returns {'theta': yaw°, 'full': head-to-toe in view, 'obs': {key: fraction of standing height},
    'lines': [(p, q)], 'problem': str|None}. Torso keys ('neck', 'chest', 'waist', 'hips') are the widths
    seen at yaw theta; estimate() turns many of them into width and depth."""
    out = {'theta': None, 'full': False, 'obs': {}, 'lines': [], 'problem': None}

    def fail(msg):
        out['problem'] = msg
        return out

    m = mask > .5
    h, w = m.shape
    if max(vis[SH_L], vis[SH_R]) < .5 or max(vis[HIP_L], vis[HIP_R]) < .5:
        return fail('Waiting for a clear view of your torso')
    sh, hp = (pts[SH_L] + pts[SH_R]) / 2, (pts[HIP_L] + pts[HIP_R]) / 2
    axis = hp - sh
    torso = np.linalg.norm(axis)
    if torso < 40:
        return fail('Too far from the camera')
    down = axis / torso
    perp = np.array([-down[1], down[0]])
    if abs(down[0]) > .25:
        return fail('Leaning — only upright frames are measured')
    knees = pts[[KN_L, KN_R]].mean(0)
    if bent(world) or (vis[[KN_L, KN_R]].min() > .5 and np.dot(knees - hp, down) < .55 * torso) or (
            vis[[KN_L, KN_R, AN_L, AN_R]].min() > .5 and foreshortened(pts)):  # thighs not hanging, or pointing at the camera
        return fail('Sitting, crouching or bending — only standing frames are measured')
    quad = np.zeros(m.shape, np.uint8)
    cv2.fillConvexPoly(quad, np.round(pts[[SH_L, SH_R, HIP_R, HIP_L]]).astype(np.int32), 1)
    if quad.any() and m[quad > 0].mean() < .7:  # landmarks without a body behind them: occluded, or a misdetection
        return fail('Your body isn\'t clearly in view')
    theta = out['theta'] = (yaw(world, SH_L, SH_R) + yaw(world, HIP_L, HIP_R)) / 2
    obs = out['obs']

    # Scale: standing height in px when head-to-toe is in view, otherwise from your torso length.
    H, head = torso / torso_ratio, (pts[EAR_L] + pts[EAR_R]) / 2
    if vis[[AN_L, AN_R]].max() > .5 and vis[[EAR_L, EAR_R]].max() > .5:
        rows_in = lambda x0, x1: np.flatnonzero(m[:, max(int(x0), 0):min(int(x1) + 1, w)].any(1))  # noqa: E731
        ax = pts[[AN_L, AN_R], 0]
        head_rows, foot_rows = rows_in(head[0] - .15 * torso, head[0] + .15 * torso), rows_in(ax.min() - .3 * torso, ax.max() + .3 * torso)
        if len(head_rows) and len(foot_rows) and head_rows[0] > 2 and foot_rows[-1] < h - 3:
            top, bottom = head_rows[0], foot_rows[-1]
            if .22 < torso / (bottom - top) < .36:
                H, out['full'] = float(bottom - top), True
                obs['torso_ratio'] = torso / H

    def arm_clash(p, a, b):
        """An arm at the edge of this slice would widen it (in front of the torso or well clear is fine)."""
        tp = np.dot(p - sh, down)
        for chain in ((SH_L, EL_L, WR_L), (SH_R, EL_R, WR_R)):
            for j0, j1 in zip(chain, chain[1:]):
                t0, t1 = np.dot(pts[j0] - sh, down), np.dot(pts[j1] - sh, down)
                if min(vis[j0], vis[j1]) < .3 or (t0 - tp) * (t1 - tp) > 0 or t0 == t1:
                    continue  # hidden behind the body, or not at this height
                off = np.dot(pts[j0] + (pts[j1] - pts[j0]) * (tp - t0) / (t1 - t0) - p, perp)
                if abs(abs(off) - (a if off > 0 else b)) < .035 * H:
                    return True
        return False

    def slice_(name, points, pick=min, cap=torso, bare=True):
        cuts = [(p, _run(m, p, perp, cap)[0], _run(m, p, -perp, cap)[0]) for p in points]
        p, a, b = pick(cuts, key=lambda c: c[1] + c[2])
        line, width = (p + perp * a, p - perp * b), (a + b) / H
        lo, hi = SEEN[name]
        if lo <= width <= hi and not arm_clash(p, a, b) and not (bare and skin is not None and skin_share(ycc, skin, *line) < .5):
            obs[name] = width
            out['lines'].append(line)
        return p

    at = lambda f: sh + axis * f  # noqa: E731
    p_neck = slice_('neck', [sh + (head - sh) * t for t in np.linspace(.2, .55, 8)], cap=.5 * torso, bare=False)
    p_chest = slice_('chest', [at(.30)], bare=not female)  # a sports bra is fine
    if female:  # under the breasts: with the bust, gives cup size; and it's the ribcage the avatar fits to
        slice_('underbust', [at(.42)], bare=False)
    p_waist = slice_('waist', [at(f) for f in (np.linspace(.5, .8, 7) if female else [.84])])  # Navy: women narrowest, men at the navel
    p_hips = slice_('hips', [at(f) for f in np.linspace(.95, 1.15, 6)], pick=max, bare=False)  # underwear is fine

    if theta < 25:
        a, b = _run(m, sh, perp, torso)[0], _run(m, sh, -perp, torso)[0]
        if PLAUSIBLE['shoulders_w'][0] <= (a + b) / H <= PLAUSIBLE['shoulders_w'][1]:
            obs['shoulders_w'] = (a + b) / H

    def limb(j0, j1, ts, cap, inner_ok, bare=True):
        if min(vis[j0], vis[j1]) < .5:
            return None
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
            line = (p + n * a, p - n * b)
            if not (bare and skin is not None and skin_share(ycc, skin, *line) < .5):
                out['lines'].append(line)
                return (a + b) / H

    if theta < 45:  # limbs side by side only when roughly facing toward or away from the camera
        for s, (S, E, HIP, K, A) in {'l': (SH_L, EL_L, HIP_L, KN_L, AN_L), 'r': (SH_R, EL_R, HIP_R, KN_R, AN_R)}.items():
            for key, v in ((f'arm_{s}_w', limb(S, E, [.5], .3, False)), (f'thigh_{s}_w', limb(HIP, K, [.4], .35, True, bare=False)),
                           (f'calf_{s}_w', limb(K, A, np.linspace(.2, .45, 6), .3, True))):
                if v is not None:
                    obs[key] = v
    if out['full']:
        lv = lambda p: (bottom - p[1]) / H  # noqa: E731
        obs.update(lv_neck=lv(p_neck), lv_shoulder=lv(sh), lv_chest=lv(p_chest), lv_waist=lv(p_waist), lv_hips=lv(p_hips),
                   lv_hip=lv(hp), lv_knee=(lv(pts[KN_L]) + lv(pts[KN_R])) / 2, lv_ankle=(lv(pts[AN_L]) + lv(pts[AN_R])) / 2)
        if theta < 45 and vis[[EL_L, EL_R, WR_L, WR_R]].min() > .5:
            seg = lambda i, j: np.linalg.norm(pts[i] - pts[j]) / H  # noqa: E731
            obs.update(upperarm=(seg(SH_L, EL_L) + seg(SH_R, EL_R)) / 2, forearm=(seg(EL_L, WR_L) + seg(EL_R, WR_R)) / 2)
    for k in [k for k in obs if k in PLAUSIBLE and not PLAUSIBLE[k][0] <= obs[k] <= PLAUSIBLE[k][1]]:
        del obs[k]  # one misread part shouldn't cost the rest of the frame
    out['H'], out['origin'] = H, hp  # px per standing height, hip centre: to store a scale-free silhouette
    return out if obs else fail('Nothing measurable in this frame')


def foreshortened(pts, limit=.7):
    """Both thighs look much shorter than the shins, or one very much shorter: they point at the camera, as when
    sitting facing it (2D, so MediaPipe's flattened depth can't hide it). Walking keeps one leg straight and never
    lifts a thigh that far."""
    thigh = [np.linalg.norm(pts[k] - pts[h]) for h, k in ((HIP_L, KN_L), (HIP_R, KN_R))]
    shin = [np.linalg.norm(pts[a] - pts[k]) for k, a in ((KN_L, AN_L), (KN_R, AN_R))]
    r = [t / max(s, 1e-6) for t, s in zip(thigh, shin)]
    return max(r) < limit or min(r) < limit - .1


def bent(world, limit=45):
    """Both hips or both knees bent past `limit` degrees, or the torso tilted over 25° from upright (MediaPipe 3D
    landmarks, y down): sitting, crouching or bending over. Walking (one leg straight) isn't."""
    def angle(a, b):
        n = np.linalg.norm(a) * np.linalg.norm(b)
        return np.degrees(np.arccos(np.clip(a @ b / n, -1, 1))) if n > 1e-9 else 0
    torso = world[[HIP_L, HIP_R]].mean(0) - world[[SH_L, SH_R]].mean(0)
    hips = [angle(torso, world[k] - world[h]) for h, k in ((HIP_L, KN_L), (HIP_R, KN_R))]
    knees = [angle(world[k] - world[h], world[a] - world[k]) for h, k, a in ((HIP_L, KN_L, AN_L), (HIP_R, KN_R, AN_R))]
    return min(hips) > limit or min(knees) > limit or angle(-torso, np.array([0, -1., 0])) > 25


def fit_ellipse(seen, ratio):
    """Semi-axes (a = half width, b = half depth) of a body cross-section from widths w seen at yaw θ:
    (w/2)² = a²cos²θ + b²sin²θ. Robust least squares once there are views both near front-on and turned
    past 30°; until then depth is assumed to be `ratio` × width."""
    th = np.radians([t for t, _ in seen])
    y = (np.array([v for _, v in seen]) / 2) ** 2
    c, s = np.cos(th) ** 2, np.sin(th) ** 2
    if (s > .25).sum() >= 5 and (s <= .25).sum() >= 5:
        X, wt = np.stack([c, s], 1), np.ones(len(y))
        for _ in range(5):  # iteratively reweighted (Huber), so stray frames can't drag the fit
            A, B = np.linalg.lstsq(X * wt[:, None], y * wt, rcond=None)[0]
            r = np.abs(y - X @ [A, B])
            wt = np.sqrt(np.minimum(1, 1.5 * (np.median(r) * 1.4826 + 1e-12) / np.maximum(r, 1e-12)))
        if A > 0 and B > 0:
            return math.sqrt(A), math.sqrt(B)
    a = math.sqrt(float(np.median(y / (c + ratio ** 2 * s))))
    return a, ratio * a


def estimate(frames):
    """Fuse frames from any angle into one body (fractions of standing height): torso cross-sections are
    ellipses fitted to the widths seen at each yaw; everything else is a median. None if too little to go on."""
    raw = {'n_frames': len(frames), 'n_angle': round(max((f['theta'] for f in frames), default=0))}
    for level, ratio in TORSO.items():
        # A turned neck's outline includes the jaw, so its depth comes from TORSO, not side views.
        seen = [(f['theta'], f['obs'][level]) for f in frames if level in f['obs'] and (level != 'neck' or f['theta'] < 30)]
        if len(seen) >= 8:
            a, b = fit_ellipse(seen, ratio)
            raw[level + '_w'], raw[level + '_d'] = 2 * a, 2 * b
    for k in {k for f in frames for k in f['obs']} - set(TORSO):
        vals = [f['obs'][k] for f in frames if k in f['obs']]
        if len(vals) >= 5:
            raw[k] = float(np.median(vals))
    return raw if {'neck_w', 'hips_w'} <= raw.keys() else None  # no waist yet: no body fat, the rest still counts


def skin_share(ycc, skin, a, b, tol=14):
    """Share of pixels along a measurement line (middle 70 %) whose Cr/Cb chroma is within tol of your skin.
    ycc: YCrCb image; skin: (Cr, Cb). A shirt or loose top shows up as a low share."""
    t = np.linspace(.15, .85, 40)[:, None]
    xy = np.rint(np.asarray(a) + (np.asarray(b) - np.asarray(a)) * t).astype(int).clip(0, [ycc.shape[1] - 1, ycc.shape[0] - 1])
    return float((np.abs(ycc[xy[:, 1], xy[:, 0], 1:].astype(int) - np.asarray(skin)) < tol).all(1).mean())


def hair_style(hair, img, face):
    """Hair colour and the closest avatar haircut from a hair mask (HxW, 0..1) of a front-on photo.
    face: (478, 2) face landmarks in px. Ponytails and braids can't be seen from the front; pick those by hand."""
    hair = hair > .5
    top, chin = face[10][1], face[152][1]
    left, right = face[:, 0].min(), face[:, 0].max()
    fh, fw = chin - top, right - left
    ys, xs = np.nonzero(hair)
    if len(ys) < .05 * fw * fh:
        return {'style': 'none', 'color': None}
    px = img[ys, xs][:, ::-1].astype(float)  # BGR -> RGB
    lum = px @ [.299, .587, .114]
    lo, hi = np.percentile(lum, [25, 60])
    rgb = px[(lum >= lo) & (lum <= hi)].mean(0)  # typical strands: the renderer adds its own highlights
    rgb = rgb.mean() + .6 * (rgb - rgb.mean())  # less saturated: a room's warm light reddens hair in a photo
    near = np.abs(xs - (left + right) / 2) < fw  # hair beside the face, not the background's
    length = (ys[near].max() - chin) / fh  # how far below the chin it reaches
    width = (xs.max() - xs.min()) / fw
    crown = (top - ys.min()) / fh  # thickness above the forehead
    style = ('long01' if length > .35 else 'bob02' if length > -.15 else 'afro01' if width > 1.7 or crown > .45
             else 'short04' if crown < .12 else 'short02')
    # A fringe: hair over the forehead (between the face's top and the brows, inner brow to inner brow).
    y0, y1, x0, x1 = int(top), int(face[9][1]), int(face[105][0]), int(face[334][0])
    fringe = bool(hair[y0:max(y1, y0 + 1), min(x0, x1):max(x0, x1) + 1].mean() > .25)
    return {'style': style, 'color': '#' + ''.join(f'{int(v):02x}' for v in rgb), 'fringe': fringe}


# How much each part's circumference typically moves per kg of fat lost/gained (cm/kg), until your own history says
# otherwise; and per kg of lean mass gained. shortcut: population-typical priors, replaced per part by your measured slope.
FAT_CM_PER_KG = {'male': {'waist': 1.1, 'hips': .6, 'chest': .5, 'neck': .2, 'thigh_l': .4, 'thigh_r': .4, 'arm_l': .25, 'arm_r': .25,
                          'calf_l': .1, 'calf_r': .1},
                 'female': {'waist': .8, 'hips': .9, 'chest': .6, 'underbust': .4, 'neck': .15, 'thigh_l': .6, 'thigh_r': .6,
                            'arm_l': .3, 'arm_r': .3, 'calf_l': .15, 'calf_r': .15}}
ESSENTIAL_FAT = {'male': 4, 'female': 12}  # % body fat; physiological minimums
LEAN_CM_PER_KG = {'chest': .8, 'shoulders': .5, 'arm_l': .6, 'arm_r': .6, 'thigh_l': .6, 'thigh_r': .6, 'calf_l': .2, 'calf_r': .2,
                  'waist': .1, 'neck': .2}


def maintenance(logs, scans, now=None):
    """Your maintenance calories: from logged intake vs weight trend when there's enough of both (what your body
    actually did), else Apple Health's active + resting energy, else None."""
    now, since = now or time.time(), (now or time.time()) - 28 * 86400
    daily = [d for d in usable([l for l in logs if l['kind'] == 'daily'], now) if d['ts'] >= since]
    intake = [d['kcal'] for d in daily if 'kcal' in d]
    change = trend_change(series('weight', scans, logs), since)
    if len(intake) >= 14 and change is not None:
        return statistics.fmean(intake) - change * 7700 / 28, 'from your logged food and weight trend'
    burn = [d['active'] + d['basal'] for d in daily if {'active', 'basal'} <= d.keys()]
    if len(burn) >= 7:
        return statistics.fmean(burn), "from Apple Health's active + resting energy"
    return None, None


def personal_slopes(scans, sex):
    """cm of change per kg of fat for each part, from your own scans once there's a usable spread, else typical."""
    out = dict(FAT_CM_PER_KG[sex])
    for part in out:
        pts = [(s['m']['fat_kg'], s['m'][part]) for s in scans if 'fat_kg' in s['m'] and part in s['m']]
        if len(pts) >= 5 and np.ptp([p[0] for p in pts]) >= 1.5:  # at least 1.5 kg of fat change to learn from
            slope = statistics.linear_regression(*zip(*pts)).slope
            if 0 < slope < 4:
                out[part] = slope
    return out


def predict(m, sex, weeks, kcal_delta, strength_per_week, protein_g_kg, height_cm, slopes=None):
    """Simulate weeks of a plan from today's body m (needs weight and bf). Daily energy balance -> fat and lean change
    (Hall's energy densities; Forbes's rule for how losses split between fat and lean; maintenance falls as you
    lose), with resistance training and protein protecting or building lean mass. Returns weekly points and the
    final metrics, including each part's circumference."""
    weight, bf = m['weight'], m['bf']
    fat, lean = weight * bf / 100, weight * (1 - bf / 100)
    trains = strength_per_week >= 2 and protein_g_kg >= 1.6
    floor = ESSENTIAL_FAT[sex] / 100  # below this the body breaks down lean tissue and organs suffer
    path, warning = [], None
    for d in range(weeks * 7 + 1):
        if d % 7 == 0:
            path.append({'week': d // 7, 'weight': round(fat + lean, 2), 'bf': round(100 * fat / (fat + lean), 2),
                         'fat_kg': round(fat, 2), 'lean_kg': round(lean, 2)})
        balance = kcal_delta + 25 * (weight - (fat + lean))  # maintenance drops ~25 kcal/day per kg lost
        p = 10.4 / (10.4 + fat)  # Forbes: lean share of weight change (leaner people lose relatively more lean)
        if balance < 0 and trains:
            p *= .3  # lifting + enough protein spares most lean mass in a deficit
        elif balance > 0:
            p = .5 if trains else .3  # a surplus goes more to muscle when you train
        d_weight = balance / (p * 1816 + (1 - p) * 9440)  # kcal per kg of that fat/lean mix
        d_fat, d_lean = (1 - p) * d_weight, p * d_weight
        if d_fat < 0 and (fat + d_fat) / (fat + lean + d_weight) < floor:
            d_fat, d_lean = 0, balance / 1816  # nothing left to lose but muscle
            warning = (f"This plan would take you below essential body fat (~{ESSENTIAL_FAT[sex]}% for "
                       f"{'women' if sex == 'female' else 'men'}): fat loss stops and you'd lose muscle. Not a safe plan.")
        if trains:  # resistance training builds some muscle even at maintenance; less as you get leaner of new gains
            d_lean += min(strength_per_week, 5) / 5 * .25 / 30  # up to ~0.25 kg/month
        fat, lean = fat + d_fat, lean + d_lean
    return {'path': path, 'm': at_composition(m, sex, fat, lean, height_cm, slopes), 'warning': warning}


def at_composition(m, sex, fat, lean, height_cm, slopes=None):
    """Today's body m (weight, bf, circumferences) with `fat` and `lean` kg instead: each part grows or shrinks by its
    fat (your measured slope, else typical) and lean centimetres per kilogram."""
    final = {**m, 'weight': fat + lean, 'fat_kg': fat, 'lean_kg': lean, 'bf': 100 * fat / (fat + lean),
             'ffmi': lean / (height_cm / 100) ** 2}
    d_fat, d_lean = fat - m['weight'] * m['bf'] / 100, lean - m['weight'] * (1 - m['bf'] / 100)
    for part, cm in (slopes or FAT_CM_PER_KG[sex]).items():
        if part in m:
            final[part] = m[part] + cm * d_fat + LEAN_CM_PER_KG.get(part, 0) * max(d_lean, 0)
    if 'shoulders' in m:
        final['shoulders'] = m['shoulders'] + LEAN_CM_PER_KG['shoulders'] * max(d_lean, 0)
    if 'waist' in final:
        final['whtr'] = final['waist'] / height_cm
    return {k: round(v, 3) for k, v in final.items()}


# A lean, muscular look: what to aim for unless you choose otherwise (body fat %).
RECOMMENDED_BF = {'male': 12, 'female': 22}


def ellipse(w, d):
    """Ramanujan's ellipse perimeter from full width and depth."""
    a, b = w / 2, d / 2
    return math.pi * (3 * (a + b) - math.sqrt((3 * a + b) * (a + 3 * b)))


def navy_bf(sex, height, neck, waist, hips):
    """US Navy circumference body-fat formula (all cm)."""
    if sex == 'female':
        return 495 / (1.29579 - .35004 * math.log10(waist + hips - neck) + .221 * math.log10(height)) - 450
    return 495 / (1.0324 - .19077 * math.log10(waist - neck) + .15456 * math.log10(height)) - 450


def derive(raw, profile, weight=None, scale_bf=None):
    """Raw height fractions -> (metrics, dims). metrics: circumferences/breadth in cm, bf %, ratios, kg.
    dims: sizes in cm for drawing the avatar."""
    H = profile['height_cm']
    dims = {k: v * H for k, v in raw.items() if isinstance(v, (int, float)) and not k.startswith('n_')}
    m = {}
    tape = raw.get('tape', {})  # tape-measured on the fitted body (fit.py): preferred over ellipses
    for p, ratio in TORSO.items():
        if p + '_w' in dims:
            dims.setdefault(p + '_d', dims[p + '_w'] * ratio)
        if p in tape or p + '_w' in dims:
            m[p] = tape.get(p) or ellipse(dims[p + '_w'], dims[p + '_d']) * SHAPE[p]
    if 'shoulders_w' in dims:
        m['shoulders'] = dims['shoulders_w']
    for p in LIMBS:
        if p in tape or p + '_w' in dims:
            m[p] = tape.get(p) or math.pi * dims[p + '_w']
    m |= {k: v for k, v in tape.items() if k.startswith('penis')}  # closeups.py: at rest, and the range seen
    if 'waist' in m:
        m['whtr'] = m['waist'] / H
        if 'hips' in m:
            m['whr'] = m['waist'] / m['hips']
    if {'neck', 'waist', 'hips'} <= m.keys() and m['waist'] > m['neck']:
        m['bf'] = m['bf_cam'] = navy_bf(profile.get('sex'), H, m['neck'], m['waist'], m['hips'])
        # Two independent estimates, each ±3–5 points: the cameras' (the US Navy formula on your measured body) and a smart
        # scale's (bioimpedance). For lean, muscular builds they err in opposite directions, so the shown number is their mean.
        if scale_bf:
            m['bf_scale'], m['bf'] = scale_bf, (m['bf'] + scale_bf) / 2
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
    logged = [(l['ts'], l[metric]) for l in logs if l.get(metric) is not None]
    if logged or metric == 'weight':
        return logged
    return [(s['ts'], s['m'][metric]) for s in scans if metric in s['m']]


def _delta(pts, since):
    pts = [p for p in pts if p[0] >= since]
    return round(pts[-1][1] - pts[0][1], 3) if len(pts) >= 2 else None


CUMULATIVE = {'kcal', 'protein', 'carbs', 'fat', 'fiber', 'sugar', 'water', 'alcohol', 'active', 'basal', 'steps', 'exercise'}
NUTRITION = {'kcal', 'protein', 'carbs', 'fat', 'fiber', 'sugar'}


def usable(daily, now):
    """Daily records with values that would mislead dropped: today's running totals (the day isn't over) and
    nutrition from days logged at under 60 % of your usual intake (a forgotten meal, not a fast)."""
    today = date.fromtimestamp(now)
    kcals = [d['kcal'] for d in daily if 'kcal' in d and date.fromtimestamp(d['ts']) != today]
    usual = statistics.median(kcals) if len(kcals) >= 5 else 0
    out = []
    for d in daily:
        drop = (CUMULATIVE if date.fromtimestamp(d['ts']) == today else set()) | (NUTRITION if d.get('kcal', usual) < .6 * usual else set())
        out.append({k: v for k, v in d.items() if k not in drop})
    return out


def trend_change(pts, since):
    """Change over the window from the median of all pairwise slopes (Theil-Sen), so one noisy or
    water-heavy reading doesn't swing it the way first-vs-last or a least-squares line would."""
    pts = [p for p in pts if p[0] >= since]
    if len(pts) < 3 or pts[-1][0] - pts[0][0] < 7 * 86400:
        return _delta(pts, since)
    slopes = [(b[1] - a[1]) / (b[0] - a[0]) for i, a in enumerate(pts) for b in pts[i + 1:] if b[0] > a[0]]
    return round(statistics.median(slopes) * (pts[-1][0] - pts[0][0]), 3)


def insights(scans, logs, goals, now=None):
    """Inputs (workouts, food, sleep) vs outputs (body metrics): weekly table, 28-day summary,
    per-part training vs change, goal projections, correlation notes, data coverage and gaps."""
    now, day = now or time.time(), 86400
    workouts = [l for l in logs if l['kind'] == 'workout']
    raw_daily = [l for l in logs if l['kind'] == 'daily']
    daily = usable(raw_daily, now)

    def load(ws):
        return round(sum(w['minutes'] * w['rpe'] for w in ws))

    def avg(xs):
        return round(statistics.fmean(xs), 2) if xs else None

    def balance(ds, min_days=1):  # energy in - out, only on fully logged days with intake, active and resting energy
        days = [d['kcal'] - d['active'] - d['basal'] for d in ds if {'kcal', 'active', 'basal'} <= d.keys()]
        return avg(days) if len(days) >= min_days else None

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
                      **{k: avg([d[k] for d in dl if k in d]) if sum(k in d for d in dl) >= 3 else None
                         for k in ('kcal', 'protein', 'sleep', 'weight', 'active', 'steps')},
                      'balance': balance(dl, 4),
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
    window = [d for d in daily if d['ts'] >= since]
    balance_days = sum({'kcal', 'active', 'basal'} <= d.keys() for d in window)
    bal = balance(window, 14)  # under half the month logged, a balance would mostly be guesswork
    summary = {'workouts': len(recent), 'load': load(recent), 'balance': bal, 'balance_days': balance_days,
               'predicted_kg': round(bal * 28 / 7700, 2) if bal is not None else None,  # ~7700 kcal per kg of fat
               'kcal': avg([d['kcal'] for d in daily if d['ts'] >= since and 'kcal' in d]),
               'changes': {k: trend_change(series(k, scans, logs), since) for k in ('bf', 'waist', 'weight', 'lean_kg')}}
    parts = {}
    for p in PARTS:
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

    # Every Health metric: last 7 days vs the 7 before, with how many days actually had data.
    last7, prev7 = [d for d in daily if d['ts'] >= now - 7 * day], [d for d in daily if now - 14 * day <= d['ts'] < now - 7 * day]
    health = {k: {'avg': avg([d[k] for d in last7 if k in d]), 'prev': avg([d[k] for d in prev7 if k in d]),
                  'days': sum(k in d for d in last7)}
              for k in sorted({k for d in raw_daily for k in d} - {'id', 'ts', 'kind'})}

    # Gaps: missing inputs make outputs wrong, so say exactly what's missing and how to fill it.
    days28 = lambda k: sum(k in d for d in window)  # noqa: E731
    partial = sum('kcal' in r and 'kcal' not in d for r, d in zip(raw_daily, daily) if r['ts'] >= since)
    gaps = []
    if days28('kcal') < 14:
        gaps.append(f"Calories logged on {days28('kcal')} of 28 days. Energy balance and food insights need a food log — "
                    "any app that writes to Apple Health (Cronometer, MyFitnessPal, Lose It, MacroFactor).")
    if partial:
        gaps.append(f"{partial} partly logged day{'s' * (partial > 1)} left out of nutrition averages.")
    weighins = sum(1 for p in series('weight', scans, logs) if p[0] >= since)
    if weighins < 8:
        gaps.append(f"{weighins} weigh-in{'s' * (weighins != 1)} in 28 days. Fat and lean mass need a weight within a week of "
                    "each scan — weigh in a few times a week (a smart scale that syncs to Health does it for you).")
    if days28('sleep') < 17:
        gaps.append(f"Sleep tracked {days28('sleep')} of 28 nights — wear your Watch to bed for recovery insights.")
    recent_scans = sum(1 for s in scans if s['ts'] >= since)
    if recent_scans < 4:
        gaps.append(f"{recent_scans} body scan{'s' * (recent_scans != 1)} in 28 days — walk past the camera a few times a week.")
    # Each day's inputs, so a missing one shows on the day it can still be fixed.
    kcals = [d['kcal'] for d in raw_daily if 'kcal' in d and date.fromtimestamp(d['ts']) != today]
    usual = statistics.median(kcals) if len(kcals) >= 5 else None

    def checks(d0):
        rec = {k: v for r in raw_daily if date.fromtimestamp(r['ts']) == d0 for k, v in r.items()}
        scan = next((s for s in scans if date.fromtimestamp(s['ts']) == d0), None)
        return {'date': d0.isoformat(), 'weight': 'weight' in rec, 'sleep': 'sleep' in rec, 'activity': 'steps' in rec or 'active' in rec,
                'food': 'kcal' in rec and (usual is None or rec['kcal'] >= .6 * usual), 'kcal': rec.get('kcal'),
                'scan': bool(scan and (scan.get('frames') or 0) >= 40), 'views': (scan or {}).get('frames') or 0}
    days = [checks(today - timedelta(days=i)) for i in range(13, -1, -1)]
    return {'weeks': weeks, 'summary': summary, 'parts': parts, 'goals': goal_out, 'notes': notes, 'health': health, 'gaps': gaps,
            'days': days, 'usual_kcal': usual}
