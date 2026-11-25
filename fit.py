"""Fit Anny's body shape to every stored camera outline: one shape shared by all frames, each frame with its own
pose and turn. Optional (PyTorch): uv run --group fit python fit.py [--db data/bodylab.db] [--day YYYY-MM-DD] [--plot DIR]

Per frame, MediaPipe's 2D landmarks, 3D landmarks and outline (all relative to the hip centre, in units of standing
height) are matched by the posed mesh, projected orthographically. Parts outside the camera's view are ignored."""
import argparse
import json
import math
import os
import sqlite3
import time
from datetime import date
from pathlib import Path

import numpy as np

ROOT = Path(__file__).parent
os.environ.setdefault('ANNY_CACHE_DIR', str(ROOT / 'models' / 'anny'))
MIN_FRAMES = 30
MAX_FRAMES = 800  # shortcut: a fit's time grows with frames (1,100 took over an hour); evenly spaced 800 keep every angle
KP_SHAPE = .3  # landmarks' weight once frames are posed
KP_FREE = ('left_shoulder', 'right_shoulder', 'left_hip', 'right_hip')  # ...and none for these: the outline places those edges
DENSITY = 1.04  # kg/L of a whole body including the air in the lungs (tissue alone ~1.06)

# COCO keypoint -> MediaPipe pose landmark
KP = {'nose': 0, 'left_ear': 7, 'right_ear': 8, 'left_shoulder': 11, 'right_shoulder': 12, 'left_elbow': 13, 'right_elbow': 14,
      'left_wrist': 15, 'right_wrist': 16, 'left_hip': 23, 'right_hip': 24, 'left_knee': 25, 'right_knee': 26,
      'left_ankle': 27, 'right_ankle': 28}
POSED = ['clavicle.L', 'clavicle.R', 'upperarm01.L', 'upperarm01.R', 'lowerarm01.L', 'lowerarm01.R', 'upperleg01.L', 'upperleg01.R',
         'lowerleg01.L', 'lowerleg01.R', 'spine03', 'spine01', 'neck01', 'head']
FACE = ('eye', 'ear', 'nose', 'mouth', 'chin', 'cheek', 'forehead', 'head', 'brow', 'jaw', 'lip', 'finger', 'hand', 'foot',
        'toe', 'temple', 'genital', 'nipple')
HEAD = ('eye', 'ear', 'nose', 'mouth', 'chin', 'cheek', 'forehead', 'head', 'brow', 'jaw', 'lip', 'temple')  # a face scan shapes these
# Anny (x: person's left, y: back, z: up) -> camera (x: right, y: down, z: away), for someone facing the camera.
TO_CAM = [[1., 0, 0], [0, 0, -1], [0, 1, 0]]


def load(db, day=None):
    c = sqlite3.connect(f'file:{db}?mode=ro', uri=True)
    day = day or c.execute("select max(date(ts, 'unixepoch', 'localtime')) from frames").fetchone()[0]
    # shortcut: the day and the two before (your shape barely changes in that time; many more angles and bare views
    # than one day gives), so a trend shows a day or two late
    rows = [json.loads(r[0]) for r in c.execute("select data from frames where date(ts, 'unixepoch', 'localtime') "
                                                "between date(?, '-2 days') and ?", (day, day))]
    prof = json.loads(c.execute("select v from settings where k = 'profile'").fetchone()[0])
    noon = time.mktime(time.strptime(day + ' 12', '%Y-%m-%d %H'))
    w = c.execute("select json_extract(data, '$.weight'), abs(ts - ?) / 86400 from logs where kind = 'daily' "
                  "and json_extract(data, '$.weight') > 0 and abs(ts - ?) < 180 * 86400 order by abs(ts - ?) limit 1", (noon,) * 3).fetchone()
    prof |= {'weight': w[0], 'weight_age': w[1]} if w else {}
    import measure
    scan = Path(db).parent / 'face_scan.json'  # TrueDepth (app.py post_face_scan): your face's landmarks in 3D
    prof['face_scan'] = json.loads(scan.read_text())['points'] if scan.exists() else None
    last = c.execute('select params from fits where day <= ? order by day desc limit 1', (day,)).fetchone()
    if last and prof.get('weight'):  # body composition from the last fit: what the muscle/fat split should look like
        t = json.loads(last[0])['tape']
        bf = measure.navy_bf(prof.get('sex'), prof['height_cm'], t['neck'], t['waist'], t['hips'])
        scale = [r[0] for r in c.execute("select json_extract(data, '$.scale_bf') from logs where kind = 'daily' and "
                                         "json_extract(data, '$.scale_bf') > 0 and abs(ts - ?) < 7 * 86400", (noon,))]
        bf = (bf + sum(scale) / len(scale)) / 2 if scale else bf  # as the app shows it (measure.derive)
        prof |= {'bf': bf, 'ffmi': prof['weight'] * (1 - bf / 100) / (prof['height_cm'] / 100) ** 2}
    return day, [f for f in rows if f.get('sil')], prof


def borders(c, eps=.004):
    """Image edges the outline runs along (the body continues past them): [xmin, xmax, ymin, ymax], inf if none."""
    lo, hi = c.min(0), c.max(0)
    b = [-np.inf, np.inf, -np.inf, np.inf]
    for k in (0, 1):
        if (np.abs(c[:, k] - lo[k]) < eps).sum() >= 4:
            b[2 * k] = lo[k]
        if (np.abs(c[:, k] - hi[k]) < eps).sum() >= 4:
            b[2 * k + 1] = hi[k]
    return b


def kabsch(a, b):
    """Rotation R minimising |R a - b| (rows are points, both centred)."""
    u, _, vt = np.linalg.svd(b.T @ a)
    d = np.sign(np.linalg.det(u @ vt))
    return u @ np.diag([1, 1, d]) @ vt


def seg_dist2(p, a, b):
    """Squared distance from points p (B,V,2) to segments a->b (B,E,2): (B,V,E)."""
    ab = (b - a)[:, None]
    t = (((p[:, :, None] - a[:, None]) * ab).sum(-1) / (ab * ab).sum(-1).clamp_min(1e-12)).clamp(0, 1)
    return ((p[:, :, None] - (a[:, None] + t[..., None] * ab)) ** 2).sum(-1)


def inside(p, a, b):
    """Even-odd test of points p (B,V,2) against polygons with edges a->b (B,E,2): (B,V) bool."""
    py, px = p[:, :, None, 1], p[:, :, None, 0]
    ay, by, ax, bx = a[:, None, :, 1], b[:, None, :, 1], a[:, None, :, 0], b[:, None, :, 0]
    cross = ((ay > py) != (by > py)) & (px < ax + (py - ay) * (bx - ax) / (by - ay + 1e-12))
    return cross.sum(-1) % 2 == 1


def body_vertices():
    """Base-mesh vertices of the visible body (no helper geometry), from the web assets."""
    meta = json.loads((ROOT / 'static/assets/body.json').read_text())
    t, off, n = meta['arrays']['body.src']
    return np.unique(np.frombuffer((ROOT / 'static/assets/body.bin').read_bytes(), {'uint16': np.uint16, 'uint32': np.uint32}[t], n, off))


def fit(frames, prof, iters=(150, 350), plot=None, log=print, want=()):
    """... want: indices of frames to return the posed body for, in each one's camera frame (for baking skin)."""
    import anny
    import roma
    import torch
    dev = 'mps' if torch.backends.mps.is_available() else 'cpu'
    model = anny.Anny(topology='makehuman', rig='makehuman', skinning_method='lbs', local_changes='default').to(dtype=torch.float32, device=dev)
    for f in Path(os.environ['ANNY_CACHE_DIR']).glob('*/*.safetensors'):
        if f.name.startswith(('load_data_', 'build_model_data_')):
            f.unlink()  # one-time build steps (~1 GB); only the compact model is read after that
    kreg = anny.KeypointsRegressor.coco(model, labels=list(KP))
    to_cam = torch.tensor(TO_CAM, device=dev)
    B, H = len(frames), prof['height_cm'] / 100
    female = prof.get('sex') == 'female'
    age = (date.today() - date.fromisoformat(prof['birth_date'])).days / 365.25 if prof.get('birth_date') else 30
    # Anny's age axis: 0 = baby (1 y), 1/3 = child (11 y), 2/3 = young adult (25 y), 1 = old (90 y)
    age_p = (2 + (age - 25) / 65) / 3 if age >= 25 else (1 + (age - 11) / 14) / 3 if age >= 11 else (age - 1) / 30

    rng = np.random.default_rng(0)
    body = torch.tensor(body_vertices().astype(np.int64), device=dev)
    verts = body[torch.tensor(rng.choice(len(body), 2500, replace=False), device=dev)]
    labels = model.local_change_labels
    # Body-shape changes only; no breast changes on a man (outlines can't tell pecs from breasts).
    local_on = torch.tensor([not any(t in k for t in FACE) and not (not female and k.startswith('breast')) for k in labels], device=dev)
    head_on = torch.tensor([any(t in k for t in HEAD) for k in labels], device=dev)  # shaped by your face scan, if any
    flocal = torch.zeros(len(labels), device=dev, requires_grad=True)
    mirror = torch.tensor([labels.index('r' + k[1:]) if k.startswith('l-') and 'r' + k[1:] in labels else
                           labels.index('l' + k[1:]) if k.startswith('r-') and 'l' + k[1:] in labels else i for i, k in enumerate(labels)], device=dev)
    posed = torch.tensor([model.bone_labels.index(b) for b in POSED], device=dev)

    # Targets, in metres in the image plane (relative to the hip centre).
    C = torch.tensor(np.array([f['sil']['c'] for f in frames]), dtype=torch.float32, device=dev) * H
    L = torch.tensor(np.array([np.array(f['sil']['l'])[list(KP.values())] for f in frames]), dtype=torch.float32, device=dev) * H
    W = np.array([np.array(f['sil']['w'])[list(KP.values())] for f in frames])
    # Ignored: image edges, and clothed legs (shorts' or trousers' outline isn't the legs'): a band from the crotch
    # (4 % of height below the hip centre) to the knees, or to the feet.
    bd = torch.tensor(np.array([borders(np.array(f['sil']['c'])) for f in frames]), dtype=torch.float32, device=dev) * H
    knee = [float(np.mean(np.array(f['sil']['l'])[[25, 26], 1])) for f in frames]
    # Clothed where the clothes check saw clothes, or ran but couldn't tell: loose shorts fitted as your hips and thighs
    # inflate them, and with your weight pinned the torso shrinks to pay for it.
    covered = lambda f, k: 'skin' in f and (f['skin'].get(k) is None or f['skin'][k] < .5)  # noqa: E731
    # Shorts cover from just above the hip line (their bulk would widen your hips) to the knees; trousers to the feet.
    # (Per-point flags miss loose fabric at the edge, so this applies with them too.)
    band = torch.tensor([[-.02, np.inf if covered(f, 'shins') else kn if covered(f, 'thighs') else -.02]
                         for f, kn in zip(frames, knee)], dtype=torch.float32, device=dev) * H
    in_view = lambda p: ((p[..., 0] > bd[:, None, 0] + .01) & (p[..., 0] < bd[:, None, 1] - .01)  # noqa: E731
                         & (p[..., 1] > bd[:, None, 2] + .01) & (p[..., 1] < bd[:, None, 3] - .01)
                         & ~((p[..., 1] > band[:, None, 0]) & (p[..., 1] < band[:, None, 1])))
    off = torch.tensor([f['sil'].get('off', [0] * len(f['sil']['c'])) for f in frames], device=dev) > 0
    # Forearms and hands: hanging by the hips they merge with the torso's outline, and wherever the posed arm is a
    # little off the torso would grow to cover it. Near them (7 cm), the outline says nothing about the body.
    el, wr = L[:, [5, 6]], L[:, [7, 8]]
    hand = wr + (wr - el) / (wr - el).norm(dim=-1, keepdim=True).clamp_min(1e-6) * .1 * H
    arm_a, arm_b = torch.cat([el, wr], 1), torch.cat([wr, hand], 1)
    near_arm = lambda p: seg_dist2(p, arm_a, arm_b).min(-1).values < (.04 * H) ** 2  # noqa: E731
    edge_ok = in_view(C) & ~off & ~near_arm(C)  # nor are outline points on an image edge, clothes or hair
    # Inside the outline, the body must be there too: contour points alone let the arms stand off a shrunken torso
    # (the gap is hidden behind the arm's outer edge). A grid over each outline's box, kept where it's inside and seen.
    lo, hi = C.min(1).values, C.max(1).values
    g = torch.stack(torch.meshgrid(torch.linspace(0, 1, 10, device=dev), torch.linspace(0, 1, 20, device=dev), indexing='ij'), -1).reshape(1, -1, 2)
    grid = lo[:, None] + g * (hi - lo)[:, None]
    grid_ok = (inside(grid, C, torch.roll(C, -1, 1)) & in_view(grid) & ~near_arm(grid)).float()
    kp_ok = in_view(L).float()

    # Shape (shared) and pose (per frame).
    phen = {k: torch.zeros((), device=dev, requires_grad=True) for k in ('muscle', 'weight', 'height', 'proportions')}
    local = torch.zeros(len(model.local_change_labels), device=dev, requires_grad=True)
    rot = torch.zeros(B, len(POSED), 3, device=dev, requires_grad=True)
    cam = torch.zeros(B, 3, device=dev, requires_grad=True)  # log scale, x, y shift

    def shape(n):
        p = {k: torch.sigmoid(v).expand(n) for k, v in phen.items()}
        p |= {'gender': torch.full((n,), 1. if female else 0., device=dev), 'age': torch.full((n,), age_p, device=dev)}  # Anny: 0 = male
        lc = (torch.tanh((local + local[mirror]) / 2) * local_on + torch.tanh((flocal + flocal[mirror]) / 2) * head_on).expand(n, -1)  # left = right
        return {'phenotype_kwargs': p, 'local_changes_kwargs': lc}

    def run():
        pose = torch.eye(4, device=dev).repeat(B, model.bone_count, 1, 1)
        pose[:, posed, :3, :3] = roma.rotvec_to_rotmat(rot)
        return model(pose_parameters=pose, **shape(B))

    # Each frame's turn, from its 3D landmarks against the rest pose's shoulders, hips and head.
    with torch.no_grad():
        rest_k = (kreg(run())[0] @ to_cam.T).cpu().numpy()
    torso = [0, 1, 2, 3, 4, 9, 10]  # nose, ears, shoulders, hips
    R0 = []
    for w in W:
        a, b = rest_k[torso] - rest_k[[9, 10]].mean(0), w[torso] - w[[9, 10]].mean(0)
        b *= np.linalg.norm(a) / np.linalg.norm(b)
        R0.append(kabsch(a, b))
    turn = roma.rotmat_to_rotvec(torch.tensor(np.array(R0), dtype=torch.float32)).to(dev).requires_grad_()

    def project(out):
        R = roma.rotvec_to_rotmat(turn) @ to_cam  # (B,3,3)
        k3 = torch.einsum('bij,bkj->bki', R, kreg(out))
        v3 = torch.einsum('bij,bkj->bki', R, out['vertices'][:, verts])
        hip = k3[:, 9:11].mean(1, keepdim=True)
        s = torch.exp(cam[:, :1])[:, None]
        shift = torch.cat([cam[:, 1:], torch.zeros(B, 1, device=dev)], 1)[:, None]
        return (k3 - hip) * s + shift, (v3 - hip) * s + shift

    tris = torch.tensor(assets()[2], device=dev)  # whole body
    volume = lambda v: torch.einsum('ij,ij->i', v[tris[:, 0]], torch.cross(v[tris[:, 1]], v[tris[:, 2]], -1)).sum().abs() / 6 * 1000  # noqa: E731
    Wt = torch.tensor(W, dtype=torch.float32, device=dev)
    Wt = Wt - Wt[:, 9:11].mean(1, keepdim=True)
    a, b = C, torch.roll(C, -1, 1)

    use = torch.ones(B, device=dev)  # frames the shape is fitted to
    # What the captures measure where outlines can't see (closeups.py: the upper arms' depth): the body must measure it
    # too, at the very slices tape() reports; which vertices bound each slice is re-picked every 50 iterations.
    targets = {k: v for k, v in prof.get('seen', {}).items() if k.startswith(('arm', 'thigh'))}  # girths
    depths = {int(k[1:]) / 100: v for k, v in prof.get('seen', {}).items() if k[0] == 'd'}  # torso depth (cm) at heights hip → shoulder
    torso_v = torch.tensor(np.unique(assets()[0]), device=dev)
    joints = [model.bone_labels.index(f'{b}.{s}') for b in ('upperleg01', 'upperarm01') for s in 'LR']
    rings = {}

    def axes(out):
        heads = out['rest_bone_heads'][0].detach().cpu().numpy()
        bone = lambda b: heads[model.bone_labels.index(b)].tolist()  # noqa: E731
        return (bone('neck01'), bone('head')), {f'{k}_{s.lower()}': (bone(f'{a}.{s}'), bone(f'{b}.{s}')) for s in 'LR' for k, a, b in
                                                (('arm', 'upperarm01', 'lowerarm01'), ('thigh', 'upperleg01', 'lowerleg01'), ('calf', 'lowerleg01', 'foot'))}

    def pick_rings(out):
        V = out['rest_vertices'][0].detach().cpu().numpy() * 100
        for k, (at, n, tri) in slices(V / 100, *axes(out), female)[0].items():
            if k in targets:
                ids = hull(V, at, n, tri)
                p = V[ids] - np.outer(V[ids] @ n, n)
                scale = ring(V, at, n, tri) / max(np.linalg.norm(p - np.roll(p, 1, 0), axis=1).sum(), 1e-6)  # vertices to tape()'s slice
                rings[k] = (torch.tensor(ids, device=dev), torch.tensor(n, dtype=torch.float32, device=dev), scale)
    kp_shape = torch.tensor([0 if k in KP_FREE else KP_SHAPE for k in KP], device=dev)
    # shortcut: linear maps from body fat % and fat-free mass index to Anny's fat and muscle axes, eyeballed on its
    # renders (lean ~10 % men show definition near muscle .65 / fat .3); refine against reference captures.
    off = (10, 3) if female else (0, 0)
    fat_t = float(np.clip(.2 + (prof['bf'] - 8 - off[0]) / 40, .1, .9)) if 'bf' in prof else .45
    muscle_t = float(np.clip(.5 + (prof['ffmi'] - 19 + off[1]) / 8, .2, 1)) if 'ffmi' in prof else .6
    # Definition the outlines can't see (the pecs' lower edge, a toned stomach): leaner shows more, ~.5 at 13 % for men.
    # shortcut: eyeballed on Anny's renders; the side views still win where they constrain it
    lean = float(np.clip((18 + off[0] - prof['bf']) / 10, 0, .7)) if 'bf' in prof else 0.
    define = {labels.index(k): lean * share for k, share in (('stomach-tone-incr', 1), ('torso-muscle-pectoral-incr', 0 if female else .25)) if k in labels}  # flat pecs, as in the captures
    # shortcut: the front chest (armpit fold to fold) hides under the arms in every outline and SAM can't part arm from
    # pec reliably; on the reference captures it's a little broad, the pec's corner under the deltoid: eyeballed at +.2
    if not female and 'measure-frontchest-dist-incr' in labels:
        define[labels.index('measure-frontchest-dist-incr')] = .2

    def loss(stage):
        out = run()
        k3, v3 = project(out)
        k2, v2 = k3[..., :2], v3[..., :2]
        # 2D landmarks where they're in view; 3D landmarks (scale-free) for depth and turn.
        e_kp = ((k2 - L) ** 2).sum(-1).mul(kp_ok)
        # Once each frame is posed, MediaPipe's shoulder and hip points (not quite where the body's edges are) mustn't
        # pull the shape: the outline decides it.
        kw = kp_shape if stage else torch.ones(len(KP), device=dev)
        l_kp = (e_kp * use[:, None] * kw).sum() / (kp_ok * use[:, None] * kw).sum()
        w = Wt * (k3.norm(dim=-1).mean(1) / Wt.norm(dim=-1).mean(1))[:, None, None]
        l_3d = (((k3 - w) ** 2).sum(-1).mean(1) * use).sum() / use.sum()
        rest = out['rest_vertices'][0]
        l_h = (rest[body, 2].max() - rest[body, 2].min() - H) ** 2
        mass = volume(rest) * DENSITY
        # Outlines can't separate muscle from fat: lean on your body composition (last fit), else a fit adult.
        l_reg = (rot ** 2).mean() * .01 + (cam[:, 0] ** 2).mean() * .1 + (torch.tanh(local) ** 2).mean() * .02 \
            + .004 * ((torch.sigmoid(phen['muscle']) - muscle_t) ** 2 + (torch.sigmoid(phen['weight']) - fat_t) ** 2) \
            + .01 * sum((torch.tanh(local[i]) - t) ** 2 for i, t in define.items())
        total = l_kp + .3 * l_3d + 10 * l_h + l_reg
        if prof.get('weight') and stage:  # outlines leave volume loose (±10 %); a weigh-in pins it, trusted less with age
            trust = min(1, max(0, 1 - (prof.get('weight_age', 0) - 4) / 180))
            total = total + .3 * trust * ((mass - prof['weight']) / prof['weight']) ** 2
        parts = {'kp': l_kp.item() ** .5 * 100, '3d': l_3d.item() ** .5 * 100, 'kg': mass.item()}
        if stage and rings:
            l_seen = 0
            for k, (ids, n, scale) in rings.items():
                p = rest[ids] * 100
                p = p - (p @ n)[:, None] * n
                l_seen = l_seen + (((p - torch.roll(p, 1, 0)).norm(dim=1).sum() * scale - targets[k]) / targets[k]) ** 2
            total = total + 50 * l_seen
            parts['seen'] = (l_seen.item() / len(rings)) ** .5 * 100  # rms % off what the captures measure
        if stage and depths:  # front to back, at heights between the hip and shoulder joints (closeups.torso)
            heads = out['rest_bone_heads'][0][joints, 2]
            hip, sh, T = heads[:2].mean(), heads[2:].mean(), rest[torso_v]
            l_depth = 0
            for f, d in depths.items():
                band = T[((T[:, 2] - (hip + f * (sh - hip))).abs() < .01).detach()]
                l_depth = l_depth + ((band[:, 1].max() - band[:, 1].min()) * 100 / d - 1) ** 2
            total = total + 50 * l_depth
            parts['depth'] = (l_depth.item() / len(depths)) ** .5 * 100
        if stage:
            # Mesh outside the outline (where the camera sees) is too big; outline nowhere near the mesh is too small.
            d2 = seg_dist2(v2, a, b).min(-1).values
            out_m = (~inside(v2, a, b) & in_view(v2) & ~near_arm(v2)).float() * use[:, None]
            # past 5 mm: each frame's arm pose is a little off, and paying for every millimetre a thick arm strays
            # outside would thin the limbs and shoulders to hedge
            l_out = ((d2.clamp_min(1e-12).sqrt() - .005).clamp_min(0) ** 2 * out_m).sum() / use.sum() / v2.shape[1]
            near = ((C[:, :, None] - v2[:, None]) ** 2).sum(-1).min(-1).values
            l_in = (near * edge_ok * use[:, None]).sum() / (edge_ok * use[:, None]).sum()
            e_sil = ((d2 * out_m).mean(1) + (near * edge_ok).sum(1) / edge_ok.sum(1).clamp_min(1)).detach().sqrt()  # per frame
            gap = (((grid[:, :, None] - v2[:, None]) ** 2).sum(-1).min(-1).values.sqrt() - .02).clamp_min(0) ** 2  # >2 cm from any body point
            l_fill = (gap * grid_ok * use[:, None]).sum() / (grid_ok * use[:, None]).sum().clamp_min(1)
            total = total + 20 * (l_out + l_in + l_fill)
            parts |= {'out': ((d2 * out_m).sum() / out_m.sum().clamp_min(1)).item() ** .5 * 100, 'in': l_in.item() ** .5 * 100, 'fill': l_fill.item() ** .5 * 100}
        return total, parts, out, e_kp, (e_sil if stage else None)

    t0 = time.time()
    for stage, n in enumerate([*iters, 120]):  # stage 2: refine without outlines the shared body can't explain
        if stage == 2:
            e = e_sil[use > 0]
            bad = (e_sil > max(.03, 2.5 * e.median().item())) & (use > 0)  # someone else, a misread mask
            if not bad.any():
                break
            use = use * ~bad
            log(f'{int(bad.sum())} frames dropped: their outline fits no shared body')
        params = [rot, turn, cam] + ([local] + list(phen.values()) if stage else [])
        opt = torch.optim.Adam(params, lr=.02 if stage else .05)
        for i in range(n):
            if stage and targets and i % 50 == 0:
                with torch.no_grad():
                    pick_rings(run())
            opt.zero_grad()
            total, parts, out, e_kp, e_sil = loss(stage)
            total.backward()
            opt.step()
            if i % 50 == 0 or i == n - 1:
                log(f'stage {stage} it {i:3d}  ' + '  '.join(f'{k} {v:.2f}' for k, v in parts.items()) + f'  ({time.time() - t0:.0f}s)')
        if not stage:  # misdetected frames (landmarks no body pose can match) don't shape the body
            e = (e_kp.sum(1) / kp_ok.sum(1).clamp_min(1)).detach().sqrt()
            use = (e < max(.03, 2.5 * e.median().item())).float()
            log(f'{int(use.sum())} of {B} frames fit the landmarks')

    if prof.get('face_scan') is not None:  # your face scan: the head's own controls, the body as fitted
        lm, ids, wts = face_anchors('female' if female else 'male')
        S = torch.tensor(np.array(prof['face_scan'])[lm], dtype=torch.float32, device=dev) @ torch.tensor([[1., 0, 0], [0, 0, -1], [0, 1, 0]], device=dev).T
        S = S - S.mean(0)  # ARKit (x: the face's left, y up, z forward) -> Anny (x left, y back, z up), centred
        ids, wts = torch.tensor(ids, device=dev), torch.tensor(wts, dtype=torch.float32, device=dev)
        eye4 = torch.eye(4, device=dev)[None, None].repeat(1, model.bone_count, 1, 1)
        opt = torch.optim.Adam([flocal], lr=.03)
        for i in range(200):
            opt.zero_grad()
            P = (model(pose_parameters=eye4, **shape(1))['rest_vertices'][0][ids] * wts[..., None]).sum(1)
            P = P - P.mean(0)
            R = torch.tensor(kabsch(S.detach().cpu().numpy(), P.detach().cpu().numpy()), dtype=torch.float32, device=dev)  # turn only
            e = ((P - S @ R.T) ** 2).sum(1)
            (e.mean() + 1e-5 * (torch.tanh(flocal) ** 2).sum()).backward()
            opt.step()
            if i in (0, 199):
                log(f'face {i:3d}  {e.sqrt().median().item() * 1000:.1f} mm off your scan (median)')

    with torch.no_grad():
        out = run()
        k3, v3 = project(out)
        rest = out['rest_vertices'][0].cpu().numpy()
        neck, limbs = axes(out)
        res = {'phenotypes': {k: round(torch.sigmoid(v).item(), 4) for k, v in phen.items()} | {'age': round(age_p, 4)},
               'local': {k: round(v, 4) for k, v in zip(labels, shape(1)['local_changes_kwargs'][0].tolist()) if abs(v) > 1e-3},
               'fit': {k: round(v, 2) for k, v in parts.items()}, 'neck_axis': neck, 'limb_axes': limbs}
        # Each camera's outline error (cm, median over its frames used): one drifting (bumped, refocused) shows here.
        cams, kept = np.array([f.get('cam', 0) for f in frames]), use.cpu().numpy() > 0
        res['fit_cams'] = {int(c): round(float(np.median(e_sil.cpu().numpy()[(cams == c) & kept])) * 100, 2)
                           for c in set(cams.tolist()) if ((cams == c) & kept).any()}
        res['dropped'] = int(B - kept.sum())
        if plot:
            draw(plot, frames, C.cpu().numpy(), out['vertices'], model, verts, project, out)
        fixed = shape(1)  # computed without gradients: the stance only moves bones
        R, kk, views = roma.rotvec_to_rotmat(turn) @ to_cam, kreg(out), {}
        for j in want:  # every vertex as the fit posed and projected it: metres from the hip centre, z away from the camera
            v, k3 = out['vertices'][j] @ R[j].T, kk[j] @ R[j].T
            q = (v - k3[9:11].mean(0)) * torch.exp(cam[j, 0])
            q[:, :2] += cam[j, 1:]
            views[j] = q.cpu().numpy()
    return res, rest, stance(model, fixed, kreg, shoulder_slope(frames)), views


def shoulder_slope(frames):
    """How your shoulders slope from the neck, seen from the front (degrees, + falling outward): the median over front
    frames of the outline's top at .65 and 1.0 of the way from between the shoulder points to each. Or None."""
    out = []
    for f in frames:
        if 20 <= abs(f.get('theta', 90)) <= 160 or 'sil' not in f:
            continue
        L, C = np.array(f['sil']['l']), np.array(f['sil']['c'])
        c, nxt = L[[11, 12]].mean(0), np.roll(C, -1, 0)
        for s in (11, 12):
            ys = []
            for x in (c[0] + .65 * (L[s][0] - c[0]), L[s][0]):
                cross = (C[:, 0] - x) * (nxt[:, 0] - x) < 0
                y = C[cross, 1] + (nxt[cross, 1] - C[cross, 1]) * (x - C[cross, 0]) / (nxt[cross, 0] - C[cross, 0])
                ys.append(y.min() if len(y) else None)
            if None not in ys and ys[0] < L[s][1]:  # image y runs down
                out.append(np.degrees(np.arctan2(ys[1] - ys[0], abs(L[s][0] - c[0]) * .35)))
    return float(np.median(out)) if len(out) >= 20 else None


def stance(model, shape, kreg, slope=None):
    """Bone transforms (posed × rest⁻¹, Anny frame) for a relaxed, confident stance: feet a little wider than the hips,
    toes turned out, weight nearly even (a little more on the left: pelvis tipped 2°, the spine bringing the head back
    upright), knees over the feet; arms 8° out and elbows softly bent (the right a little more) with the forearm's
    outward carrying angle, never inside the hips; palms toward the thighs, hands in line with the forearms, fingers
    together and curled toward the palm, more toward the little finger. The shoulder bone shares the arm's turn so the deltoid
    keeps its shape. (Your own arm angles from MediaPipe's 3D landmarks were too noisy to look natural.)"""
    import roma
    import torch
    dev = shape['local_changes_kwargs'].device
    names = [f'{b}.{s}' for b in ('upperarm01', 'lowerarm01', 'shoulder01', 'upperleg01', 'foot', 'lowerleg01') for s in 'LR'] + ['spine05', 'spine03']
    bones = torch.tensor([model.bone_labels.index(b) for b in names], device=dev)
    rot = torch.zeros(len(names), 3, device=dev, requires_grad=True)
    reg = torch.tensor([1, 1, 1, 1, 3, 3, 1, 1, 1, 1, 1, 1, 2, 2.], device=dev)[:, None]  # the shoulder takes about a quarter of the turn
    root = model.bone_labels.index('root')
    clav = torch.tensor([model.bone_labels.index(f'clavicle.{s}') for s in 'LR'], device=dev)
    lift = torch.zeros(2, device=dev, requires_grad=slope is not None)  # each collarbone's turn about the front-back axis: the shoulder's height
    tilt = torch.eye(3, device=dev)  # the pelvis, set below once the upright targets are taken
    twist_b = torch.tensor([model.bone_labels.index(f'lowerarm02.{s}') for s in 'LR'], device=dev)
    twist = torch.zeros(2, device=dev, requires_grad=True)  # forearm twist turns the palm
    wrist_b = torch.tensor([model.bone_labels.index(f'wrist.{s}') for s in 'LR'], device=dev)
    wrot = torch.zeros(2, 3, device=dev, requires_grad=True)  # the wrist only straightens the hand (aimed below)
    axes = torch.zeros(2, 3, device=dev)
    curl = {}  # finger bone -> rotvec (set below)

    def run(tw=None):
        pose = torch.eye(4, device=dev).repeat(1, model.bone_count, 1, 1)
        turns = roma.rotvec_to_rotmat(rot)
        tilt_up = roma.rotvec_to_rotmat(torch.stack([torch.zeros_like(lift), lift, torch.zeros_like(lift)], 1))
        turns = torch.cat([tilt_up.transpose(1, 2) @ turns[:2], turns[2:]])  # the upper arms (names 0, 1) turn back: they still hang
        pose[0, bones, :3, :3] = turns
        pose[0, root, :3, :3] = tilt
        pose[0, clav, :3, :3] = tilt_up
        pose[0, twist_b, :3, :3] = roma.rotvec_to_rotmat(axes * (twist if tw is None else tw)[:, None])
        pose[0, wrist_b, :3, :3] = roma.rotvec_to_rotmat(wrot)
        for b, rv in curl.items():
            pose[0, b, :3, :3] = roma.rotvec_to_rotmat(rv)
        return model(pose_parameters=pose, **shape)
    with torch.no_grad():
        out = run()
        k, V = kreg(out)[0], out['vertices'][0]  # both in the posed output's frame
        top = model.vertex_bone_indices[torch.arange(len(V)), model.vertex_bone_weights.argmax(1)]
        limb = torch.tensor([any(t in b for t in ('arm', 'hand', 'finger', 'wrist', 'shoulder', 'metacarpal', 'clavicle'))
                             for b in model.bone_labels], device=dev)[top]
        heads = out['rest_bone_heads'][0]
        at = lambda b: heads[model.bone_labels.index(b)]  # noqa: E731  a bone's head at rest
        nrm = lambda v: v / v.norm()  # noqa: E731
        for j, s in enumerate('LR'):  # the forearm turns about itself (Anny turns each bone about axes aligned with the world at rest)
            axes[j] = nrm(at(f'wrist.{s}') - at(f'lowerarm01.{s}'))
        part = lambda b: top == model.bone_labels.index(b)  # noqa: E731  vertices a bone mostly moves
        # Fingers: their spread closes most of the way to the middle finger's line, then each joint bends about its own
        # flexion axis, across the finger and along the palm (palms face down at rest), so they curl toward the palm
        # without twisting or crossing. Relaxed: index 10/18/8° (knuckle, middle, end), 3° more a finger toward the little one.
        for s in 'LR':
            palm = torch.linalg.cross(at(f'finger2-1.{s}') - at(f'wrist.{s}'), at(f'finger5-1.{s}') - at(f'wrist.{s}'))
            palm = nrm(palm) * (1 if palm[2] < 0 else -1)
            mid = nrm(at(f'finger3-3.{s}') - at(f'finger3-1.{s}'))
            for f in (2, 3, 4, 5):
                d, splay = nrm(at(f'finger{f}-3.{s}') - at(f'finger{f}-1.{s}')), torch.zeros(3, device=dev)
                if f != 3:
                    splay = nrm(torch.linalg.cross(d, mid)) * torch.arccos(torch.clamp(d @ mid, -1, 1)) * .8
                    d = roma.rotvec_to_rotmat(splay) @ d
                flex = nrm(torch.linalg.cross(d, palm))
                for jn, deg in ((1, 10), (2, 18), (3, 8)):
                    rv = flex * math.radians(deg + 3 * (f - 2))
                    curl[model.bone_labels.index(f'finger{f}-{jn}.{s}')] = roma.rotmat_to_rotvec(roma.rotvec_to_rotmat(rv) @ roma.rotvec_to_rotmat(splay)) if jn == 1 else rv
        for s in 'LR':  # thumb tucked beside the index finger (tip ~2 cm from its middle segment)
            b, tip, index = model.bone_labels.index(f'finger1-1.{s}'), part(f'finger1-3.{s}'), part(f'finger2-2.{s}')
            best = (1e9, None)
            for turn in [torch.zeros(3, device=dev)] + [e * sign * t for e in torch.eye(3, device=dev) for sign in (1, -1) for t in (.15, .3, .45)]:
                curl[b] = turn
                V1 = run()['vertices'][0]
                best = min(best, (abs((V1[tip].mean(0) - V1[index].mean(0)).norm().item() - .02), turn), key=lambda x: x[0])
            curl[b] = best[1]
        knuckles2 = [(part(f'finger2-1.{s}'), part(f'finger5-1.{s}')) for s in 'LR']  # index and little knuckles
        facing = []  # the sign that makes cross(index − wrist, little − wrist) point out of the palm (palms face down at rest)
        for j, (ix, lt) in enumerate(knuckles2):
            n0 = torch.linalg.cross(V[ix].mean(0) - k[7 + j], V[lt].mean(0) - k[7 + j])
            facing.append(1. if n0[2] < 0 else -1.)
        unit = lambda v: np.array(v) / np.linalg.norm(v)  # noqa: E731  x out (your left), y back, z up
        r = math.radians
        arms = [unit([math.tan(r(8)), -math.tan(r(2)), -1]), unit([math.tan(r(4)), -math.tan(r(20)), -1])]
        forearm = {1: arms[1], -1: unit([math.tan(r(4)), -math.tan(r(25)), -1])}  # relaxed isn't symmetric: the right a little more bent
        hand = {s: unit([math.tan(r(3)), -math.tan(r(d + 3)), -1]) for s, d in ((1, 20), (-1, 25))}  # in line with the forearm
        knuckles = [part(f'finger3-1.{s}') for s in 'LR']
        targets = []
        for side, (s, e, w) in ((1, (3, 5, 7)), (-1, (4, 6, 8))):  # KP order: shoulders 3/4, elbows 5/6, wrists 7/8
            S, Lu, Ll = k[s], (k[e] - k[s]).norm(), (k[w] - k[e]).norm()
            mirror = torch.tensor([side, 1, 1.], device=dev)
            E = S + Lu * torch.tensor(arms[0], dtype=torch.float32, device=dev) * mirror
            W = E + Ll * torch.tensor(forearm[side], dtype=torch.float32, device=dev) * mirror
            band = V[~limb & ((V[:, 2] - W[2]).abs() < .03) & (V[:, 0] * side > 0)]
            if len(band):  # never inside the hips: the wrist at least 2 cm clear of them
                W[0] = side * max(abs(W[0].item()), band[:, 0].abs().max().item() + .02)
            targets += [(e, E), (w, W)]
        for a, h, out, fwd in ((13, 9, 1.4, 0), (14, 10, 1.45, .02)):  # feet a little wider than the hips (rest: splayed 35 cm
            A = k[a].clone()                                              # apart); the right a little forward
            A[0], A[1] = k[h, 0] * out - .01, A[1] - fwd  # the hips a centimetre toward the left foot
            targets.append((a, A))
        targets += [(i, k[i].clone()) for i in (0, 1, 2)]  # the head stays where it was: the spine bends back over the hips
        soles, toes = [], []
        for s in 'LR':  # the lowest fifth of each foot: kept level, so the foot stays flat as the leg swings in
            f = torch.nonzero(part(f'foot.{s}')).squeeze(1)
            sole = f[V[f, 2] <= torch.quantile(V[f, 2], .2)]
            soles.append(sole)
            front, back = V[sole, 1] <= torch.quantile(V[sole, 1], .2), V[sole, 1] >= torch.quantile(V[sole, 1], .8)
            toes.append((sole[back], sole[front]))  # heel to toes: turned ~12° out, as feet rest
        # A little more weight on the left leg: the pelvis tips 2° (right hip down); legs and spine reach the same feet and
        # head targets around it.
        rest_root = run()['rest_bone_poses'][0][root]
        a = -math.radians(2)
        world = torch.tensor([[math.cos(a), 0, math.sin(a)], [0, 1, 0], [-math.sin(a), 0, math.cos(a)]], device=dev)
        tilt = rest_root[:3, :3].T @ world @ rest_root[:3, :3]
        pivot = torch.eye(4, device=dev)  # the same tilt about the pelvis, kept after the root is put back below
        pivot[:3, :3], pivot[:3, 3] = world, rest_root[:3, 3] - world @ rest_root[:3, 3]
    opt = torch.optim.Adam([rot, twist, wrot], lr=.03)
    for _ in range(250):
        opt.zero_grad()
        o = run()
        kp, out_v = kreg(o)[0], o['vertices'][0]
        loss = sum(((kp[i] - t) ** 2).sum() for i, t in targets) + 1e-4 * (reg * rot ** 2).sum() + 1e-4 * (wrot ** 2).sum()
        loss = loss + sum(out_v[f, 2].var() for f in soles) + 10 * (out_v[soles[0], 2].mean() - out_v[soles[1], 2].mean()) ** 2  # both on the floor
        for (heel, toe), side in zip(toes, (1, -1)):
            d = out_v[toe].mean(0)[:2] - out_v[heel].mean(0)[:2]
            aim = torch.tensor([side * math.sin(r(12)), -math.cos(r(12))], dtype=d.dtype, device=dev)
            loss = loss + .05 * (1 - d @ aim / d.norm())
        for j, side in enumerate((1, -1)):  # Anny's rest pose cocks the hand out and forward: aim it down
            d = out_v[knuckles[j]].mean(0) - kp[7 + j]
            aim = torch.tensor(hand[side] * [side, 1, 1], dtype=d.dtype, device=dev)
            loss = loss + .05 * (1 - d @ aim / d.norm())
        for j, ((ix, lt), side) in enumerate(zip(knuckles2, (1, -1))):  # palms toward the thighs, turned a little back
            n = torch.linalg.cross(out_v[ix].mean(0) - kp[7 + j], out_v[lt].mean(0) - kp[7 + j]) * facing[j]
            aim = torch.tensor([-side, .35, 0], dtype=n.dtype, device=dev)
            loss = loss + .3 * (1 - n @ aim / n.norm() / aim.norm())
        loss.backward()
        opt.step()
    if slope is not None:  # then the collarbones alone lift the shoulders until they slope as yours do (shoulder_slope)
        rot, twist, wrot = rot.detach(), twist.detach(), wrot.detach()
        opt = torch.optim.Adam([lift], lr=.02)
        for _ in range(80):
            opt.zero_grad()
            o = run()
            kp, out_v, loss = kreg(o)[0], o['vertices'][0], 0
            mid = (kp[3] + kp[4]) / 2
            for j in (3, 4):
                top = []
                for x in (mid[0] + .65 * (kp[j, 0] - mid[0]), kp[j, 0]):  # the outline's top, near the neck and at the joint
                    band = ((out_v[:, 0] - x).abs() < .005) & (out_v[:, 2] < kp[j, 2] + .12)  # below the head
                    top.append(out_v[band.detach(), 2].max())
                loss = loss + (torch.atan2(top[0] - top[1], (kp[j, 0] - mid[0]).abs() * .35) - math.radians(slope)) ** 2
            loss.backward()
            opt.step()
    with torch.no_grad():
        out = run()
        G = out['bone_poses'][0] @ torch.linalg.inv(out['rest_bone_poses'][0])
        return (pivot @ torch.linalg.inv(G[root]) @ G).cpu().numpy()  # posed output sits elsewhere: keep the root put, tilted


def assets():
    """From the web assets: body triangles without the arms (base-mesh indices), and the navel's vertex."""
    meta = json.loads((ROOT / 'static/assets/body.json').read_text())
    buf = (ROOT / 'static/assets/body.bin').read_bytes()
    types = {'uint8': np.uint8, 'uint16': np.uint16, 'uint32': np.uint32, 'float32': np.float32}
    arr = lambda k: np.frombuffer(buf, types[meta['arrays'][k][0]], meta['arrays'][k][2], meta['arrays'][k][1])  # noqa: E731
    F = arr('body.src')[arr('body.tris').reshape(-1, 3)].astype(np.int64)
    arm = arr('weights').reshape(-1, 6)[:, 2:4].sum(1) > 127
    T = arr('male').reshape(-1, 3)  # template: y up, +z front
    y0, H = T[np.unique(F), 1].min(), np.ptp(T[np.unique(F), 1])
    mid = np.flatnonzero((np.abs(T[:, 0]) < 1) & (T[:, 2] > 0) & (np.abs(T[:, 1] - y0 - .615 * H) < .04 * H))
    mid = np.intersect1d(mid, np.unique(F))
    # the navel: the front-midline vertex furthest behind its neighbours 1-3 cm above and below
    navel = min(mid, key=lambda k: T[k, 2] - np.mean([T[q, 2] for q in mid if 1 < abs(T[q, 1] - T[k, 1]) < 3] or [T[k, 2]]))
    return F[~arm[F].any(1)], int(navel), F, arr('weights').reshape(-1, 6) / 255


def face_anchors(sex):
    """Where each face landmark (MediaPipe's 478) sits on the body mesh: (landmark indices, their triangles' base-mesh
    vertex ids (K, 3), barycentric weights (K, 3)), from the UVs the face photo is mapped with (web assets)."""
    meta = json.loads((ROOT / 'static/assets/body.json').read_text())
    buf = (ROOT / 'static/assets/body.bin').read_bytes()
    types = {'uint8': np.uint8, 'uint16': np.uint16, 'uint32': np.uint32, 'float32': np.float32}
    arr = lambda k: np.frombuffer(buf, types[meta['arrays'][k][0]], meta['arrays'][k][2], meta['arrays'][k][1])  # noqa: E731
    uv, src, tris = arr('body.uv').reshape(-1, 2), arr('body.src').astype(np.int64), arr('body.tris').reshape(-1, 3).astype(np.int64)
    head = arr('weights').reshape(-1, 6)[:, 0] > 127
    tris = tris[head[src[tris]].all(1)]
    A = arr(f'face_uv.{sex}').reshape(-1, 2)
    a, b, c = uv[tris[:, 0]], uv[tris[:, 1]], uv[tris[:, 2]]
    v0, v1 = b - a, c - a
    den = v0[:, 0] * v1[:, 1] - v1[:, 0] * v0[:, 1]
    lm, ids, wts = [], [], []
    for k, q in enumerate(A):
        if np.isnan(q).any():
            continue
        v2 = q - a
        with np.errstate(divide='ignore', invalid='ignore'):
            u = (v2[:, 0] * v1[:, 1] - v1[:, 0] * v2[:, 1]) / den
            v = (v0[:, 0] * v2[:, 1] - v2[:, 0] * v0[:, 1]) / den
        hit = np.flatnonzero((u >= -1e-6) & (v >= -1e-6) & (u + v <= 1 + 1e-6))
        if len(hit):
            t = hit[0]
            lm.append(k)
            ids.append(src[tris[t]])
            wts.append([1 - u[t] - v[t], u[t], v[t]])
    return np.array(lm), np.array(ids), np.clip(np.array(wts), 0, 1)


def ring(V, at, n, tri):
    """Tape length (cm) round the slice of the triangles tri through point at, across direction n: the convex hull of the
    slice, like a tape bridging hollows. V: vertices, cm."""
    import cv2
    a, b, c = V[tri[:, 0]], V[tri[:, 1]], V[tri[:, 2]]
    da, db, dc = (a - at) @ n, (b - at) @ n, (c - at) @ n
    pts = []
    for p, q, dp, dq in ((a, b, da, db), (b, c, db, dc), (c, a, dc, da)):
        s = dp * dq < 0
        pts.append(p[s] + (q[s] - p[s]) * (dp[s] / (dp[s] - dq[s]))[:, None])
    P = np.concatenate(pts) - at
    u = np.cross(n, [1, 0, 0] if abs(n[0]) < .9 else [0, 1, 0])
    u /= np.linalg.norm(u)
    P2 = np.stack([P @ u, P @ np.cross(n, u)], 1).astype(np.float32)
    return cv2.arcLength(cv2.convexHull(P2), True) if len(P2) > 2 else 0


def slices(rest, neck=None, limbs=None, female=False):
    """Where each girth is taped on the rest-pose body (Anny frame, m): {part: (point cm, direction, triangles)}, plus the
    navel's height (fraction) and the body's height (cm). Torso slices skip the arms. Neck: narrowest below the jaw,
    across the neck's own axis (it leans forward, so a level slice catches the chin); chest: widest; waist: at the navel;
    hips: widest round the seat. limbs: {'arm_l': (shoulder, elbow), ...} bone ends: arms at their midpoint, thighs 40 %
    down, calves at their widest, each across the bone and on that limb alone."""
    F, navel, Fall, reg = assets()
    V = rest * 100
    z0, H = V[np.unique(F), 2].min(), np.ptp(V[np.unique(F), 2])
    up = np.array([0, 0, 1.])
    level = lambda f: (np.array([0, 0, z0 + f * H]), up, F)  # noqa: E731
    length = lambda s: ring(V, *s)  # noqa: E731
    out = {}
    if neck is not None:
        b0, b1 = np.asarray(neck[0]) * 100, np.asarray(neck[1]) * 100
        ax = (b1 - b0) / np.linalg.norm(b1 - b0)
        out['neck'] = min(((b0 + (b1 - b0) * t, ax, F) for t in np.linspace(.2, .8, 13)), key=length)
    else:
        out['neck'] = min((level(f) for f in np.arange(.83, .88, .005)), key=length)
    for k, (p0, p1) in (limbs or {}).items():
        p0, p1 = np.asarray(p0) * 100, np.asarray(p1) * 100
        r = ['arm_l', 'arm_r', 'leg_l', 'leg_r'].index(k.replace('thigh', 'leg').replace('calf', 'leg')) + 2
        tri = Fall[(reg[Fall, r] > .5).all(1)]
        ax = (p1 - p0) / np.linalg.norm(p1 - p0)
        ts = [.5] if k.startswith('arm') else [.4] if k.startswith('thigh') else np.linspace(.2, .45, 6)
        out[k] = max(((p0 + (p1 - p0) * t, ax, tri) for t in ts), key=length)
    chest = [level(f) for f in np.arange(.68, .76, .005)]
    widest = int(np.argmax([length(c) for c in chest]))
    out['chest'] = chest[widest]
    if female:  # underbust: just below where the breasts start, the steepest rise in girth under the fullest ring (cup size)
        f = .68 + .005 * widest
        below = [level(g) for g in np.arange(f - .06, f, .005)]
        out['underbust'] = below[int(np.argmax(np.diff([length(s) for s in below])))]
    out['waist'] = (np.array([0, 0, V[navel, 2]]), up, F)
    out['hips'] = max((level(f) for f in np.arange(.46, .56, .005)), key=length)
    return out, float((V[navel, 2] - z0) / H), float(H)


def tape(rest, neck=None, limbs=None, female=False):
    """Tape-style circumferences (cm) of the rest-pose body, at slices()."""
    sl, navel_h, H = slices(rest, neck, limbs, female)
    return {k: round(ring(rest * 100, *s), 1) for k, s in sl.items()} | {'navel_h': round(navel_h, 3), 'height': round(H, 1)}


def hull(V, at, n, tri, band=1.):
    """The mesh vertices within `band` cm of a slice that bound it, in order round it (V in cm)."""
    import cv2
    vs = np.unique(tri)
    vs = vs[np.abs((V[vs] - at) @ n) < band]
    u = np.cross(n, [1, 0, 0] if abs(n[0]) < .9 else [0, 1, 0])
    u /= np.linalg.norm(u)
    P = np.stack([(V[vs] - at) @ u, (V[vs] - at) @ np.cross(n, u)], 1).astype(np.float32)
    return vs[cv2.convexHull(P, returnPoints=False)[:, 0]]


def restance(db, day, frames, prof):
    """Recompute the stance of a stored fit from its parameters (seconds, not a refit) and save it."""
    import sqlite3 as sq
    import anny
    import torch
    with sq.connect(db) as c:
        params, blob = c.execute('select params, mesh from fits where day = ?', (day,)).fetchone()
    res = json.loads(params)
    model = anny.Anny(topology='makehuman', rig='makehuman', skinning_method='lbs', local_changes='default').to(dtype=torch.float32)
    lc = torch.zeros(1, len(model.local_change_labels))
    for k, v in res['local'].items():
        lc[0, model.local_change_labels.index(k)] = v
    p = {k: torch.tensor([v]) for k, v in res['phenotypes'].items()} | {'gender': torch.tensor([1. if prof.get('sex') == 'female' else 0.])}
    G = stance(model, {'phenotype_kwargs': p, 'local_changes_kwargs': lc}, anny.KeypointsRegressor.coco(model, labels=list(KP)), shoulder_slope(frames))
    rest = model(pose_parameters=torch.eye(4)[None, None].repeat(1, model.bone_count, 1, 1), phenotype_kwargs=p,
                 local_changes_kwargs=lc)['rest_vertices'][0].detach().numpy()
    store(db, day, res, rest, G)
    return f'{day}: stance updated'


def bake_all(db, prof):
    """Bake your skin from every reference capture (any day), when there are new ones: the body is fitted to those
    frames alone, which poses and projects it for each."""
    refs = [p for p in (Path(db).parent / 'reference').glob('*.json') if p.with_suffix('.jpg').exists()]
    done = Path(db).parent / 'skin.json'
    if not refs or (done.exists() and done.stat().st_mtime > max(p.stat().st_mtime for p in refs)):
        return
    recs = [(p, json.loads(p.read_text())) for p in refs]
    recs = [(p, r) for p, r in recs if r.get('ref') and r.get('sil')]
    if len(recs) < MIN_FRAMES:
        return
    _, _, _, views = fit([r for _, r in recs], prof, want=range(len(recs)), log=lambda *a: None)
    bake_skin(db, prof, {recs[j][1]['ts']: (recs[j][0], v) for j, v in views.items()})


def bake_skin(db, prof, views):
    """Bake your skin texture from the day's reference captures (bake.py) into skin.jpg beside the database."""
    import cv2
    import bake
    H = prof['height_cm'] / 100
    src, uv, tris, head = bake.assets()
    vs = []
    for ts, (path, cam3) in views.items():
        if not path.exists():  # replaced by a better capture meanwhile
            continue
        rec, img = json.loads(path.read_text()), cv2.imread(str(path.with_suffix('.jpg')))
        info = rec.get('ref')
        if img is None or not info:  # older captures don't record where their crop sits
            continue
        px = (np.array(info['origin']) + cam3[:, :2] / H * info['H'] - info['box'][::-1]) * info['k']
        vs.append({'img': img, 'px': px, 'cam': cam3, 'skin': bake.skin_mask(img), 'weight': info.get('score', 1)})
    if not vs:
        return
    tex, cover = bake.bake(vs, src, uv, tris, base=cv2.imread(str(ROOT / f"static/assets/skin_{prof.get('sex', 'male')}.jpg")), skip=head)
    ys, xs = bake.texels(uv, tris, 256)[:2]
    seen = cv2.resize(cover, (256, 256), interpolation=cv2.INTER_AREA)[ys, xs].mean()  # share of the body's surface, not of the image
    print(f'skin baked from {len(vs)} captures, {seen * 100:.0f} % of the body seen')
    if seen >= .5:  # patchier than that looks worse than the template
        cv2.imwrite(str(Path(db).parent / 'skin.jpg'), tex, [cv2.IMWRITE_JPEG_QUALITY, 90])
    # Your body's own tone (a face is pinker than a torso): median skin colour of the well-exposed captures (backlit or
    # blown-out ones don't show it)
    tones = [np.median(v['img'][v['skin']], 0) for v in vs if v['skin'].sum() > 2000]
    tones = [t for t in tones if 80 < t.mean() < 220]
    tone = [int(c) for c in np.median(tones, 0)[::-1]] if tones else None  # RGB
    (Path(db).parent / 'skin.json').write_text(json.dumps({'captures': len(vs), 'seen': round(float(seen), 3), 'ts': time.time(), 'tone': tone}))


def store(db, day, res, rest, G, model_skin=None):
    """Save the fit: its parameters and measurements; the rest-pose mesh in the web template's frame (y up, front +z,
    feet at 0, int16 mm); the stance's bone transforms in that frame (float32, 3×4 per bone). The skin weights the
    transforms apply with (8 bones per vertex, uint8 index + weight) go once into skin.bin."""
    import sqlite3 as sq
    F = assets()[0]
    zmin = rest[np.unique(F), 2].min()
    mesh = np.round(np.stack([rest[:, 0], rest[:, 2] - zmin, -rest[:, 1]], 1) * 1000).astype('<i2')
    C = np.array([[100., 0, 0, 0], [0, 0, 100, -100 * zmin], [0, -100, 0, 0], [0, 0, 0, 1]])  # Anny m -> template cm
    Gt = (C @ G @ np.linalg.inv(C))[:, :3].astype('<f4')
    with sq.connect(db, timeout=30) as c:
        c.execute('insert or replace into fits(day, ts, params, mesh) values (?, ?, ?, ?)', (day, time.time(), json.dumps(res), mesh.tobytes() + Gt.tobytes()))
    skin = Path(db).parent / 'skin.bin'
    if model_skin is not None and not skin.exists():
        idx, w = model_skin
        top = np.argsort(-w, 1)[:, :8]
        wq = np.round(np.take_along_axis(w, top, 1) * 255).astype(np.uint8)
        skin.write_bytes(np.take_along_axis(idx, top, 1).astype(np.uint8).tobytes() + wq.tobytes())


def draw(path, frames, C, vertices, model, verts, project, out):
    import cv2
    Path(path).mkdir(parents=True, exist_ok=True)
    _, v3 = project(out)
    v2 = v3[..., :2].cpu().numpy()
    for i in np.linspace(0, len(frames) - 1, min(8, len(frames))).astype(int):
        img = np.full((700, 500, 3), 255, np.uint8)
        tf = lambda p: (p * 300 + [250, 330]).astype(int)  # noqa: E731
        for p in tf(v2[i]):
            cv2.circle(img, tuple(p), 1, (180, 120, 60), -1)
        cv2.polylines(img, [tf(C[i])], True, (0, 0, 220), 2)
        cv2.putText(img, f"{frames[i]['theta']:.0f} deg", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, .7, (0, 0, 0), 2)
        cv2.imwrite(f'{path}/fit_{i:03d}.png', img)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--db', default=os.environ.get('BODYLAB_DB', str(ROOT / 'data' / 'bodylab.db')))
    ap.add_argument('--day')
    ap.add_argument('--plot')
    ap.add_argument('--save', help='write the fitted rest-pose vertices (.npy, metres)')
    ap.add_argument('--store', action='store_true', help="save the day's fit for the app")
    ap.add_argument('--restance', action='store_true', help="re-pose the day's stored fit (no refit)")
    a = ap.parse_args()
    day, frames, prof = load(a.db, a.day)
    print(day, len(frames), 'frames')
    if a.restance:
        raise SystemExit(restance(a.db, day, frames, prof))
    if len(frames) < MIN_FRAMES:
        raise SystemExit(f'need {MIN_FRAMES} frames')
    import closeups
    ref_dir = Path(a.db).parent / 'reference'
    prof['seen'] = closeups.limbs(ref_dir, prof['height_cm'], day) | {k: v for k, v in closeups.torso(ref_dir, prof['height_cm'], day).items()
                                                                       if k in ('d28', 'd50', 'd62')}  # in side views the arm sits inside the torso's outline
    print('from the captures:', prof['seen'] or 'too few views')
    use = [frames[i] for i in np.linspace(0, len(frames) - 1, min(len(frames), MAX_FRAMES)).round().astype(int)]
    res, rest, G, _ = fit(use, prof, plot=a.plot)
    res |= {'tape': tape(rest, res['neck_axis'], res['limb_axes'], prof.get('sex') == 'female'), 'frames': len(frames)}  # all of them: app.py refits as they grow
    if a.save:
        np.save(a.save, rest)
    if a.store:
        import anny
        if prof.get('sex') != 'female':  # from the cameras' close views
            res['tape'] |= closeups.genitals(Path(a.db).parent / 'reference', prof['height_cm'], day)
        m = anny.Anny(topology='makehuman', rig='makehuman', skinning_method='lbs', local_changes='default')
        store(a.db, day, res, rest, G, (m.vertex_bone_indices.numpy(), m.vertex_bone_weights.numpy()))
        bake_all(a.db, prof)
    print(json.dumps(res, indent=1))
