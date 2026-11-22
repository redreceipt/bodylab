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
DENSITY = 1.04  # kg/L of a whole body including the air in the lungs (tissue alone ~1.06)

# COCO keypoint -> MediaPipe pose landmark
KP = {'nose': 0, 'left_ear': 7, 'right_ear': 8, 'left_shoulder': 11, 'right_shoulder': 12, 'left_elbow': 13, 'right_elbow': 14,
      'left_wrist': 15, 'right_wrist': 16, 'left_hip': 23, 'right_hip': 24, 'left_knee': 25, 'right_knee': 26,
      'left_ankle': 27, 'right_ankle': 28}
POSED = ['upperarm01.L', 'upperarm01.R', 'lowerarm01.L', 'lowerarm01.R', 'upperleg01.L', 'upperleg01.R',
         'lowerleg01.L', 'lowerleg01.R', 'spine03', 'spine01', 'neck01', 'head']
FACE = ('eye', 'ear', 'nose', 'mouth', 'chin', 'cheek', 'forehead', 'head', 'brow', 'jaw', 'lip', 'finger', 'hand', 'foot',
        'toe', 'temple', 'genital', 'nipple')
# Anny (x: person's left, y: back, z: up) -> camera (x: right, y: down, z: away), for someone facing the camera.
TO_CAM = [[1., 0, 0], [0, 0, -1], [0, 1, 0]]


def load(db, day=None):
    c = sqlite3.connect(f'file:{db}?mode=ro', uri=True)
    day = day or c.execute("select max(date(ts, 'unixepoch', 'localtime')) from frames").fetchone()[0]
    rows = [json.loads(r[0]) for r in c.execute("select data from frames where date(ts, 'unixepoch', 'localtime') = ?", (day,))]
    prof = json.loads(c.execute("select v from settings where k = 'profile'").fetchone()[0])
    noon = time.mktime(time.strptime(day + ' 12', '%Y-%m-%d %H'))
    w = c.execute("select json_extract(data, '$.weight'), abs(ts - ?) / 86400 from logs where kind = 'daily' "
                  "and json_extract(data, '$.weight') > 0 and abs(ts - ?) < 30 * 86400 order by abs(ts - ?) limit 1", (noon,) * 3).fetchone()
    return day, [f for f in rows if f.get('sil')], prof | ({'weight': w[0], 'weight_age': w[1]} if w else {})


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


def fit(frames, prof, iters=(150, 350), plot=None, log=print):
    import anny
    import roma
    import torch
    dev = 'mps' if torch.backends.mps.is_available() else 'cpu'
    model = anny.Anny(topology='makehuman', rig='makehuman', skinning_method='lbs', local_changes='default').to(dtype=torch.float32, device=dev)
    kreg = anny.KeypointsRegressor.coco(model, labels=list(KP))
    to_cam = torch.tensor(TO_CAM, device=dev)
    B, H = len(frames), prof['height_cm'] / 100
    female = prof.get('sex') == 'female'
    age = (date.today() - date.fromisoformat(prof['birth_date'])).days / 365.25 if prof.get('birth_date') else 30
    # MakeHuman's age axis: 0 = 1 y, 0.1875 = 11 y, 0.5 = 25 y, 1 = 90 y
    age_p = .5 + (age - 25) / 130 if age >= 25 else .1875 + (age - 11) / 14 * .3125

    rng = np.random.default_rng(0)
    body = torch.tensor(body_vertices().astype(np.int64), device=dev)
    verts = body[torch.tensor(rng.choice(len(body), 2500, replace=False), device=dev)]
    local_on = torch.tensor([not any(t in k for t in FACE) for k in model.local_change_labels], device=dev)
    posed = torch.tensor([model.bone_labels.index(b) for b in POSED], device=dev)

    # Targets, in metres in the image plane (relative to the hip centre).
    C = torch.tensor(np.array([f['sil']['c'] for f in frames]), dtype=torch.float32, device=dev) * H
    L = torch.tensor(np.array([np.array(f['sil']['l'])[list(KP.values())] for f in frames]), dtype=torch.float32, device=dev) * H
    W = np.array([np.array(f['sil']['w'])[list(KP.values())] for f in frames])
    bd = torch.tensor(np.array([borders(np.array(f['sil']['c'])) for f in frames]), dtype=torch.float32, device=dev) * H
    in_view = lambda p: ((p[..., 0] > bd[:, None, 0] + .01) & (p[..., 0] < bd[:, None, 1] - .01)  # noqa: E731
                         & (p[..., 1] > bd[:, None, 2] + .01) & (p[..., 1] < bd[:, None, 3] - .01))
    edge_ok = in_view(C)  # outline points on an image edge aren't body edges
    kp_ok = in_view(L).float()

    # Shape (shared) and pose (per frame).
    phen = {k: torch.zeros((), device=dev, requires_grad=True) for k in ('muscle', 'weight', 'height', 'proportions')}
    local = torch.zeros(len(model.local_change_labels), device=dev, requires_grad=True)
    rot = torch.zeros(B, len(POSED), 3, device=dev, requires_grad=True)
    cam = torch.zeros(B, 3, device=dev, requires_grad=True)  # log scale, x, y shift

    def run(pose_only=False):
        p = {k: torch.sigmoid(v).expand(B) for k, v in phen.items()}
        p |= {'gender': torch.full((B,), 0. if female else 1., device=dev), 'age': torch.full((B,), age_p, device=dev),
              'cupsize': torch.full((B,), .5, device=dev), 'firmness': torch.full((B,), .5, device=dev)}
        p = {k: v for k, v in p.items() if k in model.phenotype_labels}
        pose = torch.eye(4, device=dev).repeat(B, model.bone_count, 1, 1)
        pose[:, posed, :3, :3] = roma.rotvec_to_rotmat(rot)
        lc = (torch.tanh(local) * local_on).expand(B, -1)
        return model(pose_parameters=pose, phenotype_kwargs=p, local_changes_kwargs=lc)

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

    tris = torch.tensor(assets()[2], device=dev)
    volume = lambda v: torch.einsum('ij,ij->i', v[tris[:, 0]], torch.cross(v[tris[:, 1]], v[tris[:, 2]], -1)).sum().abs() / 6 * 1000  # noqa: E731
    Wt = torch.tensor(W, dtype=torch.float32, device=dev)
    Wt = Wt - Wt[:, 9:11].mean(1, keepdim=True)
    a, b = C, torch.roll(C, -1, 1)

    use = torch.ones(B, device=dev)  # frames the shape is fitted to

    def loss(stage):
        out = run()
        k3, v3 = project(out)
        k2, v2 = k3[..., :2], v3[..., :2]
        # 2D landmarks where they're in view; 3D landmarks (scale-free) for depth and turn.
        e_kp = ((k2 - L) ** 2).sum(-1).mul(kp_ok)
        l_kp = (e_kp * use[:, None]).sum() / (kp_ok * use[:, None]).sum()
        w = Wt * (k3.norm(dim=-1).mean(1) / Wt.norm(dim=-1).mean(1))[:, None, None]
        l_3d = (((k3 - w) ** 2).sum(-1).mean(1) * use).sum() / use.sum()
        rest = out['rest_vertices'][0]
        l_h = (rest[body, 2].max() - rest[body, 2].min() - H) ** 2
        mass = volume(rest) * DENSITY
        l_reg = (rot ** 2).mean() * .01 + (cam[:, 0] ** 2).mean() * .1 + (torch.tanh(local) ** 2).mean() * .003
        total = l_kp + .3 * l_3d + 10 * l_h + l_reg
        if prof.get('weight') and stage:  # outlines leave volume loose (±10 %); a weigh-in pins it, fading over 30 days
            trust = min(1, max(0, (30 - prof.get('weight_age', 0)) / 26))
            total = total + .3 * trust * ((mass - prof['weight']) / prof['weight']) ** 2
        parts = {'kp': l_kp.item() ** .5 * 100, '3d': l_3d.item() ** .5 * 100, 'kg': mass.item()}
        if stage:
            # Mesh outside the outline (where the camera sees) is too big; outline nowhere near the mesh is too small.
            d2 = seg_dist2(v2, a, b).min(-1).values
            out_m = (~inside(v2, a, b) & in_view(v2)).float() * use[:, None]
            l_out = (d2 * out_m).sum() / use.sum() / v2.shape[1]
            near = ((C[:, :, None] - v2[:, None]) ** 2).sum(-1).min(-1).values
            l_in = (near * edge_ok * use[:, None]).sum() / (edge_ok * use[:, None]).sum()
            total = total + 20 * (l_out + l_in)
            parts |= {'out': ((d2 * out_m).sum() / out_m.sum().clamp_min(1)).item() ** .5 * 100, 'in': l_in.item() ** .5 * 100}
        return total, parts, out, e_kp

    t0 = time.time()
    for stage, n in enumerate(iters):
        params = [rot, turn, cam] + ([local] + list(phen.values()) if stage else [])
        opt = torch.optim.Adam(params, lr=.02 if stage else .05)
        for i in range(n):
            opt.zero_grad()
            total, parts, out, e_kp = loss(stage)
            total.backward()
            opt.step()
            if i % 50 == 0 or i == n - 1:
                log(f'stage {stage} it {i:3d}  ' + '  '.join(f'{k} {v:.2f}' for k, v in parts.items()) + f'  ({time.time() - t0:.0f}s)')
        if not stage:  # misdetected frames (landmarks no body pose can match) don't shape the body
            e = (e_kp.sum(1) / kp_ok.sum(1).clamp_min(1)).detach().sqrt()
            use = (e < max(.03, 2.5 * e.median().item())).float()
            log(f'{int(use.sum())} of {B} frames fit the landmarks')

    with torch.no_grad():
        out = run()
        k3, v3 = project(out)
        rest = out['rest_vertices'][0].cpu().numpy()
        res = {'phenotypes': {k: round(torch.sigmoid(v).item(), 4) for k, v in phen.items()} | {'age': round(age_p, 4)},
               'local': {k: round(v, 4) for k, v in zip(model.local_change_labels, (torch.tanh(local) * local_on).tolist()) if abs(v) > 1e-3},
               'fit': {k: round(v, 2) for k, v in parts.items()}}
        if plot:
            draw(plot, frames, C.cpu().numpy(), out['vertices'], model, verts, project, out)
    return res, rest


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
    return F[~arm[F].any(1)], int(navel), F


def tape(rest):
    """Tape-style circumferences (cm) of the rest-pose body: the convex hull of horizontal slices of the body
    without its arms, like a tape bridging hollows. Neck: narrowest below the jaw; chest: widest; waist: at the
    navel; hips: widest round the seat."""
    import cv2
    F, navel, _ = assets()
    V = rest * 100
    z0, H = V[np.unique(F), 2].min(), np.ptp(V[np.unique(F), 2])
    a, b, c = V[F[:, 0]], V[F[:, 1]], V[F[:, 2]]

    def ring(z):
        pts = []
        for p, q in ((a, b), (b, c), (c, a)):
            s = (p[:, 2] - z) * (q[:, 2] - z) < 0
            pts.append(p[s] + (q[s] - p[s]) * ((z - p[s, 2]) / (q[s, 2] - p[s, 2]))[:, None])
        P = np.concatenate(pts)[:, :2].astype(np.float32)
        return cv2.arcLength(cv2.convexHull(P), True) if len(P) > 2 else 0

    at = lambda lo, hi: [ring(z0 + f * H) for f in np.arange(lo, hi, .005)]  # noqa: E731
    return {'neck': round(min(at(.83, .88)), 1), 'chest': round(max(at(.68, .76)), 1), 'waist': round(ring(V[navel, 2]), 1),
            'hips': round(max(at(.46, .56)), 1), 'navel_h': round(float((V[navel, 2] - z0) / H), 3), 'height': round(float(H), 1)}


def store(db, day, res, rest):
    """Save the fit: its parameters and measurements, and the mesh in the web template's frame (y up, front +z,
    feet at 0) as int16 millimetres."""
    import sqlite3 as sq
    F, _, _ = assets()
    zmin = rest[np.unique(F), 2].min()
    mesh = np.round(np.stack([rest[:, 0], rest[:, 2] - zmin, -rest[:, 1]], 1) * 1000).astype('<i2')
    with sq.connect(db, timeout=30) as c:
        c.execute('insert or replace into fits(day, ts, params, mesh) values (?, ?, ?, ?)', (day, time.time(), json.dumps(res), mesh.tobytes()))


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
    a = ap.parse_args()
    day, frames, prof = load(a.db, a.day)
    print(day, len(frames), 'frames')
    if len(frames) < MIN_FRAMES:
        raise SystemExit(f'need {MIN_FRAMES} frames')
    res, rest = fit(frames, prof, plot=a.plot)
    res |= {'tape': tape(rest), 'frames': len(frames)}
    if a.save:
        np.save(a.save, rest)
    if a.store:
        store(a.db, day, res, rest)
    print(json.dumps(res, indent=1))
