"""Build static/assets/ (the realistic 3D body) from MakeHuman's CC0 assets (basemesh hm08).

- base mesh + default young male/female shapes (base + 1/3 each ethnicity target, as MakeHuman does)
- per-vertex rig region weights and joint positions (cm, feet at y=0, facing +z, character's left at +x)
- skin textures, hair styles, eyebrows, eyelashes, eyes and workout clothes (proxies anchored to the base mesh,
  so they follow the fit); breast and genital morphs; MakeHuman's genital helper mesh
- face anchors: the template head is rendered front-on and run through MediaPipe's face landmarker, giving each
  of the 478 face landmarks a UV on the skin texture. A selfie's landmarks then map the photo onto the skin.

Run: uv run python tools/build_assets.py   (downloads ~300 MB of sources into a temp cache)
"""
import io
import json
import tempfile
import urllib.request
import zipfile
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks.python import BaseOptions, vision

MH = 'https://raw.githubusercontent.com/makehumancommunity/makehuman/a8bc2d54ff0ac92e78ff71431b1023eda42bf482/makehuman/data/'
PACK = 'https://files.makehumancommunity.org/asset_packs/makehuman_system_assets/makehuman_system_assets_cc0.zip'
CANONICAL = ('https://raw.githubusercontent.com/google-ai-edge/mediapipe/9ce52340ad12b1be04bff9e6ae5d323c9855f80c/'
             'mediapipe/modules/face_geometry/data/canonical_face_model.obj')
FACE_MODEL = 'https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/latest/face_landmarker.task'
CACHE = Path(tempfile.gettempdir()) / 'bodylab-build'
OUT = Path(__file__).resolve().parent.parent / 'static' / 'assets'
REGIONS = ['head', 'torso', 'arm_l', 'arm_r', 'leg_l', 'leg_r']
JOINTS = {'neck': 'joint-neck', 'head': 'joint-head', 'eye_l': 'joint-l-eye', 'eye_r': 'joint-r-eye'} | {
    f'{name}_{s}': f'joint-{s}-{mh}' for s in 'lr' for name, mh in
    (('shoulder', 'shoulder'), ('elbow', 'elbow'), ('wrist', 'hand'), ('hip', 'upper-leg'), ('knee', 'knee'), ('ankle', 'ankle'))}
SKINS = {'male': 'skins/young_caucasian_male/young_lightskinned_male_diffuse.png',
         'female': 'skins/young_caucasian_female/young_lightskinned_female_diffuse.png'}
HAIRS = ['short02', 'short04', 'bob02', 'long01', 'ponytail01', 'afro01', 'braid01']
# Sparse morphs the browser blends from your measurements (deltas from the default young male/female).
SHAPES = {'cup+': 'breast/female-young-averagemuscle-averageweight-maxcup-averagefirmness',
          'cup-': 'breast/female-young-averagemuscle-averageweight-mincup-averagefirmness',
          'firm+': 'breast/female-young-averagemuscle-averageweight-averagecup-maxfirmness',
          'firm-': 'breast/female-young-averagemuscle-averageweight-averagecup-minfirmness',
          'length+': 'genitals/penis-length-incr', 'length-': 'genitals/penis-length-decr',
          'girth+': 'genitals/penis-circ-incr', 'girth-': 'genitals/penis-circ-decr',
          'testicles+': 'genitals/penis-testicles-incr', 'testicles-': 'genitals/penis-testicles-decr'}
SUIT = 'clothes/female_sportsuit01/female_sportsuit01'  # crop top + leggings: the leggings, cut down to athletic shorts
TEE = 'clothes/female_casualsuit02/female_casualsuit02'  # fitted tee + denim shorts: the tee, cut down to a sports bra
EYE_COLORS = ['brown', 'brownlight', 'blue', 'deepblue', 'bluegreen', 'green', 'grey', 'lightblue', 'ice']


def fetch(url, name=None):
    path = CACHE / (name or url.rsplit('/', 1)[-1])
    if not path.exists():
        CACHE.mkdir(exist_ok=True)
        print('downloading', url, flush=True)
        urllib.request.urlretrieve(url, path.with_suffix('.part'))
        path.with_suffix('.part').rename(path)
    return path


def region(bone):
    name, side = bone.split('.')[0], '_l' if bone.endswith('.L') else '_r'
    if name.startswith(('upperarm', 'lowerarm', 'wrist', 'finger', 'metacarpal')):
        return 'arm' + side
    if name.startswith(('upperleg', 'lowerleg', 'foot', 'toe')):
        return 'leg' + side
    if name in ('head', 'jaw') or name.startswith(('eye', 'oculi', 'orbicularis', 'oris', 'risorius', 'levator', 'special', 'tongue')):
        return 'head'
    return 'torso'  # root, spine, pelvis, clavicle, shoulder (deltoid), breast, neck


def parse_obj(text):
    """-> vertices (n,3), uvs (m,2), {group: [[(v, vt), ...], ...]} (0-based)"""
    v, vt, faces, group = [], [], {}, None
    for line in text.splitlines():
        p = line.split()
        if not p:
            continue
        if p[0] == 'v':
            v.append([float(x) for x in p[1:4]])
        elif p[0] == 'vt':
            vt.append([float(x) for x in p[1:3]])
        elif p[0] == 'g':
            group = p[1]
        elif p[0] == 'f':
            faces.setdefault(group, []).append([tuple(int(i) - 1 if i else -1 for i in (x.split('/') + [''])[:2]) for x in p[1:]])
    return np.array(v), np.array(vt), faces


def split_uv(faces, uvs):
    """Render vertices are unique (vertex, uv) pairs. -> src vertex per render vertex, uv per render vertex, triangles."""
    key, src, uv, tris = {}, [], [], []
    for f in faces:
        ids = []
        for v, t in f:
            if (v, t) not in key:
                key[v, t] = len(src)
                src.append(v)
                uv.append(uvs[t] if t >= 0 else (0, 0))
            ids.append(key[v, t])
        tris += [[ids[0], ids[k], ids[k + 1]] for k in range(1, len(ids) - 1)]
    return np.array(src), np.array(uv), np.array(tris)


def parse_mhclo(text):
    """-> refs (n,3), weights (n,3), offsets (n,3), scale refs [[v1, v2, den] for x, y, z]"""
    lines, refs, scale = text.splitlines(), [], {}
    in_verts = False
    for line in lines:
        p = line.split()
        if not p:
            continue
        if p[0] in ('x_scale', 'y_scale', 'z_scale'):
            scale[p[0][0]] = [int(p[1]), int(p[2]), float(p[3])]
        elif p[0] == 'verts':
            in_verts = True
        elif p[0] == 'delete_verts':
            in_verts = False
        elif in_verts and p[0].isdigit() and len(p) in (1, 9):
            refs.append([int(p[0])] * 3 + [1, 0, 0, 0, 0, 0] if len(p) == 1 else [int(x) for x in p[:3]] + [float(x) for x in p[3:]])
    r = np.array(refs, dtype=float)
    return r[:, :3].astype(int), r[:, 3:6], r[:, 6:9], [scale[k] for k in 'xyz']


def delete_verts(text):
    """Body vertices a garment hides ('a - b' ranges and single indices after `delete_verts`)."""
    tokens, out = ' '.join(text.split('delete_verts', 1)[1].split('\n')).split(), set()
    i = 0
    while i < len(tokens):
        if i + 2 < len(tokens) and tokens[i + 1] == '-':
            out.update(range(int(tokens[i]), int(tokens[i + 2]) + 1))
            i += 3
        else:
            out.add(int(tokens[i]))
            i += 1
    return out


def rasterize(P, Z, tris, W, H):
    """Z-buffered triangle ids and barycentrics for pixel-space points P (larger Z = nearer)."""
    zbuf, tri_id, bary = np.full((H, W), -np.inf), np.full((H, W), -1), np.zeros((H, W, 3))
    for t, (a, b, c) in enumerate(tris):
        (ax, ay), (bx, by), (cx, cy) = P[a], P[b], P[c]
        x0, x1 = max(int(min(ax, bx, cx)), 0), min(int(max(ax, bx, cx)) + 1, W - 1)
        y0, y1 = max(int(min(ay, by, cy)), 0), min(int(max(ay, by, cy)) + 1, H - 1)
        d = (bx - ax) * (cy - ay) - (by - ay) * (cx - ax)
        if x0 > x1 or y0 > y1 or abs(d) < 1e-9:
            continue
        xs, ys = np.meshgrid(np.arange(x0, x1 + 1) + .5, np.arange(y0, y1 + 1) + .5)
        w1 = ((xs - ax) * (cy - ay) - (ys - ay) * (cx - ax)) / d
        w2 = ((bx - ax) * (ys - ay) - (by - ay) * (xs - ax)) / d
        w0 = 1 - w1 - w2
        z = w0 * Z[a] + w1 * Z[b] + w2 * Z[c]
        win = (slice(y0, y1 + 1), slice(x0, x1 + 1))
        hit = (w0 >= 0) & (w1 >= 0) & (w2 >= 0) & (z > zbuf[win])
        zbuf[win][hit], tri_id[win][hit] = z[hit], t
        bary[win][hit] = np.stack([w0, w1, w2], -1)[hit]
    return tri_id, bary


def face_anchors(pos, src, uv, tris, eye_tris, eye_centers, skin, landmarker):
    """Render the template head front-on; landmarks -> skin UV (NaN where a landmark isn't on skin)."""
    rv = pos[src]  # render vertices, then eyeball vertices appended
    chin = eye_centers[:, 1].mean() - 16
    lo, hi = rv[rv[:, 1] > chin].min(0), rv[rv[:, 1] > chin].max(0)
    lo[1] = chin
    W, s = 720, 720 / (hi[0] - lo[0] + 4)
    H = int((hi[1] - lo[1] + 4) * s)
    ev = np.unique(eye_tris)
    allp = np.vstack([rv, pos[ev]])
    remap = {v: len(rv) + k for k, v in enumerate(ev)}
    all_tris = np.vstack([tris, np.vectorize(remap.get)(eye_tris)])
    P = np.stack([(allp[:, 0] - lo[0] + 2) * s, (hi[1] + 2 - allp[:, 1]) * s], 1)
    n = np.zeros_like(allp)  # vertex normals for simple shading
    fn = np.cross(allp[all_tris[:, 1]] - allp[all_tris[:, 0]], allp[all_tris[:, 2]] - allp[all_tris[:, 0]])
    for k in range(3):
        np.add.at(n, all_tris[:, k], fn)
    n /= np.linalg.norm(n, axis=1, keepdims=True).clip(1e-9)
    tri_id, bary = rasterize(P, allp[:, 2], all_tris, W, H)
    img, uvbuf = np.full((H, W, 3), 90, np.uint8), np.full((H, W, 2), np.nan)
    ys, xs = np.nonzero(tri_id >= 0)
    t = tri_id[ys, xs]
    b = bary[ys, xs]
    nz = (n[all_tris[t]] * b[..., None]).sum(1)[:, 2]
    shade = (.45 + .55 * nz.clip(0, 1))[:, None]
    body = t < len(tris)
    u = (uv[all_tris[t[body]]] * b[body][..., None]).sum(1)
    uvbuf[ys[body], xs[body]] = u
    th, tw = skin.shape[:2]
    tex = skin[((1 - u[:, 1]) * (th - 1)).astype(int), (u[:, 0] * (tw - 1)).astype(int)][:, ::-1]  # BGR -> RGB
    img[ys[body], xs[body]] = (tex * shade[body]).clip(0, 255)
    p3 = (allp[all_tris[t[~body]]] * b[~body][..., None]).sum(1)
    c = eye_centers[(p3[:, 0] < 0).astype(int)]  # [l, r]
    fwd = (p3 - c)[:, 2] / np.linalg.norm(p3 - c, axis=1)
    img[ys[~body], xs[~body]] = np.where(fwd[:, None] > .93, [15, 10, 8], np.where(fwd[:, None] > .8, [70, 45, 30], [235, 230, 225]))
    res = landmarker.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(img)))
    assert res.face_landmarks, 'face landmarker found no face on the template render'
    # UV islands (connected render triangles): a landmark only counts on the face's island, otherwise the photo
    # warp would smear across unrelated parts of the texture.
    parent = list(range(len(src)))

    def root(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i
    for a, b, c in tris:
        parent[root(b)] = parent[root(c)] = root(a)
    island = np.full((H, W), -1)
    island[tri_id >= 0] = [root(tris[t][0]) if t < len(tris) else -1 for t in tri_id[tri_id >= 0]]
    out, where = np.full((478, 2), np.nan), np.full(478, -1)
    for k, lm in enumerate(res.face_landmarks[0]):
        px, py = int(lm.x * W), int(lm.y * H)
        win = (slice(max(py - 2, 0), py + 3), slice(max(px - 2, 0), px + 3))
        ok = ~np.isnan(uvbuf[win][..., 0])
        if ok.any():  # average only the majority island: lip corners border the mouth-interior UV island
            where[k] = np.bincount(island[win][ok]).argmax()
            out[k] = uvbuf[win][ok & (island[win] == where[k])].mean(0)
    out[where != where[1]] = np.nan  # 1 = nose tip
    for k, lm in enumerate(res.face_landmarks[0]):
        cv2.circle(img, (int(lm.x * W), int(lm.y * H)), 1, (0, 255, 0) if not np.isnan(out[k, 0]) else (255, 0, 0), -1)
    cv2.imwrite(str(CACHE / f'anchors_{len(tris)}_{int(hi[1])}.png'), img[..., ::-1])
    return out


def pieces(faces, by=0):
    """Connected pieces, biggest first. by=0: faces sharing vertices; by=1: sharing UVs (a garment's sewn panels)."""
    parent = {}

    def root(i):
        parent.setdefault(i, i)
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i
    for f in faces:
        for x in f[1:]:
            parent[root(x[by])] = root(f[0][by])
    out = {}
    for f in faces:
        out.setdefault(root(f[0][by]), []).append(f)
    return sorted(out.values(), key=len, reverse=True)


def subdivide(faces, levels=2):
    """Catmull-Clark subdivision as weights: each smooth vertex is a fixed mix of original vertices, so the browser
    can recompute it after fitting. faces: lists of original vertex ids. -> (rows [{orig: weight}], quads)."""
    rows = {v: {v: 1.0} for f in faces for v in f}
    mix = lambda parts: {k: sum(w * r.get(k, 0) for w, r in parts) for k in {k for _, r in parts for k in r}}  # noqa: E731
    for _ in range(levels):
        edges, around = {}, {}
        for fi, f in enumerate(faces):
            for i, v in enumerate(f):
                e = tuple(sorted((v, f[(i + 1) % len(f)])))
                edges.setdefault(e, []).append(fi)
                around.setdefault(v, set()).add(fi)
        fp = {fi: mix([(1 / len(f), rows[v]) for v in f]) for fi, f in enumerate(faces)}
        ep, boundary = {}, {}
        for (a, b), fs in edges.items():
            if len(fs) == 2:
                ep[a, b] = mix([(.25, rows[a]), (.25, rows[b]), (.25, fp[fs[0]]), (.25, fp[fs[1]])])
            else:
                ep[a, b] = mix([(.5, rows[a]), (.5, rows[b])])
                boundary.setdefault(a, []).append(b)
                boundary.setdefault(b, []).append(a)
        vp = {}
        for v, fs in around.items():
            if v in boundary:  # open edge (where it meets the body): cubic B-spline along the border
                vp[v] = mix([(.75, rows[v])] + [(.125, rows[u]) for u in boundary[v][:2]])
            else:  # (F + 2R + (n-3)P) / n: F = mean face point, R = mean edge midpoint, n = valence
                nb = [e for e in edges if v in e]
                n = len(nb)
                vp[v] = mix([(1 / n / len(fs), fp[f]) for f in fs] + [(1 / n / n, rows[a]) for e in nb for a in e] + [((n - 3) / n, rows[v])])
        ids, new_rows, quads = {}, [], []

        def key(k, r):
            if k not in ids:
                ids[k] = len(new_rows)
                new_rows.append(r)
            return ids[k]
        for fi, f in enumerate(faces):
            c = key(('f', fi), fp[fi])
            for i, v in enumerate(f):
                prev, nxt_ = f[i - 1], f[(i + 1) % len(f)]
                quads.append([key(('v', v), vp[v]), key(('e', *sorted((v, nxt_))), ep[tuple(sorted((v, nxt_)))]), c,
                              key(('e', *sorted((prev, v))), ep[tuple(sorted((prev, v)))])])
        rows, faces = dict(enumerate(new_rows)), quads
    return [rows[i] for i in range(len(rows))], faces


def gray_rgba(rgba, size):
    """Hair/brow textures -> neutral grey (mean ~0.55) + alpha, so the material colour sets the hair colour."""
    rgba = cv2.resize(rgba, (size, size), interpolation=cv2.INTER_AREA)
    g = cv2.cvtColor(rgba[..., :3], cv2.COLOR_BGR2GRAY).astype(float)
    a = rgba[..., 3] > 128
    g = (g * (140 / max(g[a].mean(), 1))).clip(0, 255).astype(np.uint8)
    return np.dstack([g, g, g, rgba[..., 3]])


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    pack = zipfile.ZipFile(fetch(PACK))
    read = lambda name: pack.read(name)  # noqa: E731
    image = lambda name, flags=cv2.IMREAD_UNCHANGED: cv2.imdecode(np.frombuffer(read(name), np.uint8), flags)  # noqa: E731
    text = lambda url: fetch(url, url.split('makehuman/data/')[-1].replace('/', '_')).read_text()  # noqa: E731

    base, uvs, faces = parse_obj(text(MH + '3dobjs/base.obj'))
    n = len(base)
    weights = np.zeros((n, len(REGIONS)))
    for bone, pairs in json.loads(text(MH + 'rigs/default_weights.mhw'))['weights'].items():
        for i, w in pairs:
            weights[i, REGIONS.index(region(bone))] += w
    weights /= weights.sum(1, keepdims=True).clip(1e-9)

    def target(path):
        t = np.zeros_like(base)
        for line in text(MH + path).splitlines():
            if line and not line.startswith('#'):
                i, dx, dy, dz = line.split()
                t[int(i)] = float(dx), float(dy), float(dz)
        return t

    src, uv, tris = split_uv(faces['body'], uvs)
    arrays = {'weights': np.rint(weights * 255).astype('u1'), 'body.src': src.astype('<u2'),
              'body.uv': uv.astype('<f4'), 'body.tris': tris.astype('<u2')}
    meta = {'n': n, 'joints': {}, 'proxies': {}, 'license': 'Derived from MakeHuman hm08 assets (CC0 1.0)'}
    tmpl = {}
    eye_tris = np.array([[f[0][0], f[k][0], f[k + 1][0]] for g in ('helper-l-eye', 'helper-r-eye') for f in faces[g] for k in range(1, len(f) - 1)])
    group_center = lambda v, g: v[sorted({i for f in faces[g] for i, _ in f})].mean(0)  # noqa: E731
    landmarker = vision.FaceLandmarker.create_from_options(vision.FaceLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(fetch(FACE_MODEL)))))

    for sex in ('male', 'female'):
        v = base + sum(target(f'targets/macrodetails/{race}-{sex}-young.target') for race in ('african', 'asian', 'caucasian')) / 3
        v = (v - [0, group_center(v, 'joint-ground')[1], 0]) * 10  # decimetres -> cm, feet on the floor
        meta['joints'][sex] = {k: group_center(v, g).round(2).tolist() for k, g in JOINTS.items()}
        arrays[sex], tmpl[sex] = v.astype('<f4'), v
        skin = image(SKINS[sex], cv2.IMREAD_COLOR)
        cv2.imwrite(str(OUT / f'skin_{sex}.jpg'), skin, [cv2.IMWRITE_JPEG_QUALITY, 85])
        eyes = np.array([meta['joints'][sex]['eye_l'], meta['joints'][sex]['eye_r']])
        arrays[f'face_uv.{sex}'] = face_anchors(v, src, uv, tris, eye_tris, eyes, skin, landmarker).astype('<f4')
        print(sex, 'face anchors on skin:', int((~np.isnan(arrays[f'face_uv.{sex}'][:, 0])).sum()), '/ 478')

    # Selfie -> skin warp: MediaPipe's canonical face triangles, and the face outline (for a soft mask).
    canon = parse_obj(fetch(CANONICAL).read_text())[2]
    arrays['face.tris'] = np.array([[f[0][0], f[k][0], f[k + 1][0]] for g in canon.values() for f in g for k in range(1, len(f) - 1)], '<u2')
    edges = {c.start: c.end for c in vision.FaceLandmarksConnections.FACE_LANDMARKS_FACE_OVAL}
    oval = [next(iter(edges))]
    while edges[oval[-1]] != oval[0]:
        oval.append(edges[oval[-1]])
    meta['face_oval'] = oval

    # Breast and genital shape morphs, sparse, in cm.
    for name, path in SHAPES.items():
        t = target(f'targets/{path}.target') * 10
        idx = np.flatnonzero(np.abs(t).sum(1))
        arrays[f'morph.{name}.idx'], arrays[f'morph.{name}.d'] = idx.astype('<u2'), t[idx].astype('<f4')

    # Genitals: MakeHuman's helper geometry, plus the template's size and what each full morph does to it,
    # so a length or girth in cm maps to morph weights.
    rows, quads = subdivide([[v for v, _ in f] for f in faces['helper-genital']])  # the helper is low-poly; smooth it
    arrays |= {'genital.ptr': np.cumsum([0] + [len(r) for r in rows]).astype('<u4'),
               'genital.idx': np.array([k for r in rows for k in r], '<u2'), 'genital.w': np.array([w for r in rows for w in r.values()], '<f4'),
               'genital.tris': np.array([[q[0], q[k], q[k + 1]] for q in quads for k in (1, 2)], '<u2')}
    shaft = np.flatnonzero(np.abs(target('targets/genitals/penis-circ-incr.target')).sum(1))
    axis = (target('targets/genitals/penis-length-incr.target')[shaft]).sum(0)
    axis /= np.linalg.norm(axis)

    def size(v):
        p = v[shaft]
        t = p @ axis
        length = t.max() - t.min()
        mid = np.abs(t - (t.min() + t.max()) / 2) < .15 * length
        q = p[mid] - np.outer(t[mid], axis)
        return [round(float(length), 2), round(float(2 * np.pi * np.linalg.norm(q - q.mean(0), axis=1).mean()), 2)]
    vm = tmpl['male']
    meta['genital'] = {'base': size(vm)} | {k: size(vm + 10 * target(f'targets/{SHAPES[k]}.target')) for k in ('length+', 'length-', 'girth+', 'girth-')}
    print('genital template size (length, girth cm):', meta['genital'])

    # Shorts and sports bra: the sportsuit's leggings cut at mid-thigh, and a fitted tee cut under the bust. Each also lists
    # the body skin it covers, hidden while it's worn so skin can't poke through.
    refs, w, d, scale = parse_mhclo(read(SUIT + '.mhclo').decode())
    sv, suv, sfaces = parse_obj(read(SUIT + '.obj').decode())
    vf, Jf = tmpl['female'], meta['joints']['female']
    gp = (w[:, :, None] * vf[refs]).sum(1)  # garment vertices on the template (offsets don't matter for cutting)
    bottom = min(pieces([f for g in sfaces.values() for f in g])[:2], key=lambda fs: gp[[v for f in fs for v, _ in f], 1].mean())
    def along(a, b):  # position of each garment vertex along its own side's limb (0 at joint a, 1 at b)
        A_, B_ = np.where(gp[:, :1] > 0, Jf[f'{a}_l'], Jf[f'{a}_r']), np.where(gp[:, :1] > 0, Jf[f'{b}_l'], Jf[f'{b}_r'])
        return ((gp - A_) * (B_ - A_)).sum(1) / ((B_ - A_) ** 2).sum(1)
    # Cut across each limb, not horizontally, so the hems follow the garment's own edge loops (clean, not ragged).
    leg = along('hip', 'knee')

    def gap(t, piece, lo, hi):  # the widest gap between edge loops in [lo, hi]: a cut there crosses no faces raggedly
        vals = np.unique(np.round(t[[v for f in piece for v, _ in f]], 3))
        vals = vals[(vals > lo) & (vals < hi)]
        i = np.argmax(np.diff(vals))
        return (vals[i] + vals[i + 1]) / 2
    leg_cut = gap(leg, bottom, .35, .55)
    garments = {'shorts': ([f for f in bottom if all(leg[v] < leg_cut for v, _ in f)], SUIT, (refs, w, d, scale), suv, gp)}
    tee = parse_mhclo(read(TEE + '.mhclo').decode())
    _, tuv, tfaces = parse_obj(read(TEE + '.obj').decode())
    tgp = (tee[1][:, :, None] * vf[tee[0]]).sum(1)
    panels = [f for p in pieces([f for g in tfaces.values() for f in g], by=1)[:2] for f in p]  # front + back; sleeves are their own panels
    apex = tgp[[v for f in panels for v, _ in f]][:, 2].argmax()
    apex_y = tgp[[v for f in panels for v, _ in f]][apex, 1]
    bra_cut = gap(tgp[:, 1], panels, apex_y - 13, apex_y - 8)  # 8-13 cm under the bust: the band
    garments['bra'] = ([f for f in panels if all(tgp[v, 1] > bra_cut for v, _ in f)], TEE, tee, tuv, tgp)
    body_arm = weights[:, 2:4].sum(1)
    for name, (fs, src_path, (refs, w, d, scale), suv, gp) in garments.items():
        hidden = delete_verts(read(src_path + '.mhclo').decode())
        psrc, ppuv, ptris = split_uv(fs, suv)
        ys = gp[[v for f in fs for v, _ in f], 1]
        m = min(1.5, (ys.max() - ys.min()) / 5)
        hide = [i for i in hidden if ys.min() + m < vf[i, 1] < ys.max() - m and body_arm[i] < .5]  # inside the edges: no gaps
        print(f'{name}: {len(fs)} faces, y {ys.min():.2f}..{ys.max():.2f}, hides {len(hide)} body vertices')
        arrays |= {f'{name}.refs': refs.astype('<u2'), f'{name}.w': w.astype('<f4'), f'{name}.d': d.astype('<f4'),
                   f'{name}.src': psrc.astype('<u2'), f'{name}.uv': ppuv.astype('<f4'), f'{name}.tris': ptris.astype('<u2'),
                   f'{name}.hide': np.array(sorted(hide), '<u2')}
        meta['proxies'][name] = {'kind': 'garment', 'scale': scale}
    # For draping: each shorts vertex's place down the leg (0 = hip joint, 1 = hem), and the joints' vertices.
    arrays['shorts.leg'] = (leg / leg_cut).astype('<f4')
    for k in ('hip_l', 'hip_r', 'knee_l', 'knee_r'):
        arrays[f'joint.{k}'] = np.array(sorted({i for f in faces[JOINTS[k]] for i, _ in f}), '<u2')
    for name, path in (('shorts', SUIT), ('bra', TEE)):
        diffuse = image(f'{path}_diffuse.png', cv2.IMREAD_COLOR)
        if path == TEE:  # paint out the MakeHuman logo printed on the chest (keyed on anything that isn't shirt blue)
            box = diffuse[330:620, 1470:1780]
            off = (np.abs(box.astype(int) - np.median(box.reshape(-1, 3), 0)).sum(2) > 60).astype(np.uint8)
            box[:] = cv2.inpaint(box, cv2.dilate(off, np.ones((7, 7), np.uint8)), 9, cv2.INPAINT_TELEA)
        for tex, img in (('', diffuse), ('_normal', image(f'{path}_normal.png', cv2.IMREAD_COLOR))):
            cv2.imwrite(str(OUT / f'{name}{tex}.jpg'), cv2.resize(img, (1024, 1024), interpolation=cv2.INTER_AREA), [cv2.IMWRITE_JPEG_QUALITY, 88])

    proxies = {f'hair/{h}/{h}': 'hair' for h in HAIRS} | {'eyebrows/eyebrow001/eyebrow001': 'eyebrows',
                                                          'eyelashes/eyelashes01/eyelashes01': 'eyelashes', 'eyes/high-poly/high-poly': 'eyes'}
    for path, kind in proxies.items():
        name = path.rsplit('/', 1)[-1]
        refs, w, d, scale = parse_mhclo(read(path + '.mhclo').decode())
        pv, puv, pfaces = parse_obj(read(path + '.obj').decode())
        assert len(pv) == len(refs), (name, len(pv), len(refs))
        pf = [f for g in pfaces.values() for f in g]
        if kind == 'eyes':  # drop the cornea shell (UV'd to the texture's pale corner); clearcoat stands in for it
            pf = [f for f in pf if np.hypot(*(puv[[t for _, t in f]].mean(0) - [.93, .07])) > .08]
        psrc, ppuv, ptris = split_uv(pf, puv)
        arrays |= {f'{name}.refs': refs.astype('<u2'), f'{name}.w': w.astype('<f4'), f'{name}.d': d.astype('<f4'),
                   f'{name}.src': psrc.astype('<u2'), f'{name}.uv': ppuv.astype('<f4'), f'{name}.tris': ptris.astype('<u2')}
        meta['proxies'][name] = {'kind': kind, 'scale': scale}
        mat = read(path.rsplit('/', 1)[0] + f'/{name}.mhmat').decode() if kind != 'eyes' else ''
        if kind in ('hair', 'eyebrows', 'eyelashes'):
            tex = next(l.split()[1] for l in mat.splitlines() if l.startswith('diffuseTexture'))
            rgba = image(path.rsplit('/', 1)[0] + '/' + tex)
            out = gray_rgba(rgba, 1024 if kind == 'hair' else 256) if kind != 'eyelashes' else cv2.resize(rgba, (256, 256))
            cv2.imwrite(str(OUT / f'{name}.webp'), out, [cv2.IMWRITE_WEBP_QUALITY, 90])
    # Eye colours: texture + mean iris colour (irises sit at fixed spots in every MakeHuman eye texture).
    meta['eyes'] = {}
    for color in EYE_COLORS:
        tex = cv2.resize(image(f'eyes/materials/{color}_eye.png', cv2.IMREAD_COLOR), (512, 512), interpolation=cv2.INTER_AREA)
        cv2.imwrite(str(OUT / f'eye_{color}.jpg'), tex, [cv2.IMWRITE_JPEG_QUALITY, 88])
        yy, xx = np.mgrid[:512, :512]
        ring = (np.hypot(xx - 360, yy - 150) - 30).clip(0) * (np.hypot(xx - 360, yy - 150) < 52) > 0
        meta['eyes'][color] = tex[ring][:, ::-1].mean(0).round().astype(int).tolist()

    blob, meta['arrays'] = bytearray(), {}
    for name, a in arrays.items():
        a = np.ascontiguousarray(a)
        blob += b'\0' * (-len(blob) % 4)
        meta['arrays'][name] = [str(a.dtype), len(blob), a.size]
        blob += a.tobytes()
    (OUT / 'body.bin').write_bytes(bytes(blob))
    (OUT / 'body.json').write_text(json.dumps(meta))
    print(f'body.bin {len(blob) / 1e6:.2f} MB; assets in {OUT}')


if __name__ == '__main__':
    main()
