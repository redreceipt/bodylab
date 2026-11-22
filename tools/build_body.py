"""Build static/body.bin + static/body.json from MakeHuman's CC0 base mesh (hm08).

Exports the default young male and female bodies (base + the three ethnicity targets at 1/3 each, as
MakeHuman does), eyeballs, per-vertex region weights from the default rig, and joint positions, in cm
with the feet at y=0, facing +z, character's left at +x. The browser fits this template to each scan.

Run: uv run python tools/build_body.py
"""
import json
import urllib.request
from pathlib import Path

import numpy as np

COMMIT = 'a8bc2d54ff0ac92e78ff71431b1023eda42bf482'
SRC = f'https://raw.githubusercontent.com/makehumancommunity/makehuman/{COMMIT}/makehuman/data/'
OUT = Path(__file__).resolve().parent.parent / 'static'
REGIONS = ['head', 'torso', 'arm_l', 'arm_r', 'leg_l', 'leg_r']
JOINTS = {'neck': 'joint-neck', 'head': 'joint-head', 'eye_l': 'joint-l-eye', 'eye_r': 'joint-r-eye'} | {
    f'{name}_{s}': f'joint-{s}-{mh}' for s in 'lr' for name, mh in
    (('shoulder', 'shoulder'), ('elbow', 'elbow'), ('wrist', 'hand'), ('hip', 'upper-leg'), ('knee', 'knee'), ('ankle', 'ankle'))}


def fetch(path):
    return urllib.request.urlopen(SRC + path, timeout=60).read().decode()


def region(bone):
    name, side = bone.split('.')[0], '_l' if bone.endswith('.L') else '_r'
    if name.startswith(('upperarm', 'lowerarm', 'wrist', 'finger', 'metacarpal')):
        return 'arm' + side
    if name.startswith(('upperleg', 'lowerleg', 'foot', 'toe')):
        return 'leg' + side
    if name in ('head', 'jaw') or name.startswith(('eye', 'oculi', 'orbicularis', 'oris', 'risorius', 'levator', 'special', 'tongue')):
        return 'head'
    return 'torso'  # root, spine, pelvis, clavicle, shoulder (deltoid), breast, neck


def main():
    verts, faces, group = [], {}, None
    for line in fetch('3dobjs/base.obj').splitlines():
        p = line.split()
        if not p:
            continue
        if p[0] == 'v':
            verts.append([float(x) for x in p[1:4]])
        elif p[0] == 'g':
            group = p[1]
        elif p[0] == 'f':
            faces.setdefault(group, []).append([int(x.split('/')[0]) - 1 for x in p[1:]])
    base = np.array(verts)

    def target(path):
        t = np.zeros_like(base)
        for line in fetch(path).splitlines():
            if line and not line.startswith('#'):
                i, dx, dy, dz = line.split()
                t[int(i)] = float(dx), float(dy), float(dz)
        return t

    body = sorted({i for f in faces['body'] for i in f})
    assert body == list(range(len(body))), 'body vertices are expected to come first'
    eye_faces = faces['helper-l-eye'] + faces['helper-r-eye']
    eye_ids = sorted({i for f in eye_faces for i in f})
    eye_map = {v: k for k, v in enumerate(eye_ids)}
    tri = lambda fs, remap=lambda i: i: [[remap(f[0]), remap(f[k]), remap(f[k + 1])] for f in fs for k in range(1, len(f) - 1)]  # noqa: E731

    weights = np.zeros((len(body), len(REGIONS)))
    for bone, pairs in json.loads(fetch('rigs/default_weights.mhw'))['weights'].items():
        for i, w in pairs:
            if i < len(body):
                weights[i, REGIONS.index(region(bone))] += w
    weights /= weights.sum(1, keepdims=True).clip(1e-9)

    meta = {'regions': REGIONS, 'n': len(body), 'tris': len(tri(faces['body'])), 'eye_n': len(eye_ids),
            'eye_tris': len(eye_faces and tri(eye_faces)), 'joints': {}, 'license': 'MakeHuman hm08 assets, CC0 1.0'}
    chunks = {}
    for sex in ('male', 'female'):
        v = base + sum(target(f'targets/macrodetails/{race}-{sex}-young.target') for race in ('african', 'asian', 'caucasian')) / 3
        ground = v[[i for f in faces['joint-ground'] for i in f]].mean(0)[1]
        v = (v - [0, ground, 0]) * 10  # decimetres -> cm, feet on the floor
        center = lambda g: v[sorted({i for f in faces[g] for i in f})].mean(0).round(2).tolist()  # noqa: E731
        meta['joints'][sex] = {k: center(g) for k, g in JOINTS.items()}
        chunks[sex] = v[:len(body)].astype('<f4').tobytes()
        chunks[f'eye_{sex}'] = v[eye_ids].astype('<f4').tobytes()

    # Layout (little endian): male xyz f32, female xyz f32, region weights u8 x6, triangles u16,
    # eye male xyz f32, eye female xyz f32, eye triangles u16.
    blob = b''.join([chunks['male'], chunks['female'], np.rint(weights * 255).astype('u1').tobytes(),
                     np.array(tri(faces['body']), '<u2').tobytes(), chunks['eye_male'], chunks['eye_female'],
                     np.array(tri(eye_faces, eye_map.get), '<u2').tobytes()])
    (OUT / 'body.bin').write_bytes(blob)
    (OUT / 'body.json').write_text(json.dumps(meta))
    print(f"body.bin {len(blob) / 1e6:.2f} MB, {meta['n']} verts, {meta['tris']} tris, height "
          f"{max(np.frombuffer(chunks['male'], '<f4')[1::3]):.1f} cm (male)")


if __name__ == '__main__':
    main()
