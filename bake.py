"""Your skin texture from the reference captures: every texel of the body's UV layout is looked up in each capture
where the fitted, posed body shows it (visible, facing the camera, on skin rather than clothes), and the views are
blended, best-facing and sharpest first, and only their fine detail is kept (on the template's even lighting) so the
room's light doesn't show. Parts no camera saw keep the template skin.
Used by fit.py --store; only the texture is kept."""
import json
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).parent


def assets():
    meta = json.loads((ROOT / 'static/assets/body.json').read_text())
    buf = (ROOT / 'static/assets/body.bin').read_bytes()
    types = {'uint8': np.uint8, 'uint16': np.uint16, 'uint32': np.uint32, 'float32': np.float32}
    arr = lambda k: np.frombuffer(buf, types[meta['arrays'][k][0]], meta['arrays'][k][2], meta['arrays'][k][1])  # noqa: E731
    return arr('body.src').astype(np.int64), arr('body.uv').reshape(-1, 2), arr('body.tris').reshape(-1, 3).astype(np.int64)


def texels(uv, tris, size):
    """Every covered texel: (y, x), its render triangle and barycentric weights. Canvas convention: (u, 1 - v)."""
    P = np.stack([uv[:, 0] * size, (1 - uv[:, 1]) * size], 1)
    ids = np.zeros((size, size), np.int32)
    for t, tri in enumerate(tris):
        cv2.fillConvexPoly(ids, np.round(P[tri] * 4).astype(np.int32), t + 1, cv2.LINE_8, shift=2)
    ys, xs = np.nonzero(ids)
    t = ids[ys, xs] - 1
    a, b, c = P[tris[t, 0]], P[tris[t, 1]], P[tris[t, 2]]
    q = np.stack([xs + .5, ys + .5], 1)
    v0, v1, v2 = b - a, c - a, q - a
    den = v0[:, 0] * v1[:, 1] - v1[:, 0] * v0[:, 1]
    den = np.where(np.abs(den) < 1e-9, 1e-9, den)
    w1 = (v2[:, 0] * v1[:, 1] - v1[:, 0] * v2[:, 1]) / den
    w2 = (v0[:, 0] * v2[:, 1] - v2[:, 0] * v0[:, 1]) / den
    return ys, xs, t, np.clip(np.stack([1 - w1 - w2, w1, w2], 1), 0, 1)


def skin_mask(img):
    """Bare skin (body or face) per pixel, by MediaPipe's multiclass person segmenter; edges trimmed."""
    import mediapipe as mp
    from mediapipe.tasks.python import BaseOptions, vision
    seg = vision.ImageSegmenter.create_from_options(vision.ImageSegmenterOptions(
        base_options=BaseOptions(model_asset_path=str(ROOT / 'models/selfie_multiclass_256x256.tflite')), output_category_mask=True))
    cat = seg.segment(mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(img[..., ::-1]))).category_mask.numpy_view()
    m = np.isin(cat.reshape(img.shape[:2]), (2, 3)).astype(np.uint8)
    return cv2.erode(m, np.ones((7, 7), np.uint8)) > 0


def sample(img, p):
    """Bilinear colours of img at points p (N, 2), as float (remap maps must stay under 32k a side)."""
    n, w = len(p), 4096
    q = np.zeros((-(-n // w) * w, 2), np.float32)
    q[:n] = p
    out = cv2.remap(img, q[:, 0].reshape(-1, w), q[:, 1].reshape(-1, w), cv2.INTER_LINEAR)
    return out.reshape(-1, 3)[:n].astype(float)


def bake(views, src, uv, tris, size=2048, base=None):
    """views: [{'img': BGR, 'px': (V, 2) image px of every base vertex, 'cam': (V, 3) camera-frame positions (m, z away),
    'weight': quality}]. Returns (texture BGR uint8, coverage 0..1 per texel)."""
    ys, xs, t, bary = texels(uv, tris, size)
    acc, wsum, per_view = np.zeros((len(t), 3)), np.zeros(len(t)), []
    for v in views:
        img, px, c3 = v['img'], v['px'][src], v['cam'][src]  # per render vertex
        h, w = img.shape[:2]
        # Facing: outward normal toward the camera (it looks along +z).
        n = np.cross(c3[tris[:, 1]] - c3[tris[:, 0]], c3[tris[:, 2]] - c3[tris[:, 0]])
        facing = -n[:, 2] / np.maximum(np.linalg.norm(n, axis=1), 1e-12)
        # Visibility: a depth buffer of the projected body, nearest triangle wins (painter's order).
        zbuf = np.full((h, w), np.inf, np.float32)
        depth = c3[tris].mean(1)[:, 2]
        for k in np.argsort(-depth):
            if facing[k] > 0:
                cv2.fillConvexPoly(zbuf, np.round(px[tris[k]] * 4).astype(np.int32), float(depth[k]), cv2.LINE_8, shift=2)
        p = (bary[:, :, None] * px[tris[t]]).sum(1)
        z = (bary * c3[tris[t]][:, :, 2]).sum(1)
        inside = (p[:, 0] >= 0) & (p[:, 0] < w - 1) & (p[:, 1] >= 0) & (p[:, 1] < h - 1)
        pi = np.clip(np.round(p).astype(int), 0, [w - 1, h - 1])
        ok = inside & (facing[t] > .3) & (z <= zbuf[pi[:, 1], pi[:, 0]] + .015) & v['skin'][pi[:, 1], pi[:, 0]]
        col = sample(img, p[ok])
        per_view.append((ok, col, facing[t][ok] ** 2 * v['weight']))
    # Each view's colour balance matched to all views' median (the two cameras see different white balance).
    allc = np.concatenate([c for _, c, _ in per_view]) if per_view else np.zeros((0, 3))
    target = np.median(allc, 0) if len(allc) else np.array([128, 128, 128.])
    for ok, col, wt in per_view:
        if len(col) < 500:
            continue
        gain = np.clip(target / np.maximum(np.median(col, 0), 1), .7, 1.4)
        acc[ok] += col * gain * wt[:, None]
        wsum[ok] += wt
    tex = np.zeros((size, size, 3))
    cover = np.zeros((size, size))
    seen = wsum > 0
    tex[ys[seen], xs[seen]] = acc[seen] / wsum[seen, None]
    cover[ys[seen], xs[seen]] = np.clip(wsum[seen] / .3, 0, 1)
    if base is not None:  # your skin's detail on the template's even lighting (tone comes from your face photo, later)
        base = cv2.resize(base, (size, size)).astype(float)
        # Detail = your skin ÷ its own blur (~1.5 cm): marks, hair, pores, not the room's light falling across you.
        lit = cv2.GaussianBlur(tex * cover[..., None], (0, 0), 12) / np.maximum(cv2.GaussianBlur(cover, (0, 0), 12), 1e-3)[..., None]
        detail = np.clip(tex / np.maximum(lit, 1), .6, 1.6)
        a = np.minimum(cv2.GaussianBlur(cover, (0, 0), 3), cover)  # feathered inward from what was seen
        island = np.zeros((size, size), np.uint8)
        island[ys, xs] = 1  # texture sections: the blur above mixes across their edges, so detail fades out there
        a = (a * np.clip(cv2.distanceTransform(island, cv2.DIST_L2, 3) / 12, 0, 1))[..., None]
        tex = base * (1 + (detail - 1) * a)
    return np.clip(tex, 0, 255).astype(np.uint8), cover
