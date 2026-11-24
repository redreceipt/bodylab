"""Your skin texture from the reference captures: every texel of the body's UV layout is looked up in each capture
where the fitted, posed body shows it (visible, facing the camera, on skin rather than clothes), and the views are
blended, best-facing and sharpest first, and only their fine detail is kept (on the template's even lighting) so the
room's light doesn't show. Parts no camera saw keep the template skin.
Used by fit.py --store; only the texture is kept."""
import functools
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
    head = arr('weights').reshape(-1, 6)[:, 0] > 127  # the head: your face photo covers it
    return arr('body.src').astype(np.int64), arr('body.uv').reshape(-1, 2), arr('body.tris').reshape(-1, 3).astype(np.int64), head


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
    w = np.clip(np.stack([1 - w1 - w2, w1, w2], 1), 0, 1)  # texels at a triangle's rim fall just outside it
    return ys, xs, t, w / w.sum(1, keepdims=True)  # a point on the triangle: weights sum to 1


@functools.cache
def segmenter():
    import mediapipe as mp
    from mediapipe.tasks.python import BaseOptions, vision
    return mp, vision.ImageSegmenter.create_from_options(vision.ImageSegmenterOptions(
        base_options=BaseOptions(model_asset_path=str(ROOT / 'models/selfie_multiclass_256x256.tflite')), output_category_mask=True))


def skin_mask(img):
    """Bare skin and body hair per pixel, by MediaPipe's multiclass person segmenter; edges trimmed."""
    mp, seg = segmenter()
    cat = seg.segment(mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(img[..., ::-1]))).category_mask.numpy_view()
    m = np.isin(cat.reshape(img.shape[:2]), (1, 2, 3)).astype(np.uint8)  # hair too: on the body that's body hair (the head isn't baked)
    return cv2.erode(m, np.ones((7, 7), np.uint8)) > 0


def sample(img, p):
    """Bilinear colours of img at points p (N, 2), as float (remap maps must stay under 32k a side)."""
    n, w = len(p), 4096
    q = np.zeros((-(-n // w) * w, 2), np.float32)
    q[:n] = p
    out = cv2.remap(img, q[:, 0].reshape(-1, w), q[:, 1].reshape(-1, w), cv2.INTER_LINEAR)
    return out.reshape(-1, 3)[:n].astype(float)


def bake(views, src, uv, tris, size=2048, base=None, skip=None):
    """views: [{'img': BGR, 'px': (V, 2) image px of every base vertex, 'cam': (V, 3) camera-frame positions (m, z away),
    'weight': quality}]. skip: base vertices left to the template (the head). Returns (texture BGR uint8, coverage 0..1
    per texel). A view counts only if the projected body overlaps the person in it well (it's aligned), and only
    where it faces the camera squarely; per texel, samples far off the others' average are dropped."""
    ys, xs, t, bary = texels(uv, tris, size)
    keep = ~skip[src[tris[t]]].any(1) if skip is not None else np.ones(len(t), bool)
    acc, wsum, per_view = np.zeros((len(t), 3)), np.zeros(len(t)), []
    for v in views:
        img, px, c3 = v['img'], v['px'][src], v['cam'][src]  # per render vertex
        h, w = img.shape[:2]
        # Facing: outward normal toward the camera (it looks along +z).
        n = np.cross(c3[tris[:, 1]] - c3[tris[:, 0]], c3[tris[:, 2]] - c3[tris[:, 0]])
        facing = -n[:, 2] / np.maximum(np.linalg.norm(n, axis=1), 1e-12)
        f = np.flatnonzero(facing[t] > 0)  # texels facing the camera: the rest can't be seen
        p = np.einsum('nk,nkd->nd', bary[f], px[tris[t[f]]])  # projected
        z = np.einsum('nk,nk->n', bary[f], c3[tris[t[f]]][:, :, 2])
        # Visibility: the nearest depth per cell of a half-resolution grid (a few texels land in each, so no holes).
        hh, ww = (h + 1) // 2, (w + 1) // 2
        cell = np.floor(p / 2).astype(int)
        on = (cell[:, 0] >= 0) & (cell[:, 0] < ww) & (cell[:, 1] >= 0) & (cell[:, 1] < hh)
        zbuf = np.full(hh * ww, np.inf, np.float32)
        np.minimum.at(zbuf, cell[on, 1] * ww + cell[on, 0], z[on])
        zbuf = zbuf.reshape(hh, ww)
        body = cv2.morphologyEx(np.isfinite(zbuf).astype(np.uint8), cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
        person = cv2.resize((np.abs(img.astype(int) - 128).sum(2) > 6).astype(np.uint8), (ww, hh), interpolation=cv2.INTER_NEAREST)  # the crop greys out all but you
        if (body & person).sum() / max((body | person).sum(), 1) < .8:
            continue  # the fit's pose doesn't line up with this capture
        inside = (p[:, 0] >= 0) & (p[:, 0] < w - 1) & (p[:, 1] >= 0) & (p[:, 1] < h - 1)
        pi = np.clip(np.round(p).astype(int), 0, [w - 1, h - 1])
        ci = np.clip(cell, 0, [ww - 1, hh - 1])
        sel = keep[f] & inside & (facing[t[f]] > .5) & (z <= zbuf[ci[:, 1], ci[:, 0]] + .015) & v['skin'][pi[:, 1], pi[:, 0]]
        col = sample(img, p[sel])
        sel[sel] = col.max(1) - col.min(1) > 8  # colourless: the crop's grey fill or a blown highlight, not skin
        col = col[col.max(1) - col.min(1) > 8]
        ok = np.zeros(len(t), bool)
        ok[f[sel]] = True
        per_view.append((ok, col, facing[t][ok] ** 2 * v['weight']))
    # Each view's colour balance matched to all views' median (the two cameras see different white balance).
    allc = np.concatenate([c for _, c, _ in per_view]) if per_view else np.zeros((0, 3))
    target = np.median(allc, 0) if len(allc) else np.array([128, 128, 128.])
    per_view = [(ok, col * np.clip(target / np.maximum(np.median(col, 0), 1), .7, 1.4), wt) for ok, col, wt in per_view if len(col) >= 500]
    for ok, col, wt in per_view:
        acc[ok] += col * wt[:, None]
        wsum[ok] += wt
    mean = acc / np.maximum(wsum, 1e-9)[:, None]
    acc[:], wsum[:] = 0, 0
    for ok, col, wt in per_view:  # again without samples over 20 % brighter or darker than the average (shadows, glare)
        good = np.abs(col.sum(1) / np.maximum(mean[ok].sum(1), 1) - 1) < .2
        acc[np.flatnonzero(ok)[good]] += col[good] * wt[good, None]
        wsum[np.flatnonzero(ok)[good]] += wt[good]
    tex = np.zeros((size, size, 3))
    cover = np.zeros((size, size))
    seen = wsum > 0
    tex[ys[seen], xs[seen]] = acc[seen] / wsum[seen, None]
    cover[ys[seen], xs[seen]] = np.clip(wsum[seen] / .3, 0, 1)
    if base is not None:  # your skin's colour and detail on the template's even lighting (the overall tone comes later)
        base = cv2.resize(base, (size, size)).astype(float)
        # Detail = your skin ÷ its own blur (~1.5 cm): marks, hair, pores, not the room's light falling across you.
        lit = cv2.GaussianBlur(tex * cover[..., None], (0, 0), 12) / np.maximum(cv2.GaussianBlur(cover, (0, 0), 12), 1e-3)[..., None]
        detail = 1 + (np.clip(tex / np.maximum(lit, 1), .8, 1.25) - 1) * .8  # gentle: skin detail, not grime
        # Muscle definition (~2-5 cm: ab grooves, the pecs' lower edge): shading every view agrees on, gentler still.
        broad = cv2.GaussianBlur(tex * cover[..., None], (0, 0), 36) / np.maximum(cv2.GaussianBlur(cover, (0, 0), 36), 1e-3)[..., None]
        detail = detail * (1 + (np.clip(lit / np.maximum(broad, 1), .85, 1.18) - 1) * .7)
        a = np.minimum(cv2.GaussianBlur(cover, (0, 0), 3), cover)  # feathered inward from what was seen
        island = np.zeros((size, size), np.uint8)
        island[ys, xs] = 1  # texture sections: the blur above mixes across their edges, so detail fades out there
        edge = cv2.distanceTransform(island, cv2.DIST_L2, 3)
        # Your skin's own colour across the body (~5 cm and up: tan lines, darker forearms, body hair), relative to your
        # overall tone, which tints the whole texture later; fades out near section edges, where the blur mixes parts.
        seen = cover > .5
        # darkening capped harder: light falls off toward the floor, so legs photograph darker than they are
        colour = np.clip(broad / np.maximum(np.median(tex[seen], 0) if seen.any() else 128, 1), .75, 1.35)
        c = (np.minimum(cv2.GaussianBlur(cover, (0, 0), 12), cover) * np.clip(edge / 36, 0, 1))[..., None]
        a = (a * np.clip(edge / 12, 0, 1))[..., None]
        tex = base * (1 + (colour - 1) * c) * (1 + (detail - 1) * a)
    return np.clip(tex, 0, 255).astype(np.uint8), cover

