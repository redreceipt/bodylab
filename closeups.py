"""Measurements SAM 2 (Meta, Apache-2.0, via transformers) takes from the reference captures, where outlines can't:
- genitals at rest (front views): SAM is prompted across the pelvis, since the shaft hangs to one side, and the mask
  shaped like a hanging shaft is kept: its length along its axis (the visible shaft, from where it leaves the pubic hair
  to the tip) and its width across the middle half; girth ≈ π × width.
- the upper arms' and thighs' girth: the body fit can't see an arm's depth (in side views it covers the torso), nor the
  outer thigh where the hand hangs beside it, and otherwise leaves them thin."""
import functools
import json
from datetime import date, timedelta
from pathlib import Path

import numpy as np

REPO = 'facebook/sam2.1-hiera-small'


@functools.cache
def sam():
    from transformers import Sam2Model, Sam2Processor
    return Sam2Processor.from_pretrained(REPO), Sam2Model.from_pretrained(REPO).eval()


def shaft(img, rec, height_cm):
    """(length cm, width cm, mask quality 0..1, cm per px) of the shaft in one capture, or None."""
    import torch
    from PIL import Image
    ref, L = rec['ref'], np.array(rec['sil']['l'])
    to_px = lambda q: (np.array(ref['origin']) + np.asarray(q) * ref['H'] - np.array(ref['box'][::-1])) * ref['k']  # noqa: E731
    hip, knee = to_px(L[[23, 24]].mean(0)), to_px(L[[25, 26]].mean(0))
    h, w = img.shape[:2]
    if not (0 <= hip[1] and knee[1] < h and 0 <= hip[0] < w):  # the crotch must be in the capture
        return None
    span, cm_px = np.linalg.norm(knee - hip), height_cm / (ref['H'] * ref['k'])
    x0, y0 = int(max(hip[0] - .45 * span, 0)), int(max(hip[1] - .15 * span, 0))
    crop = img[y0:int(hip[1] + .75 * span), x0:int(hip[0] + .45 * span)]
    cx, cy = hip[0] - x0, hip[1] - y0
    neg = [[cx - .32 * span, cy + .3 * span], [cx + .32 * span, cy + .3 * span], [cx, cy - .05 * span]]  # thighs, belly
    cands = [[[cx + dx * span, cy + .2 * span], [cx + dx * span, cy + .28 * span]] for dx in (-.16, -.11, -.06, -.02, .02, .06, .11, .16)]
    proc, model = sam()
    inp = proc(images=Image.fromarray(np.ascontiguousarray(crop[..., ::-1])), input_points=[[c + neg for c in cands]],
               input_labels=[[[1, 1, 0, 0, 0]] * len(cands)], return_tensors='pt')
    with torch.no_grad():
        out = model(**inp, multimask_output=True)
    masks, quality = proc.post_process_masks(out.pred_masks, inp['original_sizes'])[0].numpy(), out.iou_scores[0].numpy()
    best = None
    for mk, q in zip(masks.reshape(-1, *masks.shape[2:]), quality.reshape(-1)):
        ys, xs = np.nonzero(mk)
        if len(xs) < 50:
            continue
        P = np.c_[xs, ys].astype(float)
        c = P.mean(0)
        ax = np.linalg.svd(P - c, full_matrices=False)[2][0]
        along, across = (P - c) @ ax, (P - c) @ np.array([-ax[1], ax[0]])
        length = np.ptp(along)
        widths = [np.ptp(across[np.abs(along - a) < 1.5]) for a in np.linspace(along.min() + length / 4, along.max() - length / 4, 9)
                  if (np.abs(along - a) < 1.5).sum() > 1]
        lc, wc = length * cm_px, float(np.median(widths)) * cm_px if widths else 0
        # a hanging shaft: 5-22 cm long, 2.5-6 cm wide, within 35° of vertical, starting at the pubic hair
        if 5 < lc < 22 and 2.5 < wc < 6 and abs(ax[0]) < .57 and ys.min() < cy + .25 * span and (not best or q > best[2]):
            best = (lc, wc, float(q), cm_px)
    return best


def person(img):
    """You in a reference capture, by SAM 2 prompted with your box: the crop's own cut-out has a halo of background."""
    import torch
    from PIL import Image
    proc, model = sam()
    ys, xs = np.nonzero(np.abs(img.astype(int) - 128).sum(2) > 6)
    box = [float(xs.min()), float(ys.min()), float(xs.max()), float(ys.max())]
    inp = proc(images=Image.fromarray(np.ascontiguousarray(img[..., ::-1])), input_boxes=[[box]], return_tensors='pt')
    with torch.no_grad():
        out = model(**inp, multimask_output=False)
    return proc.post_process_masks(out.pred_masks, inp['original_sizes'])[0][0, 0].numpy().astype(np.uint8)


def genitals(ref_dir, height_cm, day):
    """From front captures of the day and the two before (fit.py's window), with bare thighs (shorts look like skin to
    SAM): the median at rest, 'penis_len' and 'penis_girth' (cm), and the range seen, '_lo'/'_hi' (10th, 90th
    percentile); or {} with under 3 confident masks."""
    import cv2
    days = {(date.fromisoformat(day) - timedelta(d)).isoformat() for d in range(3)}
    found = []
    for p in sorted(Path(ref_dir).glob('*.json')):
        if p.name[:10] not in days or not p.with_suffix('.jpg').exists():
            continue
        rec = json.loads(p.read_text())
        if (rec.get('ref') and abs(rec.get('theta', 90)) < 25 and ((rec.get('skin') or {}).get('thighs') or 0) >= .9
                and (r := shaft(cv2.imread(str(p.with_suffix('.jpg'))), rec, height_cm)) and r[2] >= .6):
            found.append(r)
    if len(found) < 3:
        return {}
    out = {'penis_n': len(found)}
    for k, v in (('penis_len', [f[0] for f in found]), ('penis_girth', [f[1] * np.pi for f in found])):
        lo, mid, hi = np.percentile(v, [10, 50, 90])
        out |= {k: round(float(mid), 1), f'{k}_lo': round(float(lo), 1), f'{k}_hi': round(float(hi), 1)}
    return out



def limbs(ref_dir, height_cm, day):
    """{'arm_l', 'arm_r', 'thigh_l', 'thigh_r'}: girth (cm) from captures of the day and the two before, where SAM's
    mask of the limb runs across it (prompted along it, the torso or other leg and the hand marked as not it). Upper
    arms at mid-humerus: front-view width and side-view depth make an ellipse. Thighs 40 % down: the front-view width as
    a circle's (shortcut: side views have the hand beside the thigh; measure its depth if they stop doing that). Or
    none of a limb with too few views."""
    import cv2
    import torch
    from PIL import Image
    days = {(date.fromisoformat(day) - timedelta(d)).isoformat() for d in range(3)}
    proc, model = sam()
    seen = {}
    for p in sorted(Path(ref_dir).glob('*.json')):
        if p.name[:10] not in days or not p.with_suffix('.jpg').exists():
            continue
        rec = json.loads(p.read_text())
        th = abs(rec.get('theta', 90))
        view = 'front' if th < 30 or th > 150 else 'side' if 60 < th < 120 else None
        if not view or not rec.get('ref'):
            continue
        img, ref, L = cv2.imread(str(p.with_suffix('.jpg'))), rec['ref'], np.array(rec['sil']['l'])
        to_px = lambda q: (np.array(ref['origin']) + np.asarray(q) * ref['H'] - np.array(ref['box'][::-1])) * ref['k']  # noqa: E731
        P = {i: to_px(L[i]) for i in (11, 12, 13, 14, 15, 16, 23, 24, 25, 26)}
        torso, (h, w) = (P[11] + P[12] + P[23] + P[24]) / 4, img.shape[:2]
        pil = Image.fromarray(np.ascontiguousarray(img[..., ::-1]))
        bare_legs, bare_arms = (((rec.get('skin') or {}).get(k) or 0) >= g for k, g in (('thighs', .9), ('torso', .95)))
        limbs_ = [('arm', 11, 13, .5, [torso]), ('arm', 12, 14, .5, [torso])] if bare_arms else []
        if view == 'front' and bare_legs:
            limbs_ += [('thigh', 23, 25, .4, [P[26] * .6 + P[23] * .4, P[15]]), ('thigh', 24, 26, .4, [P[25] * .6 + P[24] * .4, P[16]])]
        for kind, s_, e_, t, negs in limbs_:
            S, E = P[s_], P[e_]
            span = np.linalg.norm(E - S)
            if span < 20:
                continue
            ax = (E - S) / span
            n = np.array([-ax[1], ax[0]])
            M = S + t * (E - S)
            if kind == 'arm':  # and the background beside it, away from the torso
                n = -n if (torso - M) @ n > 0 else n
                negs = negs + [M + n * .45 * span]
            pts = [S + u * (E - S) for u in (t - .15, t, t + .15)] + negs
            if any(not (0 <= q[0] < w and 0 <= q[1] < h) for q in pts):
                continue
            inp = proc(images=pil, input_points=[[[list(map(float, q)) for q in pts]]], input_labels=[[[1, 1, 1] + [0] * len(negs)]], return_tensors='pt')
            with torch.no_grad():
                out = model(**inp, multimask_output=True)
            masks, quality = proc.post_process_masks(out.pred_masks, inp['original_sizes'])[0][0].numpy(), out.iou_scores[0, 0].numpy()
            line = M + np.arange(-.6 * span, .6 * span)[:, None] * n
            ok = (line[:, 0] >= 0) & (line[:, 0] < w) & (line[:, 1] >= 0) & (line[:, 1] < h)
            best = None
            for mk, q in zip(masks, quality):
                if q < .5 or any(mk[int(z[1]), int(z[0])] for z in negs if 0 <= z[0] < w and 0 <= z[1] < h):
                    continue  # unsure, or it took in what isn't the limb
                on = np.zeros(len(line), bool)
                on[ok] = mk[line[ok, 1].astype(int), line[ok, 0].astype(int)]
                a = b = len(line) // 2
                if not on[a]:
                    continue
                while a > 0 and on[a - 1]:
                    a -= 1
                while b < len(line) - 1 and on[b + 1]:
                    b += 1
                if b - a + 1 < .7 * span and 0 < a and b < len(line) - 1 and (best is None or q > best[1]):  # the limb alone
                    best = ((b - a + 1) * height_cm / (ref['H'] * ref['k']), q)
            if best:
                seen.setdefault((kind, view), []).append(best[0])
    out = {}
    arm_w, arm_d, thigh_w = (seen.get(k, []) for k in (('arm', 'front'), ('arm', 'side'), ('thigh', 'front')))
    if len(arm_w) >= 5 and len(arm_d) >= 3:
        w, d = np.median(arm_w), np.median(arm_d)
        out |= dict.fromkeys(('arm_l', 'arm_r'), round(float(np.pi * np.sqrt((w * w + d * d) / 2)), 1))
    if len(thigh_w) >= 5:
        out |= dict.fromkeys(('thigh_l', 'thigh_r'), round(float(np.pi * np.median(thigh_w)), 1))
    return out


def protrusion(you, hip, knee, fwd, cm_px):
    """What stands out forward of the body at hip height in a profile mask `you` (hip, knee: image points; fwd ±1: the
    way you face): (length cm, girth cm, angle ° above level), or None. Rows reaching 3+ cm past the line from belly to
    thigh make bands (a hand in front makes its own, lower); the one at hip height must reach 6+ cm. Along it, each
    column's middle traces the shaft (its slope: the angle) and its thickness × cos(angle) is the diameter."""
    h, w = you.shape
    span = np.linalg.norm(np.subtract(knee, hip))

    def front(y):  # the body's front edge on row y
        xs = np.flatnonzero(you[int(np.clip(y, 0, h - 1))])
        return (xs.max() if fwd > 0 else xs.min()) if len(xs) else None
    yb, yt = hip[1] - .2 * span, hip[1] + .45 * span  # belly and thigh, above and below
    belly, thigh = front(yb), front(yt)
    if None in (belly, thigh):
        return None
    line = lambda y: belly + (thigh - belly) * (y - yb) / (yt - yb)  # the body's front line
    rows = [(y, (f - line(y)) * fwd * cm_px) for y in range(int(hip[1] - .1 * span), int(hip[1] + .3 * span)) if (f := front(y)) is not None]
    bands, run = [], []
    for r in rows + [(0, 0)]:
        if r[1] >= 3:
            run.append(r)
        elif run:
            bands, run = bands + [run], []
    near = [b for b in bands if abs(np.mean([r[0] for r in b]) - hip[1]) * cm_px < 6]
    if not near:
        return None
    band = max(near, key=len)
    reach, y0, y1 = max(r[1] for r in band), band[0][0], band[-1][0]
    if reach < 6:
        return None
    x0 = line((y0 + y1) / 2)
    mid, thick = [], []
    for f in np.linspace(.25, .85, 13):  # columns along it, clear of the root and the tip's rounding
        x = round(x0 + fwd * f * reach / cm_px)
        ys = np.flatnonzero(you[y0:y1 + 1, int(np.clip(x, 0, w - 1))])
        if len(ys):
            mid.append((f * reach, -(y0 + ys.mean()) * cm_px))  # cm out, cm up
            thick.append(len(ys) * cm_px)
    if len(mid) < 5:
        return None
    angle = np.degrees(np.arctan(np.polyfit(*np.array(mid).T, 1)[0]))
    c = np.cos(np.radians(angle))
    return reach / c, np.pi * np.median(thick) * c, angle


def erect_views(ref_dir, height_cm, day):
    """Every true profile up to the day (any before it: erections are occasional) where an erect penis stands out forward of your outline
    (protrusion; at ~14 px across, SAM's masks of the shaft itself were less reliable): (length cm from where it leaves
    the body, girth cm, angle ° above level, capture name)."""
    import cv2
    out = []
    for p in sorted(Path(ref_dir).glob('*.json')):
        if p.name[:10] > day or not p.with_suffix('.jpg').exists():
            continue
        rec = json.loads(p.read_text())
        if not (75 < abs(rec.get('theta', 0)) < 105 and rec.get('ref') and ((rec.get('skin') or {}).get('thighs') or 0) >= .8):
            continue
        img, ref, L = cv2.imread(str(p.with_suffix('.jpg'))), rec['ref'], np.array(rec['sil']['l'])
        q = (np.array(ref['origin']) + L[[0, 23, 24, 25, 26]] * ref['H'] - np.array(ref['box'][::-1])) * ref['k']  # in the image
        nose, hip, knee = q[0], q[1:3].mean(0), q[3:].mean(0)
        fwd = 1 if nose[0] > hip[0] else -1  # the way you face
        # SAM's outline: the crop's own cut-out has a halo that would thicken it by ~3 cm
        r = protrusion(person(img) > 0, hip, knee, fwd, height_cm / (ref['H'] * ref['k']))
        if r and 2.5 < r[1] / np.pi < 6:  # a shaft's thickness, not a hand's
            out.append((*(round(float(x), 1) for x in r), p.name))
    return out


def erect(ref_dir, height_cm, day):
    """{'penis_erect_len', 'penis_erect_girth', 'penis_erect_angle', 'penis_erect_n'}: medians over erect_views (2+), or {}."""
    v = erect_views(ref_dir, height_cm, day)
    if len(v) < 2:
        return {}
    return {'penis_erect_len': round(float(np.median([x[0] for x in v])), 1), 'penis_erect_girth': round(float(np.median([x[1] for x in v])), 1),
            'penis_erect_angle': round(float(np.median([x[2] for x in v])), 1), 'penis_erect_n': len(v)}


if __name__ == '__main__':
    import sys
    print(*(f(sys.argv[1], float(sys.argv[2]), sys.argv[3]) for f in (genitals, erect, limbs)))
