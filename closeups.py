"""Measurements SAM 2 (Meta, Apache-2.0, via transformers) takes from the reference captures, where outlines can't:
- genitals at rest (front views): SAM is prompted across the pelvis, since the shaft hangs to one side, and the mask
  shaped like a hanging shaft is kept: its length along its axis (the visible shaft, from where it leaves the pubic hair
  to the tip) and its width across the middle half; girth ≈ π × width.
- the upper arms' girth: their width across in front views and their depth in side views, where the arm overlaps the
  torso and no outline shows it (the body fit otherwise leaves the arms thin)."""
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



def arms(ref_dir, height_cm, day):
    """{'arm_l', 'arm_r'}: upper-arm girth (cm) at mid-humerus, from captures of the day and the two before: SAM's arm
    mask across the humerus, prompted along it and away from the torso; the width in front views and depth in side
    views (medians over both arms) make an ellipse. Or {} with too few views."""
    import cv2
    import torch
    from PIL import Image
    days = {(date.fromisoformat(day) - timedelta(d)).isoformat() for d in range(3)}
    proc, model = sam()
    seen = {'front': [], 'side': []}
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
        P = {i: to_px(L[i]) for i in (11, 12, 13, 14, 23, 24)}  # shoulders, elbows, hips
        torso, (h, w) = (P[11] + P[12] + P[23] + P[24]) / 4, img.shape[:2]
        pil = Image.fromarray(np.ascontiguousarray(img[..., ::-1]))
        for s_, e_ in ((11, 13), (12, 14)):
            S, E = P[s_], P[e_]
            span = np.linalg.norm(E - S)
            if span < 20:
                continue
            ax = (E - S) / span
            n = np.array([-ax[1], ax[0]])
            n = -n if (torso - (S + E) / 2) @ n > 0 else n  # away from the torso
            pts = [S + t * (E - S) for t in (.35, .5, .65)] + [torso, (S + E) / 2 + n * .45 * span]
            if any(not (0 <= q[0] < w and 0 <= q[1] < h) for q in pts):
                continue
            inp = proc(images=pil, input_points=[[[list(map(float, q)) for q in pts]]], input_labels=[[[1, 1, 1, 0, 0]]], return_tensors='pt')
            with torch.no_grad():
                out = model(**inp, multimask_output=True)
            masks, quality = proc.post_process_masks(out.pred_masks, inp['original_sizes'])[0][0].numpy(), out.iou_scores[0, 0].numpy()
            line = (S + E) / 2 + np.arange(-.6 * span, .6 * span)[:, None] * n
            ok = (line[:, 0] >= 0) & (line[:, 0] < w) & (line[:, 1] >= 0) & (line[:, 1] < h)
            best = None
            for mk, q in zip(masks, quality):
                on = np.zeros(len(line), bool)
                on[ok] = mk[line[ok, 1].astype(int), line[ok, 0].astype(int)]
                a = b = len(line) // 2
                if not on[a] or q < .5:
                    continue
                while a > 0 and on[a - 1]:
                    a -= 1
                while b < len(line) - 1 and on[b + 1]:
                    b += 1
                if b - a + 1 < .7 * span and 0 < a and b < len(line) - 1 and (best is None or q > best[1]):  # the arm alone
                    best = ((b - a + 1) * height_cm / (ref['H'] * ref['k']), q)
            if best:
                seen[view].append(best[0])
    if len(seen['front']) < 5 or len(seen['side']) < 3:
        return {}
    w, d = np.median(seen['front']), np.median(seen['side'])
    g = round(float(np.pi * np.sqrt((w * w + d * d) / 2)), 1)
    return {'arm_l': g, 'arm_r': g}


if __name__ == '__main__':
    import sys
    print(genitals(sys.argv[1], float(sys.argv[2]), sys.argv[3]), arms(sys.argv[1], float(sys.argv[2]), sys.argv[3]))
