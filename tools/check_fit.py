"""How well the fitted body matches your reference captures: the body is fitted to them as fit.py fits a day (with what
closeups.py measures), then, posed and projected as in each capture, its silhouette is compared with yours (SAM 2) row
by row, the filled length at heights from the hip joints (0) to the shoulder joints (1), front views and side views apart.
Ratios near 1 are a good fit; a region consistently off points at something the fit can't see.

Run: uv run --group fit python tools/check_fit.py [--db data/bodylab.db] [--day 2026-11-24] [--sheet /tmp/check.jpg]
"""
import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import closeups  # noqa: E402
import fit  # noqa: E402

LEVELS = (-.15, .1, .28, .5, .62, .72, .85, .95)


def widths(img, px, tris, L, info, levels):
    """Body ÷ photo silhouette, row by row, at each level where both are whole (not cut off by the crop); and the body's mask."""
    h, w = img.shape[:2]
    you = closeups.person(img)
    body = np.zeros((h, w), np.uint8)
    for t in px[tris].round().astype(np.int32):
        cv2.fillConvexPoly(body, t, 1)  # one by one: overlapping polygons in one call leave holes
    to_px = lambda q: (np.array(info['origin']) + np.asarray(q) * info['H'] - np.array(info['box'][::-1])) * info['k']  # noqa: E731
    sh, hp = to_px(L[[11, 12]].mean(0)), to_px(L[[23, 24]].mean(0))

    out = {}
    for f in levels:  # filled length of the row (a few rows averaged): a small gap between arm and torso counts as itself
        y = int(round((hp + f * (sh - hp))[1]))
        if 3 <= y < h - 3 and not you[y - 3:y + 4, [0, -1]].any() and (a := you[y - 3:y + 4].sum()):
            out[f] = body[y - 3:y + 4].sum() / a
    return out, body, (sh, hp)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--db', default=str(fit.ROOT / 'data' / 'bodylab.db'))
    ap.add_argument('--day')
    ap.add_argument('--sheet', help='save overlays of a few captures (the body outlined in cyan)')
    a = ap.parse_args()
    day, _, prof = fit.load(a.db, a.day)
    ref_dir = Path(a.db).parent / 'reference'
    prof['seen'] = closeups.limbs(ref_dir, prof['height_cm'], day)
    recs = []
    for p in sorted(ref_dir.glob('*.json')):
        r = json.loads(p.read_text())
        th = abs(r.get('theta', 90))
        if (p.name[:10] <= day and r.get('ref') and r.get('sil') and p.with_suffix('.jpg').exists()
                and ((r.get('skin') or {}).get('torso') or 0) >= .95 and (th < 25 or th > 155 or 75 < th < 105)):
            recs.append((p, r))
    recs = [recs[i] for i in np.linspace(0, len(recs) - 1, min(len(recs), 240)).round().astype(int)] if recs else []
    if len(recs) < fit.MIN_FRAMES:
        raise SystemExit(f'need {fit.MIN_FRAMES} bare front or side captures, have {len(recs)}')
    _, _, _, views = fit.fit([r for _, r in recs], prof, want=range(len(recs)), log=lambda *_: None)
    tris, H = fit.assets()[2], prof['height_cm'] / 100
    ratios, tiles = {'front': {f: [] for f in LEVELS}, 'side': {f: [] for f in LEVELS}}, []
    for j, (p, rec) in enumerate(recs):
        img, info = cv2.imread(str(p.with_suffix('.jpg'))), rec['ref']
        px = (np.array(info['origin']) + views[j][:, :2] / H * info['H'] - info['box'][::-1]) * info['k']
        r, body, (sh, hp) = widths(img, px, tris, np.array(rec['sil']['l']), info, LEVELS)
        view = 'side' if 75 < abs(rec.get('theta', 90)) < 105 else 'front'
        for f, v in r.items():
            ratios[view][f].append(v)
        if a.sheet and j % max(1, len(recs) // 8) == 0:
            cs, _ = cv2.findContours(body, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            cv2.drawContours(img, cs, -1, (255, 255, 0), 2)
            tiles.append(cv2.resize(img, (int(img.shape[1] * 420 / img.shape[0]), 420)))
    print(f'{len(recs)} captures; body ÷ you, filled length of the silhouette row (median), hip joints 0 → shoulder joints 1:')
    for view, by in ratios.items():
        print(f'  {view:5}', '  '.join(f'{f:+.2f}: {np.median(v):.3f}' for f, v in by.items() if v))
    if tiles:
        cv2.imwrite(a.sheet, np.hstack(tiles), [cv2.IMWRITE_JPEG_QUALITY, 80])


if __name__ == '__main__':
    main()
