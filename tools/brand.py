"""Brand images: the template body's silhouette in BodyLab cyan, circled by a scan ring, on the app's navy.
Writes the iPhone app icon and launch mark, and the web icon. Run: uv run python tools/brand.py"""
import json
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).parent.parent
XC = ROOT / 'ios/BodyLabSync/Assets.xcassets'
CYAN, BLUE, NAVY = np.array([255, 240, 62.]), np.array([255, 150, 30.]), (20, 11, 7)  # BGR: #3ef0ff, #1e96ff, #070b14


def silhouette(size):
    """Front view of the male template body, filling ~70 % of a size×size square: float mask 0..1."""
    meta = json.loads((ROOT / 'static/assets/body.json').read_text())
    buf = (ROOT / 'static/assets/body.bin').read_bytes()
    arr = lambda k: np.frombuffer(buf, {'uint16': np.uint16, 'float32': np.float32}[meta['arrays'][k][0]],  # noqa: E731
                                  meta['arrays'][k][2], meta['arrays'][k][1])
    v = arr('male').reshape(-1, 3)
    tris = arr('body.src')[arr('body.tris').reshape(-1, 3)]
    scale = .7 * size / np.ptp(v[np.unique(tris), 1])
    xy = np.stack([size / 2 + v[:, 0] * scale, size * .86 - v[:, 1] * scale], 1)
    mask = np.zeros((size * 4, size * 4), np.uint8)  # 4× supersampled
    for t in tris:
        cv2.fillConvexPoly(mask, np.round(xy[t] * 16).astype(np.int32), 255, cv2.LINE_8, shift=2)
    return cv2.resize(mask, (size, size), interpolation=cv2.INTER_AREA) / 255


def mark(size):
    """(colour BGR float, alpha) of the figure, its glow and the scan ring, without background."""
    body = silhouette(size)
    grad = np.linspace(0, 1, size)[:, None, None]
    color = np.zeros((size, size, 3))
    alpha = np.zeros((size, size))

    def over(c, a):  # composite c (H,W,3 or 3) with alpha a over what's there
        nonlocal color, alpha
        color = c * a[..., None] + color * (1 - a[..., None])
        alpha = a + alpha * (1 - a)
    ring = [np.zeros((size, size)) for _ in range(2)]  # back half, front half
    centre, axes, w = (size // 2, int(size * .55)), (int(size * .36), int(size * .085)), max(2, size // 70)
    cv2.ellipse(ring[0], centre, axes, 0, 180, 360, 1, w, cv2.LINE_AA)
    cv2.ellipse(ring[1], centre, axes, 0, 0, 180, 1, w, cv2.LINE_AA)
    glow = lambda m, r: np.clip(cv2.GaussianBlur(m, (0, 0), size * r), 0, 1)  # noqa: E731
    over(CYAN, glow(ring[0], .02) * .5)
    over(CYAN, ring[0] * .55)
    over(CYAN, glow(body, .03) * .55)
    over(CYAN * (1 - grad) + BLUE * grad, body * .97)
    over(CYAN, glow(ring[1], .02) * .7)
    over(np.array([255, 252, 200.]), ring[1])
    return color, alpha


def icon(size):
    yy, xx = np.mgrid[:size, :size] / size
    r = np.hypot(xx - .5, yy - .42) / .75
    bg = np.array([63, 35, 19.]) * (1 - r[..., None]).clip(0, 1) + np.array(NAVY, float) * r[..., None].clip(0, 1)  # #13233f centre
    c, a = mark(size)
    return (c * a[..., None] + bg * (1 - a[..., None])).clip(0, 255).astype(np.uint8)


def write_set(folder, files, contents):
    folder.mkdir(parents=True, exist_ok=True)
    for name, img in files.items():
        cv2.imwrite(str(folder / name), img)
    (folder / 'Contents.json').write_text(json.dumps(contents | {'info': {'author': 'xcode', 'version': 1}}, indent=2))


if __name__ == '__main__':
    write_set(XC / 'AppIcon.appiconset', {'icon.png': icon(1024)},
              {'images': [{'filename': 'icon.png', 'idiom': 'universal', 'platform': 'ios', 'size': '1024x1024'}]})
    c, a = mark(600)
    write_set(XC / 'LaunchMark.imageset', {'mark.png': np.dstack([c, a * 255]).clip(0, 255).astype(np.uint8)},
              {'images': [{'filename': 'mark.png', 'idiom': 'universal', 'scale': '3x'}]})
    write_set(XC / 'LaunchBackground.colorset', {}, {'colors': [{'idiom': 'universal', 'color': {'color-space': 'srgb', 'components': {
        'red': f'{NAVY[2] / 255:.3f}', 'green': f'{NAVY[1] / 255:.3f}', 'blue': f'{NAVY[0] / 255:.3f}', 'alpha': '1.000'}}}]})
    (XC / 'Contents.json').write_text(json.dumps({'info': {'author': 'xcode', 'version': 1}}, indent=2))
    cv2.imwrite(str(ROOT / 'static/assets/icon.png'), cv2.resize(icon(1024), (180, 180), interpolation=cv2.INTER_AREA))
    print('brand images written')
