// The avatar's look: MakeHuman skin texture tinted to you, your face photo warped onto it,
// plus eye and hair colour read from the photo.
import { assets } from './body.js';

const SIZE = 2048; // skin texture size
const CHEEKS = [50, 280, 101, 330, 205, 425]; // MediaPipe landmarks on the cheeks
// Inner lip contour: triangles entirely on it would paint teeth / an open mouth onto the closed template lips.
const MOUTH = new Set([78, 191, 80, 81, 82, 13, 312, 311, 310, 415, 308, 324, 318, 402, 317, 14, 87, 178, 88, 95]);
export const loadImage = (src) => new Promise((ok, fail) => {
  const i = new Image();
  i.onload = () => ok(i);
  i.onerror = fail;
  i.src = src;
});
const canvas = (w, h = w) => Object.assign(document.createElement('canvas'), { width: w, height: h });
const hex = (h) => [1, 3, 5].map((i) => parseInt(h.slice(i, i + 2), 16));
const toHex = (rgb) => '#' + rgb.map((v) => Math.round(v).toString(16).padStart(2, '0')).join('');

// Mean colour of r×r patches around points.
function sample(ctx, pts, r = 8) {
  const sum = [0, 0, 0];
  let n = 0;
  for (const [x, y] of pts) {
    const d = ctx.getImageData(Math.round(x - r / 2), Math.round(y - r / 2), r, r).data;
    for (let i = 0; i < d.length; i += 4) { sum[0] += d[i]; sum[1] += d[i + 1]; sum[2] += d[i + 2]; n++; }
  }
  return sum.map((v) => v / n);
}

// Affine-map photo triangle s onto canvas triangle d (expanded ~1 px so neighbours overlap instead of seaming).
function warpTriangle(ctx, img, s, d) {
  const cx = (d[0][0] + d[1][0] + d[2][0]) / 3, cy = (d[0][1] + d[1][1] + d[2][1]) / 3;
  const grow = d.map(([x, y]) => { const l = Math.hypot(x - cx, y - cy) || 1; return [x + (x - cx) / l, y + (y - cy) / l]; });
  const [[x0, y0], [x1, y1], [x2, y2]] = s, [[u0, v0], [u1, v1], [u2, v2]] = d;
  const den = (x1 - x0) * (y2 - y0) - (x2 - x0) * (y1 - y0);
  if (!den) return;
  const a = ((u1 - u0) * (y2 - y0) - (u2 - u0) * (y1 - y0)) / den, b = ((u2 - u0) * (x1 - x0) - (u1 - u0) * (x2 - x0)) / den;
  const c = ((v1 - v0) * (y2 - y0) - (v2 - v0) * (y1 - y0)) / den, e = ((v2 - v0) * (x1 - x0) - (v1 - v0) * (x2 - x0)) / den;
  ctx.save();
  ctx.beginPath();
  grow.forEach(([x, y], i) => (i ? ctx.lineTo(x, y) : ctx.moveTo(x, y)));
  ctx.clip();
  ctx.setTransform(a, c, b, e, u0 - a * x0 - b * y0, v0 - c * x0 - e * y0);
  ctx.drawImage(img, 0, 0);
  ctx.restore();
}

// Skin texture canvas for `sex`: tinted to the photo's cheeks (or `tone`), with the photo's face baked in.
export async function skinCanvas(sex, { face, photo, tone }) {
  const A = assets(), anchors = A.arr(`face_uv.${sex}`);
  const at = (k) => [anchors[k * 2] * SIZE, (1 - anchors[k * 2 + 1]) * SIZE];
  const c = canvas(SIZE), ctx = c.getContext('2d', { willReadFrequently: true });
  ctx.drawImage(await loadImage(`/static/assets/skin_${sex}.jpg`), 0, 0, SIZE, SIZE);
  let pc = null, px = null;
  if (face) {
    pc = canvas(photo.naturalWidth, photo.naturalHeight).getContext('2d', { willReadFrequently: true });
    pc.drawImage(photo, 0, 0);
    px = (k) => [face.landmarks[k][0] * photo.naturalWidth, face.landmarks[k][1] * photo.naturalHeight];
  }
  const target = face ? sample(pc, CHEEKS.map(px)) : tone ? hex(tone) : null;
  if (target) { // multiply the whole skin so body and face meet without a seam
    const base = sample(ctx, CHEEKS.map(at)), k = target.map((v, i) => v / Math.max(base[i], 1));
    const im = ctx.getImageData(0, 0, SIZE, SIZE), d = im.data;
    for (let i = 0; i < d.length; i += 4) { d[i] *= k[0]; d[i + 1] *= k[1]; d[i + 2] *= k[2]; }
    ctx.putImageData(im, 0, 0);
  }
  if (face) {
    const layer = canvas(SIZE), lc = layer.getContext('2d'), tris = A.arr('face.tris');
    for (let t = 0; t < tris.length; t += 3) {
      const ks = [tris[t], tris[t + 1], tris[t + 2]];
      if (!ks.some((k) => Number.isNaN(anchors[k * 2])) && !ks.every((k) => MOUTH.has(k))) warpTriangle(lc, photo, ks.map(px), ks.map(at));
    }
    // Feathered mask: the face outline pulled 10 % toward its centre, blurred.
    const oval = A.meta.face_oval.filter((k) => !Number.isNaN(anchors[k * 2])).map(at);
    const mx = oval.reduce((s, p) => s + p[0], 0) / oval.length, my = oval.reduce((s, p) => s + p[1], 0) / oval.length;
    const mask = canvas(SIZE), mc = mask.getContext('2d');
    mc.filter = 'blur(18px)';
    mc.beginPath();
    oval.forEach(([x, y], i) => mc[i ? 'lineTo' : 'moveTo'](mx + (x - mx) * 0.9, my + (y - my) * 0.9));
    mc.fill();
    lc.globalCompositeOperation = 'destination-in';
    lc.drawImage(mask, 0, 0);
    ctx.drawImage(layer, 0, 0);
  }
  return c;
}

// Eye colour (nearest MakeHuman iris) and hair colour (just above the hairline) from the photo.
export function photoColors(photo, face) {
  const pc = canvas(photo.naturalWidth, photo.naturalHeight).getContext('2d', { willReadFrequently: true });
  pc.drawImage(photo, 0, 0);
  const P = (k) => [face.landmarks[k][0] * photo.naturalWidth, face.landmarks[k][1] * photo.naturalHeight];
  const iris = [468, 473].flatMap((c) => {
    const [x, y] = P(c), r = [1, 2, 3, 4].reduce((s, j) => s + Math.hypot(P(c + j)[0] - x, P(c + j)[1] - y), 0) / 4;
    return [[x + r * 0.6, y], [x - r * 0.6, y], [x, y + r * 0.6]];
  });
  const eyeRGB = sample(pc, iris, 3), eyes = assets().meta.eyes;
  const eye = Object.keys(eyes).reduce((a, b) => (dist(eyes[b], eyeRGB) < dist(eyes[a], eyeRGB) ? b : a));
  // Just above the hairline (10, 109, 338 = top of forehead; 152 = chin), median of three spots.
  const [cx, cy] = P(152), spots = [10, 109, 338].map((k) => { const [x, y] = P(k); return sample(pc, [[x + (x - cx) * 0.07, y + (y - cy) * 0.07]], 10); });
  const hairRGB = [0, 1, 2].map((c) => spots.map((s) => s[c]).sort((a, b) => a - b)[1]), skinRGB = sample(pc, CHEEKS.map(P));
  return { eye, hair: dist(hairRGB, skinRGB) > 60 ? toHex(hairRGB) : null }; // null: looks like skin (bald / high hairline)
}
const dist = (a, b) => Math.hypot(a[0] - b[0], a[1] - b[1], a[2] - b[2]);
