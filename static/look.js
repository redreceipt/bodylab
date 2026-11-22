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
// Separable box blur, 3 passes ≈ Gaussian. buf: Float32Array of w*h*ch.
function blur(buf, w, h, ch, r) {
  r = Math.max(1, Math.round(r));
  let a = buf, b = new Float32Array(buf.length);
  for (let pass = 0; pass < 6; pass++) {
    const horiz = pass % 2 === 0, n = horiz ? w : h, m = horiz ? h : w;
    for (let j = 0; j < m; j++) for (let c = 0; c < ch; c++) {
      const at = (i) => ((horiz ? j * w + i : i * w + j) * ch + c);
      let sum = 0;
      for (let i = -r; i <= r; i++) sum += a[at(Math.min(n - 1, Math.max(0, i)))];
      for (let i = 0; i < n; i++) {
        b[at(i)] = sum / (2 * r + 1);
        sum += a[at(Math.min(n - 1, i + r + 1))] - a[at(Math.max(0, i - r))];
      }
    }
    [a, b] = [b, a];
  }
  return a;
}

// Per-channel median of the pixels where alpha > 0.
function median(d) {
  const hist = [new Uint32Array(256), new Uint32Array(256), new Uint32Array(256)];
  let n = 0;
  for (let i = 0; i < d.length; i += 4) if (d[i + 3] > 200) { hist[0][d[i]]++; hist[1][d[i + 1]]++; hist[2][d[i + 2]]++; n++; }
  return hist.map((h) => { let acc = 0, v = 0; while (v < 255 && (acc += h[v]) < n / 2) v++; return v; });
}

// Skin texture canvas for `sex`, tinted to your skin and with your face photo baked in.
// The photo keeps only its fine detail (features, pores, brows); broad colour and shading come from the
// skin texture, so its studio lighting doesn't make the face look pasted on. Scalp under hair takes the hair colour.
export async function skinCanvas(sex, { face, photo, tone, hairColor, baseUrl }) {
  const A = assets(), anchors = A.arr(`face_uv.${sex}`);
  const at = (k) => [anchors[k * 2] * SIZE, (1 - anchors[k * 2 + 1]) * SIZE];
  const c = canvas(SIZE), ctx = c.getContext('2d', { willReadFrequently: true });
  ctx.drawImage(await loadImage(baseUrl ?? `/static/assets/skin_${sex}.jpg`), 0, 0, SIZE, SIZE); // baseUrl: your skin, baked from the cameras
  const oval = A.meta.face_oval.filter((k) => !Number.isNaN(anchors[k * 2])).map(at);
  const mx = oval.reduce((s, p) => s + p[0], 0) / oval.length, my = oval.reduce((s, p) => s + p[1], 0) / oval.length;
  const ovalPath = (g, k, dx = 0, dy = 0) => {
    g.beginPath();
    oval.forEach(([x, y], i) => g[i ? 'lineTo' : 'moveTo'](mx + (x - mx) * k - dx, my + (y - my) * k - dy));
    g.fill();
  };
  // Face region box (texture px), padded
  const xs = oval.map((p) => p[0]), ys = oval.map((p) => p[1]), pad = 40;
  const bx = Math.max(0, Math.floor(Math.min(...xs) - pad)), by = Math.max(0, Math.floor(Math.min(...ys) - pad));
  const bw = Math.min(SIZE, Math.ceil(Math.max(...xs) + pad)) - bx, bh = Math.min(SIZE, Math.ceil(Math.max(...ys) + pad)) - by;

  let layer = null;
  if (face) {
    const px = (k) => [face.landmarks[k][0] * photo.naturalWidth, face.landmarks[k][1] * photo.naturalHeight];
    layer = canvas(SIZE);
    const lc = layer.getContext('2d', { willReadFrequently: true }), tris = A.arr('face.tris');
    for (let t = 0; t < tris.length; t += 3) {
      const ks = [tris[t], tris[t + 1], tris[t + 2]];
      if (!ks.some((k) => Number.isNaN(anchors[k * 2])) && !ks.every((k) => MOUTH.has(k))) warpTriangle(lc, photo, ks.map(px), ks.map(at));
    }
  }
  // Tint the whole skin to your tone: median of the photo's face (or the chosen tone) vs the texture's face.
  const inner = canvas(bw, bh), ic = inner.getContext('2d', { willReadFrequently: true });
  ovalPath(ic, 0.75, bx, by);
  ic.globalCompositeOperation = 'source-in';
  ic.drawImage(c, bx, by, bw, bh, 0, 0, bw, bh);
  const base = median(ic.getImageData(0, 0, bw, bh).data);
  let target = tone ? hex(tone) : null;
  if (face) {
    ic.clearRect(0, 0, bw, bh);
    ic.globalCompositeOperation = 'source-over';
    ovalPath(ic, 0.75, bx, by);
    ic.globalCompositeOperation = 'source-in';
    ic.drawImage(layer, bx, by, bw, bh, 0, 0, bw, bh);
    target = median(ic.getImageData(0, 0, bw, bh).data);
  }
  if (target) {
    const k = target.map((v, i) => v / Math.max(base[i], 1));
    const im = ctx.getImageData(0, 0, SIZE, SIZE), d = im.data;
    for (let i = 0; i < d.length; i += 4) { d[i] *= k[0]; d[i + 1] *= k[1]; d[i + 2] *= k[2]; }
    ctx.putImageData(im, 0, 0);
  }
  if (face) {
    // Low-pass both at quarter resolution; photo out = photo × blur(skin) / blur(photo), feathered into the skin.
    const q = 0.25, sw = Math.ceil(bw * q), sh = Math.ceil(bh * q), r = 0.08 * bw * q;
    const small = (src, k = null) => {
      const g = canvas(sw, sh).getContext('2d', { willReadFrequently: true });
      if (k) { g.fillStyle = '#fff'; ovalPath(g, k, bx, by); g.globalCompositeOperation = 'source-in'; }
      if (src) g.drawImage(src, bx, by, bw, bh, 0, 0, sw, sh);
      return g.getImageData(0, 0, sw, sh).data;
    };
    const lowT = blur(Float32Array.from(small(c)), sw, sh, 4, r);
    const pRaw = small(layer), pre = new Float32Array(pRaw.length);
    for (let i = 0; i < pRaw.length; i += 4) {
      const a = pRaw[i + 3] / 255;
      pre[i] = pRaw[i] * a; pre[i + 1] = pRaw[i + 1] * a; pre[i + 2] = pRaw[i + 2] * a; pre[i + 3] = a;
    }
    const lowP = blur(pre, sw, sh, 4, r); // premultiplied: dividing by blurred alpha normalises at the edges
    const mg = canvas(sw, sh).getContext('2d', { willReadFrequently: true });
    mg.scale(q, q);
    mg.fillStyle = '#fff';
    ovalPath(mg, 0.92, bx, by);
    const mraw = mg.getImageData(0, 0, sw, sh).data, mask = new Float32Array(sw * sh);
    for (let i = 0; i < mask.length; i++) mask[i] = mraw[i * 4 + 3] / 255;
    const soft = blur(mask, sw, sh, 1, 0.04 * bw * q);
    const sampleAt = (buf, ch, x, y, k) => { // bilinear
      const fx = Math.min(sw - 1.001, Math.max(0, x * q - 0.5)), fy = Math.min(sh - 1.001, Math.max(0, y * q - 0.5));
      const x0 = fx | 0, y0 = fy | 0, tx = fx - x0, ty = fy - y0, i = (y0 * sw + x0) * ch + k;
      return (buf[i] * (1 - tx) + buf[i + ch] * tx) * (1 - ty) + (buf[i + sw * ch] * (1 - tx) + buf[i + sw * ch + ch] * tx) * ty;
    };
    const T = ctx.getImageData(bx, by, bw, bh), P = layer.getContext('2d').getImageData(bx, by, bw, bh), t = T.data, p = P.data;
    for (let y = 0; y < bh; y++) for (let x = 0; x < bw; x++) {
      const i = (y * bw + x) * 4, alpha = (p[i + 3] / 255) * sampleAt(soft, 1, x, y, 0);
      if (alpha <= 0) continue;
      const pa = Math.max(sampleAt(lowP, 4, x, y, 3), 1e-3);
      for (let k = 0; k < 3; k++) {
        const ratio = Math.min(2.5, Math.max(0.4, sampleAt(lowT, 4, x, y, k) / Math.max(sampleAt(lowP, 4, x, y, k) / pa, 1)));
        t[i + k] += (p[i + k] * ratio - t[i + k]) * alpha;
      }
    }
    ctx.putImageData(T, bx, by);
  }
  if (hairColor) { // hair-coloured scalp so gaps between hair strands don't show bare skin
    const sc = canvas(SIZE), sg = sc.getContext('2d'), scalp = A[sex].scalp, uv = (r) => [A.uv[r * 2] * SIZE, (1 - A.uv[r * 2 + 1]) * SIZE];
    sg.fillStyle = sg.strokeStyle = hairColor;
    for (let t = 0; t < A.tris.length; t += 3) {
      const rs = [A.tris[t], A.tris[t + 1], A.tris[t + 2]];
      if (!rs.every((r) => scalp[A.src[r]])) continue;
      sg.beginPath();
      rs.map(uv).forEach(([x, y], i) => sg[i ? 'lineTo' : 'moveTo'](x, y));
      sg.closePath();
      sg.fill();
      sg.stroke(); // closes hairline-thin cracks between triangles
    }
    ctx.globalAlpha = 0.8;
    ctx.drawImage(sc, 0, 0);
    ctx.globalAlpha = 1;
  }
  return { canvas: c, tone: target ?? base };
}

// Eye colour from the photo: the nearest MakeHuman iris. (Hair is analysed on the server.)
export function photoColors(photo, face) {
  const pc = canvas(photo.naturalWidth, photo.naturalHeight).getContext('2d', { willReadFrequently: true });
  pc.drawImage(photo, 0, 0);
  const P = (k) => [face.landmarks[k][0] * photo.naturalWidth, face.landmarks[k][1] * photo.naturalHeight];
  const iris = [468, 473].flatMap((c) => {
    const [x, y] = P(c), r = [1, 2, 3, 4].reduce((s, j) => s + Math.hypot(P(c + j)[0] - x, P(c + j)[1] - y), 0) / 4;
    return [[x + r * 0.6, y], [x - r * 0.6, y], [x, y + r * 0.6]];
  });
  const eyeRGB = sample(pc, iris, 3), eyes = assets().meta.eyes;
  return { eye: Object.keys(eyes).reduce((a, b) => (dist(eyes[b], eyeRGB) < dist(eyes[a], eyeRGB) ? b : a)) };
}
const dist = (a, b) => Math.hypot(a[0] - b[0], a[1] - b[1], a[2] - b[2]);
