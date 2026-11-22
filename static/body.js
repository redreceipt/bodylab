// Realistic body: MakeHuman's CC0 template (tools/build_assets.py) fitted to a scan's measurements.
// Torso: width/depth scaled per level (neck, shoulders, chest, waist, hips). Limbs: scaled radially around
// their bones. Heights: remapped to the scan's ankle/knee/hip/shoulder levels. Rig weights blend it all.
// Hair, brows, lashes and eyes are MakeHuman proxies anchored to base-mesh vertices, so they follow the fit.
// Pure math (no three.js / DOM) so test_body.mjs can run it in Node.

let T = null;
const REG = { head: 0, torso: 1, arm_l: 2, arm_r: 3, leg_l: 4, leg_r: 5 };
const TYPES = { float32: Float32Array, uint8: Uint8Array, uint16: Uint16Array, uint32: Uint32Array };
const lerp = (a, b, t) => a + (b - a) * t;
const clamp = (v, lo, hi) => Math.min(hi, Math.max(lo, v));
// Piecewise-linear lookup in [[x, ...values]] sorted by x; clamps outside.
function interp(knots, x) {
  if (x <= knots[0][0]) return knots[0].slice(1);
  for (let k = 1; k < knots.length; k++) {
    if (x <= knots[k][0]) {
      const t = (x - knots[k - 1][0]) / (knots[k][0] - knots[k - 1][0] || 1);
      return knots[k].slice(1).map((v, j) => lerp(knots[k - 1][j + 1], v, t));
    }
  }
  return knots.at(-1).slice(1);
}

// meta/buf: static/assets/body.json + body.bin
export function initBody(meta, buf) {
  const arr = (name) => { const [type, offset, size] = meta.arrays[name]; return new TYPES[type](buf, offset, size); };
  T = { meta, arr, n: meta.n, reg: arr('weights'), src: arr('body.src'), uv: arr('body.uv'), tris: arr('body.tris') };
  T.bodyN = T.src.reduce((a, b) => Math.max(a, b), 0) + 1; // body vertices come first; helpers after
  T.male = template(arr('male'), meta.joints.male, 'male');
  T.female = template(arr('female'), meta.joints.female, 'female');
  return T;
}

// Measure the template exactly the way measure.py measures a camera silhouette.
function template(pos, J, sex) {
  const { bodyN: n, reg } = T, w = (i, r) => reg[i * 6 + r] / 255;
  const sh = (J.shoulder_l[1] + J.shoulder_r[1]) / 2, hip = (J.hip_l[1] + J.hip_r[1]) / 2, len = sh - hip;
  let H = 0;
  for (let i = 0; i < n; i++) H = Math.max(H, pos[i * 3 + 1]);
  const notArm = (i) => w(i, REG.arm_l) + w(i, REG.arm_r) < 0.5;
  const slab = (y, use = notArm) => {
    let x0 = Infinity, x1 = -Infinity, z0 = Infinity, z1 = -Infinity;
    for (let i = 0; i < n; i++) {
      if (Math.abs(pos[i * 3 + 1] - y) > 1.2 || !use(i)) continue;
      x0 = Math.min(x0, pos[i * 3]); x1 = Math.max(x1, pos[i * 3]);
      z0 = Math.min(z0, pos[i * 3 + 2]); z1 = Math.max(z1, pos[i * 3 + 2]);
    }
    return { y, f: (sh - y) / len, w: x1 - x0, d: z1 - z0, cz: (z0 + z1) / 2 };
  };
  const range = (a, b, k) => Array.from({ length: k }, (_, j) => lerp(a, b, j / (k - 1)));
  const pick = (ys, best) => ys.map((y) => slab(y)).reduce((a, b) => (best(b.w, a.w) ? b : a));
  const yAt = (f) => sh - f * len;
  const limbWidth = (A, B, ts, r) => Math.max(...ts.map((t) => {
    const ax = B[0] - A[0], ay = B[1] - A[1], L = Math.hypot(ax, ay), dx = ax / L, dy = ay / L;
    let lo = Infinity, hi = -Infinity;
    for (let i = 0; i < n; i++) {
      if (w(i, r) < 0.5) continue;
      const px = pos[i * 3] - A[0], py = pos[i * 3 + 1] - A[1];
      if (Math.abs(px * dx + py * dy - t * L) > 1.2) continue;
      const across = -px * dy + py * dx;
      lo = Math.min(lo, across); hi = Math.max(hi, across);
    }
    return hi - lo;
  }));
  const t = {
    pos, J, H, sh, hip, len,
    neck: pick(range(sh + 0.2 * (J.eye_l[1] - sh), sh + 0.55 * (J.eye_l[1] - sh), 8), (a, b) => a < b),
    shoulders: slab(sh, () => true),
    chest: slab(yAt(0.3)),
    waist: sex === 'female' ? pick(range(yAt(0.5), yAt(0.8), 7), (a, b) => a < b) : slab(yAt(0.69)), // men: the navel
    underbust: slab(yAt(0.42)),
    hips: pick(range(yAt(0.95), yAt(1.1), 6), (a, b) => a > b),
  };
  for (const s of ['l', 'r']) {
    t[`arm_${s}`] = limbWidth(J[`shoulder_${s}`], J[`elbow_${s}`], [0.5], REG[`arm_${s}`]);
    t[`thigh_${s}`] = limbWidth(J[`hip_${s}`], J[`knee_${s}`], [0.4], REG[`leg_${s}`]);
    t[`calf_${s}`] = limbWidth(J[`knee_${s}`], J[`ankle_${s}`], range(0.2, 0.45, 6), REG[`leg_${s}`]);
  }
  // Per-vertex labels for picking/highlighting.
  t.parts = new Array(n);
  t.scalp = new Uint8Array(n); // head skin a haircut covers: above the forehead, and the back of the head to the nape
  for (let i = 0; i < n; i++) {
    let r = 0;
    for (let k = 1; k < 6; k++) if (reg[i * 6 + k] > reg[i * 6 + r]) r = k;
    const y = pos[i * 3 + 1], f = (sh - y) / len, side = r % 2 ? 'r' : 'l';
    t.parts[i] = r === REG.head ? 'overall' : r === REG.torso
      ? (f < t.neck.f / 2 ? 'neck' : f < 0.15 ? 'shoulders' : f < 0.5 ? 'chest' : f < 0.87 ? 'waist' : 'hips')
      : r <= REG.arm_r ? `arm_${side}` : (y > J[`knee_${side}`][1] ? `thigh_${side}` : `calf_${side}`);
    const eyeY = J.eye_l[1];
    t.scalp[i] = r === REG.head && (y > eyeY + 0.045 * H || (pos[i * 3 + 2] < J.head[2] - 3 && y > eyeY - 0.035 * H)) ? 1 : 0;
  }
  return t;
}

// Breast and genital morph weights from sizes: cup = bust minus underbust (cm), penis length/girth (cm),
// firmness and testicles -1..1. Lengths are converted to the template's scale first.
function morphWeights(sex, shape, s) {
  const w = {}, set = (k, v) => { if (v > 0) w[`${k}+`] = Math.min(v, 1.5); else if (v < 0) w[`${k}-`] = Math.min(-v, 1.5); };
  if (sex === 'female') {
    // shortcut: MakeHuman's "average cup" taken as ~2.5 in of difference, min ~0 in, max ~6 in; refine against tape fits
    if (shape.cup != null) { const inch = shape.cup / 2.54; set('cup', inch >= 2.5 ? (inch - 2.5) / 3.5 : (inch - 2.5) / 2.5); }
    set('firm', shape.firmness ?? 0);
  } else {
    const g = T.meta.genital, toward = (v, base, up, down) => (v >= base ? (v - base) / (up - base) : (v - base) / (base - down));
    if (shape.length) set('length', toward(shape.length / s, g.base[0], g['length+'][0], g['length-'][0]));
    if (shape.girth) set('girth', toward(shape.girth / s, g.base[1], g['girth+'][1], g['girth-'][1]));
    set('testicles', shape.testicles ?? 0);
  }
  return w;
}

// dims: scan sizes in cm (from /api/state); shape: see morphWeights. Returns fitted positions of every
// base-mesh vertex for height H. fitted: the day's body-shape fit (same vertices, cm), used as is.
export function fitBody(dims, H, sex, shape = {}, fitted = null) {
  const t = T[sex === 'female' ? 'female' : 'male'], d = dims || {}, s = H / t.H, n = T.n;
  const ratio = (scan, tmpl) => (scan ? clamp(scan / (tmpl * s), 0.6, 1.8) : 1);
  if (fitted) {  // the fit already shaped the bust; genital size still comes from tape
    const pos = Float32Array.from(fitted);
    for (const [name, wt] of Object.entries(morphWeights(sex, shape, s))) {
      if (/^(cup|firm)/.test(name)) continue;
      const idx = T.arr(`morph.${name}.idx`), dd = T.arr(`morph.${name}.d`);
      for (let j = 0; j < idx.length; j++) for (let k = 0; k < 3; k++) pos[idx[j] * 3 + k] += wt * s * dd[j * 3 + k];
    }
    return { pos, parts: t.parts };
  }
  let pos = t.pos;
  const morphs = Object.entries(morphWeights(sex, shape, s));
  if (morphs.length) {
    pos = Float32Array.from(t.pos);
    for (const [name, wt] of morphs) {
      const idx = T.arr(`morph.${name}.idx`), dd = T.arr(`morph.${name}.d`);
      for (let j = 0; j < idx.length; j++) for (let k = 0; k < 3; k++) pos[idx[j] * 3 + k] += wt * dd[j * 3 + k];
    }
  }
  // torso knots by f (0 = shoulders, 1 = hip joints): [f, x scale, z scale, z centre]. With breasts, the fit
  // follows the ribcage under them (underbust) and the cup morph supplies the bust.
  const chest = sex === 'female' && d.underbust_w
    ? [t.underbust.f, ratio(d.underbust_w, t.underbust.w), ratio(d.underbust_d, t.underbust.d), t.underbust.cz]
    : [0.3, ratio(d.chest_w, t.chest.w), ratio(d.chest_d, t.chest.d), t.chest.cz];
  const torso = [
    [t.neck.f, ratio(d.neck_w, t.neck.w), ratio(d.neck_d, t.neck.d), t.neck.cz],
    [0, ratio(d.shoulders_w, t.shoulders.w), chest[2], t.shoulders.cz],
    chest,
    [t.waist.f, ratio(d.waist_w, t.waist.w), ratio(d.waist_d, t.waist.d), t.waist.cz],
    [t.hips.f, ratio(d.hips_w, t.hips.w), ratio(d.hips_d, t.hips.d), t.hips.cz],
  ];
  const J = (k) => t.J[k].map((v) => v * s);
  const limbs = ['l', 'r'].flatMap((side) => {
    const rArm = ratio(d[`arm_${side}_w`], t[`arm_${side}`]), rFore = 1 + (rArm - 1) * 0.5;
    const rThigh = ratio(d[`thigh_${side}_w`], t[`thigh_${side}`]), rCalf = ratio(d[`calf_${side}_w`], t[`calf_${side}`]);
    const shift = (torso[1][1] - 1) * t.J[`shoulder_${side}`][0] * s; // arms ride out with the shoulders
    return [
      { r: REG[`arm_${side}`], joints: [J(`shoulder_${side}`), J(`elbow_${side}`), J(`wrist_${side}`)], shift,
        knots: [[0, rArm], [0.8, rArm], [1, (rArm + rFore) / 2], [1.3, rFore], [2, 1 + (rFore - 1) * 0.3], [2.2, 1]] },
      { r: REG[`leg_${side}`], joints: [J(`hip_${side}`), J(`knee_${side}`), J(`ankle_${side}`)], shift: 0,
        knots: [[0, rThigh], [0.7, rThigh], [1, (rThigh + rCalf) / 2], [1.3, rCalf], [1.6, rCalf], [2, 1 + (rCalf - 1) * 0.3], [2.2, 1]] },
    ];
  });
  // Vertical proportions: follow the scan's landmark heights, each segment within ±15 % of the template's.
  const tmplY = [0, t.J.ankle_l[1], t.J.knee_l[1], t.hip, t.sh, t.H].map((v) => v * s);
  const scanY = [0, d.lv_ankle, d.lv_knee, d.lv_hip, d.lv_shoulder, H];
  const outY = [0];
  for (let k = 1; k < tmplY.length; k++) {
    const seg = tmplY[k] - tmplY[k - 1], want = scanY[k] != null && scanY[k - 1] != null ? (scanY[k] - scanY[k - 1]) / seg : 1;
    outY.push(outY[k - 1] + seg * clamp(want, 0.85, 1.15));
  }
  const remap = [[-100, -100], ...tmplY.map((y, k) => [y, outY[k] * (H / outY.at(-1))]), [t.H * s + 100, H + 100]];

  const out = new Float32Array(n * 3);
  for (let i = 0; i < n; i++) {
    const x = pos[i * 3] * s, y = pos[i * 3 + 1] * s, z = pos[i * 3 + 2] * s;
    const wt = (r) => T.reg[i * 6 + r] / 255;
    let X = 0, Y = 0, Z = 0;
    const add = (wgt, px, py, pz) => { X += wgt * px; Y += wgt * py; Z += wgt * pz; };
    add(wt(REG.head), x, y, z);
    if (wt(REG.torso)) {
      const [rx, rz, cz] = interp(torso, (t.sh - pos[i * 3 + 1]) / t.len);
      add(wt(REG.torso), x * rx, y, cz * s + (z - cz * s) * rz);
    }
    for (const L of limbs) {
      const wgt = wt(L.r);
      if (!wgt) continue;
      // project onto upper then lower bone; u = 0 at the root joint, 1 at elbow/knee, 2 at wrist/ankle
      let [A, B] = L.joints, u0 = 0;
      let ax = B[0] - A[0], ay = B[1] - A[1], az = B[2] - A[2], len = Math.hypot(ax, ay, az);
      let tt = ((x - A[0]) * ax + (y - A[1]) * ay + (z - A[2]) * az) / (len * len);
      if (tt > 1) {
        [A, B] = [L.joints[1], L.joints[2]]; u0 = 1;
        ax = B[0] - A[0]; ay = B[1] - A[1]; az = B[2] - A[2]; len = Math.hypot(ax, ay, az);
        tt = ((x - A[0]) * ax + (y - A[1]) * ay + (z - A[2]) * az) / (len * len);
      }
      const qx = A[0] + ax * tt, qy = A[1] + ay * tt, qz = A[2] + az * tt, [r] = interp(L.knots, u0 + tt);
      add(wgt, qx + (x - qx) * r + L.shift, qy + (y - qy) * r, qz + (z - qz) * r);
    }
    let sum = 0; // byte-quantised weights don't sum to exactly 1
    for (let r = 0; r < 6; r++) sum += T.reg[i * 6 + r];
    sum /= 255;
    out[i * 3] = X / sum; out[i * 3 + 1] = interp(remap, Y / sum)[0]; out[i * 3 + 2] = Z / sum;
  }
  return { pos: out, parts: t.parts };
}

// MakeHuman proxy (hair, brows, lashes, eyes): each vertex = weighted 3 base vertices + offset scaled by body size.
export function fitProxy(name, pos) {
  const refs = T.arr(`${name}.refs`), w = T.arr(`${name}.w`), d = T.arr(`${name}.d`);
  const scale = T.meta.proxies[name].scale.map(([a, b, den], k) => Math.abs(pos[a * 3 + k] - pos[b * 3 + k]) / den);
  const out = new Float32Array(refs.length);
  for (let j = 0; j < refs.length; j += 3) {
    for (let k = 0; k < 3; k++) {
      out[j + k] = w[j] * pos[refs[j] * 3 + k] + w[j + 1] * pos[refs[j + 1] * 3 + k] + w[j + 2] * pos[refs[j + 2] * 3 + k] + d[j + k] * scale[k];
    }
  }
  return out;
}

// Render mesh: vertices split at UV seams (src -> original vertex), with normals averaged over the welded
// original vertices so seams don't show in the shading.
export function mesh(pos, src, tris) {
  const normal = new Float32Array(pos.length);
  for (let t = 0; t < tris.length; t += 3) {
    const [a, b, c] = [src[tris[t]] * 3, src[tris[t + 1]] * 3, src[tris[t + 2]] * 3];
    const ux = pos[b] - pos[a], uy = pos[b + 1] - pos[a + 1], uz = pos[b + 2] - pos[a + 2];
    const vx = pos[c] - pos[a], vy = pos[c + 1] - pos[a + 1], vz = pos[c + 2] - pos[a + 2];
    const nx = uy * vz - uz * vy, ny = uz * vx - ux * vz, nz = ux * vy - uy * vx;
    for (const i of [a, b, c]) { normal[i] += nx; normal[i + 1] += ny; normal[i + 2] += nz; }
  }
  const P = new Float32Array(src.length * 3), N = new Float32Array(src.length * 3);
  for (let r = 0; r < src.length; r++) {
    const i = src[r] * 3, l = Math.hypot(normal[i], normal[i + 1], normal[i + 2]) || 1;
    for (let k = 0; k < 3; k++) { P[r * 3 + k] = pos[i + k]; N[r * 3 + k] = normal[i + k] / l; }
  }
  return { position: P, normal: N };
}

// Clothes hang instead of being painted on. Shorts stand off the skin and flare toward the hem (less on the inner
// thigh, so the legs don't collide); a top falls straight from the bust instead of tucking under it.
export function drape(name, pos, g) {
  const out = Float32Array.from(g), at = (k) => { const ids = T.arr(`joint.${k}`), c = [0, 0, 0];
    for (const i of ids) for (let k2 = 0; k2 < 3; k2++) c[k2] += pos[i * 3 + k2] / ids.length; return c; };
  if (name === 'shorts') {
    const leg = T.arr('shorts.leg'), axes = { 1: [at('hip_l'), at('knee_l')], [-1]: [at('hip_r'), at('knee_r')] };
    for (let i = 0; i < out.length / 3; i++) {
      const p = [out[i * 3], out[i * 3 + 1], out[i * 3 + 2]], side = p[0] >= 0 ? 1 : -1, [A, B] = axes[side];
      const ax = B.map((v, k) => v - A[k]), L2 = ax.reduce((s2, v) => s2 + v * v, 0);
      const t = clamp(((p[0] - A[0]) * ax[0] + (p[1] - A[1]) * ax[1] + (p[2] - A[2]) * ax[2]) / L2, 0, 1);
      const r = p.map((v, k) => v - (A[k] + ax[k] * t)), len = Math.hypot(...r) || 1;
      const u = clamp(leg[i], 0, 1), inner = r[0] * side < 0 ? 0.35 : 1;  // toward the other leg: barely
      const extra = (0.5 + 2.8 * u * u) * inner;  // cm: a little ease at the waist, ~3 cm of flare at the hem
      for (let k = 0; k < 3; k++) out[i * 3 + k] += (r[k] / len) * extra;
    }
  } else if (name === 'bra') {
    // Fabric bridges hollows instead of clinging: the cleavage and the fold under the bust are spanned while the band
    // stays snug. Each front vertex rises to its neighbours' average but never sinks into the skin.
    const src = T.arr('bra.src'), tris = T.arr('bra.tris'), nb = new Map(), link = (i, j) => (nb.get(i) ?? nb.set(i, new Set()).get(i)).add(j);
    for (let t = 0; t < tris.length; t += 3) for (let a = 0; a < 3; a++) {
      const i = src[tris[t + a]], j = src[tris[t + (a + 1) % 3]];
      if (i !== j) { link(i, j); link(j, i); }
    }
    let hem = Infinity;
    for (const i of nb.keys()) hem = Math.min(hem, g[i * 3 + 1]);
    const free = [...nb.keys()].filter((i) => g[i * 3 + 2] > 0 && g[i * 3 + 1] > hem + 2);
    for (let it = 0; it < 300; it++) for (const i of free) {
      let z = 0;
      for (const j of nb.get(i)) z += out[j * 3 + 2];
      out[i * 3 + 2] = Math.max(g[i * 3 + 2], z / nb.get(i).size);
    }
  }
  return out;
}

// Smoothed genital mesh: each vertex is a fixed mix of helper vertices (Catmull-Clark weights from the build).
export function genitalPositions(pos) {
  const ptr = T.arr('genital.ptr'), idx = T.arr('genital.idx'), w = T.arr('genital.w'), out = new Float32Array((ptr.length - 1) * 3);
  for (let r = 0; r < ptr.length - 1; r++) for (let j = ptr[r]; j < ptr[r + 1]; j++) for (let k = 0; k < 3; k++) out[r * 3 + k] += w[j] * pos[idx[j] * 3 + k];
  return out;
}

export const assets = () => T;
