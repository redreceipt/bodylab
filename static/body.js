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

// Breast morph weights: cup = bust minus underbust (cm), firmness -1..1.
function morphWeights(sex, shape) {
  const w = {}, set = (k, v) => { if (v > 0) w[`${k}+`] = Math.min(v, 1.5); else if (v < 0) w[`${k}-`] = Math.min(-v, 1.5); };
  if (sex === 'female') {
    // shortcut: MakeHuman's "average cup" taken as ~2.5 in of difference, min ~0 in, max ~6 in; refine against tape fits
    if (shape.cup != null) { const inch = shape.cup / 2.54; set('cup', inch >= 2.5 ? (inch - 2.5) / 3.5 : (inch - 2.5) / 2.5); }
    set('firm', shape.firmness ?? 0);
  }
  return w;
}

// dims: scan sizes in cm (from /api/state); shape: see morphWeights. Returns fitted positions of every
// base-mesh vertex for height H. fitted: a day's body-shape fit (same vertices, cm) to start from instead of the
// template, with ref = the dims it was seen at: dims then reshape it by dims ÷ ref (a prediction), not by the template.
export function fitBody(dims, H, sex, shape = {}, fitted = null, ref = {}) {
  const t = T[sex === 'female' ? 'female' : 'male'], d = dims || {}, s = H / t.H, n = T.n;
  const ratio = (scan, tmpl) => (scan && tmpl ? clamp(scan / (tmpl * s), 0.6, 1.8) : 1);
  const size = (k, tmpl) => (fitted ? ref[k] / s : tmpl);
  let pos = fitted ? Float32Array.from(fitted, (v) => v / s) : t.pos;
  const morphs = Object.entries(morphWeights(sex, shape)).filter(([k]) => !(fitted && /^(cup|firm)/.test(k))); // fit has the bust
  if (morphs.length) {
    pos = Float32Array.from(pos);
    for (const [name, wt] of morphs) {
      const idx = T.arr(`morph.${name}.idx`), dd = T.arr(`morph.${name}.d`);
      for (let j = 0; j < idx.length; j++) for (let k = 0; k < 3; k++) pos[idx[j] * 3 + k] += wt * dd[j * 3 + k];
    }
  }
  // torso knots by f (0 = shoulders, 1 = hip joints): [f, x scale, z scale, z centre]. With breasts, the fit
  // follows the ribcage under them (underbust) and the cup morph supplies the bust.
  const r = (k, slab, wd) => ratio(d[`${k}_${wd}`], size(`${k}_${wd}`, slab[wd]));
  const chest = sex === 'female' && d.underbust_w
    ? [t.underbust.f, r('underbust', t.underbust, 'w'), r('underbust', t.underbust, 'd'), t.underbust.cz]
    : [0.3, r('chest', t.chest, 'w'), r('chest', t.chest, 'd'), t.chest.cz];
  const torso = [
    [t.neck.f, r('neck', t.neck, 'w'), r('neck', t.neck, 'd'), t.neck.cz],
    [0, r('shoulders', t.shoulders, 'w'), chest[2], t.shoulders.cz],
    chest,
    [t.waist.f, r('waist', t.waist, 'w'), r('waist', t.waist, 'd'), t.waist.cz],
    [t.hips.f, r('hips', t.hips, 'w'), r('hips', t.hips, 'd'), t.hips.cz],
  ];
  const J = (k) => t.J[k].map((v) => v * s);
  const limbs = ['l', 'r'].flatMap((side) => {
    const rArm = ratio(d[`arm_${side}_w`], size(`arm_${side}_w`, t[`arm_${side}`])), rFore = 1 + (rArm - 1) * 0.5;
    const rThigh = ratio(d[`thigh_${side}_w`], size(`thigh_${side}_w`, t[`thigh_${side}`]));
    const rCalf = ratio(d[`calf_${side}_w`], size(`calf_${side}_w`, t[`calf_${side}`]));
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
  const scanY = fitted ? tmplY : [0, d.lv_ankle, d.lv_knee, d.lv_hip, d.lv_shoulder, H]; // a fit has its own heights
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

// Dual-quaternion skinning: G = 3×4 rigid bone transforms (posed × rest⁻¹), skin = 8 (bone, weight) bytes per
// vertex. Unlike averaging matrices (linear blend), blending rotations keeps volume at joints: shoulders and
// elbows don't collapse when the arms come down from the rest pose's spread.
export function pose(pos, G, skin) {
  const B = G.length / 12, dq = new Float32Array(B * 8);
  for (let b = 0; b < B; b++) { // rotation matrix -> quaternion (w, x, y, z); translation -> dual part
    const m = (r, c) => G[b * 12 + r * 4 + c], tr = m(0, 0) + m(1, 1) + m(2, 2);
    let w, x, y, z;
    if (tr > 0) { const s = Math.sqrt(tr + 1) * 2; w = s / 4; x = (m(2, 1) - m(1, 2)) / s; y = (m(0, 2) - m(2, 0)) / s; z = (m(1, 0) - m(0, 1)) / s; }
    else if (m(0, 0) > m(1, 1) && m(0, 0) > m(2, 2)) { const s = Math.sqrt(1 + m(0, 0) - m(1, 1) - m(2, 2)) * 2; w = (m(2, 1) - m(1, 2)) / s; x = s / 4; y = (m(0, 1) + m(1, 0)) / s; z = (m(0, 2) + m(2, 0)) / s; }
    else if (m(1, 1) > m(2, 2)) { const s = Math.sqrt(1 + m(1, 1) - m(0, 0) - m(2, 2)) * 2; w = (m(0, 2) - m(2, 0)) / s; x = (m(0, 1) + m(1, 0)) / s; y = s / 4; z = (m(1, 2) + m(2, 1)) / s; }
    else { const s = Math.sqrt(1 + m(2, 2) - m(0, 0) - m(1, 1)) * 2; w = (m(1, 0) - m(0, 1)) / s; x = (m(0, 2) + m(2, 0)) / s; y = (m(1, 2) + m(2, 1)) / s; z = s / 4; }
    const tx = m(0, 3), ty = m(1, 3), tz = m(2, 3);
    dq.set([w, x, y, z, -0.5 * (tx * x + ty * y + tz * z), 0.5 * (tx * w + ty * z - tz * y), 0.5 * (-tx * z + ty * w + tz * x), 0.5 * (tx * y - ty * x + tz * w)], b * 8);
  }
  const out = Float32Array.from(pos), n = pos.length / 3, wt = skin.subarray(n * 8), q = new Float32Array(8);
  for (let i = 0; i < n; i++) {
    q.fill(0);
    let first = -1;
    for (let k = i * 8; k < i * 8 + 8; k++) {
      if (!wt[k]) continue;
      const d = skin[k] * 8;
      if (first < 0) first = d;
      const sgn = dq[first] * dq[d] + dq[first + 1] * dq[d + 1] + dq[first + 2] * dq[d + 2] + dq[first + 3] * dq[d + 3] < 0 ? -1 : 1; // shortest arc
      for (let c = 0; c < 8; c++) q[c] += sgn * wt[k] * dq[d + c];
    }
    if (first < 0) continue;
    const len = Math.hypot(q[0], q[1], q[2], q[3]);
    const [w, x, y, z] = [q[0] / len, q[1] / len, q[2] / len, q[3] / len], [e0, e1, e2, e3] = [q[4] / len, q[5] / len, q[6] / len, q[7] / len];
    const px = pos[i * 3], py = pos[i * 3 + 1], pz = pos[i * 3 + 2];
    // rotate: v + 2 r × (r × v + w v), r = (x, y, z)
    const cx = y * pz - z * py + w * px, cy = z * px - x * pz + w * py, cz = x * py - y * px + w * pz;
    const rx = px + 2 * (y * cz - z * cy), ry = py + 2 * (z * cx - x * cz), rz = pz + 2 * (x * cy - y * cx);
    // translate: t = 2 (w e - e0 r + r × e), e = (e1, e2, e3)
    out[i * 3] = rx + 2 * (w * e1 - e0 * x + y * e3 - z * e2);
    out[i * 3 + 1] = ry + 2 * (w * e2 - e0 * y + z * e1 - x * e3);
    out[i * 3 + 2] = rz + 2 * (w * e3 - e0 * z + x * e2 - y * e1);
  }
  return out;
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

// Underwear cut from the body's own surface (see tools/build_assets.py underwear()): each vertex sits on a body edge,
// then stands 3 mm off the skin, and fabric tension bridges hollows (cleavage, between the buttocks) while the edges
// stay snug. Returns positions for the garment's triangles `${name}.tris` and uv `${name}.uv`.
const adjacency = {};
export function underwear(name, pos) {
  const ij = T.arr(`${name}.ij`), t = T.arr(`${name}.t`), tris = T.arr(`${name}.tris`), n = t.length, p = new Float32Array(n * 3);
  for (let k = 0; k < n; k++) for (let c = 0; c < 3; c++) p[k * 3 + c] = pos[ij[k * 2] * 3 + c] * (1 - t[k]) + pos[ij[k * 2 + 1] * 3 + c] * t[k];
  const nb = (adjacency[name] ??= (() => {
    const a = Array.from({ length: n }, () => new Set());
    for (let k = 0; k < tris.length; k += 3) for (let e = 0; e < 3; e++) { a[tris[k + e]].add(tris[k + (e + 1) % 3]); a[tris[k + (e + 1) % 3]].add(tris[k + e]); }
    return a.map((s) => [...s]);
  })());
  const normals = new Float32Array(n * 3);
  for (let k = 0; k < tris.length; k += 3) {
    const [a, b, c] = [tris[k], tris[k + 1], tris[k + 2]], e1 = [0, 1, 2].map((d) => p[b * 3 + d] - p[a * 3 + d]), e2 = [0, 1, 2].map((d) => p[c * 3 + d] - p[a * 3 + d]);
    const nx = e1[1] * e2[2] - e1[2] * e2[1], ny = e1[2] * e2[0] - e1[0] * e2[2], nz = e1[0] * e2[1] - e1[1] * e2[0];
    for (const v of [a, b, c]) { normals[v * 3] += nx; normals[v * 3 + 1] += ny; normals[v * 3 + 2] += nz; }
  }
  for (let k = 0; k < n; k++) {
    const l = Math.hypot(normals[k * 3], normals[k * 3 + 1], normals[k * 3 + 2]) || 1;
    for (let c = 0; c < 3; c++) { normals[k * 3 + c] /= l; p[k * 3 + c] += 0.3 * normals[k * 3 + c]; }
  }
  const edge = (k) => ij[k * 2] !== ij[k * 2 + 1];
  for (let it = 0; it < 30; it++) for (let k = 0; k < n; k++) {
    if (edge(k)) continue;
    let d = 0;
    for (const q of nb[k]) for (let c = 0; c < 3; c++) d += (p[q * 3 + c] - p[k * 3 + c]) * normals[k * 3 + c];
    d /= nb[k].length;
    if (d > 0) for (let c = 0; c < 3; c++) p[k * 3 + c] += d * normals[k * 3 + c];
  }
  return p;
}

// Genitals: a CC0 community model (modelled erect, ~22 cm) on the fitted body (cm), its shaft sized to length and girth
// (cm) and bent: up 0 hangs, 1 is erect at `angle` (° above level). The first 1.5 cm curve to where the root points; the
// rest follows, straight hanging or erect, drooping part-way up. Returns the smoothed mesh's positions and each vertex's
// place along the shaft (0 base … 1 tip; -1: scrotum).
export function genitalMesh(pos, { length, girth, up, angle = 0 }) {
  const P0 = fitProxy('genitals', pos), w = T.arr('genitals.shaft'), { tip, rings, base } = T.meta.genitals, n = w.length;
  const v = (P, i) => [P[i * 3], P[i * 3 + 1], P[i * 3 + 2]], dot = (x, y) => x[0] * y[0] + x[1] * y[1] + x[2] * y[2];
  const add = (x, y, k = 1) => [x[0] + k * y[0], x[1] + k * y[1], x[2] + k * y[2]], unit = (x) => add([0, 0, 0], x, 1 / Math.hypot(...x));
  const [c1, c2] = rings.map((r) => r.reduce((sum, i) => add(sum, v(P0, i), 1 / r.length), [0, 0, 0]));
  const a = unit(add(c2, c1, -1)), B = add(c1, a, -base * Math.hypot(...add(c2, c1, -1))); // axis; where it leaves the body
  const u = unit(add([0, -1, 0], a, a[1])), e = [a[1] * u[2] - a[2] * u[1], a[2] * u[0] - a[0] * u[2], a[0] * u[1] - a[1] * u[0]]; // down, across
  const ptr = T.arr('genitals.ptr'), idx = T.arr('genitals.idx'), mix = T.arr('genitals.mix'), P = new Float32Array(n * 3);
  for (let r = 0; r < n; r++) for (let j = ptr[r]; j < ptr[r + 1]; j++) for (let k = 0; k < 3; k++) P[r * 3 + k] += mix[j] * P0[idx[j] * 3 + k]; // smoothed
  // The model's scrotum is broad and held high (as when erect): drawn in round its middle (86 % across, 90 % otherwise) and
  // hanging 1.2 cm lower, its seam with the body kept; less so as it rises (erect, as modelled).
  const free = T.arr('genitals.free'), sw = (i) => (1 - w[i] / 255) * free[i] / 255;
  let sum = [0, 0, 0], tot = 0;
  for (let i = 0; i < n; i++) { sum = add(sum, v(P, i), sw(i)); tot += sw(i); }
  const mid = sum.map((x) => x / tot);
  for (let i = 0; i < n; i++) {
    const p = add(v(P, i), mid, -1), q = add(add(add(mid, a, 0.9 * dot(p, a)), u, 0.9 * dot(p, u) + 1.2), e, 0.86 * dot(p, e));
    for (let m = 0; m < 3; m++) P[i * 3 + m] += (q[m] - P[i * 3 + m]) * sw(i) * (1 - up);
  }
  const L0 = dot(add(v(P0, tip), B, -1), a), radii = [];
  for (let i = 0; i < n; i++) {
    const p = add(v(P, i), B, -1), s = dot(p, a);
    if (w[i] === 255 && Math.abs(s / L0 - 0.5) < 0.15) radii.push(Math.hypot(...add(p, a, -s)));
  }
  const G0 = 2 * Math.PI * radii.sort((x, y) => x - y)[radii.length >> 1];
  // Turns down from the modelled axis: aimed at `aim` on average, the root droop / 2 above it and the tip as far below.
  const rad = Math.PI / 180, aim = -88 + (angle + 88) * up, droop = 120 * up * (1 - up) * rad; // up to 30°, half way up
  const kL = length / L0, kG = girth / G0, k1 = (Math.asin(a[1]) - aim * rad - droop / 2) / 1.5, k2 = droop / Math.max(length - 1.5, 1);
  const arc = (t0, k, l) => (Math.abs(k) < 1e-6 ? [Math.cos(t0) * l, Math.sin(t0) * l] // along a and u, turning by k per cm
    : [(Math.sin(t0 + k * l) - Math.sin(t0)) / k, (Math.cos(t0) - Math.cos(t0 + k * l)) / k]);
  const along = new Float32Array(n);
  for (let i = 0; i < n; i++) {
    const p = add(v(P, i), B, -1), s = dot(p, a), f = w[i] / 255;
    along[i] = f > 0.5 ? Math.max(0, s / L0) : -1;
    if (s <= 0 || f === 0) continue; // inside the body, or scrotum: as modelled
    const S = s * kL, S1 = Math.min(S, 1.5), S2 = Math.max(0, S - 1.5), t = k1 * S1 + k2 * S2, g = 1 + (kG - 1) * Math.min(1, s / 2); // girth eases in over 2 cm: the root meets the body
    const under = add([0, 0, 0], u, Math.cos(t)).map((x, m) => x - a[m] * Math.sin(t)), [x1, y1] = arc(0, k1, S1), [x2, y2] = arc(k1 * 1.5, k2, S2);
    const c = add(add([0, 0, 0], a, x1 + x2), u, y1 + y2); // centreline
    const q = add(add(add(B, c), under, dot(p, u) * g), e, dot(p, e) * g);
    for (let m = 0; m < 3; m++) P[i * 3 + m] += (q[m] - P[i * 3 + m]) * f;
  }
  return { pos: P, along };
}

export const assets = () => T;
