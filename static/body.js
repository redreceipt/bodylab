// Realistic body: MakeHuman's CC0 template (tools/build_assets.py) fitted to a scan's measurements.
// Torso: width/depth scaled per level (neck, shoulders, chest, waist, hips). Limbs: scaled radially around
// their bones. Heights: remapped to the scan's ankle/knee/hip/shoulder levels. Rig weights blend it all.
// Hair, brows, lashes and eyes are MakeHuman proxies anchored to base-mesh vertices, so they follow the fit.
// Pure math (no three.js / DOM) so test_body.mjs can run it in Node.

let T = null;
const REG = { head: 0, torso: 1, arm_l: 2, arm_r: 3, leg_l: 4, leg_r: 5 };
const TYPES = { float32: Float32Array, uint8: Uint8Array, uint16: Uint16Array };
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
    waist: sex === 'female' ? pick(range(yAt(0.5), yAt(0.8), 7), (a, b) => a < b) : slab(yAt(0.73)),
    hips: pick(range(yAt(0.95), yAt(1.1), 6), (a, b) => a > b),
  };
  for (const s of ['l', 'r']) {
    t[`arm_${s}`] = limbWidth(J[`shoulder_${s}`], J[`elbow_${s}`], [0.5], REG[`arm_${s}`]);
    t[`thigh_${s}`] = limbWidth(J[`hip_${s}`], J[`knee_${s}`], [0.4], REG[`leg_${s}`]);
    t[`calf_${s}`] = limbWidth(J[`knee_${s}`], J[`ankle_${s}`], range(0.2, 0.45, 6), REG[`leg_${s}`]);
  }
  // Per-vertex labels for picking/highlighting, and workout clothes (shorts; sports top for women).
  t.parts = new Array(n);
  t.cloth = new Float32Array(n); // 0 = skin, 1 = fabric, soft edges
  t.scalp = new Uint8Array(n); // head skin a haircut covers: above the forehead, and the back of the head to the nape
  const smooth = (a, b, x) => { const k = clamp((x - a) / (b - a), 0, 1); return k * k * (3 - 2 * k); };
  const band = (y, lo, hi) => smooth(lo - 1.5, lo + 1.5, y) * (1 - smooth(hi - 1.5, hi + 1.5, y));
  const kneeY = (J.knee_l[1] + J.knee_r[1]) / 2;
  for (let i = 0; i < n; i++) {
    let r = 0;
    for (let k = 1; k < 6; k++) if (reg[i * 6 + k] > reg[i * 6 + r]) r = k;
    const y = pos[i * 3 + 1], f = (sh - y) / len, side = r % 2 ? 'r' : 'l';
    t.parts[i] = r === REG.head ? 'overall' : r === REG.torso
      ? (f < t.neck.f / 2 ? 'neck' : f < 0.15 ? 'shoulders' : f < 0.5 ? 'chest' : f < 0.87 ? 'waist' : 'hips')
      : r <= REG.arm_r ? `arm_${side}` : (y > J[`knee_${side}`][1] ? `thigh_${side}` : `calf_${side}`);
    const eyeY = J.eye_l[1];
    t.scalp[i] = r === REG.head && (y > eyeY + 0.045 * H || (pos[i * 3 + 2] < J.head[2] - 3 && y > eyeY - 0.035 * H)) ? 1 : 0;
    const top = sex === 'female' ? band(y, sh - 0.42 * len, sh - 0.1 * len) : 0;
    t.cloth[i] = Math.max(band(y, hip - 0.38 * (hip - kneeY), sh - 0.84 * len), top) * clamp(1 - w(i, REG.arm_l) - w(i, REG.arm_r), 0, 1);
  }
  return t;
}

// dims: scan sizes in cm (from /api/state). Returns fitted positions of every base-mesh vertex for height H.
export function fitBody(dims, H, sex) {
  const t = T[sex === 'female' ? 'female' : 'male'], d = dims || {}, s = H / t.H, n = T.n;
  const ratio = (scan, tmpl) => (scan ? clamp(scan / (tmpl * s), 0.6, 1.8) : 1);
  // torso knots by f (0 = shoulders, 1 = hip joints): [f, x scale, z scale, z centre]
  const torso = [
    [t.neck.f, ratio(d.neck_w, t.neck.w), ratio(d.neck_d, t.neck.d), t.neck.cz],
    [0, ratio(d.shoulders_w, t.shoulders.w), ratio(d.chest_d, t.chest.d), t.shoulders.cz],
    [0.3, ratio(d.chest_w, t.chest.w), ratio(d.chest_d, t.chest.d), t.chest.cz],
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
    const x = t.pos[i * 3] * s, y = t.pos[i * 3 + 1] * s, z = t.pos[i * 3 + 2] * s;
    const wt = (r) => T.reg[i * 6 + r] / 255;
    let X = 0, Y = 0, Z = 0;
    const add = (wgt, px, py, pz) => { X += wgt * px; Y += wgt * py; Z += wgt * pz; };
    add(wt(REG.head), x, y, z);
    if (wt(REG.torso)) {
      const [rx, rz, cz] = interp(torso, (t.sh - t.pos[i * 3 + 1]) / t.len);
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
  return { pos: out, parts: t.parts, cloth: t.cloth };
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

export const assets = () => T;
