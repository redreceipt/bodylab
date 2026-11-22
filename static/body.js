// Realistic body: MakeHuman's CC0 template (tools/build_body.py) fitted to a scan's measurements.
// Torso: width/depth scaled per level (neck, shoulders, chest, waist, hips). Limbs: scaled radially around
// their bones. Heights: remapped to the scan's ankle/knee/hip/shoulder levels. Rig weights blend it all.
// Pure math (no three.js) so test_body.mjs can run it in Node.

let T = null;
const REG = { head: 0, torso: 1, arm_l: 2, arm_r: 3, leg_l: 4, leg_r: 5 };
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

// meta/buf: static/body.json + static/body.bin. Returns the shared triangle indices and templates.
export function initBody(meta, buf) {
  let o = 0;
  const take = (Type, n) => { const a = new Type(buf, o, n); o += n * Type.BYTES_PER_ELEMENT; return a; };
  const male = take(Float32Array, meta.n * 3), female = take(Float32Array, meta.n * 3), reg = take(Uint8Array, meta.n * 6);
  const index = take(Uint16Array, meta.tris * 3), eyeM = take(Float32Array, meta.eye_n * 3), eyeF = take(Float32Array, meta.eye_n * 3);
  const eyeIndex = take(Uint16Array, meta.eye_tris * 3);
  T = { n: meta.n, reg, index, eyeIndex };
  T.male = template(male, eyeM, meta.joints.male, 'male');
  T.female = template(female, eyeF, meta.joints.female, 'female');
  return { index, eyeIndex, male: T.male, female: T.female };
}

// Measure the template exactly the way measure.py measures a camera silhouette.
function template(pos, eye, J, sex) {
  const { n, reg } = T, w = (i, r) => reg[i * 6 + r] / 255;
  const sh = (J.shoulder_l[1] + J.shoulder_r[1]) / 2, hip = (J.hip_l[1] + J.hip_r[1]) / 2, len = sh - hip;
  let H = 0;
  for (let i = 1; i < pos.length; i += 3) H = Math.max(H, pos[i]);
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
    pos, eye, J, H, sh, hip, len,
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
    const top = sex === 'female' ? band(y, sh - 0.42 * len, sh - 0.1 * len) : 0;
    t.cloth[i] = Math.max(band(y, hip - 0.38 * (hip - kneeY), sh - 0.84 * len), top) * clamp(1 - w(i, REG.arm_l) - w(i, REG.arm_r), 0, 1);
  }
  return t;
}

// dims: scan sizes in cm (from /api/state). Returns vertex positions for a person of height H.
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
  // Eyes follow the head; iris/pupil coloured by how far each vertex faces forward.
  const ep = new Float32Array(t.eye.length), ec = new Float32Array(t.eye.length);
  for (let i = 0; i < t.eye.length; i += 3) {
    const side = t.eye[i] > 0 ? 'eye_l' : 'eye_r', c = t.J[side];
    const v = [t.eye[i] - c[0], t.eye[i + 1] - c[1], t.eye[i + 2] - c[2]], fwd = v[2] / Math.hypot(...v);
    const col = fwd > 0.93 ? [0.02, 0.02, 0.02] : fwd > 0.8 ? [0.23, 0.15, 0.09] : [0.92, 0.9, 0.88];
    ep.set([t.eye[i] * s, interp(remap, t.eye[i + 1] * s)[0], t.eye[i + 2] * s], i);
    ec.set(col, i);
  }
  return { pos: out, eyePos: ep, eyeColors: ec, parts: t.parts, cloth: t.cloth };
}
