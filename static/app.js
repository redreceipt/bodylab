import * as THREE from 'three';
import { OrbitControls } from 'three/addons/OrbitControls.js';

const $ = (s) => document.querySelector(s);
const esc = (s) => String(s).replace(/[&<>"']/g, (c) => `&#${c.charCodeAt(0)};`);
let S = null;      // /api/state
let live = null;   // {m, dims} while someone stands in front of the camera
let idx = -1;      // selected scan
let view = null;   // open panel: {tab} or {part}
let hovered = null, selectedPart = null, lastScanId = null, lastState = null, lastDimsKey = '';

// ---------- metrics & units ----------
const META = {
  bf: { label: 'Body fat', unit: '%' }, weight: { label: 'Weight', unit: 'kg' }, lean_kg: { label: 'Lean mass', unit: 'kg' },
  fat_kg: { label: 'Fat mass', unit: 'kg' }, ffmi: { label: 'Fat-free mass index', unit: '' },
  whtr: { label: 'Waist ÷ height', unit: '', dec: 3 }, whr: { label: 'Waist ÷ hips', unit: '', dec: 3 },
  neck: { label: 'Neck', unit: 'cm' }, shoulders: { label: 'Shoulder width', unit: 'cm' }, chest: { label: 'Chest', unit: 'cm' },
  waist: { label: 'Waist', unit: 'cm' }, hips: { label: 'Hips', unit: 'cm' },
  arm_l: { label: 'Left arm', unit: 'cm' }, arm_r: { label: 'Right arm', unit: 'cm' },
  thigh_l: { label: 'Left thigh', unit: 'cm' }, thigh_r: { label: 'Right thigh', unit: 'cm' },
  calf_l: { label: 'Left calf', unit: 'cm' }, calf_r: { label: 'Right calf', unit: 'cm' },
};
const imperial = () => S?.profile?.units === 'imperial';
const K = { cm: 1 / 2.54, kg: 2.20462 };
const toUnit = (v, u) => (imperial() && K[u] ? v * K[u] : v);
const fromUnit = (v, u) => (imperial() && K[u] ? v / K[u] : v);
const uLabel = (u) => (imperial() ? { cm: 'in', kg: 'lb' }[u] ?? u : u);
function fmt(k, v, signed = false) {
  if (v == null || Number.isNaN(v)) return '—';
  const { unit = '', dec = 1 } = META[k] || {};
  const x = toUnit(v, unit), u = uLabel(unit);
  return (signed && x > 0 ? '+' : '') + x.toFixed(dec) + (u === '%' ? '%' : u ? ' ' + u : '');
}
const day = (ts) => new Date(ts * 1000).toLocaleDateString(undefined, { month: 'short', day: 'numeric' });
const isoDay = (iso) => new Date(iso + 'T12:00').toLocaleDateString(undefined, { month: 'short', day: 'numeric', year: 'numeric' });
const cur = () => live?.m ?? S.scans[idx]?.m ?? {};
const base = () => S.scans[0]?.m ?? {};
const series = (k) => (k === 'weight' ? S.logs.filter((l) => l.weight).map((l) => [l.ts, l.weight])
  : S.scans.filter((s) => s.m[k] != null).map((s) => [s.ts, s.m[k]]));
// good/bad colouring only where the user said which way they want a metric to go
function tone(k, delta) {
  const g = S.insights.goals[k];
  if (!g || !delta || g.start == null || g.target === g.start) return '';
  return Math.sign(delta) === Math.sign(g.target - g.start) ? 'good' : 'bad';
}

async function api(method, url, body) {
  const r = await fetch(url, { method, headers: { 'Content-Type': 'application/json' }, body: body && JSON.stringify(body) });
  const j = await r.json();
  if (!r.ok) throw new Error(j.error || r.statusText);
  return j;
}
function toast(msg, err = false) {
  const t = $('#toast');
  t.textContent = msg; t.className = err ? 'err' : ''; t.style.display = 'block';
  clearTimeout(toast.t); toast.t = setTimeout(() => (t.style.display = 'none'), 3500);
}

// ---------- 3D scene ----------
const renderer = new THREE.WebGLRenderer({ canvas: $('#c'), antialias: true, alpha: true });
renderer.setPixelRatio(devicePixelRatio);
renderer.shadowMap.enabled = true;
const scene = new THREE.Scene();
const camera = new THREE.PerspectiveCamera(32, 1, 0.1, 100);
camera.position.set(0.6, 1.25, 4.6);
const controls = new OrbitControls(camera, renderer.domElement);
Object.assign(controls, { enableDamping: true, autoRotate: true, autoRotateSpeed: 0.7, minDistance: 1.2, maxDistance: 9 });
controls.target.set(0, 0.9, 0);
controls.addEventListener('start', () => (controls.autoRotate = false));
scene.add(new THREE.HemisphereLight('#c4dcff', '#141c2c', 1.4));
const keyLight = new THREE.DirectionalLight('#ffffff', 2.4);
keyLight.position.set(2, 4, 3);
keyLight.castShadow = true;
keyLight.shadow.mapSize.set(1024, 1024);
Object.assign(keyLight.shadow.camera, { left: -1.5, right: 1.5, top: 2.5, bottom: -0.5 });
scene.add(keyLight);
const rim = new THREE.DirectionalLight('#3ef0ff', 1.3);
rim.position.set(-3, 2.5, -3);
scene.add(rim);
const floor = new THREE.Mesh(new THREE.CircleGeometry(0.95, 72), new THREE.MeshStandardMaterial({ color: '#0c1524', roughness: 0.9 }));
floor.rotation.x = -Math.PI / 2;
floor.receiveShadow = true;
scene.add(floor, new THREE.PolarGridHelper(0.95, 8, 5, 72, '#1c3454', '#1c3454'));
const halo = new THREE.Mesh(new THREE.RingGeometry(0.93, 0.96, 96), new THREE.MeshBasicMaterial({ color: '#3ef0ff', transparent: true, opacity: 0.7 }));
halo.rotation.x = -Math.PI / 2;
halo.position.y = 0.002;
scene.add(halo);
const scanRing = new THREE.Mesh(new THREE.TorusGeometry(0.42, 0.004, 8, 96), new THREE.MeshBasicMaterial({ color: '#3ef0ff', transparent: true, opacity: 0.85 }));
scanRing.rotation.x = Math.PI / 2;
scanRing.visible = false;
scene.add(scanRing);

let avatar = null, ghost = null, scanning = false;

// Proportions as fractions of height, used for anything a scan hasn't measured.
const DEF = {
  male: { neck_w: .07, shoulders_w: .265, chest_w: .183, waist_w: .166, hips_w: .194, arm_w: .057, thigh_w: .097, calf_w: .069 },
  female: { neck_w: .062, shoulders_w: .245, chest_w: .178, waist_w: .153, hips_w: .215, arm_w: .055, thigh_w: .104, calf_w: .07 },
  levels: { lv_neck: .855, lv_shoulder: .818, lv_chest: .72, lv_waist: .6, lv_hips: .5, lv_hip: .52, lv_knee: .285, lv_ankle: .045, upperarm: .186, forearm: .146 },
};
const DEPTH = { neck: 1, chest: .72, waist: .78, hips: .75 };
function fullDims(dims, H, sex) {
  const d = {};
  for (const [k, v] of Object.entries({ ...DEF[sex === 'female' ? 'female' : 'male'], ...DEF.levels })) d[k] = v * H;
  for (const p of ['arm', 'thigh', 'calf']) for (const s of ['l', 'r']) d[`${p}_${s}_w`] = d[`${p}_w`];
  Object.assign(d, dims);
  for (const [p, r] of Object.entries(DEPTH)) d[`${p}_d`] = dims?.[`${p}_d`] ?? d[`${p}_w`] * r;
  d.H = H;
  return d;
}

// Smooth loft through elliptical-ish rings [{y, w, d}] (bottom→top); superellipse makes the torso less tube-like.
function loft(rings, power = 2.5) {
  const curve = new THREE.CatmullRomCurve3(rings.map((r) => new THREE.Vector3(r.w, r.y, r.d)), false, 'centripetal');
  const N = 64, SEG = 40, e = 2 / power, pos = [], index = [];
  for (let i = 0; i <= N; i++) {
    const c = curve.getPoint(i / N);
    for (let j = 0; j <= SEG; j++) {
      const t = (j / SEG) * Math.PI * 2, ct = Math.cos(t), st = Math.sin(t);
      pos.push(Math.sign(ct) * Math.abs(ct) ** e * c.x / 2, c.y, Math.sign(st) * Math.abs(st) ** e * c.z / 2);
    }
  }
  for (let i = 0; i < N; i++) for (let j = 0; j < SEG; j++) {
    const a = i * (SEG + 1) + j, b = a + SEG + 1;
    index.push(a, b, a + 1, b, b + 1, a + 1);
  }
  const g = new THREE.BufferGeometry();
  g.setAttribute('position', new THREE.Float32BufferAttribute(pos, 3));
  g.setIndex(index);
  g.computeVertexNormals();
  return g;
}
// Limb from a radius profile [[r, y]] with y ascending from 0.
function lathe(profile) {
  const pts = new THREE.SplineCurve(profile.map(([r, y]) => new THREE.Vector2(r, y))).getPoints(56);
  return new THREE.LatheGeometry([new THREE.Vector2(0, profile[0][1]), ...pts, new THREE.Vector2(0, profile.at(-1)[1])], 32);
}

const COLORS = { base: new THREE.Color('#9db7d6'), good: new THREE.Color('#4cff9a'), bad: new THREE.Color('#ff6b6b'), sel: new THREE.Color('#3ef0ff') };
function partColor(p) {
  if (p === selectedPart) return COLORS.sel;
  const ch = S?.insights.parts[p]?.change, t = S && tone(p, ch);
  const c = COLORS.base.clone();
  if (t) c.lerp(COLORS[t], 0.55);
  if (p === hovered) c.lerp(new THREE.Color('#ffffff'), 0.35);
  return c;
}

function buildAvatar(dims, sex, isGhost) {
  const H = S.profile?.height_cm ?? 175, d = fullDims(dims, H, sex);
  if (isGhost) for (const k in d) if (/_[wd]$/.test(k)) d[k] -= 1.2; // hide sub-6 mm differences (and z-fighting)
  const group = new THREE.Group();
  // Ghost = first scan as a translucent shell that loses depth ties, so it only shows where the body has shrunk since.
  const mat = isGhost
    ? new THREE.MeshBasicMaterial({ color: '#3ef0ff', transparent: true, opacity: 0.22, depthWrite: false,
      polygonOffset: true, polygonOffsetFactor: 2, polygonOffsetUnits: 8 })
    : new THREE.MeshStandardMaterial({ vertexColors: true, roughness: 0.42, metalness: 0.12 });
  const add = (geo, partAt, place) => {
    const mesh = new THREE.Mesh(geo, mat);
    if (!isGhost) {
      const y = geo.attributes.position;
      geo.userData.parts = Array.from({ length: y.count }, (_, i) => partAt(y.getY(i)));
      geo.setAttribute('color', new THREE.Float32BufferAttribute(new Float32Array(y.count * 3), 3));
      mesh.castShadow = true;
    }
    mesh.userData.partAt = partAt;
    place?.(mesh);
    group.add(mesh);
    return mesh;
  };
  const along = (from, to) => (mesh) => { // put a +Y-built limb from point `from` toward `to`
    mesh.position.copy(from);
    mesh.quaternion.setFromUnitVectors(new THREE.Vector3(0, 1, 0), to.clone().sub(from).normalize());
  };

  // torso + neck
  const shY = d.lv_shoulder, neckY = Math.max(d.lv_neck, shY + .045 * H);
  const rings = [
    { y: d.lv_hips - .07 * H, w: d.hips_w * .7, d: d.hips_d * .6 },
    { y: d.lv_hips - .04 * H, w: d.hips_w * .96, d: d.hips_d * .9 },
    { y: d.lv_hips, w: d.hips_w, d: d.hips_d },
    { y: d.lv_waist, w: d.waist_w, d: d.waist_d },
    { y: d.lv_chest, w: d.chest_w, d: d.chest_d },
    { y: shY - .03 * H, w: Math.max(d.chest_w, d.shoulders_w * .8), d: d.chest_d * .92 },
    { y: shY + .012 * H, w: d.shoulders_w * .72, d: d.chest_d * .6 },
    { y: shY + .028 * H, w: d.neck_w * 1.45, d: d.neck_d * 1.1 },
    { y: neckY, w: d.neck_w, d: d.neck_d },
    { y: neckY + .035 * H, w: d.neck_w * .95, d: d.neck_d * .95 },
  ].sort((a, b) => a.y - b.y);
  const torsoCuts = [['hips', (d.lv_hips + d.lv_waist) / 2], ['waist', (d.lv_waist + d.lv_chest) / 2],
    ['chest', shY - .035 * H], ['shoulders', shY + .022 * H], ['neck', Infinity]];
  const torso = add(loft(rings, 2.2), (y) => torsoCuts.find(([, top]) => y < top)[0]);
  torso.userData.breathe = true;

  // head
  const hw = .085 * H, hh = .118 * H, hd = .108 * H, headGeo = new THREE.SphereGeometry(1, 40, 28), hp = headGeo.attributes.position;
  for (let i = 0; i < hp.count; i++) { // taper the lower half into a jaw
    const k = 1 - .3 * Math.max(0, -hp.getY(i)) ** 1.6;
    hp.setXYZ(i, hp.getX(i) * k * hw / 2, hp.getY(i) * hh / 2, hp.getZ(i) * (1 - .12 * Math.max(0, -hp.getY(i))) * hd / 2);
  }
  headGeo.computeVertexNormals();
  add(headGeo, () => 'overall', (m) => m.position.set(0, H - hh / 2, .012 * H));
  if (!isGhost) for (const s of [-1, 1]) {
    const eye = new THREE.Mesh(new THREE.SphereGeometry(.0055 * H, 12, 8), new THREE.MeshBasicMaterial({ color: '#0a1a2a' }));
    eye.position.set(s * hw * .19, H - hh * .44, .012 * H + hd * .47);
    group.add(eye);
  }

  for (const [s, x] of [['l', 1], ['r', -1]]) { // the avatar faces +z, so its left side is +x
    // legs: ankle → hip joint, slight A stance
    const rT = d[`thigh_${s}_w`] / 2, rC = d[`calf_${s}_w`] / 2;
    const hipX = Math.max(rT * .98, d.hips_w / 2 - rT * 1.05);
    const hip = new THREE.Vector3(x * hipX, d.lv_hip, 0), ankle = new THREE.Vector3(x * hipX * 1.12, d.lv_ankle, 0);
    const L = hip.distanceTo(ankle), tk = (d.lv_knee - d.lv_ankle) / (d.lv_hip - d.lv_ankle) * L;
    add(lathe([[rC * .5, 0], [rC * .6, tk * .25], [rC, tk * .68], [rC * .88, tk * .9], [rC * .92, tk],
      [rT * .82, tk + (L - tk) * .3], [rT, tk + (L - tk) * .6], [rT * 1.03, L], [rT * .9, L + .03 * H]]),
    (y) => (y < tk ? `calf_${s}` : `thigh_${s}`), along(ankle, hip));
    add(new THREE.SphereGeometry(1, 20, 14), () => `calf_${s}`, (m) => {
      m.scale.set(rC * .85, .022 * H, .075 * H);
      m.position.set(ankle.x, .022 * H, .035 * H);
    });
    // arms: wrist → shoulder joint, ~28° A-pose
    const rA = d[`arm_${s}_w`] / 2, Lu = d.upperarm, Lf = d.forearm, La = Lu + Lf, a = THREE.MathUtils.degToRad(28);
    const sh = new THREE.Vector3(x * (d.shoulders_w / 2 - rA), shY, 0);
    const wrist = sh.clone().add(new THREE.Vector3(x * Math.sin(a) * La, -Math.cos(a) * La, 0));
    add(lathe([[rA * .42, 0], [rA * .55, Lf * .3], [rA * .72, Lf * .7], [rA * .62, Lf], [rA * .9, Lf + Lu * .35],
      [rA, Lf + Lu * .5], [rA * 1.08, Lf + Lu * .82], [rA * .95, La], [rA * .55, La + rA * .7]]),
    () => `arm_${s}`, along(wrist, sh));
    const dir = wrist.clone().sub(sh).normalize();
    add(new THREE.SphereGeometry(1, 16, 12), () => `arm_${s}`, (m) => {
      m.scale.set(rA * .42, .05 * H, rA * .28);
      m.position.copy(wrist).addScaledVector(dir, .045 * H);
      m.quaternion.setFromUnitVectors(new THREE.Vector3(0, 1, 0), dir);
    });
  }
  group.scale.setScalar(0.01); // cm → m
  return group;
}

function paint() {
  avatar?.traverse((o) => {
    const parts = o.geometry?.userData.parts;
    if (!parts) return;
    const col = o.geometry.attributes.color, cache = {};
    parts.forEach((p, i) => { const c = (cache[p] ??= partColor(p)); col.setXYZ(i, c.r, c.g, c.b); });
    col.needsUpdate = true;
  });
}

function rebuildAvatar(force = false) {
  const sex = S.profile?.sex, dims = live?.dims ?? S.scans[idx]?.dims ?? {};
  const key = JSON.stringify([dims, sex, S.profile?.height_cm]);
  if (!force && key === lastDimsKey) return paint();
  lastDimsKey = key;
  for (const g of [avatar, ghost]) if (g) { scene.remove(g); g.traverse((o) => o.geometry?.dispose()); }
  avatar = buildAvatar(dims, sex, false);
  scene.add(avatar);
  const showGhost = $('#ghost').checked && S.scans.length > 1 && (live || idx > 0);
  ghost = showGhost ? buildAvatar(S.scans[0].dims, sex, true) : null;
  if (ghost) scene.add(ghost);
  controls.target.y = (S.profile?.height_cm ?? 175) / 200;
  paint();
}

function resize() {
  renderer.setSize(innerWidth, innerHeight, false);
  camera.aspect = innerWidth / innerHeight;
  camera.updateProjectionMatrix();
}
addEventListener('resize', resize);
resize();
renderer.setAnimationLoop((t) => {
  const s = Math.sin(t / 650);
  avatar?.children.forEach((m) => m.userData.breathe && m.scale.set(1 + s * .008, 1, 1 + s * .014));
  const H = (S?.profile?.height_cm ?? 175) / 100;
  scanRing.visible = scanning;
  if (scanning) scanRing.position.y = H * (0.5 + 0.48 * Math.sin(t / 700));
  halo.material.opacity = 0.5 + 0.25 * Math.sin(t / 900);
  controls.update();
  renderer.render(scene, camera);
});

// picking
const ray = new THREE.Raycaster(), ndc = new THREE.Vector2();
function pick(e) {
  if (!avatar) return null;
  ndc.set((e.clientX / innerWidth) * 2 - 1, -(e.clientY / innerHeight) * 2 + 1);
  ray.setFromCamera(ndc, camera);
  const hit = ray.intersectObjects(avatar.children.filter((o) => o.userData.partAt), false)[0];
  return hit ? hit.object.userData.partAt(hit.object.worldToLocal(hit.point.clone()).y) : null;
}
const canvas = renderer.domElement;
let downAt = null;
canvas.addEventListener('pointermove', (e) => {
  const p = pick(e), tip = $('#tip');
  if (p !== hovered) { hovered = p; paint(); }
  canvas.style.cursor = p ? 'pointer' : 'grab';
  if (!p) return (tip.style.display = 'none');
  const k = p === 'overall' ? 'bf' : p;
  tip.innerHTML = `<b>${p === 'overall' ? 'Overall' : META[p].label}</b> · ${fmt(k, cur()[k])}`;
  Object.assign(tip.style, { display: 'block', left: e.clientX + 14 + 'px', top: e.clientY + 14 + 'px' });
});
canvas.addEventListener('pointerdown', (e) => (downAt = [e.clientX, e.clientY]));
canvas.addEventListener('pointerup', (e) => {
  if (!downAt || Math.hypot(e.clientX - downAt[0], e.clientY - downAt[1]) > 5) return;
  const p = pick(e);
  if (p) open({ part: p });
});

// ---------- HUD, timeline ----------
function renderHud() {
  const m = cur(), b = base();
  $('#stats').innerHTML = ['bf', 'weight', 'lean_kg', 'whtr'].map((k) => {
    const d = S.scans.length > 1 && m[k] != null && b[k] != null ? m[k] - b[k] : null;
    return `<button class="stat" data-part="${k === 'whtr' ? 'waist' : 'overall'}"><div class="k">${META[k].label}</div>
      <div class="v">${fmt(k, m[k])}${d ? `<span class="d ${tone(k, d)}">${fmt(k, d, true)}</span>` : ''}</div></button>`;
  }).join('');
}
function renderTimeline() {
  const n = S.scans.length;
  $('#scrub').max = Math.max(0, n - 1);
  $('#scrub').value = idx;
  $('#scrub').disabled = n < 2;
  $('#scanLabel').textContent = live ? 'live view' : n ? `scan ${idx + 1}/${n} · ${day(S.scans[idx].ts)}` : 'no scans yet';
  $('#liveTag').hidden = !live;
}
$('#scrub').addEventListener('input', (e) => { idx = +e.target.value; renderAll(); });
$('#ghost').addEventListener('change', () => rebuildAvatar(true));
$('#stats').addEventListener('click', (e) => { const b = e.target.closest('[data-part]'); if (b) open({ part: b.dataset.part }); });

// ---------- panel ----------
function spark(k) {
  const pts = series(k);
  if (pts.length < 2) return '<p class="muted">Trend appears after 2+ data points.</p>';
  const u = META[k].unit, ys = pts.map((p) => toUnit(p[1], u)), xs = pts.map((p) => p[0]);
  const g = S.goals[k] && toUnit(S.goals[k].target, u);
  const lo = Math.min(...ys, g ?? Infinity), hi = Math.max(...ys, g ?? -Infinity);
  const W = 360, H = 80, sx = (x) => 6 + ((x - xs[0]) / (xs.at(-1) - xs[0] || 1)) * (W - 12), sy = (y) => H - 8 - ((y - lo) / (hi - lo || 1)) * (H - 16);
  return `<svg class="spark" viewBox="0 0 ${W} ${H}" role="img" aria-label="${META[k].label} history">
    ${g != null ? `<line x1="0" x2="${W}" y1="${sy(g)}" y2="${sy(g)}"/>` : ''}
    <polyline points="${xs.map((x, i) => `${sx(x)},${sy(ys[i])}`).join(' ')}"/>
    <circle r="3.5" cx="${sx(xs.at(-1))}" cy="${sy(ys.at(-1))}"/></svg>`;
}
function goalBlock(k) {
  const g = S.goals[k], gi = S.insights.goals[k];
  if (!g) return `<form class="inline" data-goal="${k}"><input name="target" type="number" step="any" placeholder="Target (${uLabel(META[k].unit) || 'value'})" required aria-label="Goal for ${META[k].label}"><button>Set goal</button></form>`;
  const pace = gi.progress === 1 ? '🎉 Goal reached'
    : gi.eta ? `On pace for <b>${isoDay(gi.eta)}</b> (${fmt(k, gi.rate_per_week, true)}/week)`
      : gi.rate_per_week != null ? `Trend ${fmt(k, gi.rate_per_week, true)}/week — not heading toward the goal yet` : 'Projection needs ~3 data points over a few days';
  return `<div class="goal"><div class="row"><span>Goal <b>${fmt(k, g.target)}</b></span><button class="link" data-delgoal="${k}">remove</button></div>
    <div class="bar"><i style="width:${(gi.progress ?? 0) * 100}%"></i></div><div class="muted">${pace}</div></div>`;
}
function metricBlock(k) {
  const m = cur(), b = base(), d = m[k] != null && b[k] != null && S.scans.length > 1 ? m[k] - b[k] : null;
  const c28 = S.insights.parts[k]?.change ?? S.insights.summary.changes[k];
  return `<div class="metric"><div class="row"><h3>${META[k].label}</h3><span class="big">${fmt(k, m[k])}</span></div>
    <div class="muted">Since first scan <span class="${tone(k, d)}">${fmt(k, d, true)}</span> · last 28 days <span class="${tone(k, c28)}">${fmt(k, c28, true)}</span></div>
    ${spark(k)}${goalBlock(k)}</div>`;
}
function partPanel(k) {
  if (k === 'overall') {
    const m = cur();
    return `<h2>Overall</h2>${['bf', 'weight', 'lean_kg'].map(metricBlock).join('')}
      <table>${['fat_kg', 'ffmi', 'whtr', 'whr'].map((x) => `<tr><td>${META[x].label}</td><td>${fmt(x, m[x])}</td></tr>`).join('')}</table>
      <p class="muted">Body fat uses the US Navy circumference formula (height, neck, waist${S.profile?.sex === 'female' ? ', hips' : ''}). Expect ±3–4 points; tape-calibrating neck and waist helps most. Mass values need a weigh-in within 7 days of the scan.</p>`;
  }
  const p = S.insights.parts[k] || {}, cal = S.profile?.calib?.[k], dims = (live?.dims ?? S.scans[idx]?.dims) || {};
  const torso = DEPTH[k] && dims[`${k}_w`] ? `<p class="muted">Measured width ${fmt('neck', dims[`${k}_w`])}, depth ${fmt('neck', dims[`${k}_d`])} ${S.scans[idx]?.side && !live ? '(side view)' : '(estimated — turn sideways during a scan to measure it)'}.</p>` : '';
  return `${metricBlock(k)}${torso}
    <h4>Training → result · last 28 days</h4>
    <p>${p.workouts ? `<b>${p.workouts}</b> workouts targeted this area (load ${p.load}).` : 'No logged workouts targeted this area.'}
    ${p.change != null ? ` Measured change <b class="${tone(k, p.change)}">${fmt(k, p.change, true)}</b>.` : ''}</p>
    <h4>Calibrate</h4><p class="muted">Tape-measure once; all scans (past and future) are corrected.${cal ? ` Current correction ×${cal.toFixed(3)}.` : ''}</p>
    <form class="inline" data-cal="${k}"><input name="value" type="number" step="any" required placeholder="Tape (${uLabel('cm')})" aria-label="Tape measurement"><button>Calibrate</button>
    ${cal ? `<button type="button" class="link" data-calreset="${k}">reset</button>` : ''}</form>`;
}
const FOCUS = ['full body', 'chest', 'back', 'shoulders', 'arms', 'core', 'legs', 'glutes', 'cardio'];
function logItem(l) {
  const txt = l.kind === 'workout'
    ? `🏋️ ${l.type} · ${l.minutes} min · RPE ${l.rpe}${l.focus?.length ? ' · ' + l.focus.join(', ') : ''}${l.notes ? ' · ' + l.notes : ''}`
    : '📋 ' + [l.weight && fmt('weight', l.weight), l.kcal && `${Math.round(l.kcal)} kcal`, l.protein && `${Math.round(l.protein)} g protein`,
      l.sleep && `${l.sleep.toFixed(1)} h sleep`, l.active && `${Math.round(l.active)} active kcal`].filter(Boolean).join(' · ');
  return `<li><span>${day(l.ts)}</span> ${esc(txt)} <button class="link" data-dellog="${l.id}">delete</button></li>`;
}
function logPanel() {
  const today = new Date().toLocaleDateString('en-CA');
  return `<h2>Log inputs</h2>
    <form id="fWorkout" class="stack"><h3>Workout</h3>
      <div class="grid2"><label>Date<input type="date" name="date" value="${today}"></label>
      <label>Type<select name="type">${['Strength', 'Cardio', 'HIIT', 'Sports', 'Mobility'].map((t) => `<option>${t}</option>`).join('')}</select></label>
      <label>Minutes<input type="number" name="minutes" min="1" max="600" value="45" required></label>
      <label>Effort (RPE 1–10)<input type="number" name="rpe" min="1" max="10" value="7" required></label></div>
      <fieldset class="chips"><legend class="muted">Focus</legend>${FOCUS.map((f) => `<label><input type="checkbox" name="focus" value="${f}">${f}</label>`).join('')}</fieldset>
      <input name="notes" placeholder="Notes (optional)" aria-label="Notes"><button>Save workout</button></form>
    <form id="fDaily" class="stack"><h3>Daily check-in</h3>
      <div class="grid2"><label>Date<input type="date" name="date" value="${today}"></label>
      <label>Weight (${uLabel('kg')})<input type="number" step="any" name="weight"></label>
      <label>Calories (kcal)<input type="number" name="kcal"></label>
      <label>Protein (g)<input type="number" name="protein"></label>
      <label>Sleep (h)<input type="number" step="any" name="sleep"></label></div><button>Save check-in</button></form>
    <h3>Recent</h3><ul class="list">${S.logs.slice(-15).reverse().map(logItem).join('') || '<li class="muted">Nothing logged yet</li>'}</ul>
    <h3>Scans</h3><ul class="list">${S.scans.slice(-10).reverse().map((s) => `<li><span>${new Date(s.ts * 1000).toLocaleString()}</span> · body fat ${fmt('bf', s.m.bf)}${s.side ? ' · front+side' : ''} <button class="link" data-delscan="${s.id}">delete</button></li>`).join('') || '<li class="muted">No scans yet</li>'}</ul>`;
}
function goalsPanel() {
  const free = Object.keys(META).filter((k) => !S.goals[k]);
  return `<h2>Goals</h2>${Object.keys(S.goals).map(metricBlock).join('') || '<p class="muted">No goals yet. Goals colour the body: green = moving toward it, red = away.</p>'}
    <form id="fGoal" class="inline"><select name="metric" aria-label="Metric">${free.map((k) => `<option value="${k}">${META[k].label}</option>`).join('')}</select>
    <input name="target" type="number" step="any" placeholder="Target" required aria-label="Target"><button>Add goal</button></form>`;
}
function weekChart(weeks) {
  const W = 380, H = 140, bw = W / weeks.length, maxL = Math.max(1, ...weeks.map((w) => w.load));
  const bfs = weeks.map((w) => w.bf).filter((v) => v != null), lo = Math.min(...bfs), hi = Math.max(...bfs);
  const y = (v) => H - 22 - ((v - lo) / (hi - lo || 1)) * (H - 44);
  const pts = weeks.map((w, i) => (w.bf == null ? null : [i * bw + bw / 2, y(w.bf)])).filter(Boolean);
  return `<svg class="chart" viewBox="0 0 ${W} ${H}" role="img" aria-label="Weekly training load and body fat">
    ${weeks.map((w, i) => { const h = (w.load / maxL) * (H - 44); return `<rect x="${i * bw + 5}" y="${H - 16 - h}" width="${bw - 10}" height="${h}" rx="3"/>`; }).join('')}
    <polyline points="${pts.map((p) => p.join(',')).join(' ')}"/>${pts.map(([cx, cy]) => `<circle r="3" cx="${cx}" cy="${cy}"/>`).join('')}
    ${weeks.map((w, i) => `<text x="${i * bw + bw / 2}" y="${H - 3}">${w.week.slice(5)}</text>`).join('')}</svg>
    <div class="legend"><span><i></i>training load (min × RPE)</span><span><i class="l"></i>body fat</span></div>`;
}
function insightsPanel() {
  const I = S.insights, s = I.summary, c = s.changes;
  const outs = ['bf', 'waist', 'weight', 'lean_kg'].filter((k) => c[k] != null).map((k) => `${META[k].label.toLowerCase()} <b class="${tone(k, c[k])}">${fmt(k, c[k], true)}</b>`);
  const lines = [`<b>Inputs</b> (28 days): ${s.workouts} workouts, load ${s.load}${s.kcal ? `, ${Math.round(s.kcal)} kcal/day avg` : ''}.`,
    `<b>Outputs</b>: ${outs.length ? outs.join(', ') : 'need 2+ scans in this window'}.`,
    ...Object.entries(I.goals).filter(([, g]) => g.eta).map(([k, g]) => `At this pace: ${META[k].label} ${fmt(k, g.target)} by <b>${isoDay(g.eta)}</b>.`),
    ...I.notes];
  const cell = (k, v, dec = 0) => (v == null ? '—' : toUnit(v, META[k]?.unit).toFixed(META[k]?.dec ?? dec));
  return `<h2>Inputs → Outputs</h2><div class="callout">${lines.map((l) => `<p>${l}</p>`).join('')}</div>${weekChart(I.weeks)}
    <table><tr><th>Week of</th><th>Wkts</th><th>Load</th><th>kcal</th><th>Prot g</th><th>Sleep h</th><th>Wt ${uLabel('kg')}</th><th>BF %</th><th>Waist ${uLabel('cm')}</th></tr>
    ${I.weeks.map((w) => `<tr><td>${isoDay(w.week).replace(/, \d{4}$/, '')}</td><td>${w.workouts}</td><td>${w.load}</td><td>${cell(0, w.kcal)}</td>
      <td>${cell(0, w.protein)}</td><td>${cell(0, w.sleep, 1)}</td><td>${cell('weight', w.weight)}</td><td>${cell('bf', w.bf)}</td><td>${cell('waist', w.waist)}</td></tr>`).join('')}</table>
    <p class="muted">Correlations show up after ~4 weeks of scans and logs. They're hints, not proof.</p>`;
}
function profilePanel() {
  const p = S.profile || {}, imp = p.units ? imperial() : true;
  const h = p.height_cm ? (imp ? p.height_cm / 2.54 : p.height_cm).toFixed(1) : '';
  return `<h2>${p.height_cm ? 'Profile' : 'Welcome to BodyLab'}</h2>${p.height_cm ? '' : '<p>Enter your height — it is the ruler every camera measurement is scaled by.</p>'}
    <form id="fProfile" class="stack">
      <label>Units<select name="units"><option value="imperial" ${imp ? 'selected' : ''}>Imperial (in, lb)</option><option value="metric" ${imp ? '' : 'selected'}>Metric (cm, kg)</option></select></label>
      <label><span>Height (<span id="hu">${imp ? 'in' : 'cm'}</span>)</span><input name="height" type="number" step="any" value="${h}" required></label>
      <label>Body-fat formula<select name="sex"><option value="male">Male</option><option value="female" ${p.sex === 'female' ? 'selected' : ''}>Female</option></select></label>
      <label>Camera URL<input name="camera_url" value="${esc(p.camera_url ?? '')}" placeholder="http://&lt;pi-tailscale-ip&gt;:8000/stream.mjpg"></label>
      <label>Camera rotation<select name="rotate">${[0, 90, 180, 270].map((r) => `<option value="${r}" ${(p.rotate ?? 0) === r ? 'selected' : ''}>${r ? r + '° clockwise' : 'None'}</option>`).join('')}</select></label>
      <button>Save</button></form>
    <h3>iPhone · Apple Health sync</h3>
    <p class="muted">In the BodyLab Sync app, enter this server and token. It pulls weight, calories, protein, sleep, activity and workouts automatically.</p>
    <table><tr><td>Server</td><td><code>${esc(S.sync.url ?? 'https://<this-mac>.<tailnet>.ts.net:8443')}</code></td></tr>
      <tr><td>Token</td><td><code>${esc(S.sync.token)}</code></td></tr></table>
    <h3>Scanning tips</h3><ul class="tips"><li>Camera ~chest height, 2.5–3.5 m away, whole body in frame.</li>
      <li>Fitted clothes (or shorts / sports bra), hair off the neck.</li><li>Stand tall, feet hip-width, arms ~30° out.</li>
      <li>Hold ~5 s facing the camera, then turn 90° for ~3 s.</li><li>Same time of day (e.g. morning, before eating) for clean trends.</li></ul>
    <p class="muted">Video never leaves this Mac or gets saved — only the measurements.</p>`;
}
function renderPanel() {
  if (!view) return ($('#panel').hidden = true);
  $('#panel').hidden = false;
  $('#pbody').innerHTML = view.part ? partPanel(view.part)
    : { log: logPanel, goals: goalsPanel, insights: insightsPanel, profile: profilePanel }[view.tab]();
  document.querySelectorAll('#nav button').forEach((b) => b.setAttribute('aria-pressed', b.dataset.tab === view.tab));
}
function open(v) {
  view = v;
  selectedPart = v?.part && v.part !== 'overall' ? v.part : null;
  renderPanel();
  paint();
}
$('#nav').addEventListener('click', (e) => { const t = e.target.closest('[data-tab]')?.dataset.tab; if (t) open(view?.tab === t ? null : { tab: t }); });
$('#close').addEventListener('click', () => open(null));
addEventListener('keydown', (e) => e.key === 'Escape' && open(null));
$('#panel').addEventListener('change', (e) => {
  if (e.target.name !== 'units') return; // keep the typed height meaning the same thing
  const h = e.target.form.height, imp = e.target.value === 'imperial';
  if (h.value) h.value = (imp ? h.value / 2.54 : h.value * 2.54).toFixed(1);
  $('#hu').textContent = imp ? 'in' : 'cm';
});
$('#panel').addEventListener('submit', async (e) => {
  e.preventDefault();
  const f = e.target, fd = new FormData(f), o = Object.fromEntries(fd);
  try {
    if (f.id === 'fWorkout') await api('POST', '/api/logs', { kind: 'workout', ...o, focus: fd.getAll('focus') });
    else if (f.id === 'fDaily') await api('POST', '/api/logs', { kind: 'daily', ...o, weight: o.weight && fromUnit(+o.weight, 'kg') });
    else if (f.id === 'fProfile') await api('POST', '/api/profile', { ...o, height_cm: o.units === 'imperial' ? o.height * 2.54 : +o.height });
    else if (f.id === 'fGoal' || f.dataset.goal) {
      const k = f.dataset.goal || o.metric;
      await api('POST', '/api/goals', { metric: k, target: fromUnit(+o.target, META[k].unit) });
    } else if (f.dataset.cal) await api('POST', '/api/calibrate', { metric: f.dataset.cal, value: fromUnit(+o.value, 'cm') });
    await refresh();
    toast('Saved ✓');
  } catch (err) { toast(err.message, true); }
});
$('#panel').addEventListener('click', async (e) => {
  const b = e.target.closest('button[data-dellog],button[data-delscan],button[data-delgoal],button[data-calreset]');
  if (!b) return;
  const { dellog, delscan, delgoal, calreset } = b.dataset;
  if (delscan && !confirm('Delete this scan?')) return;
  try {
    if (dellog) await api('DELETE', `/api/logs/${dellog}`);
    if (delscan) await api('DELETE', `/api/scans/${delscan}`);
    if (delgoal) await api('DELETE', `/api/goals/${encodeURIComponent(delgoal)}`);
    if (calreset) await api('POST', '/api/calibrate', { metric: calreset, value: null });
    await refresh();
  } catch (err) { toast(err.message, true); }
});

// ---------- camera status ----------
$('#live').addEventListener('click', () => $('#cam').classList.toggle('big'));
$('#rescan').addEventListener('click', async () => { await api('POST', '/api/rescan', {}); toast('Ready — step in front of the camera'); });
async function poll() {
  try {
    const st = await (await fetch('/api/status')).json();
    $('#camMsg').textContent = st.msg;
    $('#dot').className = st.state;
    $('#barF').style.width = Math.min(1, st.front ?? 0) * (st.side ? 50 : 100) + '%';
    $('#barS').style.width = Math.min(1, st.side ?? 0) * 50 + '%';
    scanning = ['scanning', 'saved', 'live', 'adjust'].includes(st.state);
    const wasLive = !!live;
    live = st.live ?? null;
    if (st.scan_id && st.scan_id !== lastScanId || (st.state === 'done' && lastState !== 'done')) {
      if (st.scan_id !== lastScanId && lastScanId !== null) toast('Scan saved ✓');
      lastScanId = st.scan_id;
      idx = Infinity; // jump to newest
      await refresh();
    } else if (live || wasLive) renderAll(false);
    lastState = st.state;
  } catch { /* server restarting */ }
  setTimeout(poll, 700);
}

function renderAll(withPanel = true) {
  idx = Math.min(Math.max(idx, 0), S.scans.length - 1);
  renderHud();
  renderTimeline();
  rebuildAvatar();
  if (withPanel) renderPanel();
}
async function refresh() {
  const atEnd = !S || idx >= S.scans.length - 1;
  S = await (await fetch('/api/state')).json();
  if (atEnd) idx = S.scans.length - 1;
  renderAll();
}

await refresh();
$('#live').src = '/api/live.mjpg';
if (!S.profile?.height_cm) open({ tab: 'profile' });
poll();
