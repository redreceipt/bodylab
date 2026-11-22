import * as THREE from 'three';
import { OrbitControls } from 'three/addons/OrbitControls.js';
import { RoomEnvironment } from 'three/addons/RoomEnvironment.js';
import { fitBody, fitProxy, initBody, mesh } from './body.js';
import { loadImage, photoColors, skinCanvas } from './look.js';
import { renderSVG } from './vendor/uqr.mjs';

const $ = (s) => document.querySelector(s);
const esc = (s) => String(s).replace(/[&<>"']/g, (c) => `&#${c.charCodeAt(0)};`);
let S = null;      // /api/state
let live = null;   // {m, dims} while someone stands in front of the camera
let idx = -1;      // selected scan
let view = null;   // open panel: {tab} or {part}
let hovered = null, selectedPart = null, lastScanId = null, lastState = null, lastRev = null, lastDimsKey = '';

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
// Daily inputs (manual or Apple Health): [label, unit, decimals]
const HEALTH = {
  scale_bf: ['Smart-scale body fat', '%', 1], scale_lean: ['Smart-scale lean mass', 'kg', 1], tape_waist: ['Tape waist', 'cm', 1],
  kcal: ['Calories eaten', 'kcal', 0], protein: ['Protein', 'g', 0], carbs: ['Carbs', 'g', 0], fat: ['Fat', 'g', 0], fiber: ['Fiber', 'g', 0],
  sugar: ['Sugar', 'g', 0], water: ['Water', 'L', 1], alcohol: ['Alcoholic drinks', '', 1], active: ['Active energy', 'kcal', 0],
  basal: ['Resting energy', 'kcal', 0], steps: ['Steps', '', 0], exercise: ['Exercise', 'min', 0], rhr: ['Resting heart rate', 'bpm', 0],
  hrv: ['Heart-rate variability', 'ms', 0], vo2max: ['VO₂ max', 'ml/kg/min', 1], walk_hr: ['Walking heart rate', 'bpm', 0],
  resp: ['Respiratory rate', '/min', 1], sleep: ['Sleep', 'h', 1], sleep_deep: ['Deep sleep', 'h', 1], sleep_rem: ['REM sleep', 'h', 1],
  balance: ['Energy balance', 'kcal', 0],
};
for (const [k, [label, unit, dec]] of Object.entries(HEALTH)) META[k] ??= { label, unit, dec };
const GROUPS = [['Body', ['weight', 'scale_bf', 'scale_lean', 'tape_waist']], ['Fuel', ['kcal', 'protein', 'carbs', 'fat', 'fiber', 'sugar', 'water', 'alcohol']],
  ['Activity', ['active', 'basal', 'steps', 'exercise']], ['Heart & recovery', ['rhr', 'hrv', 'vo2max', 'walk_hr', 'resp']], ['Sleep', ['sleep', 'sleep_deep', 'sleep_rem']]];
function healthTable() {
  const H = S.insights.health, rows = GROUPS.map(([g, keys]) => {
    const ks = keys.filter((k) => H[k]?.avg != null);
    return ks.length ? `<tr><th colspan="4">${g}</th></tr>` + ks.map((k) => `<tr><td>${META[k].label}</td><td>${fmt(k, H[k].avg)}</td>
      <td class="muted">${H[k].prev != null ? fmt(k, H[k].avg - H[k].prev, true) : ''}</td>
      <td class="${H[k].days < 4 ? 'bad' : 'muted'}" title="days with data in the last week">${H[k].days}/7 d</td></tr>`).join('') : '';
  }).join('');
  return rows ? `<h4>Apple Health · 7-day average (change vs the week before · days with data)</h4><table>${rows}</table>
    <p class="muted">Today and partly logged food days are left out, so gaps don't drag averages down.</p>`
    : '<p class="muted">Pair the iPhone app (Profile) to bring in Apple Health data.</p>';
}
const imperial = () => S?.profile?.units === 'imperial';
const K = { cm: 1 / 2.54, kg: 2.20462 };
const toUnit = (v, u) => (imperial() && K[u] ? v * K[u] : v);
const fromUnit = (v, u) => (imperial() && K[u] ? v / K[u] : v);
const uLabel = (u) => (imperial() ? { cm: 'in', kg: 'lb' }[u] ?? u : u);
function fmt(k, v, signed = false) {
  if (v == null || Number.isNaN(v)) return '—';
  const { unit = '', dec = 1 } = META[k] || {};
  const x = toUnit(v, unit), u = uLabel(unit);
  return (signed && x > 0 ? '+' : '') + x.toLocaleString(undefined, { minimumFractionDigits: dec, maximumFractionDigits: dec }) + (u === '%' ? '%' : u ? ' ' + u : '');
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
controls.target.set(0, 0.88, 0);
controls.addEventListener('start', () => { controls.autoRotate = false; focus.target = null; });
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
const A = initBody(...await Promise.all([fetch('/static/assets/body.json').then((r) => r.json()),
  fetch('/static/assets/body.bin').then((r) => r.arrayBuffer())]));
renderer.toneMapping = THREE.ACESFilmicToneMapping;
scene.environment = new THREE.PMREMGenerator(renderer).fromScene(new RoomEnvironment(), 0.04).texture;
scene.environmentIntensity = 0.45;

const textures = {};
const texture = (url) => (textures[url] ??= Object.assign(new THREE.TextureLoader().load(url), { colorSpace: THREE.SRGBColorSpace, anisotropy: 8 }));
const geometry = (pos, src, uv, tris) => {
  const m = mesh(pos, src, tris), g = new THREE.BufferGeometry();
  g.setAttribute('position', new THREE.BufferAttribute(m.position, 3));
  g.setAttribute('normal', new THREE.BufferAttribute(m.normal, 3));
  g.setAttribute('uv', new THREE.BufferAttribute(uv, 2));
  g.setIndex(new THREE.BufferAttribute(tris, 1));
  return g;
};
const proxy = (name, pos) => geometry(fitProxy(name, pos), A.arr(`${name}.src`), A.arr(`${name}.uv`), A.arr(`${name}.tris`));
const HAIRS = { short02: 'Short', short04: 'Buzz', bob02: 'Bob', long01: 'Long', ponytail01: 'Ponytail', afro01: 'Afro', braid01: 'Braid' };
function glowMaterial(opacity) {
  return new THREE.MeshBasicMaterial({ vertexColors: true, transparent: true, opacity, depthWrite: false,
    polygonOffset: true, polygonOffsetFactor: -1, polygonOffsetUnits: -4 });
}
const strands = (map, color) => new THREE.MeshStandardMaterial({ map, color, alphaTest: 0.4, alphaToCoverage: true, side: THREE.DoubleSide, roughness: 0.7 });
const mats = {
  skin: new THREE.MeshPhysicalMaterial({ vertexColors: true, roughness: 0.6, sheen: 0.3, sheenRoughness: 0.7, sheenColor: '#ffcfb8' }),
  // Additive glow over the selected / hovered part, so the skin itself stays untouched.
  selected: glowMaterial(0.45), hovered: glowMaterial(0.25),
  hair: strands(null, '#3b2a20'), brows: strands(texture('/static/assets/eyebrow001.webp'), '#3b2a20'),
  lashes: strands(texture('/static/assets/eyelashes01.webp'), '#ffffff'),
  eyes: new THREE.MeshPhysicalMaterial({ roughness: 0.05, clearcoat: 1 }),
  ghost: new THREE.MeshBasicMaterial({ color: '#3ef0ff', transparent: true, opacity: 0.22, depthWrite: false,
    polygonOffset: true, polygonOffsetFactor: 2, polygonOffsetUnits: 8 }),
};
let look = { hair: 'short02', brows: true }, lookKey = '';
// Skin texture, face photo, hair and eyes. Rebuilt only when the profile's look or the face photo changes.
async function updateLook() {
  const p = S.profile || {}, sex = p.sex === 'female' ? 'female' : 'male', face = S.face;
  const key = JSON.stringify([sex, p.skin, p.hair, p.hair_color, p.eye_color, face?.ts]);
  if (key === lookKey) return;
  lookKey = key;
  const photo = face && await loadImage(`/api/face.jpg?${face.ts}`);
  const auto = face ? photoColors(photo, face) : {}, hair = p.hair ?? (sex === 'female' ? 'long01' : 'short02');
  const hairColor = p.hair_color ?? auto.hair ?? '#3b2a20';
  const skin = new THREE.CanvasTexture(await skinCanvas(sex, { face, photo, tone: p.skin, hairColor: HAIRS[hair] ? hairColor : null }));
  Object.assign(skin, { colorSpace: THREE.SRGBColorSpace, anisotropy: 8 });
  mats.skin.map?.dispose();
  mats.skin.map = skin;
  mats.skin.needsUpdate = true;
  look = { hair, brows: !face, auto };
  if (HAIRS[hair]) mats.hair.map = texture(`/static/assets/${hair}.webp`);
  for (const m of [mats.hair, mats.brows]) m.color.set(hairColor);
  mats.eyes.map = texture(`/static/assets/eye_${p.eye_color && p.eye_color !== 'auto' ? p.eye_color : auto.eye ?? 'brown'}.jpg`);
  mats.hair.needsUpdate = mats.eyes.needsUpdate = true;
  rebuildAvatar(true);
}

const COLORS = { cloth: new THREE.Color('#262b36'), good: new THREE.Color('#4cff9a'), bad: new THREE.Color('#ff6b6b'), sel: new THREE.Color('#3ef0ff'), white: new THREE.Color('#ffffff') };
function partColor(p, cloth) { // multiplies the skin texture: white = true skin colour
  const c = cloth ? COLORS.cloth.clone() : COLORS.white.clone(), t = S && p !== 'overall' && tone(p, S.insights.parts[p]?.change);
  return t ? c.lerp(COLORS[t], 0.35) : c;
}

function buildAvatar(dims, sex, isGhost) {
  // Ghost = first scan as a translucent shell that loses depth ties, so it only shows where the body has shrunk.
  // Shrinking it 6 mm per side hides noise-level differences (and z-fighting).
  const d = isGhost ? Object.fromEntries(Object.entries(dims || {}).map(([k, v]) => [k, /_[wd]$/.test(k) ? v - 1.2 : v])) : dims;
  const fit = fitBody(d, S.profile?.height_cm ?? 175, sex);
  const body = geometry(fit.pos, A.src, A.uv, A.tris);
  const group = new THREE.Group();
  if (isGhost) {
    group.add(new THREE.Mesh(body, mats.ghost));
  } else {
    body.setAttribute('color', new THREE.BufferAttribute(new Float32Array(A.src.length * 3), 3));
    body.userData = { parts: fit.parts, cloth: fit.cloth };
    const skin = new THREE.Mesh(body, mats.skin), glow = (m) => {
      const g = new THREE.BufferGeometry();
      g.setAttribute('position', body.attributes.position);
      g.setAttribute('color', new THREE.BufferAttribute(new Float32Array(A.src.length * 4), 4)); // cyan, alpha fades at the edge
      return Object.assign(new THREE.Mesh(g, m), { renderOrder: 1 });
    };
    skin.castShadow = true;
    group.userData.glow = { selected: glow(mats.selected), hovered: glow(mats.hovered) };
    group.add(skin, group.userData.glow.selected, group.userData.glow.hovered, new THREE.Mesh(proxy('high-poly', fit.pos), mats.eyes), new THREE.Mesh(proxy('eyelashes01', fit.pos), mats.lashes));
    if (look.brows) group.add(new THREE.Mesh(proxy('eyebrow001', fit.pos), mats.brows));
    if (HAIRS[look.hair]) {
      const hair = new THREE.Mesh(proxy(look.hair, fit.pos), mats.hair);
      hair.castShadow = true;
      group.add(hair);
    }
  }
  group.scale.setScalar(0.01); // cm → m
  return group;
}

function paint() {
  const g = avatar?.children[0].geometry, col = g?.attributes.color;
  if (!col) return;
  const { parts, cloth } = g.userData, cache = {};
  for (let r = 0; r < A.src.length; r++) {
    const i = A.src[r], [a, b] = (cache[parts[i]] ??= [partColor(parts[i], false), partColor(parts[i], true)]), k = cloth[i];
    col.setXYZ(r, a.r + (b.r - a.r) * k, a.g + (b.g - a.g) * k, a.b + (b.b - a.b) * k);
  }
  col.needsUpdate = true;
  for (const [which, part] of [['selected', selectedPart], ['hovered', hovered !== selectedPart && hovered]]) {
    const g = avatar.userData.glow[which].geometry, rgba = g.attributes.color, on = (r) => parts[A.src[r]] === part;
    for (let r = 0; r < A.src.length; r++) rgba.setXYZW(r, 0.02, 0.6, 0.9, on(r) ? 1 : 0);
    rgba.needsUpdate = true;
    const tris = [];
    if (part) for (let t = 0; t < A.tris.length; t += 3) if (on(A.tris[t]) || on(A.tris[t + 1]) || on(A.tris[t + 2])) tris.push(A.tris[t], A.tris[t + 1], A.tris[t + 2]);
    g.setIndex(tris);
  }
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
  paint();
}

// Camera glides to the selected part (close-up for the head); closing the panel returns to the full body.
const focus = { target: null, dist: 0, front: false };
function focusOn(part) {
  const H = (S.profile?.height_cm ?? 175) / 100;
  focus.target = new THREE.Vector3(0, H / 2, 0);
  focus.dist = 4.6;
  focus.front = part === 'overall'; // face close-ups turn to the front
  if (part) controls.autoRotate = false;
  const g = part && avatar?.children[0].geometry;
  if (!g) return;
  const sum = new THREE.Vector3(), p = g.attributes.position;
  let n = 0;
  for (let r = 0; r < A.src.length; r++) if (g.userData.parts[A.src[r]] === part) { sum.add(new THREE.Vector3().fromBufferAttribute(p, r)); n++; }
  if (n) { focus.target = sum.divideScalar(n * 100); focus.dist = part === 'overall' ? 0.75 : 2.2; }
  if (part === 'overall') focus.target.y += 0.02;
}

function resize() {
  renderer.setSize(innerWidth, innerHeight, false);
  camera.aspect = innerWidth / innerHeight;
  camera.updateProjectionMatrix();
}
addEventListener('resize', resize);
resize();
renderer.setAnimationLoop((t) => {
  const H = (S?.profile?.height_cm ?? 175) / 100;
  scanRing.visible = scanning;
  if (scanning) scanRing.position.y = H * (0.5 + 0.48 * Math.sin(t / 700));
  halo.material.opacity = 0.5 + 0.25 * Math.sin(t / 900);
  if (focus.target) {
    const offset = camera.position.clone().sub(controls.target), len = offset.length();
    if (focus.front) offset.lerp(new THREE.Vector3(0, 0.08, 1).setLength(len), 0.08);
    controls.target.lerp(focus.target, 0.08);
    camera.position.copy(controls.target).add(offset.setLength(len + (focus.dist - len) * 0.08));
    if (controls.target.distanceTo(focus.target) < 0.002 && Math.abs(len - focus.dist) < 0.01) focus.target = null;
  }
  controls.update();
  renderer.render(scene, camera);
});

// picking
const ray = new THREE.Raycaster(), ndc = new THREE.Vector2();
function pick(e) {
  if (!avatar) return null;
  ndc.set((e.clientX / innerWidth) * 2 - 1, -(e.clientY / innerHeight) * 2 + 1);
  ray.setFromCamera(ndc, camera);
  const hit = ray.intersectObjects(avatar.children, false)[0];
  return hit ? hit.object.geometry.userData.parts?.[A.src[hit.face.a]] ?? 'overall' : null;
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
      ${healthTable()}
      <p class="muted">Body fat uses the US Navy circumference formula (height, neck, waist${S.profile?.sex === 'female' ? ', hips' : ''}). Expect ±3–4 points; tape-calibrating neck and waist helps most. Mass values need a weigh-in within 7 days of the scan.</p>`;
  }
  const p = S.insights.parts[k] || {}, cal = S.profile?.calib?.[k], dims = (live?.dims ?? S.scans[idx]?.dims) || {};
  const torso = ['neck', 'chest', 'waist', 'hips'].includes(k) && dims[`${k}_w`] ? `<p class="muted">Measured width ${fmt('neck', dims[`${k}_w`])}, depth ${fmt('neck', dims[`${k}_d`])} ${S.scans[idx]?.side && !live ? '(side view)' : '(estimated — turn sideways during a scan to measure it)'}.</p>` : '';
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
      l.sleep && `${l.sleep.toFixed(1)} h sleep`, l.steps && `${Math.round(l.steps)} steps`, l.active && `${Math.round(l.active)} active kcal`].filter(Boolean).join(' · ');
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
    <h3>Scans</h3>${S.hidden_scans ? `<p class="muted">${S.hidden_scans} scan${S.hidden_scans > 1 ? 's' : ''} hidden: the proportions weren't human (a misread pose).</p>` : ''}<ul class="list">${S.scans.slice(-10).reverse().map((s) => `<li><span>${new Date(s.ts * 1000).toLocaleString()}</span> · body fat ${fmt('bf', s.m.bf)}${s.side ? ' · front+side' : ''} <button class="link" data-delscan="${s.id}">delete</button></li>`).join('') || '<li class="muted">No scans yet</li>'}</ul>`;
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
    s.balance != null
      ? `<b>Energy balance</b> ${s.balance > 0 ? '+' : ''}${Math.round(s.balance)} kcal/day (${s.balance_days} fully logged days) → predicts <b>${fmt('weight', s.predicted_kg, true)}</b> over 28 days${c.weight != null ? `; measured trend ${fmt('weight', c.weight, true)}` : ''}.`
      : `<b>Energy balance</b>: needs 14+ fully logged days in the last 28 (${s.balance_days} so far).`,
    `<b>Outputs</b>: ${outs.length ? outs.join(', ') : 'need 2+ scans in this window'}.`,
    ...Object.entries(I.goals).filter(([, g]) => g.eta).map(([k, g]) => `At this pace: ${META[k].label} ${fmt(k, g.target)} by <b>${isoDay(g.eta)}</b>.`),
    ...I.notes];
  const cell = (k, v, dec = 0) => (v == null ? '—' : toUnit(v, META[k]?.unit).toFixed(META[k]?.dec ?? dec));
  const gaps = I.gaps.length ? `<div class="callout gaps"><b>Data gaps</b> — fill these for trustworthy results<ul>${I.gaps.map((g) => `<li>${esc(g)}</li>`).join('')}</ul></div>` : '';
  return `<h2>Inputs → Outputs</h2><div class="callout">${lines.map((l) => `<p>${l}</p>`).join('')}</div>${gaps}${weekChart(I.weeks)}
    <div class="scroll"><table><tr><th>Week of</th><th>Wkts</th><th>Load</th><th>kcal</th><th>Bal</th><th>Steps</th><th>Prot g</th><th>Sleep h</th><th>Wt ${uLabel('kg')}</th><th>BF %</th><th>Waist ${uLabel('cm')}</th></tr>
    ${I.weeks.map((w) => `<tr><td>${isoDay(w.week).replace(/, \d{4}$/, '')}</td><td>${w.workouts}</td><td>${w.load}</td><td>${cell(0, w.kcal)}</td><td>${cell(0, w.balance)}</td><td>${cell(0, w.steps)}</td>
      <td>${cell(0, w.protein)}</td><td>${cell(0, w.sleep, 1)}</td><td>${cell('weight', w.weight)}</td><td>${cell('bf', w.bf)}</td><td>${cell('waist', w.waist)}</td></tr>`).join('')}</table></div>
    <p class="muted">Correlations show up after ~4 weeks of scans and logs. They're hints, not proof.</p>`;
}
const EYES = ['brown', 'brownlight', 'blue', 'deepblue', 'bluegreen', 'green', 'grey', 'lightblue', 'ice'];
function pairBlock() {
  if (!S.sync.url) return '<p class="muted">Run <code>tailscale serve --bg --https=8443 http://127.0.0.1:8765</code> so your iPhone can reach BodyLab, then reload.</p>';
  const link = `bodylab://pair?server=${encodeURIComponent(S.sync.url)}&token=${encodeURIComponent(S.sync.token)}`;
  return `<div class="qr" title="Pairing code">${renderSVG(link, { pixelSize: 4 })}</div>
    <p class="muted">Scan with the iPhone Camera to pair the BodyLab Sync app. Apple Health then fills in your height, sex and age and keeps weight, food, activity, heart, sleep and workouts flowing in.</p>`;
}
function profilePanel() {
  const p = S.profile || {}, imp = p.units ? imperial() : true, first = !p.height_cm;
  const h = p.height_cm ? (imp ? p.height_cm / 2.54 : p.height_cm).toFixed(1) : '';
  const opt = (value, label, on) => `<option value="${value}" ${on ? 'selected' : ''}>${label}</option>`;
  return `<h2>${first ? 'Welcome to BodyLab' : 'Profile'}</h2>
    ${first ? `<p>Fastest setup: pair your iPhone and let Apple Health fill in the rest.</p>${pairBlock()}<p class="muted">Or enter your height yourself:</p>` : ''}
    <form id="fProfile" class="stack">
      <label>Units<select name="units">${opt('imperial', 'Imperial (in, lb)', imp)}${opt('metric', 'Metric (cm, kg)', !imp)}</select></label>
      <label><span>Height (<span id="hu">${imp ? 'in' : 'cm'}</span>)</span><input name="height" type="number" step="any" value="${h}" required></label>
      <label>Sex (body shape &amp; body-fat formula)<select name="sex">${opt('male', 'Male')}${opt('female', 'Female', p.sex === 'female')}</select></label>
      ${p.birth_date ? `<p class="muted">Born ${esc(p.birth_date)} (from Apple Health)</p>` : ''}
      <label>Camera URL<input name="camera_url" value="${esc(p.camera_url ?? '')}" placeholder="http://&lt;pi-tailscale-ip&gt;:8000/stream.mjpg"></label>
      <label>Camera rotation<select name="rotate">${[0, 90, 180, 270].map((r) => opt(r, r ? r + '° clockwise' : 'None', (p.rotate ?? 0) === r)).join('')}</select></label>
      <button>Save</button></form>
    <h3>Scanning</h3>
    <form id="fScan" class="stack">
      <label class="check"><input type="checkbox" name="bare_only" ${p.bare_only === false ? '' : 'checked'}> Only count scans in minimal clothing (underwear is fine) — clothes inflate every measurement</label>
      <label>Live preview<select name="preview">${opt('silhouette', 'Silhouette only (private)', p.preview !== 'camera')}${opt('camera', 'Camera image (for aiming the camera)', p.preview === 'camera')}</select></label>
      <p class="muted">${S.face ? 'Scans are saved only for you: your face must match your photo, and only one person can be in the frame.' : 'Add a face photo below so scans are saved only for you. Until then, anyone alone in the frame gets scanned.'} Frames are never stored.</p>
      <button>Save</button></form>
    <h3>Look</h3>
    <div class="row" style="justify-content:flex-start">${S.face ? `<img class="facepic" src="/api/face.jpg?${S.face.ts}" alt="Your face photo">` : '<div class="facepic">no photo</div>'}
      <label class="filebtn">${S.face ? 'Replace' : 'Add'} face photo<input type="file" id="faceFile" accept="image/*" hidden></label>
      ${S.face ? '<button type="button" class="link" data-delface="1">remove</button>' : ''}</div>
    <p class="muted">Straight on, evenly lit, neutral face with mouth closed, like a passport photo (the iPhone app can take it). It's mapped onto your 3D face and sets skin, eye and hair colour. Stays on this Mac.</p>
    <form id="fLook" class="stack"><div class="grid2">
      <label>Hair<select name="hair">${[['none', 'None'], ...Object.entries(HAIRS)].map(([k, v]) => opt(k, v, look.hair === k)).join('')}</select></label>
      <label>Hair colour<input type="color" name="hair_color" value="${esc(p.hair_color ?? look.auto?.hair ?? '#3b2a20')}"></label>
      <label>Eyes<select name="eye_color">${opt('auto', S.face ? 'From photo' : 'Brown', !p.eye_color || p.eye_color === 'auto')}${EYES.map((e) => opt(e, e, p.eye_color === e)).join('')}</select></label>
      <label>Skin tone${S.face ? ' (from photo)' : ''}<input type="color" name="skin" value="${esc(p.skin ?? '#c99478')}" ${S.face ? 'disabled' : ''}></label>
    </div><button>Save look</button></form>
    ${first ? '' : `<h3>iPhone · Apple Health</h3>${pairBlock()}`}
    <h3>Scanning tips</h3><ul class="tips"><li>Camera ~chest height, 2.5–3.5 m away, whole body in frame.</li>
      <li>Minimal clothing (underwear / sports bra) — clothes inflate every measurement. Hair off the neck.</li><li>Stand tall, feet hip-width, arms ~30° out.</li>
      <li>Hold ~5 s facing the camera, then turn 90° for ~3 s.</li><li>Same time of day (e.g. morning, before eating) for clean trends.</li></ul>
    <p class="muted">Camera video never leaves this Mac or gets saved — only the measurements.</p>`;
}
function renderPanel() {
  if (!view) return ($('#panel').hidden = true);
  $('#panel').hidden = false;
  $('#pbody').innerHTML = view.part ? partPanel(view.part)
    : { log: logPanel, goals: goalsPanel, insights: insightsPanel, profile: profilePanel }[view.tab]();
  document.querySelectorAll('#nav button').forEach((b) => b.setAttribute('aria-pressed', b.dataset.tab === view.tab));
}
function open(v) {
  if (v?.part || (!v && view?.part)) focusOn(v?.part);
  view = v;
  selectedPart = v?.part && v.part !== 'overall' ? v.part : null;
  renderPanel();
  paint();
}
$('#nav').addEventListener('click', (e) => { const t = e.target.closest('[data-tab]')?.dataset.tab; if (t) open(view?.tab === t ? null : { tab: t }); });
$('#close').addEventListener('click', () => open(null));
addEventListener('keydown', (e) => e.key === 'Escape' && open(null));
$('#panel').addEventListener('change', async (e) => {
  if (e.target.id === 'faceFile' && e.target.files[0]) {
    toast('Reading your face…');
    const r = await fetch('/api/face', { method: 'POST', headers: { 'Content-Type': e.target.files[0].type || 'image/jpeg' }, body: e.target.files[0] });
    const j = await r.json().catch(() => ({}));
    if (!r.ok) return toast(j.error || 'Upload failed', true);
    toast('Face photo saved ✓');
    return refresh();
  }
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
    else if (f.id === 'fLook') await api('POST', '/api/profile', o);
    else if (f.id === 'fScan') await api('POST', '/api/profile', { bare_only: fd.has('bare_only'), preview: o.preview });
    else if (f.id === 'fGoal' || f.dataset.goal) {
      const k = f.dataset.goal || o.metric;
      await api('POST', '/api/goals', { metric: k, target: fromUnit(+o.target, META[k].unit) });
    } else if (f.dataset.cal) await api('POST', '/api/calibrate', { metric: f.dataset.cal, value: fromUnit(+o.value, 'cm') });
    await refresh();
    toast('Saved ✓');
  } catch (err) { toast(err.message, true); }
});
$('#panel').addEventListener('click', async (e) => {
  const b = e.target.closest('button[data-dellog],button[data-delscan],button[data-delgoal],button[data-calreset],button[data-delface]');
  if (!b) return;
  const { dellog, delscan, delgoal, calreset, delface } = b.dataset;
  if (delscan && !confirm('Delete this scan?')) return;
  try {
    if (dellog) await api('DELETE', `/api/logs/${dellog}`);
    if (delscan) await api('DELETE', `/api/scans/${delscan}`);
    if (delgoal) await api('DELETE', `/api/goals/${encodeURIComponent(delgoal)}`);
    if (calreset) await api('POST', '/api/calibrate', { metric: calreset, value: null });
    if (delface) await api('DELETE', '/api/face');
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
    } else if (st.rev !== lastRev && lastRev !== null) {
      await refresh(!$('#panel').contains(document.activeElement)); // iPhone sync / face photo landed; don't clobber typing
    } else if (live || wasLive) renderAll(false);
    lastState = st.state;
    lastRev = st.rev;
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
async function refresh(withPanel = true) {
  const atEnd = !S || idx >= S.scans.length - 1;
  S = await (await fetch('/api/state')).json();
  if (atEnd) idx = S.scans.length - 1;
  renderAll(withPanel);
  updateLook().catch((e) => toast(`Couldn't build the avatar's look: ${e.message}`, true));
}

await refresh();
$('#live').src = '/api/live.mjpg';
if (!S.profile?.height_cm) open({ tab: 'profile' });
poll();
