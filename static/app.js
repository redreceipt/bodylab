import * as THREE from 'three';
import { OrbitControls } from 'three/addons/OrbitControls.js';
import { RoomEnvironment } from 'three/addons/RoomEnvironment.js';
import { drape, fitBody, fitProxy, genitalPositions, initBody, mesh, pose } from './body.js';
import { loadImage, photoColors, skinCanvas } from './look.js';
import { renderSVG } from './vendor/uqr.mjs';

const $ = (s) => document.querySelector(s);
const esc = (s) => String(s).replace(/[&<>"']/g, (c) => `&#${c.charCodeAt(0)};`);
let S = null;      // /api/state
let live = null;   // {m, dims} while someone stands in front of the camera
let idx = -1;      // selected scan
let view = null;   // open panel: {tab} or {part}
let hovered = null, selectedPart = null, lastRev = null, lastDimsKey = '';

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
Object.assign(HEALTH, { underbust: ['Underbust', 'cm', 1], tape_bust: ['Bust (tape)', 'cm', 1], tape_underbust: ['Underbust (tape)', 'cm', 1] });
for (const [k, [label, unit, dec]] of Object.entries(HEALTH)) META[k] ??= { label, unit, dec };
// US cup letter from bust minus underbust (one letter per inch).
const cupLetter = (cm) => ['AA', 'A', 'B', 'C', 'D', 'DD', 'DDD/F', 'G', 'H'][Math.max(0, Math.min(8, Math.round(cm / 2.54)))];
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
const series = (k) => (LOGGED.has(k) ? S.logs.filter((l) => l[k] != null).map((l) => [l.ts, l[k]])
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
  ...Object.fromEntries(['shorts', 'bra'].map((g) => [g, new THREE.MeshStandardMaterial({ map: texture(`/static/assets/${g}.jpg`),
    normalMap: new THREE.TextureLoader().load(`/static/assets/${g}_normal.jpg`), roughness: 0.85, side: THREE.DoubleSide,
    polygonOffset: true, polygonOffsetFactor: -2, polygonOffsetUnits: -2 })])),
  genitals: new THREE.MeshPhysicalMaterial({ roughness: 0.6, sheen: 0.3, sheenRoughness: 0.7, sheenColor: '#ffcfb8' }),
  ghost: new THREE.MeshBasicMaterial({ color: '#3ef0ff', transparent: true, opacity: 0.22, depthWrite: false,
    polygonOffset: true, polygonOffsetFactor: 2, polygonOffsetUnits: 8 }),
};
let look = { hair: 'short02', brows: true, clothes: false }, lookKey = '';
// Skin texture, face photo, hair and eyes. Rebuilt only when the profile's look or the face photo changes.
async function updateLook() {
  const p = S.profile || {}, sex = p.sex === 'female' ? 'female' : 'male', face = S.face;
  look.clothes = p.clothes === true; // nude unless censored
  $('#censored').checked = look.clothes;
  const key = JSON.stringify([sex, p.skin, p.hair, p.hair_color, p.eye_color, face?.ts]);
  if (key === lookKey) return;
  lookKey = key;
  const photo = face && await loadImage(`/api/face.jpg?${face.ts}`);
  const auto = { ...(face ? photoColors(photo, face) : {}), hair: face?.hair?.color };
  const hair = p.hair ?? face?.hair?.style ?? (sex === 'female' ? 'long01' : 'short02'), hairColor = p.hair_color ?? auto.hair ?? '#3b2a20';
  const { canvas: skinTex, tone } = await skinCanvas(sex, { face, photo, tone: p.skin, hairColor: HAIRS[hair] ? hairColor : null });
  mats.genitals.color.set(`rgb(${tone.map(Math.round).join(',')})`).multiplyScalar(0.93); // a touch darker than the face's tone
  const skin = new THREE.CanvasTexture(skinTex);
  Object.assign(skin, { colorSpace: THREE.SRGBColorSpace, anisotropy: 8 });
  mats.skin.map?.dispose();
  mats.skin.map = skin;
  mats.skin.needsUpdate = true;
  look = { hair, brows: !face, auto, clothes: look.clothes };
  if (HAIRS[hair]) mats.hair.map = texture(`/static/assets/${hair}.webp`);
  for (const m of [mats.hair, mats.brows]) m.color.set(hairColor).multiplyScalar(3.5); // strand textures average ~0.28 (linear); this restores the chosen colour
  mats.eyes.map = texture(`/static/assets/eye_${p.eye_color && p.eye_color !== 'auto' ? p.eye_color : auto.eye ?? 'brown'}.jpg`);
  mats.hair.needsUpdate = mats.eyes.needsUpdate = true;
  rebuildAvatar(true);
}

const COLORS = { good: new THREE.Color('#4cff9a'), bad: new THREE.Color('#ff6b6b'), white: new THREE.Color('#ffffff') };
function partColor(p) { // multiplies the skin texture: white = true skin colour
  const t = S && p !== 'overall' && tone(p, S.insights.parts[p]?.change);
  return t ? COLORS.white.clone().lerp(COLORS[t], 0.35) : COLORS.white;
}

// Breast shape: your latest tape measurements, else the camera's bust and underbust.
const latest = (k) => S.logs.findLast((l) => l[k] != null)?.[k];
function shape(m) {
  const bust = latest('tape_bust') ?? m.chest, under = latest('tape_underbust') ?? m.underbust;
  return { cup: bust && under ? bust - under : null };
}
// Body triangles under the worn garments are dropped so skin can't poke through.
const dressed = {};
function bodyTris(garments) {
  if (!garments.length) return A.tris;
  return (dressed[garments] ??= (() => {
    const hide = new Set(garments.flatMap((g) => Array.from(A.arr(`${g}.hide`))));
    return A.tris.filter((_, t) => { const b = t - (t % 3); return ![0, 1, 2].every((k) => hide.has(A.src[A.tris[b + k]])); });
  })());
}

// A day's fitted body: rest mesh (int16 mm) and its stance's bone transforms, fetched once with the shared skin
// weights; the avatar is rebuilt when they arrive.
const fits = {};
let skin = null;
function fitted(scan) {
  if (!scan?.fit) return null;
  if (!(scan.day in fits)) {
    fits[scan.day] = null;
    Promise.all([fetch(`/api/fit/${scan.day}`).then((r) => (r.ok ? r.arrayBuffer() : null)),
      skin ?? fetch('/api/skin').then((r) => (r.ok ? r.arrayBuffer() : null))]).then(([b, s]) => {
      if (!b) return;
      skin = s && new Uint8Array(s);
      const n = A.n * 3;
      fits[scan.day] = { rest: Float32Array.from(new Int16Array(b, 0, n), (v) => v / 10), G: b.byteLength > n * 2 ? new Float32Array(b.slice(n * 2)) : null };
      rebuildAvatar(true);
    });
  }
  return fits[scan.day];
}

function buildAvatar(dims, sex, isGhost, m = {}, mesh = null, ref = {}) {
  // Ghost = first scan as a translucent shell that loses depth ties, so it only shows where the body has shrunk.
  // Shrinking it 6 mm per side hides noise-level differences (and z-fighting).
  const d = isGhost ? Object.fromEntries(Object.entries(dims || {}).map(([k, v]) => [k, /_[wd]$/.test(k) ? v - 1.2 : v])) : dims;
  const garments = look.clothes && !isGhost ? (sex === 'female' ? ['shorts', 'bra'] : ['shorts']) : [];
  const fit = fitBody(d, S.profile?.height_cm ?? 175, sex, shape(m), mesh?.rest, ref);
  if (mesh?.G && skin) fit.pos = pose(fit.pos, mesh.G, skin); // relaxed stance instead of the rest pose's spread arms
  const body = geometry(fit.pos, A.src, A.uv, bodyTris(garments));
  const group = new THREE.Group();
  if (isGhost) {
    group.add(new THREE.Mesh(body, mats.ghost));
  } else {
    body.setAttribute('color', new THREE.BufferAttribute(new Float32Array(A.src.length * 3), 3));
    body.userData = { parts: fit.parts };
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
    for (const g of garments) {
      const mesh = geometry(drape(g, fit.pos, fitProxy(g, fit.pos)), A.arr(`${g}.src`), A.arr(`${g}.uv`), A.arr(`${g}.tris`));
      group.add(Object.assign(new THREE.Mesh(mesh, mats[g]), { castShadow: true }));
    }
    if (!look.clothes && sex !== 'female') {
      const g = new THREE.BufferGeometry();
      g.setAttribute('position', new THREE.BufferAttribute(genitalPositions(fit.pos), 3));
      g.setIndex(new THREE.BufferAttribute(A.arr('genital.tris'), 1));
      g.computeVertexNormals();
      group.add(new THREE.Mesh(g, mats.genitals));
    }
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
  const { parts } = g.userData, cache = {};
  for (let r = 0; r < A.src.length; r++) {
    const c = (cache[parts[A.src[r]]] ??= partColor(parts[A.src[r]]));
    col.setXYZ(r, c.r, c.g, c.b);
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
  const sex = S.profile?.sex, today = live?.dims ?? S.scans[idx]?.dims ?? {}, dims = predicted?.dims ?? today;
  const m = predicted?.m ?? live?.m ?? S.scans[idx]?.m ?? {};
  const mesh = fitted(predicted || live ? S.scans.at(-1) : S.scans[idx]); // predictions reshape today's fit
  const key = JSON.stringify([dims, sex, S.profile?.height_cm, look.clothes, shape(m), !!predicted, !!mesh]);
  if (!force && key === lastDimsKey) return paint();
  lastDimsKey = key;
  for (const g of [avatar, ghost]) if (g) { scene.remove(g); g.traverse((o) => o.geometry?.dispose()); }
  avatar = buildAvatar(dims, sex, false, m, mesh, predicted ? today : dims);
  scene.add(avatar);
  const showGhost = predicted || ($('#ghost').checked && S.scans.length > 1 && (live || idx > 0));
  ghost = showGhost ? (predicted ? buildAvatar(today, sex, true, {}, mesh, today) // predicted: outline = today
    : buildAvatar(S.scans[0].dims, sex, true, {}, fitted(S.scans[0]), S.scans[0].dims)) : null;
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
  return hit ? hit.object.userData.part ?? hit.object.geometry.userData.parts?.[A.src[hit.face.a]] ?? 'overall' : null;
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

// ---------- main screen: summary, today, timeline ----------
const LOGGED = new Set(['weight', 'tape_bust', 'tape_underbust']);
const valueOf = (k) => (LOGGED.has(k) ? latest(k) : cur()[k]);
function todayCard() {
  const t = S.insights.days.at(-1), usual = S.insights.usual_kcal;
  const items = [
    ['scan', 'Body scan', t.views ? `${t.views} views so far — keep walking past` : 'Walk past the camera; turning helps'],
    ['weight', 'Weigh-in', 'Step on the scale, or log it in BodyLab Sync'],
    ['food', 'Food', t.kcal ? `${Math.round(t.kcal).toLocaleString()}${usual ? ` of ~${Math.round(usual).toLocaleString()}` : ''} kcal logged` : 'Log meals in BodyLab Sync or any food app'],
    ['sleep', 'Sleep', 'Wear your Watch to bed'],
    ['activity', 'Activity', 'Wear your Watch'],
  ];
  const done = items.filter(([k]) => t[k]).length;
  return `<div class="row"><b>Today</b><span class="muted">${done} of ${items.length}</span></div>
    <ul>${items.map(([k, label, hint]) => `<li class="${t[k] ? 'ok' : ''}"><span>${t[k] ? '✓' : '○'}</span><div>${label}${t[k] ? '' : `<small>${hint}</small>`}</div></li>`).join('')}</ul>`;
}
function renderHud() {
  const m = cur(), c = S.insights.summary.changes;
  $('#summary').innerHTML = m.bf != null
    ? `<div class="k">Body fat</div><div class="v">${fmt('bf', m.bf)}</div>
       <div class="muted">${[valueOf('weight') != null && fmt('weight', valueOf('weight')), c.bf != null && `${fmt('bf', c.bf, true)} in 4 weeks`].filter(Boolean).join(' · ')}</div>`
    : '<div class="k">Your body</div><div class="muted">No estimate yet. Walk past the camera — every view helps.</div>';
  $('#today').innerHTML = todayCard();
}
function renderTimeline() {
  const n = S.scans.length;
  $('#scrub').max = Math.max(0, n - 1);
  $('#scrub').value = idx;
  $('#scrub').hidden = $('#ghostLabel').hidden = n < 2 || !!predicted;
  $('#scanLabel').textContent = predicted ? `predicted · in ${plan.weeks} weeks` : live ? 'today, live' : n ? day(S.scans[idx].ts) : 'no estimate yet';
}
$('#scrub').addEventListener('input', (e) => { idx = +e.target.value; renderAll(); });
$('#ghost').addEventListener('change', () => rebuildAvatar(true));
$('#censored').addEventListener('change', async (e) => {
  look.clothes = e.target.checked;
  rebuildAvatar(true);
  await api('POST', '/api/profile', { clothes: look.clothes });
});
$('#summary').addEventListener('click', () => open({ part: 'overall' }));

// ---------- panels ----------
function spark(k) {
  const pts = series(k);
  if (pts.length < 2) return '';
  const u = META[k].unit, ys = pts.map((p) => toUnit(p[1], u)), xs = pts.map((p) => p[0]);
  const g = S.goals[k] && toUnit(S.goals[k].target, u);
  const lo = Math.min(...ys, g ?? Infinity), hi = Math.max(...ys, g ?? -Infinity);
  const W = 360, H = 70, sx = (x) => 6 + ((x - xs[0]) / (xs.at(-1) - xs[0] || 1)) * (W - 12), sy = (y) => H - 8 - ((y - lo) / (hi - lo || 1)) * (H - 16);
  return `<svg class="spark" viewBox="0 0 ${W} ${H}" role="img" aria-label="${META[k].label} history">
    ${g != null ? `<line x1="0" x2="${W}" y1="${sy(g)}" y2="${sy(g)}"/>` : ''}
    <polyline points="${xs.map((x, i) => `${sx(x)},${sy(ys[i])}`).join(' ')}"/>
    <circle r="3.5" cx="${sx(xs.at(-1))}" cy="${sy(ys.at(-1))}"/></svg>`;
}
function goalBlock(k) {
  const g = S.goals[k], gi = S.insights.goals[k];
  if (!g) return `<form class="inline" data-goal="${k}"><input name="target" type="number" step="any" placeholder="Goal (${uLabel(META[k].unit) || 'value'})" required aria-label="Goal for ${META[k].label}"><button>Set goal</button></form>`;
  const pace = gi.progress === 1 ? '🎉 Goal reached'
    : gi.eta ? `On pace for <b>${isoDay(gi.eta)}</b> (${fmt(k, gi.rate_per_week, true)}/week)`
      : gi.rate_per_week != null ? `Trend ${fmt(k, gi.rate_per_week, true)}/week — not heading toward it yet` : 'A pace estimate needs a few days of data';
  return `<div class="goal"><div class="row"><span>Goal <b>${fmt(k, g.target)}</b></span><button class="link" data-delgoal="${k}">remove</button></div>
    <div class="bar"><i style="width:${(gi.progress ?? 0) * 100}%"></i></div><div class="muted">${pace}</div></div>`;
}
function metricBlock(k, withGoal = true) {
  const pts = series(k), now = valueOf(k), c28 = S.insights.parts[k]?.change ?? S.insights.summary.changes[k];
  const since = pts.length > 1 ? now - pts[0][1] : null;
  return `<div class="metric"><div class="row"><h3>${META[k].label}</h3><span class="big">${fmt(k, now)}</span></div>
    ${since != null || c28 != null ? `<div class="muted">${c28 != null ? `last 4 weeks <span class="${tone(k, c28)}">${fmt(k, c28, true)}</span>` : ''}${since != null ? ` · since you started <span class="${tone(k, since)}">${fmt(k, since, true)}</span>` : ''}</div>` : ''}
    ${spark(k)}${withGoal ? goalBlock(k) : ''}</div>`;
}
function partPanel(k) {
  if (k === 'overall') {
    const m = cur();
    return `<h2>Overall</h2>${['bf', 'weight', 'lean_kg'].map((x) => metricBlock(x)).join('')}
      <details><summary>More numbers</summary>
        <table>${['fat_kg', 'ffmi', 'whtr', 'whr'].map((x) => `<tr><td>${META[x].label}</td><td>${fmt(x, m[x])}</td></tr>`).join('')}</table>
        ${healthTable()}
        <p class="muted">Body fat uses the US Navy circumference formula. Expect ±3–4 points; tape-calibrating neck and waist (tap them) tightens it.</p>
      </details>`;
  }
  const p = S.insights.parts[k] || {}, cal = S.calibration?.[k];
  const breasts = k === 'chest' && S.profile?.sex === 'female' ? (() => {
    const m = cur(), bust = latest('tape_bust') ?? m.chest, under = latest('tape_underbust') ?? m.underbust;
    return `${metricBlock('underbust', false)}<p>${bust && under ? `Cup size about <b>${cupLetter(bust - under)}</b>.` : 'Cup size appears once bust and underbust are measured.'}</p>`;
  })() : '';
  return `${metricBlock(k, false)}${breasts}
    ${p.workouts || p.change != null ? `<p>${p.workouts ? `${p.workouts} workouts trained this in the last 4 weeks.` : ''}</p>` : ''}
    <details><summary>Goal</summary>${goalBlock(k)}</details>
    <p class="muted">${cal ? `Calibrated to your tape measurements (×${cal.toFixed(3)}).` : 'Tape-measure this in BodyLab Sync to calibrate the camera.'}</p>`;
}
function logItem(l) {
  const txt = l.kind === 'measure' ? '📏 ' + Object.keys(l).filter((x) => META[x]).map((x) => `${META[x].label} ${fmt(x, l[x])}`).join(' · ')
    : l.kind === 'workout'
    ? `🏋️ ${l.type} · ${l.minutes} min${l.focus?.length ? ' · ' + l.focus.join(', ') : ''}${l.notes ? ' · ' + l.notes : ''}`
    : '📋 ' + [l.weight && fmt('weight', l.weight), l.kcal && `${Math.round(l.kcal)} kcal`, l.protein && `${Math.round(l.protein)} g protein`,
      l.sleep && `${l.sleep.toFixed(1)} h sleep`, l.steps && `${Math.round(l.steps)} steps`].filter(Boolean).join(' · ');
  return `<li><span>${day(l.ts)}</span> ${esc(txt)} <button class="link" data-dellog="${l.id}">delete</button></li>`;
}
function weekChart(weeks) {
  const W = 380, H = 130, bw = W / weeks.length, maxL = Math.max(1, ...weeks.map((w) => w.load));
  const bfs = weeks.map((w) => w.bf).filter((v) => v != null), lo = Math.min(...bfs), hi = Math.max(...bfs);
  const y = (v) => H - 22 - ((v - lo) / (hi - lo || 1)) * (H - 44);
  const pts = weeks.map((w, i) => (w.bf == null ? null : [i * bw + bw / 2, y(w.bf)])).filter(Boolean);
  return `<svg class="chart" viewBox="0 0 ${W} ${H}" role="img" aria-label="Weekly training and body fat">
    ${weeks.map((w, i) => { const h = (w.load / maxL) * (H - 44); return `<rect x="${i * bw + 5}" y="${H - 16 - h}" width="${bw - 10}" height="${h}" rx="3"/>`; }).join('')}
    <polyline points="${pts.map((p) => p.join(',')).join(' ')}"/>${pts.map(([cx, cy]) => `<circle r="3" cx="${cx}" cy="${cy}"/>`).join('')}
    ${weeks.map((w, i) => `<text x="${i * bw + bw / 2}" y="${H - 3}">${w.week.slice(5)}</text>`).join('')}</svg>
    <div class="legend"><span><i></i>training</span><span><i class="l"></i>body fat</span></div>`;
}
function daysGrid() { // which inputs each of the last 14 days has, so gaps show while they can still be fixed
  const rows = [['scan', 'Scan'], ['weight', 'Weigh-in'], ['food', 'Food'], ['sleep', 'Sleep'], ['activity', 'Activity']];
  return `<table class="days"><tr><th></th>${S.insights.days.map((d) => `<th>${'SMTWTFS'[new Date(d.date + 'T12:00').getDay()]}</th>`).join('')}</tr>
    ${rows.map(([k, label]) => `<tr><td>${label}</td>${S.insights.days.map((d) => `<td title="${d.date}">${d[k] ? '●' : '<span class="miss">○</span>'}</td>`).join('')}</tr>`).join('')}</table>`;
}
function progressPanel() {
  const I = S.insights, s = I.summary, c = s.changes;
  const outs = ['bf', 'waist', 'weight', 'lean_kg'].filter((k) => c[k] != null).map((k) => `${META[k].label.toLowerCase()} <b class="${tone(k, c[k])}">${fmt(k, c[k], true)}</b>`);
  const lines = [outs.length ? `Last 4 weeks: ${outs.join(', ')}.` : 'Changes appear after a few days of body estimates.',
    s.balance != null ? `You ate about <b>${Math.abs(Math.round(s.balance))} kcal/day ${s.balance < 0 ? 'under' : 'over'}</b> what you burned, which predicts ${fmt('weight', s.predicted_kg, true)}${c.weight != null ? `; the scale says ${fmt('weight', c.weight, true)}` : ''}.` : '',
    ...Object.entries(I.goals).filter(([, g]) => g.eta).map(([k, g]) => `On pace: ${META[k].label.toLowerCase()} ${fmt(k, g.target)} by <b>${isoDay(g.eta)}</b>.`),
    ...I.notes].filter(Boolean);
  const free = Object.keys(META).filter((k) => !S.goals[k]);
  return `<h2>Progress</h2><div class="callout">${lines.map((l) => `<p>${l}</p>`).join('')}</div>
    ${I.gaps.length ? `<div class="callout gaps"><b>Missing data</b><ul>${I.gaps.map((g) => `<li>${esc(g)}</li>`).join('')}</ul></div>` : ''}
    <h3>Last 2 weeks</h3>${daysGrid()}
    <h3>Goals</h3>${Object.keys(S.goals).map((k) => metricBlock(k)).join('') || '<p class="muted">Goals colour the body: green = moving toward them, red = away.</p>'}
    <form id="fGoal" class="inline"><select name="metric" aria-label="Metric">${free.map((k) => `<option value="${k}">${META[k].label}</option>`).join('')}</select>
      <input name="target" type="number" step="any" placeholder="Target" required aria-label="Target"><button>Add</button></form>
    <details><summary>Training vs body fat, week by week</summary>${weekChart(I.weeks)}</details>
    <details><summary>Recent entries</summary><ul class="list">${S.logs.slice(-20).reverse().map(logItem).join('') || '<li class="muted">Nothing yet</li>'}</ul></details>`;
}
// What if: a plan's effect on your body, simulated from your own data.
const plan = { kcal: -300, strength: 3, protein: 1.6, weeks: 12 };
let predicted = null, predictTimer = null;
function whatifPanel() {
  const r = (name, label, min, max, step, unit) => `<label>${label} <b>${plan[name]}${unit}</b><input type="range" name="${name}" min="${min}" max="${max}" step="${step}" value="${plan[name]}"></label>`;
  return `<h2>What if…</h2><p class="muted">The avatar shows you after this plan; the outline is you today.</p>
    <form id="fPlan" class="stack">${r('kcal', 'Calories vs maintenance', -1000, 800, 50, ' kcal/day')}${r('strength', 'Strength workouts', 0, 6, 1, '/week')}
      ${r('protein', 'Protein', 0.8, 2.4, 0.1, ' g/kg')}${r('weeks', 'For', 1, 52, 1, ' weeks')}</form><div id="planOut" class="callout"><p class="muted">Simulating…</p></div>`;
}
async function runPlan() {
  try {
    const out = await api('POST', '/api/predict', plan), m = out.m;
    predicted = { dims: out.dims, m };
    $('#planOut').innerHTML = `${out.warning ? `<p class="bad"><b>${esc(out.warning)}</b></p>` : ''}<p>${['weight', 'bf', 'lean_kg', 'waist'].filter((k) => m[k] != null)
      .map((k) => `${META[k].label} <b>${fmt(k, m[k])}</b> (${fmt(k, m[k] - (out.start[k] ?? cur()[k] ?? m[k]), true)})`).join('<br>')}</p>
      <p class="muted">${out.maintenance ? `Your maintenance ≈ ${out.maintenance.toLocaleString()} kcal/day, ${out.maintenance_source}.` : 'Log food and weight in Apple Health and this learns your real maintenance.'}
      Each body part changes the way yours has, once there's enough history; until then, by typical patterns.</p>`;
  } catch (e) {
    predicted = null;
    $('#planOut').innerHTML = `<p>${esc(e.message)}</p>`;
  }
  renderTimeline();
  rebuildAvatar();
}
const EYES = ['brown', 'brownlight', 'blue', 'deepblue', 'bluegreen', 'green', 'grey', 'lightblue', 'ice'];
function pairBlock() {
  if (!S.sync.url) return '<p class="muted">Run <code>tailscale serve --bg --https=8443 http://127.0.0.1:8765</code> so your iPhone can reach BodyLab, then reload.</p>';
  const link = `bodylab://pair?server=${encodeURIComponent(S.sync.url)}&token=${encodeURIComponent(S.sync.token)}`;
  return `<div class="qr" title="Pairing code">${renderSVG(link, { pixelSize: 4 })}</div>
    <p class="muted">Scan with the iPhone Camera to pair BodyLab Sync. Apple Health fills in your height, sex and age, then keeps everything flowing in.</p>`;
}
function settingsPanel() {
  const p = S.profile || {}, imp = p.units ? imperial() : true, first = !p.height_cm;
  const h = p.height_cm ? (imp ? p.height_cm / 2.54 : p.height_cm).toFixed(1) : '';
  const opt = (value, label, on) => `<option value="${value}" ${on ? 'selected' : ''}>${label}</option>`;
  return `<h2>${first ? 'Welcome to BodyLab' : 'Settings'}</h2>
    ${first ? `<p>Install BodyLab Sync on your iPhone and scan this code. Everything about you comes from Apple Health and the app.</p>${pairBlock()}`
    : `<p class="muted">${[h && `${h} ${imp ? 'in' : 'cm'}`, p.sex, p.birth_date && `born ${esc(p.birth_date)}`, imp ? 'imperial units' : 'metric units'].filter(Boolean).join(' · ')}
       — from Apple Health and your phone. Log anything else in BodyLab Sync.</p>`}
    <details><summary>Your look</summary>
      <div class="row" style="justify-content:flex-start">${S.face ? `<img class="facepic" src="/api/face.jpg?${S.face.ts}" alt="Your face photo">` : '<div class="facepic">no photo</div>'}
        <label class="filebtn">${S.face ? 'Replace' : 'Add'} face photo<input type="file" id="faceFile" accept="image/*" hidden></label>
        ${S.face ? '<button type="button" class="link" data-delface="1">remove</button>' : ''}</div>
      <p class="muted">Straight on, even light, neutral face, mouth closed. Sets your face, skin, eyes and hair.</p>
      <form id="fLook" class="stack"><div class="grid2">
        <label>Hair<select name="hair">${[['none', 'None'], ...Object.entries(HAIRS)].map(([k, v]) => opt(k, v, look.hair === k)).join('')}</select></label>
        <label>Hair colour<input type="color" name="hair_color" value="${esc(p.hair_color ?? look.auto?.hair ?? '#3b2a20')}"></label>
        <label>Eyes<select name="eye_color">${opt('auto', S.face ? 'From photo' : 'Brown', !p.eye_color || p.eye_color === 'auto')}${EYES.map((e) => opt(e, e, p.eye_color === e)).join('')}</select></label>
        ${S.face ? '' : `<label>Skin tone<input type="color" name="skin" value="${esc(p.skin ?? '#c99478')}"></label>`}
      </div><button>Save look</button></form></details>
    <details><summary>Camera</summary>
      <form id="fCamera" class="stack">
        <label>Camera URL<input name="camera_url" value="${esc(p.camera_url ?? '')}" placeholder="http://&lt;pi-tailscale-ip&gt;:8000/stream.mjpg"></label>
        <label>Rotation<select name="rotate">${[0, 90, 180, 270].map((r) => opt(r, r ? r + '° clockwise' : 'None', (p.rotate ?? 0) === r)).join('')}</select></label>
        <label class="check"><input type="checkbox" name="bare_only" ${p.bare_only === false ? '' : 'checked'}> Only measure bare skin (underwear is fine): clothes inflate measurements</label>
        <label>Preview<select name="preview">${opt('silhouette', 'Silhouette only (private)', p.preview !== 'camera')}${opt('camera', 'Camera image (for aiming)', p.preview === 'camera')}</select></label>
        <button>Save</button></form>
      <p class="muted">Only you are measured (your face must match your photo; one person in view). Frames are never stored.</p>
      <button type="button" id="clear">Discard today's measurements</button></details>
    ${first ? '' : `<details><summary>iPhone and Apple Health</summary>${pairBlock()}</details>`}`;
}
function renderPanel() {
  if (!view) return ($('#panel').hidden = true);
  $('#panel').hidden = false;
  $('#pbody').innerHTML = view.part ? partPanel(view.part)
    : { progress: progressPanel, whatif: whatifPanel, settings: settingsPanel }[view.tab]();
  if (view.tab === 'whatif') runPlan();
  document.querySelectorAll('#nav button').forEach((b) => b.setAttribute('aria-pressed', b.dataset.tab === view.tab));
}
function open(v) {
  if (v?.part || (!v && view?.part)) focusOn(v?.part);
  if (view?.tab === 'whatif' && v?.tab !== 'whatif') { predicted = null; renderTimeline(); rebuildAvatar(); }
  view = v;
  selectedPart = v?.part && v.part !== 'overall' ? v.part : null;
  renderPanel();
  paint();
}
$('#nav').addEventListener('click', (e) => { const t = e.target.closest('[data-tab]')?.dataset.tab; if (t) open(view?.tab === t ? null : { tab: t }); });
$('#close').addEventListener('click', () => open(null));
addEventListener('keydown', (e) => e.key === 'Escape' && open(null));
$('#panel').addEventListener('input', (e) => {
  if (e.target.form?.id !== 'fPlan') return;
  plan[e.target.name] = +e.target.value;
  e.target.previousElementSibling.textContent = e.target.previousElementSibling.textContent.replace(/^-?[\d.]+/, e.target.value);
  clearTimeout(predictTimer);
  predictTimer = setTimeout(runPlan, 250);
});
$('#panel').addEventListener('change', async (e) => {
  if (e.target.id === 'faceFile' && e.target.files[0]) {
    toast('Reading your face…');
    const r = await fetch('/api/face', { method: 'POST', headers: { 'Content-Type': e.target.files[0].type || 'image/jpeg' }, body: e.target.files[0] });
    const j = await r.json().catch(() => ({}));
    if (!r.ok) return toast(j.error || 'Upload failed', true);
    toast('Face photo saved ✓');
    return refresh();
  }
});
$('#panel').addEventListener('submit', async (e) => {
  e.preventDefault();
  const f = e.target, fd = new FormData(f), o = Object.fromEntries(fd);
  try {
    if (f.id === 'fLook') await api('POST', '/api/profile', o);
    else if (f.id === 'fCamera') await api('POST', '/api/profile', { ...o, bare_only: fd.has('bare_only') });
    else if (f.id === 'fGoal' || f.dataset.goal) {
      const k = f.dataset.goal || o.metric;
      await api('POST', '/api/goals', { metric: k, target: fromUnit(+o.target, META[k].unit) });
    }
    await refresh();
    toast('Saved ✓');
  } catch (err) { toast(err.message, true); }
});
$('#panel').addEventListener('click', async (e) => {
  if (e.target.id === 'clear') {
    if (!confirm("Discard everything measured today? Earlier days aren't touched.")) return;
    await api('POST', '/api/clear-today', {});
    return toast("Today's measurements cleared");
  }
  const b = e.target.closest('button[data-dellog],button[data-delscan],button[data-delgoal],button[data-delface]');
  if (!b) return;
  const { dellog, delscan, delgoal, delface } = b.dataset;
  if (delscan && !confirm('Delete this scan?')) return;
  try {
    if (dellog) await api('DELETE', `/api/logs/${dellog}`);
    if (delscan) await api('DELETE', `/api/scans/${delscan}`);
    if (delgoal) await api('DELETE', `/api/goals/${encodeURIComponent(delgoal)}`);
    if (delface) await api('DELETE', '/api/face');
    await refresh();
  } catch (err) { toast(err.message, true); }
});

// ---------- camera status ----------
$('#cam').addEventListener('click', () => {
  const show = $('#preview').hidden;
  $('#preview').hidden = !show;
  $('#live').src = show ? '/api/live.mjpg' : ''; // only stream while you're looking
});
async function poll() {
  try {
    const st = await (await fetch('/api/status')).json();
    $('#camMsg').textContent = st.msg;
    $('#dot').className = st.state;
    $('#barF').style.width = Math.min(1, (st.frames ?? 0) / 150) * 50 + '%'; // views today (150 = plenty)
    $('#barS').style.width = Math.min(1, (st.angle ?? 0) / 75) * 50 + '%'; // how far you've turned: sets depth
    $('#evidence').title = `${st.frames ?? 0} views today, turned up to ${st.angle ?? 0}°`;
    scanning = st.state === 'scanning';
    const wasLive = !!live;
    live = st.live ?? null;
    if (st.rev !== lastRev && lastRev !== null) {
      if (live) idx = Infinity; // today's estimate changed: follow it
      await refresh(!$('#panel').contains(document.activeElement)); // new evidence, iPhone sync, face photo; don't clobber typing
    } else if (live || wasLive) renderAll(false);
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
if (!S.profile?.height_cm) open({ tab: 'settings' });
poll();
