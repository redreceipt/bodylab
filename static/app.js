import * as THREE from 'three';
import { OrbitControls } from 'three/addons/OrbitControls.js';
import { RoomEnvironment } from 'three/addons/RoomEnvironment.js';
import { fitBody, initBody } from './body.js';

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
const BODY = initBody(...await Promise.all([fetch('/static/body.json').then((r) => r.json()),
  fetch('/static/body.bin').then((r) => r.arrayBuffer())]));
function geometry(pos, index, color) {
  const g = new THREE.BufferGeometry();
  g.setAttribute('position', new THREE.BufferAttribute(pos, 3));
  if (color) g.setAttribute('color', new THREE.BufferAttribute(color, 3));
  g.setIndex(new THREE.BufferAttribute(index, 1));
  g.computeVertexNormals();
  return g;
}
renderer.toneMapping = THREE.ACESFilmicToneMapping;
scene.environment = new THREE.PMREMGenerator(renderer).fromScene(new RoomEnvironment(), 0.04).texture;
scene.environmentIntensity = 0.45;

const COLORS = { cloth: new THREE.Color('#262b36'), good: new THREE.Color('#4cff9a'), bad: new THREE.Color('#ff6b6b'), sel: new THREE.Color('#3ef0ff'), white: new THREE.Color('#ffffff') };
function partColor(p, cloth) {
  const c = cloth ? COLORS.cloth.clone() : new THREE.Color(S?.profile?.skin || '#c99478');
  if (p === selectedPart) return c.lerp(COLORS.sel, 0.55);
  const t = S && p !== 'overall' && tone(p, S.insights.parts[p]?.change);
  if (t) c.lerp(COLORS[t], 0.45);
  if (p === hovered) c.lerp(COLORS.white, 0.3);
  return c;
}

function buildAvatar(dims, sex, isGhost) {
  // Ghost = first scan as a translucent shell that loses depth ties, so it only shows where the body has shrunk.
  // Shrinking it 6 mm per side hides noise-level differences (and z-fighting).
  const d = isGhost ? Object.fromEntries(Object.entries(dims || {}).map(([k, v]) => [k, /_[wd]$/.test(k) ? v - 1.2 : v])) : dims;
  const fit = fitBody(d, S.profile?.height_cm ?? 175, sex);
  const body = geometry(fit.pos, BODY.index, isGhost ? null : new Float32Array(fit.pos.length));
  body.userData = { parts: fit.parts, cloth: fit.cloth };
  const group = new THREE.Group();
  if (isGhost) {
    group.add(new THREE.Mesh(body, new THREE.MeshBasicMaterial({ color: '#3ef0ff', transparent: true, opacity: 0.22,
      depthWrite: false, polygonOffset: true, polygonOffsetFactor: 2, polygonOffsetUnits: 8 })));
  } else {
    const eyes = geometry(fit.eyePos, BODY.eyeIndex, fit.eyeColors);
    const skin = new THREE.Mesh(body, new THREE.MeshPhysicalMaterial({ vertexColors: true, roughness: 0.58, sheen: 0.35,
      sheenRoughness: 0.7, sheenColor: '#ffcfb8', clearcoat: 0.06, clearcoatRoughness: 0.5 }));
    skin.castShadow = true;
    group.add(skin, new THREE.Mesh(eyes, new THREE.MeshPhysicalMaterial({ vertexColors: true, roughness: 0.08, clearcoat: 1 })));
  }
  group.scale.setScalar(0.01); // cm → m
  return group;
}

function paint() {
  const g = avatar?.children[0].geometry, col = g?.attributes.color;
  if (!col) return;
  const { parts, cloth } = g.userData, cache = {};
  for (let i = 0; i < parts.length; i++) {
    const [a, b] = (cache[parts[i]] ??= [partColor(parts[i], false), partColor(parts[i], true)]), k = cloth[i];
    col.setXYZ(i, a.r + (b.r - a.r) * k, a.g + (b.g - a.g) * k, a.b + (b.b - a.b) * k);
  }
  col.needsUpdate = true;
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
  const hit = ray.intersectObjects(avatar.children, false)[0];
  return hit ? hit.object.geometry.userData.parts?.[hit.face.a] ?? 'overall' : null;
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
      <label>Skin tone<input type="color" name="skin" value="${esc(p.skin ?? '#c99478')}"></label>
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
