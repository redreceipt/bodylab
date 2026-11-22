// Self-check for static/body.js (template fitting). Run: node test_body.mjs
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { fitBody, fitProxy, genitalPositions, initBody } from './static/body.js';

const file = (f) => readFileSync(new URL(`./static/assets/${f}`, import.meta.url));
const bin = file('body.bin');
const A = initBody(JSON.parse(file('body.json')), bin.buffer.slice(bin.byteOffset, bin.byteOffset + bin.length));
for (const k of Object.keys(A.meta.arrays)) assert(A.arr(k).length > 0, k); // every array loads (e.g. uint32)
const t = A.male;

// x-extent of vertices whose part passes `use`, within ±1.2 cm of height y
function width(fit, y, use) {
  let lo = Infinity, hi = -Infinity;
  for (let i = 0; i < fit.parts.length; i++) {
    if (Math.abs(fit.pos[i * 3 + 1] - y) > 1.2 || !use(fit.parts[i])) continue;
    lo = Math.min(lo, fit.pos[i * 3]); hi = Math.max(hi, fit.pos[i * 3]);
  }
  return hi - lo;
}
const notArm = (p) => !p.startsWith('arm');
const near = (a, b, tol) => assert.ok(Math.abs(a / b - 1) < tol, `${a.toFixed(2)} vs ${b.toFixed(2)}`);

// No measurements at the template's own height leaves the template untouched.
const same = fitBody({}, t.H, 'male');
assert.ok(same.pos.every((v, i) => Math.abs(v - t.pos[i]) < 1e-3));

// Height scales the whole body.
const tall = fitBody({}, t.H * 1.1, 'male');
near(Math.max(...tall.pos.filter((_, i) => i % 3 === 1 && i < t.parts.length * 3)), t.H * 1.1, 0.002);

// A 20 % wider waist measurement makes the waist 20 % wider, and leaves the neck alone.
const wide = fitBody({ waist_w: t.waist.w * 1.2 }, t.H, 'male');
near(width(wide, t.waist.y, notArm), width(same, t.waist.y, notArm) * 1.2, 0.03);
near(width(wide, t.neck.y, notArm), width(same, t.neck.y, notArm), 0.01);

// A 30 % bigger left thigh grows only the left thigh.
const thighY = t.J.hip_l[1] - 0.4 * (t.J.hip_l[1] - t.J.knee_l[1]);
const big = fitBody({ thigh_l_w: t.thigh_l * 1.3 }, t.H, 'male');
near(width(big, thighY, (p) => p === 'thigh_l'), width(same, thighY, (p) => p === 'thigh_l') * 1.3, 0.1);
near(width(big, thighY, (p) => p === 'thigh_r'), width(same, thighY, (p) => p === 'thigh_r'), 0.01);
// Hair follows the head: 10 % taller body lifts the hair's centre ~10 %.
const centreY = (p) => { let y = 0; for (let i = 1; i < p.length; i += 3) y += p[i]; return y / (p.length / 3); };
const hair = centreY(fitProxy('short02', same.pos));
assert.ok(hair > t.J.eye_l[1], 'hair sits above the eyes');
near(centreY(fitProxy('short02', tall.pos)), hair * 1.1, 0.01);
// Genitals come out at the measured size (absolute cm), whatever the body's height.
const shaft = A.arr('morph.girth+.idx'), grow = A.arr('morph.length+.d'), ax = [0, 0, 0];
const at = Object.fromEntries(Array.from(A.arr('morph.length+.idx'), (v, j) => [v, j]));
for (const v of shaft) if (at[v] != null) for (let k = 0; k < 3; k++) ax[k] += grow[at[v] * 3 + k];
const al = Math.hypot(...ax);
function penis(pos) { // length along the shaft axis, girth around its middle (as the build measures the template)
  const p = Array.from(shaft, (v) => [pos[v * 3], pos[v * 3 + 1], pos[v * 3 + 2]]), t = p.map((q) => (q[0] * ax[0] + q[1] * ax[1] + q[2] * ax[2]) / al);
  const lo = Math.min(...t), hi = Math.max(...t), mid = p.filter((_, i) => Math.abs(t[i] - (lo + hi) / 2) < 0.15 * (hi - lo));
  const ring = mid.map((q, i) => q.map((c, k) => c - t[p.indexOf(q)] * ax[k] / al)), c = [0, 1, 2].map((k) => ring.reduce((a, q) => a + q[k], 0) / ring.length);
  return [hi - lo, 2 * Math.PI * ring.reduce((a, q) => a + Math.hypot(q[0] - c[0], q[1] - c[1], q[2] - c[2]), 0) / ring.length];
}
for (const H of [t.H, t.H * 1.1]) {
  const [len, girth] = penis(fitBody({}, H, 'male', { length: 13, girth: 11 }).pos);
  near(len, 13, 0.05);
  near(girth, 11, 0.07);
}
console.log('body ok');
assert(genitalPositions(same.pos).every(Number.isFinite)); // nude mode builds
// A fitted mesh is used as is, and a prediction reshapes it by predicted ÷ today.
const base = fitBody({}, t.H, 'male').pos, today = { waist_w: 30, waist_d: 22 };
const asIs = fitBody(today, t.H, 'male', {}, base, today).pos;
assert(asIs.every((v, i) => Math.abs(v - base[i]) < 1e-3));
const thinner = fitBody({ waist_w: 27, waist_d: 22 }, t.H, 'male', {}, base, today);
near(width(thinner, t.waist.y, notArm), width({ pos: base, parts: same.parts }, t.waist.y, notArm) * 0.9, 0.03);
