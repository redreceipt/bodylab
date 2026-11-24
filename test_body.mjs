// Self-check for static/body.js (template fitting). Run: node test_body.mjs
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { fitBody, fitProxy, genitalMesh, initBody, pose, underwear } from './static/body.js';

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
// Genitals: erect, the shaft comes out at the measured length and girth; at rest it hangs, tip below its base.
{
  const { rings } = A.meta.genitals, P0 = fitProxy('genitals', same.pos);
  const mean = (r, P) => [0, 1, 2].map((k) => r.reduce((s, i) => s + P[i * 3 + k], 0) / r.length);
  const erect = genitalMesh(same.pos, { length: 15, girth: 13, up: 1 }), rest = genitalMesh(same.pos, { length: 10, girth: 11, up: 0 });
  assert(erect.pos.every(Number.isFinite) && rest.pos.every(Number.isFinite));
  const tipAt = (g) => { let best = 0; g.along.forEach((x, i) => { if (x > g.along[best]) best = i; }); return [g.pos[best * 3], g.pos[best * 3 + 1], g.pos[best * 3 + 2]]; };
  const base = mean(rings[0], P0);
  assert(tipAt(rest)[1] < base[1] - 4 && tipAt(erect)[1] > base[1] - 3, 'hangs at rest, rises erect');
  assert(Math.hypot(...tipAt(erect).map((x, k) => x - base[k])) > Math.hypot(...tipAt(rest).map((x, k) => x - base[k])), 'erect reaches further');
  const c2 = mean(rings[1], P0), d = Math.hypot(...c2.map((x, k) => x - base[k])), root = base.map((x, k) => x - (c2[k] - x) / d * A.meta.genitals.base * d);
  near(Math.hypot(...tipAt(erect).map((x, k) => x - root[k])), 15, 0.04); // where it leaves the body to the tip
}
console.log('body ok');
// A fitted mesh is used as is, and a prediction reshapes it by predicted ÷ today.
const base = fitBody({}, t.H, 'male').pos, today = { waist_w: 30, waist_d: 22 };
const asIs = fitBody(today, t.H, 'male', {}, base, today).pos;
assert(asIs.every((v, i) => Math.abs(v - base[i]) < 1e-3));
const thinner = fitBody({ waist_w: 27, waist_d: 22 }, t.H, 'male', {}, base, today);
near(width(thinner, t.waist.y, notArm), width({ pos: base, parts: same.parts }, t.waist.y, notArm) * 0.9, 0.03);
// Underwear is built on any body: every vertex placed, standing off the skin.
for (const g of ['trunks', 'briefs', 'bra']) { const u = underwear(g, same.pos); assert(u.length === A.arr(`${g}.t`).length * 3 && u.every(Number.isFinite), g); }
{ // Dual-quaternion skinning: one rigid bone moves every vertex exactly by its transform (and two equal ones do too).
  const a = Math.PI / 6, c = Math.cos(a), s = Math.sin(a), G = new Float32Array(24);
  G.set([c, 0, s, 1, 0, 1, 0, 2, -s, 0, c, 3], 0); G.set([c, 0, s, 1, 0, 1, 0, 2, -s, 0, c, 3], 12);
  const P = new Float32Array([1, 2, 3, -4, 5, 0.5]), sk = new Uint8Array(32);
  sk[0] = 0; sk[8 + 0] = 0; sk[8 + 1] = 1; sk[16 + 0] = 255; sk[24 + 0] = 128; sk[24 + 1] = 127; // vertex 0: bone 0; vertex 1: bones 0 and 1
  const out = pose(P, G, sk);
  for (let i = 0; i < 2; i++) {
    const [x, y, z] = [P[i * 3], P[i * 3 + 1], P[i * 3 + 2]];
    near(out[i * 3] + 10, c * x + s * z + 1 + 10, 1e-4); near(out[i * 3 + 1] + 10, y + 2 + 10, 1e-4); near(out[i * 3 + 2] + 10, -s * x + c * z + 3 + 10, 1e-4);
  }
}
