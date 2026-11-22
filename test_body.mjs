// Self-check for static/body.js (template fitting). Run: node test_body.mjs
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { fitBody, fitProxy, initBody } from './static/body.js';

const file = (f) => readFileSync(new URL(`./static/assets/${f}`, import.meta.url));
const bin = file('body.bin');
const { male: t } = initBody(JSON.parse(file('body.json')), bin.buffer.slice(bin.byteOffset, bin.byteOffset + bin.length));

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
console.log('body ok');
