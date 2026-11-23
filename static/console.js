// BodyLab's developer console: set-up and the tools for checking and tuning the measuring. The app is /?app=1.
import { renderSVG } from './vendor/uqr.mjs';

const $ = (s) => document.querySelector(s);
const esc = (s) => String(s ?? '').replace(/[&<>"]/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' })[c]);
const api = async (method, url, body) => {
  const r = await fetch(url, { method, headers: { 'Content-Type': 'application/json' }, body: body && JSON.stringify(body) });
  const j = await r.json().catch(() => ({}));
  if (!r.ok) throw new Error(j.error || r.statusText);
  return j;
};
const toast = (msg) => { const t = $('#toast'); t.textContent = msg; t.style.display = 'block'; setTimeout(() => (t.style.display = 'none'), 2500); };
let S = null;
const imp = () => S.profile?.units === 'imperial';
const cm = (v) => (v == null ? '—' : imp() ? `${(v / 2.54).toFixed(1)} in` : `${v.toFixed(1)} cm`);
const kg = (v) => (v == null ? '—' : imp() ? `${(v * 2.2046).toFixed(1)} lb` : `${v.toFixed(1)} kg`);
const fromUnit = (v, u) => (imp() ? (u === 'cm' ? v * 2.54 : u === 'kg' ? v / 2.2046 : v) : v);
const unit = (u) => (u === '%' ? '%' : imp() ? { cm: 'in', kg: 'lb' }[u] : u);
const show = (v, u) => (u === '%' ? `${(+v).toFixed(1)}%` : u === 'kg' ? kg(v) : cm(v));
const opt = (v, label, on) => `<option value="${v}" ${on ? 'selected' : ''}>${label}</option>`;

// [key, label, unit, where to measure, BodyLab's estimate it's compared with]. Tape snug, not pressing; standing,
// relaxed, after a normal breath out.
const REFS = [
  ['ref_neck', 'Neck', 'cm', "Just below the Adam's apple, tape tilted slightly down at the front", 'neck'],
  ['ref_chest', 'Chest / bust', 'cm', 'Fullest part: across the nipples, under the armpits, arms down', 'chest'],
  ['ref_underbust', 'Underbust', 'cm', 'Directly under the breasts, level all round', 'underbust'],
  ['ref_waist', 'Waist at navel', 'cm', "Level around the belly button; don't suck in", 'waist'],
  ['ref_hips', 'Hips', 'cm', 'Widest part of the buttocks, feet together', 'hips'],
  ['ref_arm', 'Upper arm', 'cm', 'Left arm hanging relaxed, halfway between the shoulder tip and the elbow', 'arm_l'],
  ['ref_thigh', 'Thigh', 'cm', 'Left thigh, halfway between the groin crease and the top of the kneecap', 'thigh_l'],
  ['ref_calf', 'Calf', 'cm', 'Left calf at its widest, weight on both feet', 'calf_l'],
  ['ref_weight', 'Weight', 'kg', 'Morning, after the bathroom, before eating', 'weight'],
  ['ref_bf', 'Body fat', '%', 'From DEXA, calipers or a smart scale (note which)', 'bf'],
  ['ref_penis_len', 'Penis length, resting', 'cm', 'Standing, from the base at the pubic bone along the top to the tip; not stretched', 'penis_len'],
  ['ref_penis_girth', 'Penis girth, resting', 'cm', 'Around the middle of the shaft', 'penis_girth']];

function cameras() {
  const p = S.profile || {}, rot = (k) => [0, 90, 180, 270].map((r) => opt(r, r ? `${r}° clockwise` : 'None', (p[k] ?? 0) === r)).join('');
  return `<section><h2>Cameras</h2>
    <div id="camStatus" class="muted">…</div>
    <form id="fCamera"><div class="grid">
      ${['', '2', '3'].map((k, i) => `<label>Camera ${i + 1} URL<input name="camera${k}_url" value="${esc(p[`camera${k}_url`])}"></label><label>Rotation<select name="rotate${k}">${rot(`rotate${k}`)}</select></label>`).join('')}</div>
      <label><input type="checkbox" name="bare_only" ${p.bare_only === false ? '' : 'checked'}> Only measure a bare torso (underwear is fine)</label>
      <label>Preview<select name="preview">${opt('silhouette', 'Silhouette only (private)', p.preview !== 'camera')}${opt('camera', 'Camera image (for aiming)', p.preview === 'camera')}</select></label>
      <button>Save</button> <button type="button" id="clear">Discard today's frames</button></form>
    <details id="preview"><summary>Preview</summary><select id="previewCam">${opt(1, 'Camera 1')}${opt(2, 'Camera 2')}${opt(3, 'Camera 3')}</select><img id="live" alt=""></details></section>`;
}
function fit() {
  const f = S.fit_info, scan = S.scans.at(-1), m = scan?.m ?? {};
  const rows = [['Body fat', m.bf != null ? `${m.bf.toFixed(1)}%` : '—'], ['Neck', cm(m.neck)], ['Chest', cm(m.chest)], ['Underbust', m.underbust && cm(m.underbust)],
    ['Waist', cm(m.waist)], ['Hips', cm(m.hips)], ['Upper arm', cm(m.arm_l)], ['Thigh', cm(m.thigh_l)], ['Calf', cm(m.calf_l)]].filter(([, v]) => v);
  return `<section><h2>Body fit</h2>
    <p class="muted">Frames today: ${Object.entries(S.frames_today).map(([c, n]) => `camera ${+c + 1}: ${n}`).join(', ') || 'none'}.
      Vision check today: ${Object.entries(S.checks_today ?? {}).map(([k, n]) => `${esc(k)} ${n}`).join(', ') || 'nothing checked (Ollama off?)'}.
      ${f ? `Fit ${esc(f.day)} from ${f.frames} frames; outline within ${f.fit?.out?.toFixed(1)} / ${f.fit?.in?.toFixed(1)} cm (mesh outside / outline unreached); body ${kg(f.fit?.kg)}; muscle ${f.phenotypes?.muscle?.toFixed(2)}, fat ${f.phenotypes?.weight?.toFixed(2)}.` : 'No body fit yet.'}</p>
    <table>${rows.map(([k, v]) => `<tr><td>${k}</td><td>${v}</td></tr>`).join('')}</table>
    ${S.fit_history.length ? `<details><summary>Fit history (outline error, cm)</summary><table><tr><th>Day</th><th>Frames</th><th>Dropped</th><th>Mesh out / outline in</th><th>Per camera</th></tr>
      ${S.fit_history.map((h) => `<tr><td>${h.day}</td><td>${h.frames ?? ''}</td><td>${h.dropped ?? ''}</td><td>${h.fit ? `${h.fit.out} / ${h.fit.in}` : ''}</td>
        <td>${h.fit_cams ? Object.entries(h.fit_cams).map(([c, v]) => `cam ${+c + 1}: ${v}`).join(', ') : ''}</td></tr>`).join('')}</table></details>` : ''}
    <p class="muted">Skin: ${S.skin ? `baked from ${S.skin.captures} captures, ${Math.round(S.skin.seen * 100)}% of the body seen${S.skin.seen < 0.5 ? ' (used from 50%)' : ''}` : 'not baked yet'}.</p></section>`;
}
function references() {
  const r = S.references;
  return `<section><h2>Reference images</h2>
    <label><input type="checkbox" id="refImages" ${S.profile?.reference ? 'checked' : ''}> Save the best frames of you from every angle (cropped to you), up to 1 GB on this Mac</label>
    <p class="muted">${r.count} saved (${r.mb.toFixed(0)} MB). Latest:</p>
    <div class="thumbs">${r.latest.map((n) => `<a href="/api/reference/${n}" target="_blank"><img loading="lazy" src="/api/reference/${n}" alt=""></a>`).join('')}</div></section>`;
}
function measurements() {
  const sex = S.profile?.sex;
  const fields = REFS.filter(([k]) => !(k === 'ref_underbust' && sex !== 'female') && !(k.startsWith('ref_penis') && sex === 'female'));
  const rows = S.logs.filter((l) => l.kind === 'reference').reverse().flatMap((l) => {
    const d = new Date(l.ts * 1000), key = `${d.getFullYear()}-${String(d.getMonth() + 1).padStart(2, '0')}-${String(d.getDate()).padStart(2, '0')}`;
    const scan = S.scans.find((s) => s.day === key);
    return REFS.filter(([k]) => l[k] != null).map(([k, label, u, , est]) => {
      const mine = est === 'weight' ? S.weight : est && scan?.m[est];
      return `<tr><td>${key}</td><td>${label}</td><td>${show(l[k], u)}</td><td>${mine != null ? show(mine, u) : '—'}</td><td>${mine != null ? `${mine >= l[k] ? '+' : '−'}${show(Math.abs(mine - l[k]), u)}` : ''}</td></tr>`;
    });
  });
  return `<section><h2>Reference measurements</h2>
    <p class="muted">Ground truth for tuning, never used for your numbers. Each is compared with BodyLab's estimate from the same day only: a spot check of the camera, not a trend.</p>
    <form id="fRef">${fields.map(([k, label, u, how]) => `<label>${label} (${unit(u)})<input name="${k}" type="number" step="any"><small>${how}</small></label>`).join('')}<button>Save</button></form>
    ${rows.length ? `<table><tr><th>Day</th><th>Part</th><th>Yours</th><th>BodyLab</th><th>Off by</th></tr>${rows.join('')}</table>` : ''}</section>`;
}
function setup() {
  const p = S.profile || {}, link = S.sync.url && `bodylab://pair?server=${encodeURIComponent(S.sync.url)}&token=${encodeURIComponent(S.sync.token)}`;
  return `<section><h2>iPhone and profile</h2>
    ${link ? `<div style="width:180px;background:#fff;padding:6px;border-radius:6px">${renderSVG(link, { pixelSize: 4 })}</div><p class="muted">Scan with the iPhone Camera to pair the BodyLab app.</p>`
    : '<p class="muted">Run <code>tailscale serve --bg --https=8443 http://127.0.0.1:8765</code> so the iPhone can reach BodyLab.</p>'}
    <p class="muted">From Apple Health: ${[p.height_cm && cm(p.height_cm), p.sex, p.birth_date && `born ${esc(p.birth_date)}`, p.units].filter(Boolean).join(' · ') || 'nothing yet'}.</p>
    <details><summary>Look overrides</summary><form id="fLook"><div class="grid">
      <label>Hair<select name="hair">${['auto', 'none', 'short02', 'short04', 'bob02', 'long01', 'ponytail01', 'braid01', 'afro01'].map((h) => opt(h === 'auto' ? '' : h, h, (p.hair ?? '') === (h === 'auto' ? '' : h))).join('')}</select></label>
      <label>Hair colour<input type="color" name="hair_color" value="${esc(p.hair_color ?? S.face?.hair?.color ?? '#3b2a20')}"></label></div><button>Save</button></form></details></section>`;
}
async function load() {
  S = await (await fetch('/api/state')).json();
  $('#main').innerHTML = cameras() + fit() + references() + measurements() + setup();
}
async function poll() {
  try {
    const st = await (await fetch('/api/status')).json(), line = (n, s) => `<div><span class="dot ${s.state}"></span>Camera ${n}: ${esc(s.msg)}${s.match != null ? ` (face match ${s.match})` : ''}</div>`;
    if ($('#camStatus')) $('#camStatus').innerHTML = line(1, st) + (st.cam2 ? line(2, st.cam2) : '') + (st.cam3 ? line(3, st.cam3) : '');
  } catch { /* restarting */ }
  setTimeout(poll, 2000);
}
document.addEventListener('submit', async (e) => {
  e.preventDefault();
  const f = e.target, fd = new FormData(f), o = Object.fromEntries(fd);
  try {
    if (f.id === 'fCamera') await api('POST', '/api/profile', { ...o, bare_only: fd.has('bare_only') });
    if (f.id === 'fLook') await api('POST', '/api/profile', o);
    if (f.id === 'fRef') {
      await api('POST', '/api/logs', { kind: 'reference', ...Object.fromEntries(Object.entries(o).filter(([, v]) => v !== '')
        .map(([k, v]) => [k, fromUnit(+v, REFS.find(([r]) => r === k)[2])])) });
    }
    toast('Saved');
    load();
  } catch (err) { toast(err.message); }
});
document.addEventListener('change', async (e) => {
  if (e.target.id === 'refImages') { await api('POST', '/api/profile', { reference: e.target.checked }); toast('Saved'); }
  if (e.target.id === 'previewCam') $('#live').src = `/api/live.mjpg?cam=${e.target.value}`;
});
document.addEventListener('toggle', (e) => { if (e.target.id === 'preview') $('#live').src = e.target.open ? `/api/live.mjpg?cam=${$('#previewCam').value}` : ''; }, true);
document.addEventListener('click', async (e) => {
  if (e.target.id === 'clear' && confirm("Discard today's frames and estimate?")) { await api('POST', '/api/clear-today'); toast('Discarded'); load(); }
});
load();
poll();
