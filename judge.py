"""A local vision decision model looks at the person in a frame and says what rules the frame out of measuring: not
standing (sitting, lying, reclining), on furniture, under a blanket, holding something against the body (its edges
would be taken for yours), clothes on, someone else in view, a child. The
rule-based checks miss these (a reclining body under a sheet passes them). The model is Cloudflare's Clef-flash
(Apache-2.0, 9B, ~1.5 s a frame on an M4) on MLX: it gives a probability for every answer, so close calls can lean
the right way. The image stays on this computer and isn't kept. Optional: without the `vision` dependency group,
ask() returns None and the rule-based checks stand alone.

python judge.py: check every reference capture not yet checked; ruled-out ones go, with their frames."""
import functools
import io
import json
import os
import sqlite3
import sys
from pathlib import Path

REPO, REVISION = 'mlx-community/clef-flash-8bit', '5a195dc147926b63b1d139691ed4480ca9f92c0e'  # pinned: it ships the code that runs it
WINDOW = 2.5  # s: a ruled-out frame takes its camera's frames this close to it along
STATE = 'A picture from a home camera; the background may be greyed out. Nudity is expected. Answer about what is visible.'
QUESTIONS = {k: {'type': 'noul', 'instructions': q} for k, q in {
    'several': 'Is more than one person visible, even partly (another person, their limbs or feet)?',
    'upright': 'Is the main person standing or walking upright on their own feet? (Not sitting, lying, reclining, kneeling, crouching or bending over.)',
    'furniture': 'Is the main person sitting, lying or leaning on a bed, couch, chair or table?',
    'blanket': 'Does a blanket, sheet or towel cover part of the main person?',
    'holding': 'Is the main person holding something (a laptop, tablet, bag, box or clothes) in front of or against their body?',
    'clothed_torso': "Is the main person's torso covered by clothing such as a shirt or top? Bare skin or underwear is not clothing.",
    'child': 'Is the main person a child?'}.items()}


@functools.cache
def model():
    """Clef-flash on MLX (downloaded once, 10.7 GB), or None without the `vision` group."""
    try:
        from huggingface_hub import snapshot_download
        try:
            path = snapshot_download(REPO, revision=REVISION, local_files_only=True)
        except Exception:  # not downloaded yet
            path = snapshot_download(REPO, revision=REVISION)
        sys.path.insert(0, path)
        import clef_mlx
        return clef_mlx.load(path)
    except ImportError:
        return None


def ask(jpg):
    """Probability of yes to each of QUESTIONS for a JPEG (bytes), or None if the model isn't available."""
    m = model()
    if m is None:
        return None
    from PIL import Image
    p = m.predict({'state': STATE, 'images': [Image.open(io.BytesIO(jpg)).convert('RGB')], 'questions': QUESTIONS})
    return {k: round(v['true'], 3) for k, v in p.items()}


def problem(v, bare=True):
    """Why these answers rule the frame out, or None."""
    if v['several'] > .5:
        return 'someone else in view'
    if v['child'] > .7:  # a backstop: the face check already turns away anyone who isn't you, so only clear calls
        return 'a child'
    if v['upright'] < .3:  # shortcut: thresholds from 172 captures (4 close calls); retune as reviewed cases build up
        return 'not standing'
    if v['furniture'] > .5:
        return 'on furniture'
    if v['blanket'] > .5:
        return 'under a blanket or towel'
    if v.get('holding', 0) > .5:  # its edges would be taken for yours
        return 'holding something'
    if bare and v['clothed_torso'] > .5:
        return 'clothes on'
    return None


def drop(db, cam, ts, window=WINDOW):
    """Remove camera cam's frames and reference captures within `window` s of ts."""
    with sqlite3.connect(db, timeout=30) as c:
        c.execute("delete from frames where ts between ? and ? and coalesce(json_extract(data, '$.cam'), 0) = ?", (ts - window, ts + window, cam))
    for p in (Path(db).parent / 'reference').glob(f'*_cam{cam + 1}.json'):  # named day_turn_ts_camN
        if abs(int(p.stem.split('_')[2]) - ts) < window + 1:
            p.unlink(missing_ok=True)
            p.with_suffix('.jpg').unlink(missing_ok=True)


def _demo():
    ok = {'several': .01, 'upright': .9, 'furniture': .01, 'blanket': .01, 'holding': .02, 'clothed_torso': .01, 'child': .02}
    assert problem(ok) is None
    assert problem(ok | {'several': .8}) == 'someone else in view'
    assert problem(ok | {'upright': .14}) == 'not standing' and problem(ok | {'upright': .34}) is None
    assert problem(ok | {'blanket': .7}) == 'under a blanket or towel'
    assert problem(ok | {'holding': .9}) == 'holding something'
    assert problem(ok | {'clothed_torso': .9}) == 'clothes on' and problem(ok | {'clothed_torso': .9}, bare=False) is None


if __name__ == '__main__':
    _demo()
    db = Path(os.environ.get('BODYLAB_DB') or Path(__file__).parent / 'data' / 'bodylab.db')
    for p in sorted((db.parent / 'reference').glob('*.json')):
        if not p.exists() or not p.with_suffix('.jpg').exists():  # dropped along with an earlier one
            continue
        rec = json.loads(p.read_text())
        v = rec.get('clef') if set(QUESTIONS) <= set(rec.get('clef') or {}) else ask(p.with_suffix('.jpg').read_bytes())  # re-asked when there's a new question
        if v is None:
            raise SystemExit('Clef is not available: uv sync --group vision')
        p.write_text(json.dumps({k: x for k, x in rec.items() if k != 'check'} | {'clef': v}))
        if why := problem(v):
            print(p.stem, why, v, flush=True)
            drop(db, rec.get('cam', 0), rec['ts'])
