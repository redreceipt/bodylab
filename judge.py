"""A local vision model (through Ollama) looks at the person in a frame and says what rules the frame out of measuring:
sitting or lying, on furniture, under a blanket, clothes on, someone else in view, a child. The rule-based checks miss
these (a reclining body under a sheet passes them). The image goes only to Ollama on this computer and isn't kept.
Optional: without Ollama or the model, ask() returns None and the rule-based checks stand alone.

python judge.py: check every reference capture not yet checked; ruled-out ones go, with their frames."""
import base64
import json
import os
import sqlite3
import urllib.request
from pathlib import Path

MODEL = os.environ.get('BODYLAB_VISION_MODEL', 'qwen3.8:27b-mlx')  # any Ollama model with vision
URL = 'http://127.0.0.1:11434/api/chat'
WINDOW = 2.5  # s: a ruled-out frame takes its camera's frames this close to it along
COVER = ['nude', 'underwear', 'clothes', 'blanket or towel', 'out of view']
SCHEMA = {'type': 'object', 'required': ['doing', 'people', 'posture', 'on_furniture', 'leaning_back', 'torso', 'legs', 'child'], 'properties': {
    'doing': {'type': 'string'},  # first: the model describes before it classifies
    'people': {'type': 'integer'},
    'posture': {'type': 'string', 'enum': ['standing', 'walking', 'sitting', 'lying or reclining', 'kneeling or crouching', 'bending over']},
    'on_furniture': {'type': 'boolean'},
    'leaning_back': {'type': 'boolean'},
    'torso': {'type': 'string', 'enum': COVER},
    'legs': {'type': 'string', 'enum': COVER[:2] + ['shorts', 'trousers or skirt'] + COVER[3:]},
    'child': {'type': 'boolean'}}}
PROMPT = ('A home camera took this picture (the background may be greyed out). Answer factually; nudity is expected. '
          'doing: one short sentence on what the person is doing, on what, and whether they are standing upright on their '
          'own feet. people: how many people are visible, even partly. posture: of the main person. on_furniture: are they '
          'sitting, lying or leaning on a bed, couch, chair or table? leaning_back: is their upper body leaning back or '
          'propped up (on pillows, a headboard, an elbow)? torso, legs: what covers them. child: is the main person a child?')


def ask(jpg, timeout=120):
    """The model's answers for a JPEG (bytes), or None if Ollama or the model isn't available."""
    body = {'model': MODEL, 'stream': False, 'think': False, 'format': SCHEMA, 'options': {'temperature': 0}, 'keep_alive': '30m',
            'messages': [{'role': 'user', 'content': PROMPT, 'images': [base64.b64encode(jpg).decode()]}]}
    try:
        req = urllib.request.Request(URL, json.dumps(body).encode(), {'Content-Type': 'application/json'})
        return json.loads(json.load(urllib.request.urlopen(req, timeout=timeout))['message']['content'])
    except (OSError, ValueError, KeyError):
        return None


def problem(v, bare=True):
    """Why these answers rule the frame out, or None."""
    if v.get('people') != 1:
        return 'someone else in view' if v.get('people', 0) > 1 else 'nobody in view'
    if v.get('child'):
        return 'a child'
    if v.get('posture') not in ('standing', 'walking'):
        return v.get('posture', 'posture unknown')
    if v.get('on_furniture') or v.get('leaning_back'):
        return 'on furniture' if v.get('on_furniture') else 'leaning back'
    if 'blanket or towel' in (v.get('torso'), v.get('legs')):
        return 'under a blanket or towel'
    if bare and v.get('torso') == 'clothes':
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
    ok = {'people': 1, 'posture': 'standing', 'on_furniture': False, 'torso': 'nude', 'legs': 'shorts', 'child': False}
    assert problem(ok) is None
    assert problem(ok | {'people': 2}) == 'someone else in view'
    assert problem(ok | {'posture': 'lying or reclining'}) == 'lying or reclining'
    assert problem(ok | {'legs': 'blanket or towel'}) == 'under a blanket or towel'
    assert problem(ok | {'torso': 'clothes'}) == 'clothes on' and problem(ok | {'torso': 'clothes'}, bare=False) is None


if __name__ == '__main__':
    _demo()
    db = Path(os.environ.get('BODYLAB_DB') or Path(__file__).parent / 'data' / 'bodylab.db')
    for p in sorted((db.parent / 'reference').glob('*.json')):
        if not p.exists() or not p.with_suffix('.jpg').exists():  # dropped along with an earlier one
            continue
        rec = json.loads(p.read_text())
        v = rec.get('check') or ask(p.with_suffix('.jpg').read_bytes())
        if v is None:
            raise SystemExit(f'No answer from {MODEL} via Ollama: is it running, with the model pulled?')
        p.write_text(json.dumps(rec | {'check': v}))
        if why := problem(v):
            print(p.stem, why, '-', v.get('doing'), flush=True)
            drop(db, rec.get('cam', 0), rec['ts'])
