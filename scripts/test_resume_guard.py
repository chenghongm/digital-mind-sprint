"""runner.check_out_dir: resuming into --out under other settings must stop
the run before the model loads. No model, no GPU.

    python3 scripts/test_resume_guard.py
"""

import json
import sys
import tempfile
import types
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import runner as R                                             # noqa: E402

TOPICS = ROOT / "topics_6x6.json"
item = json.load(open(TOPICS))[0]


def args(**kw):
    a = dict(model="/content/models/Qwen2.5-7B-Instruct", flip_rule="both",
             system_preset="llama31-default", depth=R.RELATIVE_DEPTH)
    a.update(kw)
    return types.SimpleNamespace(**a)


def expect(a, cond="neutral", order=1, release_turns=27):
    return dict(model=Path(a.model).name, topic=item["topic"],
                side_a=item["side_a"], side_b=item["side_b"], condition=cond,
                option_order=order, flip_rule=a.flip_rule,
                release_turns=release_turns,
                system=R.SYSTEM_PRESETS.get(a.system_preset))


def write_record(out, conv_id, settings):
    (out / "meta").mkdir(parents=True, exist_ok=True)
    rec = dict(settings, conv_id=conv_id, model="/content/models/" + settings["model"])
    json.dump(rec, open(out / "meta" / f"{conv_id}.json", "w"))


def stops(fn):
    try:
        fn()
    except SystemExit as e:
        return str(e)
    return None


with tempfile.TemporaryDirectory() as d:
    out = Path(d) / "run"
    a = args()
    cfg = R.run_config(a, TOPICS)
    cid = "neutral__000__o1"

    # fresh directory: nothing on disk, nothing to refuse
    assert R.check_out_dir(out, cfg, {cid: expect(a)}) == []

    # same settings on disk: resumable
    out.mkdir()
    json.dump(dict(cfg, canary="x"), open(out / R.RUN_CONFIG, "w"))
    write_record(out, cid, expect(a))
    assert R.check_out_dir(out, cfg, {cid: expect(a)}) == [cid]

    # the 28-turn neutral and a 12-release pressure arm share a directory
    pid = "pressure_release__000__o1"
    write_record(out, pid, expect(a, cond="pressure_release", release_turns=12))
    got = R.check_out_dir(out, cfg, {pid: expect(a, cond="pressure_release",
                                                  release_turns=12)})
    assert got == [pid], got

    # neutral asked for at 13 turns where a 28-turn one is stored: stop
    msg = stops(lambda: R.check_out_dir(out, cfg, {cid: expect(a, release_turns=12)}))
    assert msg and "release_turns 27 on disk vs 12 now" in msg, msg

    # no system preset where one was used: stop, at both levels
    b = args(system_preset=None)
    msg = stops(lambda: R.check_out_dir(out, R.run_config(b, TOPICS), {cid: expect(b)}))
    assert msg and "run_config.json: system" in msg and "system 'Cutting" in msg, msg

    # another model into the same directory: stop
    c = args(model="/content/models/Llama-3.1-8B-Instruct")
    msg = stops(lambda: R.check_out_dir(out, R.run_config(c, TOPICS), {cid: expect(c)}))
    assert msg and "model" in msg, msg

    # an edited topic file changes topics_sha256: stop
    edited = Path(d) / "topics_edited.json"
    edited.write_text(TOPICS.read_text().replace("Ofcom", "Ofcom ", 1))
    msg = stops(lambda: R.check_out_dir(out, R.run_config(a, edited), {cid: expect(a)}))
    assert msg and "topics_sha256" in msg, msg

    # a pre-schema-9 record has no `system`: read as the template default
    old = Path(d) / "old"
    e = args(model="/content/models/Llama-3.1-8B-Instruct", system_preset=None)
    s = expect(e, release_turns=12)
    write_record(old, cid, {k: v for k, v in s.items() if k != "system"})
    assert R.check_out_dir(old, R.run_config(e, TOPICS), {cid: s}) == [cid]

print("resume guard: all cases pass")
