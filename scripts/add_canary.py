"""Add the canary field to every JSON object file under runs/.

    python3 scripts/add_canary.py            # rewrite files that lack it
    python3 scripts/add_canary.py --check    # list them, change nothing

Only runner.py writes the field itself. The other writers stamp a hash of
their own source into their output (source_sha256), so editing them would
break that provenance; run this after any of them writes new output instead.

Each file is re-serialised with the indent and ensure_ascii setting it was
written with, so the diff is the one added line. A file whose formatting
cannot be reproduced exactly, or whose top level is not an object, is
reported and left alone.
"""

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from canary import CANARY   # noqa: E402

FORMATS = [(ind, ea) for ind in (2, 1, None, 4) for ea in (False, True)]


def detect_format(d, raw):
    body = raw.rstrip("\n")
    for ind, ea in FORMATS:
        if json.dumps(d, indent=ind, ensure_ascii=ea) == body:
            return ind, ea, raw[len(body):]
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=str(ROOT / "runs"))
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args()

    added, present, skipped = [], 0, []
    for path in sorted(Path(args.root).rglob("*.json")):
        raw = path.read_text(encoding="utf-8")
        d = json.loads(raw)
        if not isinstance(d, dict):
            skipped.append((path, "top level is not an object"))
            continue
        if d.get("canary") == CANARY:
            present += 1
            continue
        if "canary" in d:
            skipped.append((path, "has a different 'canary' value"))
            continue
        fmt = detect_format(d, raw)
        if fmt is None:
            skipped.append((path, "formatting not reproducible"))
            continue
        ind, ea, tail = fmt
        d["canary"] = CANARY
        if not args.check:
            path.write_text(json.dumps(d, indent=ind, ensure_ascii=ea) + tail,
                            encoding="utf-8")
        added.append(path)

    verb = "would add" if args.check else "added"
    print(f"{verb} {len(added)}, already present {present}, "
          f"skipped {len(skipped)}")
    for path, why in skipped:
        print(f"  skipped {path.relative_to(ROOT)}: {why}")
    return 1 if args.check and added else 0


if __name__ == "__main__":
    sys.exit(main())
