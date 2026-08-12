import json
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

_EVAL_DIR = Path(__file__).parent.parent / "eval"
eval_set = json.loads((_EVAL_DIR / "eval_set.json").read_text(encoding="utf-8"))
calib = json.loads((_EVAL_DIR / "calibration_dataset.json").read_text(encoding="utf-8"))

keywords_by_query = {c["query"]: c["expected_keywords"] for c in eval_set}

targets = [
    d for d in calib
    if d["needs_manual_review"] or (d["audit_status"] == "passed" and not d["auto_correct"])
]

print(f"Total cases to review: {len(targets)}\n")

for i, d in enumerate(targets, 1):
    print("=" * 90)
    print(f"Case {i}")
    print(f"Query: {d['query']}")
    print(f"Expected keywords: {keywords_by_query.get(d['query'])}")
    print(f"Source method: {d['source_method']}")
    print(f"Audit status: {d['audit_status']}")
    print(f"needs_manual_review: {d['needs_manual_review']}")
    print(f"auto_correct: {d['auto_correct']}")
    print(f"\nFinal answer:\n{d['final_answer']}")
    print()
