"""Paired comparison of a matched GPU baseline and LoRA runs.

Validate all candidate sizes first. Freeze `selection.json` before uploading or
reading test.jsonl on the GPU. McNemar's exact two-sided p is descriptive;
this small synthetic benchmark does not prove broad calculus generalization.
"""
import argparse
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "output"
SIZES = (128, 512, 2048, 4096)


def load(label):
    path = OUT / (label + ".jsonl")
    if not path.exists():
        return None
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if sorted(x["index"] for x in rows) != list(range(len(rows))):
        raise ValueError(f"Missing or duplicate indices in {path}")
    return sorted(rows, key=lambda row: row["index"])


def paired(base, adapted):
    if len(base) != len(adapted) or any(
        a["expression"] != b["expression"] or a["expected"] != b["expected"]
        for a, b in zip(base, adapted)
    ):
        raise ValueError("Unpaired evaluation rows")
    gains = sum(not a["correct"] and b["correct"] for a, b in zip(base, adapted))
    losses = sum(a["correct"] and not b["correct"] for a, b in zip(base, adapted))
    discordant = gains + losses
    p = (min(1, 2 * sum(math.comb(discordant, i) for i in range(min(gains, losses) + 1))
             / 2 ** discordant) if discordant else 1.0)
    return {"base_correct": sum(x["correct"] for x in base),
            "lora_correct": sum(x["correct"] for x in adapted), "n": len(base),
            "gains": gains, "losses": losses, "net": gains - losses,
            "mcnemar_exact_two_sided_p": p}


def report_validation():
    base = load("base-validation")
    if not base or len(base) != 200:
        raise ValueError("Wait for the complete 200-case GPU baseline")
    results = {}
    for size in SIZES:
        adapted = load(f"lora-{size}-validation")
        if adapted is None:
            continue
        if len(adapted) != 200:
            raise ValueError(f"Size {size}: incomplete validation evaluation")
        results[size] = paired(base, adapted)
    print(json.dumps({"split": "validation", "matched_base": sum(x["correct"] for x in base),
                      "comparisons": results}, indent=2))
    if len(results) == len(SIZES):
        # Choose by validation accuracy, tie breaking toward the less expensive run.
        selected = max(SIZES, key=lambda size: (results[size]["lora_correct"], -size))
        selection = {"selected_size": selected, "rule": "highest validation accuracy, ties to smaller set",
                     "validation_correct_by_size": {str(k): v["lora_correct"] for k, v in results.items()},
                     "test_seen": False}
        path = OUT / "selection.json"
        if path.exists() and json.loads(path.read_text()) != selection:
            raise ValueError(f"Existing {path} disagrees; do not change selection after seeing test")
        path.write_text(json.dumps(selection, indent=2) + "\n")
        print("SELECTED", json.dumps(selection))


def report_test():
    selection = json.loads((OUT / "selection.json").read_text())
    size = selection["selected_size"]
    base, adapted = load("base-test"), load(f"lora-{size}-test")
    if base is None or adapted is None or len(base) != 240 or len(adapted) != 240:
        raise ValueError("Wait for both complete 240-case final-test runs")
    grouped = {}
    for family in sorted({x["family"] for x in base}):
        b = [x for x in base if x["family"] == family]
        a = [x for x in adapted if x["family"] == family]
        grouped[family] = paired(b, a)
    result = {"selected_size": size, "split": "test", "overall": paired(base, adapted),
              "by_family": grouped}
    (OUT / "final-report.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("split", choices=("validation", "test"))
    args = parser.parse_args()
    (report_validation if args.split == "validation" else report_test)()
