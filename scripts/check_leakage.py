"""Leakage audit for the frozen calculus splits.

Checks, against the files the model actually saw:
  1. SHA-256 of each split matches manifest.json.
  2. Exact expression-string overlap between splits.
  3. Mathematically equivalent expressions between splits (canonical SymPy form).
  4. Held-out structural family leaking into train/validation.
  5. Answers (derivatives) shared between splits — answer leakage.
"""
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import sympy as sp

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
if not DATA.exists():  # repository layout: scripts/ and data/ are siblings
    DATA = ROOT.parent / "data"
OUTPUT = ROOT / "output"
if not OUTPUT.exists():
    OUTPUT = ROOT.parent / "output"
X = sp.Symbol("x")


def load(name):
    return [json.loads(line) for line in (DATA / f"{name}.jsonl").read_text().splitlines()]


def canonical(expr):
    parsed = sp.sympify(expr)
    return sp.srepr(sp.simplify(parsed)), parsed


def has_poly_trig_product(expr):
    """True if the expression is a product of a non-constant non-trig factor and a trig factor."""
    parsed = sp.sympify(expr)
    if not isinstance(parsed, sp.Mul):
        return False
    trig = lambda f: f.has(sp.sin) or f.has(sp.cos)
    has_trig = any(trig(f) for f in parsed.args)
    has_nonconst_other = any(f.has(X) and not trig(f) for f in parsed.args)
    return has_trig and has_nonconst_other


def main():
    manifest = json.loads((DATA / "manifest.json").read_text())
    splits = {name: load(name) for name in ("train", "validation", "test")}

    print("== 1. file integrity")
    for name, rows in splits.items():
        digest = hashlib.sha256((DATA / f"{name}.jsonl").read_bytes()).hexdigest()
        ok = digest == manifest["sha256"][name]
        print(f"  {name}: {len(rows)} rows, sha256 match = {ok}")
        assert ok

    print("== 2. exact string overlap")
    for a, b in (("train", "validation"), ("train", "test"), ("validation", "test")):
        overlap = {r["expression"] for r in splits[a]} & {r["expression"] for r in splits[b]}
        print(f"  {a} ∩ {b}: {len(overlap)}")
        assert not overlap

    print("== 3. symbolic equivalence between splits")
    canon = {}
    for name, rows in splits.items():
        canon[name] = defaultdict(list)
        for r in rows:
            key, _ = canonical(r["expression"])
            canon[name][key].append(r["expression"])
    for a, b in (("train", "validation"), ("train", "test"), ("validation", "test")):
        shared = set(canon[a]) & set(canon[b])
        print(f"  {a} ~ {b}: {len(shared)} equivalent expressions")
        for key in list(shared)[:5]:
            print("    ", canon[a][key][0], "==", canon[b][key][0])
        assert not shared

    print("== 4. held-out structure in train/validation")
    for name in ("train", "validation"):
        bad = [r["expression"] for r in splits[name] if has_poly_trig_product(r["expression"])]
        families = sorted({r["family"] for r in splits[name]})
        print(f"  {name}: polynomial×trig products = {len(bad)}, families = {families}")
        assert len(bad) == 0
    test_families = sorted({r["family"] for r in splits["test"]})
    ood = [r for r in splits["test"] if r["family"] == "product_trig_chain"]
    print(f"  test families = {test_families}, held-out rows = {len(ood)}")

    print("== 5. training order digest")
    import random
    groups = defaultdict(list)
    for r in splits["train"]:
        groups[r["family"]].append(r)
    rng = random.Random(20260927)
    for family in ("polynomial", "product", "power_chain", "trig_chain", "rational"):
        rng.shuffle(groups[family])
    ordered = []
    while any(groups.values()):
        for family in ("polynomial", "product", "power_chain", "trig_chain", "rational"):
            if groups[family]:
                ordered.append(groups[family].pop()["expression"])
    digest = hashlib.sha256(json.dumps(ordered).encode()).hexdigest()
    print("  recomputed order digest:", digest)
    summaries = sorted(OUTPUT.glob("train-*-summary.json")) if OUTPUT.exists() else []
    if not summaries:
        results = ROOT.parent / "results"
        summaries = sorted(results.glob("train-*-summary.json")) if results.exists() else []
    if not summaries:
        print("  (no training summaries found next to this script)")
    for path in summaries:
        recorded = json.loads(path.read_text())["training_order_sha256"]
        print(f"  {path.name}: match = {recorded == digest}")
        assert recorded == digest

    print("== 6. answer leakage (same derivative across splits)")
    for name, rows in splits.items():
        canon[name + "_ans"] = defaultdict(list)
        for r in rows:
            key, _ = canonical(r["answer"])
            canon[name + "_ans"][key].append(r["expression"])
    for a, b in (("train", "validation"), ("train", "test"), ("validation", "test")):
        shared = set(canon[a + "_ans"]) & set(canon[b + "_ans"])
        examples = [(canon[a + '_ans'][k][0], canon[b + '_ans'][k][0]) for k in list(shared)[:3]]
        print(f"  {a} answer ≈ {b} answer: {len(shared)} shared derivatives, e.g. {examples}")

    total = sum(len(r) for r in splits.values())
    print(f"== total rows checked: {total}; no exact or symbolic expression leakage found")


if __name__ == "__main__":
    main()
