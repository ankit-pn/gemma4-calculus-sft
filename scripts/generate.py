"""Freeze a fresh, symbolic differentiation dataset with a structural holdout."""
import hashlib
import json
import random
from pathlib import Path

import sympy as sp

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
SEED = 20260927
X = sp.Symbol("x")
TRAIN_FAMILIES = ("polynomial", "product", "power_chain", "trig_chain", "rational")
OOD_FAMILY = "product_trig_chain"  # this composition is not in train or validation


def nz(rng, bound=9):
    return rng.choice([v for v in range(-bound, bound + 1) if v != 0])


def expression(family, rng):
    a, b, c, d = (nz(rng) for _ in range(4))
    if family == "polynomial":
        n = rng.randint(3, 6)
        return a * X**n + b * X**(n - 2) + c * X + d
    if family == "product":
        return sp.Mul(a * X + b, c * X**rng.randint(2, 4) + d, evaluate=False)
    if family == "power_chain":
        return sp.Pow(a * X**2 + b * X + c, rng.randint(2, 5), evaluate=False)
    if family == "trig_chain":
        arg = a * X**2 + b * X + c if rng.randrange(2) else a * X + b
        return sp.sin(arg) if rng.randrange(2) else sp.cos(arg)
    if family == "rational":
        return sp.Mul(a * X + b, sp.Pow(c * X + d, -1, evaluate=False), evaluate=False)
    if family == OOD_FAMILY:
        return sp.Mul(a * X + b, sp.sin(c * X**2 + d), evaluate=False)
    raise ValueError(family)


def make_row(family, rng, seen):
    for _ in range(1000):
        expr = expression(family, rng)
        source = sp.sstr(expr)
        if source in seen:
            continue
        seen.add(source)
        derivative = sp.diff(expr, X)
        assert sp.simplify(derivative - sp.diff(expr, X)) == 0
        return {
            "family": family,
            "expression": source,
            "question": (
                f"Differentiate f(x) = {source} with respect to x. "
                "Return one line: Final answer: <SymPy expression using x, +, -, *, /, **, sin, cos>. "
                "Do not explain."
            ),
            "answer": sp.sstr(derivative),
        }
    raise RuntimeError("Could not draw a unique expression")


def save(name, rows):
    path = DATA / f"{name}.jsonl"
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    DATA.mkdir(parents=True, exist_ok=True)
    rng = random.Random(SEED)
    seen = set()
    # All families present in training and validation. The held-out composition
    # tests transfer of the product and chain rules, not memorized templates.
    train = [make_row(family, rng, seen)
             for i in range(4096) for family in [TRAIN_FAMILIES[i % len(TRAIN_FAMILIES)]]]
    rng.shuffle(train)
    validation = [make_row(family, rng, seen)
                  for i in range(200) for family in [TRAIN_FAMILIES[i % len(TRAIN_FAMILIES)]]]
    rng.shuffle(validation)
    test = [make_row(family, rng, seen)
            for i in range(160) for family in [TRAIN_FAMILIES[i % len(TRAIN_FAMILIES)]]]
    test.extend(make_row(OOD_FAMILY, rng, seen) for _ in range(80))
    rng.shuffle(test)
    hashes = {name: save(name, rows) for name, rows in
              (("train", train), ("validation", validation), ("test", test))}
    manifest = {
        "seed": SEED, "generator": "generate.py",
        "sympy_version": sp.__version__, "split_sizes": {"train": 4096, "validation": 200, "test": 240},
        "training_families": TRAIN_FAMILIES, "structural_holdout_family": OOD_FAMILY,
        "sha256": hashes,
        "policy": "Evaluate base and SFT checkpoints on validation; reserve test for the final pre/post comparison.",
    }
    (DATA / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))
    print("example:", json.dumps(train[0]))


if __name__ == "__main__":
    main()
