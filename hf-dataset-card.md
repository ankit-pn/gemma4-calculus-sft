---
language:
- en
license: apache-2.0
task_categories:
- text-generation
tags:
- math
- calculus
- differentiation
- synthetic
- sft
size_categories:
- 1K<n<10K
configs:
- config_name: default
  data_files:
  - split: train
    path: data/train.jsonl
  - split: validation
    path: data/validation.jsonl
  - split: test
    path: data/test.jsonl
---

# Gemma 4 Calculus Differentiation (synthetic, symbolic)

A frozen synthetic dataset for **single-variable differentiation** in SymPy syntax, generated with `sympy 1.14` using a fixed seed. It was built for a controlled supervised fine-tuning learning curve on Gemma 4 E2B — see [GitHub](https://github.com/ankit-pn/gemma4-calculus-sft) and the resulting [LoRA adapter](https://huggingface.co/ankit-pn/gemma4-e2b-calculus-lora).

## Splits

| Split | Rows | Description |
|---|---:|---|
| `train` | 4,096 | Five families, unique expressions |
| `validation` | 200 | Same families, 40 per family |
| `test` | 240 | 160 same families + **80 product × trig × chain composites** absent from train/validation |

Every split has a SHA-256 hash frozen in the generator manifest; all expressions are unique across the dataset.

## Fields

```json
{
  "family": "product",
  "expression": "(2*x - 4)*(9*x**4 - 8)",
  "question": "Differentiate f(x) = (2*x - 4)*(9*x**4 - 8) with respect to x. Return one line: Final answer: <SymPy expression using x, +, -, *, /, **, sin, cos>. Do not explain.",
  "answer": "(2*x - 4)*(36*x**3) + 9*x**4*(2) - 8*(2)"
}
```

- `family`: `polynomial`, `product`, `power_chain`, `trig_chain`, `rational`, or `product_trig_chain` (test only)
- `answer`: symbolic derivative in SymPy syntax; graded by equivalence, not string match
- Terms are non-zero integers in [-9, 9]; powers up to 6

## Generation

```python
import sympy as sp
sp.sstr(sp.diff(sp.sympify(row["expression"]), sp.Symbol("x")))
```

The exact generator (`generate.py`) and grader (`evaluate.py`) are in the GitHub repository. The structural hold-out (`product_trig_chain`) tests whether the product and chain rules transfer to a composition the model never saw in training. In the reference experiment the fine-tuned adapter improved every trained family but **regressed on this hold-out** (14/80 base → 5/80 adapter), a clear negative-transfer signal — the split is doing its job.

## Intended use

- Small-scale SFT experiments and evaluation of symbolic differentiation
- The test split should be evaluated **once**, after any model selection, to keep it meaningful

## Limitations

- Synthetic and narrow: no exponentials, logarithms, inverse trig, multivariable, or higher-order derivatives
- Base models may have seen similar textbook derivatives during pretraining; this dataset measures task adaptation on a fresh distribution, not calculus acquisition
- English prompts only

## License

Apache-2.0
