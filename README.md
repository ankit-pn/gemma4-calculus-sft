# Gemma 4 E2B — Calculus Differentiation SFT (Learning Curve)

A small, end-to-end supervised fine-tuning study: **how many training examples does it take before Gemma 4 E2B reliably differentiates?**

I generated a fresh synthetic differentiation dataset (no public benchmark examples), froze train/validation/test splits, measured the base model, then trained independent LoRA adapters on **nested** prefixes of 128, 512, 2,048 and 4,096 examples with an identical recipe. The training size was selected on validation only, and the reserved test set was evaluated exactly once.

## Headline result

| Model (same 4-bit base, same runtime) | Validation (200) | Reserved test (240) |
|---|---:|---:|
| Base `unsloth/gemma-4-E2B-it` | 79 (39.5%) | 77 (32.1%) |
| LoRA, 128 examples | 67 (33.5%) | — |
| LoRA, 512 examples | 57 (28.5%) | — |
| LoRA, 2,048 examples | 126 (63.0%) | — |
| **LoRA, 4,096 examples** (selected) | **147 (73.5%)** | **121 (50.4%)** |

**Learning only appears once the dataset is large enough; small fine-tuning runs made the model worse.** At 128 and 512 examples the LoRA regressed (paired McNemar p = 0.036 and 0.002), while 2,048 and 4,096 examples improved substantially (p < 1e-7). On the reserved test set the selected adapter gains 71 questions and loses 27 (p ≈ 1.0e-05).

### The held-out composition regressed

The test split includes 80 **product × trig × chain** composites that never appear in training. The base model scored 14/80 there; the adapter scored **5/80** (paired p = 0.049). The fine-tune improved every family it trained on, but **negative transfer** on the unseen composition: it learned the training distribution's answer patterns more than a general product/chain rule. This is the main caveat of the study.

Reserved test by family:

| Family | Base | + adapter | McNemar p |
|---|---:|---:|---:|
| Polynomial (32) | 32 | 32 | 1.0 |
| Power / chain (32) | 17 | 28 | 0.019 |
| Product rule (32) | 4 | 13 | 0.049 |
| Rational (32) | 2 | 26 | 8.0e-07 |
| Trig chain (32) | 8 | 17 | 0.064 |
| **Product × trig × chain (80, hold-out)** | **14** | **5** | **0.049** |

Validation accuracy by problem family (n = 40 each):

| Family | Base | 128 | 512 | 2,048 | 4,096 |
|---|---:|---:|---:|---:|---:|
| Polynomial | 40 | 40 | 39 | 40 | 40 |
| Power / chain rule | 22 | 11 | 5 | 36 | 38 |
| Product rule | 10 | 9 | 0 | 7 | 11 |
| Rational functions | 3 | 2 | 1 | 19 | 32 |
| Trigonometric chain rule | 4 | 5 | 12 | 24 | 26 |

Reserved test results (240 problems, including a structural hold-out composition not present in training) are in [`results/final-report.json`](results/final-report.json).

## What "correct" means

Answers are graded by **symbolic equivalence** with SymPy, not string matching. A `Final answer:` line is required; an equivalent but differently written derivative counts as correct.

## Dataset

- 4,096 train / 200 validation / 240 test **generated fresh** with `scripts/generate.py` (seed `20260927`, SymPy 1.14).
- Five families: polynomial, product, power/chain, trig/chain, rational.
- The test set additionally contains **product × trig × chain composites** that never appear in train or validation — a structural hold-out to test transfer of the product and chain rules.
- SHA-256 hashes of every split are frozen in `data/manifest.json`.
- Samples are unique; `generate.py` checks and deduplicates.

## Training recipe (identical for all sizes)

- Base: `unsloth/gemma-4-E2B-it`, 4-bit QLoRA, bf16 compute
- Adapters: r = 8, alpha = 8, dropout = 0, attention + MLP
- 1 epoch, effective batch 4 (1 × 4 grad accumulation), lr 1e-4 linear decay, warmup 3%
- Loss on assistant answers only (`train_on_responses_only`)
- Nested prefixes: the 128/512/2,048 runs are exact subsets of the 4,096 run, shuffled within family then interleaved for balance
- The single prompt template and decoding (greedy) are shared by base and LoRA evaluations

## Reproduce

```bash
pip install -r requirements.txt
python scripts/generate.py                      # regenerates the frozen splits
python scripts/evaluate.py --split validation   # CPU/GGUF evaluation path (needs a local server)
python scripts/train_curve.py --phase baseline  # GPU baseline in the training runtime
python scripts/train_curve.py --phase train --size 4096
python scripts/report.py validation             # freezes the selected size
```

On a Runpod pod, `scripts/remote-run-curve.sh` runs the probe, baseline and all four sizes; `scripts/pod-constraints.txt` pins pip to the image's CUDA 12.8 torch stack.

## Files

```
scripts/generate.py         dataset generator + SHA-256 manifest
scripts/evaluate.py         symbolic grader + GGUF/OpenAI-compatible evaluation
scripts/train_curve.py      GPU baseline, LoRA training, evaluation, final test
scripts/report.py           paired comparison + McNemar exact test
scripts/test_grader.py      grader regression tests
data/                       frozen train/validation/test splits + manifest
results/                    summaries, per-question outputs, final report
```

## Caveats

- One seed, one small model, one synthetic distribution. McNemar p-values describe this benchmark only; they are not evidence of broad calculus ability.
- The base model already saw some calculus during pretraining; this measures task adaptation on a fresh distribution, not learning calculus from scratch.
- **Negative transfer on the held-out composition** (14/80 → 5/80): the adapter fits the training distribution more than it learns a general product/chain rule. Any claim of "learning calculus" from this run would be wrong.
- Product-rule accuracy stays weak even at 4,096 examples (11/40 validation, 13/32 test).
- 128 training examples were enough to make the model reliably worse (67 vs 79 validation); the curve is not monotonically "more data helps" at the small end here.

## License

Apache-2.0 (scripts and data). Model weights follow the Gemma license.
