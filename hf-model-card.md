---
base_model: unsloth/gemma-4-E2B-it
library_name: peft
tags:
- lora
- peft
- math
- calculus
- gemma
datasets:
- ankit-pn/gemma4-calculus-differentiation
---

# Gemma 4 E2B — Calculus Differentiation LoRA (4,096 examples)

A LoRA adapter that teaches [`unsloth/gemma-4-E2B-it`](https://huggingface.co/unsloth/gemma-4-E2B-it) to return one-line derivatives in SymPy syntax.

This adapter is the **4,096-example** run selected by validation accuracy from a controlled learning-curve study (128 / 512 / 2,048 / 4,096 nested training subsets, identical recipe). Code, data and the full analysis: [GitHub — gemma4-calculus-sft](https://github.com/ankit-pn/gemma4-calculus-sft).

## Results (same 4-bit base, same runtime, greedy decoding)

| Model | Validation (200, symbolic equivalence) | Reserved test (240) |
|---|---:|---:|
| Base Gemma 4 E2B | 79 (39.5%) | 77 (32.1%) |
| + this adapter | **147 (73.5%)** | **121 (50.4%)** |

Paired gains/losses vs. base: validation **+79 / −11** (McNemar exact p ≈ 7.8e-14); test **+71 / −27** (p ≈ 1.0e-05).

Validation by family (n = 40 each):

| Family | Base | + this adapter |
|---|---:|---:|
| Polynomial | 40 | 40 |
| Power / chain rule | 22 | 38 |
| Product rule | 10 | 11 |
| Rational functions | 3 | 32 |
| Trig chain rule | 4 | 26 |

Test by family (n = 32, hold-out 80):

| Family | Base | + this adapter | McNemar p |
|---|---:|---:|---:|
| Polynomial | 32 | 32 | 1.0 |
| Power / chain | 17 | 28 | 0.019 |
| Product rule | 4 | 13 | 0.049 |
| Rational | 2 | 26 | 8.0e-07 |
| Trig chain | 8 | 17 | 0.064 |
| Product × trig × chain (hold-out) | **14** | **5** | 0.049 |

**Important:** the structurally held-out composition regressed — the adapter improved every trained family but scored worse than the base model on product × trig × chain problems it never saw. The gain reflects the training distribution, not a fully general derivative rule.

Small-data caveat: the 128- and 512-example adapters from the same study were **worse** than the base model. The improvement is specific to the larger dataset.

## Usage

```python
from peft import PeftModel
from unsloth import FastModel
from unsloth.chat_templates import get_chat_template

model, tokenizer = FastModel.from_pretrained(
    model_name="unsloth/gemma-4-E2B-it", load_in_4bit=True, max_seq_length=512,
)
model = PeftModel.from_pretrained(model, "ankit-pn/gemma4-e2b-calculus-lora")
tokenizer = get_chat_template(tokenizer, chat_template="gemma-4")

question = (
    "Differentiate f(x) = 3*x**4 - 2*x with respect to x. "
    "Return one line: Final answer: <SymPy expression using x, +, -, *, /, **, sin, cos>. "
    "Do not explain."
)
inputs = tokenizer.apply_chat_template(
    [{"role": "user", "content": [{"type": "text", "text": question}]}],
    tokenize=True, add_generation_prompt=True, return_dict=True,
    return_tensors="pt", enable_thinking=False,
).to("cuda")
print(tokenizer.decode(model.generate(**inputs, max_new_tokens=96, do_sample=False)[0]))
```

The adapter was trained with the `gemma-4` chat template, thinking disabled. Use the same template at inference time.

## Training

- Base: `unsloth/gemma-4-E2B-it`, 4-bit QLoRA, bf16
- LoRA: r = 8, alpha = 8, dropout = 0, attention + MLP, `random_state` 3407
- 1 epoch, 1,024 steps, effective batch 4, lr 1e-4 (linear decay, 3% warmup), `adamw_8bit`
- Loss on assistant responses only; max length 512
- Data: [ankit-pn/gemma4-calculus-differentiation](https://huggingface.co/datasets/ankit-pn/gemma4-calculus-differentiation) train split (first 4,096 in the published order)

## Limitations

- Trained and evaluated on **synthetic, single-variable** differentiation (polynomials, products, powers, simple trig, rationals). It does not make the model a general calculus tutor.
- Product-rule accuracy remains low (~28% on validation) even after fine-tuning.
- Training data contains no chains beyond one composition level and no exponentials, logarithms or inverse trig.
- Evaluation uses SymPy symbolic equivalence of a one-line answer; multi-step reasoning is out of scope.
