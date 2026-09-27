"""Four independent Gemma 4 E2B LoRA SFT sizes, with a matched GPU baseline.

Usage on a CUDA pod:
    python train_curve.py --phase baseline
    python train_curve.py --phase train --size 128
    python train_curve.py --phase train --size 512
    python train_curve.py --phase train --size 2048
    python train_curve.py --phase train --size 4096
    python train_curve.py --phase final-base
    python train_curve.py --phase final-lora --size SELECTED_SIZE

Each invocation reloads the same unmodified pretrained 4-bit model. Final
phases require a separate explicit transfer of the reserved test file.
"""
import argparse
import hashlib
import json
import random
import time
from collections import defaultdict
from pathlib import Path

from unsloth import FastModel
from unsloth.chat_templates import get_chat_template, train_on_responses_only
import torch
from datasets import Dataset
from trl import SFTConfig, SFTTrainer

from evaluate import grade

ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / "output"
DATA = ROOT / "data"
MODEL = "unsloth/gemma-4-E2B-it"
FAMILIES = ("polynomial", "product", "power_chain", "trig_chain", "rational")
SEED = 20260927


def read_rows(split):
    with (DATA / f"{split}.jsonl").open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def training_order():
    groups = defaultdict(list)
    for row in read_rows("train"):
        groups[row["family"]].append(row)
    rng = random.Random(SEED)
    for family in FAMILIES:
        rng.shuffle(groups[family])
    selected = []
    while any(groups.values()):
        for family in FAMILIES:
            if groups[family]:
                selected.append(groups[family].pop())
    assert len(selected) == 4096
    order = [row["expression"] for row in selected]
    sha = hashlib.sha256(json.dumps(order).encode()).hexdigest()
    print("train_order_sha256", sha, flush=True)
    return selected, sha


def evaluation(model, tokenizer, split, label, limit=0):
    model.eval()
    rows = read_rows(split)
    if limit:
        rows = rows[:limit]
    target = OUTPUT / (label + ".jsonl")
    saved = {}
    if target.exists():
        for line in target.read_text(encoding="utf-8").splitlines():
            if line.strip():
                record = json.loads(line)
                i = record["index"]
                if i >= len(rows) or record["expression"] != rows[i]["expression"]:
                    raise RuntimeError(f"Existing {label} result does not match frozen {split} split")
                saved[i] = record
    with target.open("a", encoding="utf-8", buffering=1) as f, torch.inference_mode():
        for i, row in enumerate(rows):
            if i in saved:
                continue
            inputs = tokenizer.apply_chat_template(
                [{"role": "user", "content": [{"type": "text", "text": row["question"]}]}],
                tokenize=True, add_generation_prompt=True, return_dict=True,
                return_tensors="pt", enable_thinking=False,
            ).to("cuda")
            out = model.generate(
                **inputs, max_new_tokens=192, do_sample=False, use_cache=True,
                pad_token_id=tokenizer.tokenizer.eos_token_id,
            )
            reply = tokenizer.decode(
                out[0, inputs["input_ids"].shape[-1]:], skip_special_tokens=True,
            )
            correct, error = grade(reply, row["answer"])
            record = {"index": i, "family": row["family"], "expression": row["expression"],
                      "expected": row["answer"], "reply": reply,
                      "correct": correct, "error": error}
            saved[i] = record
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            if (i + 1) % 40 == 0:
                print(label, i + 1, "/", len(rows), "correct", sum(x["correct"] for x in saved.values()), flush=True)
    records = [saved[i] for i in range(len(rows))]
    for row in records:
        row["correct"], row["error"] = grade(row["reply"], row["expected"])
    groups = defaultdict(list)
    for row in records:
        groups[row["family"]].append(row)
    summary = {"split": split, "n": len(records), "correct": sum(x["correct"] for x in records),
               "accuracy": sum(x["correct"] for x in records) / len(records),
               "parse_errors": sum(bool(x["error"]) for x in records),
               "by_family": {k: {"correct": sum(x["correct"] for x in v), "n": len(v)}
                             for k, v in sorted(groups.items())}}
    (OUTPUT / (label + "-summary.json")).write_text(json.dumps(summary, indent=2) + "\n")
    print("EVAL_SUMMARY", label, json.dumps(summary), flush=True)


def training(model, tokenizer, size):
    ordered, digest = training_order()
    selected = ordered[:size]
    print("train_size", size, "family_counts",
          {f: sum(r["family"] == f for r in selected) for f in FAMILIES}, flush=True)

    def format_row(row):
        messages = [
            {"role": "user", "content": [{"type": "text", "text": row["question"]}]},
            {"role": "assistant", "content": [{"type": "text", "text": "Final answer: " + row["answer"]}]},
        ]
        return {"text": tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False,
            enable_thinking=False,
        ).removeprefix("<bos>")}

    dataset = Dataset.from_list([format_row(row) for row in selected])
    print("first_example", dataset[0]["text"][:450], flush=True)
    model = FastModel.get_peft_model(
        model, finetune_vision_layers=False, finetune_language_layers=True,
        finetune_attention_modules=True, finetune_mlp_modules=True,
        r=8, lora_alpha=8, lora_dropout=0, bias="none", random_state=3407,
    )
    trainer = SFTTrainer(
        model=model, processing_class=tokenizer, train_dataset=dataset,
        args=SFTConfig(
            output_dir=str(OUTPUT / f"checkpoint-{size}"),
            dataset_text_field="text", max_length=512,
            per_device_train_batch_size=1, gradient_accumulation_steps=4,
            num_train_epochs=1, learning_rate=1e-4,
            warmup_ratio=0.03, lr_scheduler_type="linear",
            optim="adamw_8bit", logging_steps=max(1, size // 64),
            save_strategy="no", bf16=torch.cuda.is_bf16_supported(),
            fp16=not torch.cuda.is_bf16_supported(),
            seed=3407, report_to="none",
        ),
    )
    trainer = train_on_responses_only(
        trainer, instruction_part="<|turn>user\n", response_part="<|turn>model\n",
    )
    labels = trainer.train_dataset[0]["labels"]
    response_text = tokenizer.decode([x for x in labels if x != -100])
    if "Final answer:" not in response_text or len(response_text) > 250:
        raise RuntimeError("Incorrect assistant-only loss mask: " + response_text[:250])
    print("masked_response_check", response_text[:200], flush=True)
    started = time.monotonic()
    result = trainer.train()
    dest = OUTPUT / f"adapter-{size}"
    model.save_pretrained(str(dest))
    tokenizer.save_pretrained(str(dest))
    summary = {"train_size": size, "steps": result.global_step,
               "train_loss": result.training_loss,
               "training_seconds": round(time.monotonic() - started, 1),
               "training_order_sha256": digest, "model": MODEL,
               "r": 8, "learning_rate": 1e-4, "epochs": 1}
    (OUTPUT / f"train-{size}-summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print("TRAIN_SUMMARY", json.dumps(summary), flush=True)
    evaluation(model, tokenizer, "validation", f"lora-{size}-validation")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=("probe", "baseline", "train", "final-base", "final-lora"), required=True)
    parser.add_argument("--size", type=int, choices=(128, 512, 2048, 4096))
    args = parser.parse_args()
    if args.phase in ("train", "final-lora") and not args.size:
        parser.error("--size is required for this phase")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA not available; refusing to use paid CPU pod")
    OUTPUT.mkdir(exist_ok=True)
    print("GPU", torch.cuda.get_device_name(0), "PHASE", args.phase, flush=True)
    model, tokenizer = FastModel.from_pretrained(
        model_name=MODEL, max_seq_length=512, load_in_4bit=True, full_finetuning=False,
    )
    tokenizer = get_chat_template(tokenizer, chat_template="gemma-4")
    if args.phase == "probe":
        evaluation(model, tokenizer, "validation", "probe-validation", limit=5)
    elif args.phase == "baseline":
        evaluation(model, tokenizer, "validation", "base-validation")
    elif args.phase == "train":
        training(model, tokenizer, args.size)
    elif args.phase == "final-base":
        evaluation(model, tokenizer, "test", "base-test")
    else:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, str(OUTPUT / f"adapter-{args.size}"))
        evaluation(model, tokenizer, "test", f"lora-{args.size}-test")


if __name__ == "__main__":
    main()
