"""Evaluate Gemma's differentiation accuracy with a safe symbolic checker.

Requires requests + sympy. Accepts ONLY an explicitly marked final expression;
parses a restricted Python AST (not sympify of untrusted model text).
"""
import argparse
import ast
import json
import re
import time
from collections import defaultdict
from pathlib import Path

import requests
import sympy as sp

ROOT = Path(__file__).resolve().parent
X = sp.Symbol("x")


def safe_expression(text):
    if len(text) > 500:
        raise ValueError("expression too long")
    text = text.replace("^", "**")
    # School-math shorthand: 16x, 3(x+1), (x+1)(x-2), x(x+1).
    # Only permit these explicit rewrites; never eval untrusted text.
    text = re.sub(r"(?<=\d)(?=x\b|\()", "*", text)
    text = re.sub(r"(?<=\))(?=\(|x\b)", "*", text)
    text = re.sub(r"(?<=x)(?=\()", "*", text)
    node = ast.parse(text, mode="eval").body

    def visit(n, depth=0):
        if depth > 35:
            raise ValueError("expression too deep")
        if isinstance(n, ast.Name) and n.id == "x":
            return X
        if isinstance(n, ast.Constant) and type(n.value) is int and abs(n.value) <= 1000000:
            return sp.Integer(n.value)
        if isinstance(n, ast.UnaryOp) and isinstance(n.op, (ast.UAdd, ast.USub)):
            value = visit(n.operand, depth + 1)
            return value if isinstance(n.op, ast.UAdd) else -value
        if isinstance(n, ast.BinOp) and isinstance(n.op, (ast.Add, ast.Sub, ast.Mult, ast.Div, ast.Pow)):
            a, b = visit(n.left, depth + 1), visit(n.right, depth + 1)
            if isinstance(n.op, ast.Add): return a + b
            if isinstance(n.op, ast.Sub): return a - b
            if isinstance(n.op, ast.Mult): return a * b
            if isinstance(n.op, ast.Div): return a / b
            if b.is_Integer and abs(b) > 16:
                raise ValueError("power too large")
            if not b.is_Integer:
                raise ValueError("non-integer exponent")
            return a ** b
        if (isinstance(n, ast.Call) and isinstance(n.func, ast.Name)
                and n.func.id in ("sin", "cos") and len(n.args) == 1 and not n.keywords):
            return getattr(sp, n.func.id)(visit(n.args[0], depth + 1))
        raise ValueError("unsupported syntax")

    result = visit(node)
    if not result.free_symbols.issubset({X}):
        raise ValueError("unexpected symbol")
    return result


def final_expression(reply):
    lines = [line.strip() for line in reply.splitlines() if line.strip()]
    matches = [line.partition(":")[2].strip().strip("`$") for line in lines
               if re.match(r"(?i)^final answer\s*:", line)]
    if not matches:
        raise ValueError("missing Final answer: line")
    return safe_expression(matches[-1])


def grade(reply, expected):
    try:
        predicted = final_expression(reply)
        gold = safe_expression(expected)
        return bool(sp.simplify(predicted - gold) == 0), None
    except Exception as exc:
        return False, f"{type(exc).__name__}: {str(exc)[:100]}"


def completion(url, prompt, max_tokens):
    body = {
        "model": "local", "messages": [{"role": "user", "content": prompt}],
        "temperature": 0, "top_p": 1, "max_tokens": max_tokens,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    r = requests.post(url.rstrip("/") + "/chat/completions", json=body, timeout=(10, 240))
    r.raise_for_status()
    response = r.json()
    msg = response["choices"][0]["message"]
    return msg.get("content") or "", response["choices"][0].get("finish_reason")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=("validation", "test"), default="validation")
    ap.add_argument("--url", default="http://127.0.0.1:8083/v1")
    ap.add_argument("--output", default="results/gemma4-e2b-q4-baseline-validation.jsonl")
    ap.add_argument("--max-tokens", type=int, default=256)
    ap.add_argument("--limit", type=int, default=0, help="0 means all rows")
    args = ap.parse_args()
    with (ROOT / "data" / f"{args.split}.jsonl").open(encoding="utf-8") as f:
        rows = [json.loads(line) for line in f]
    if args.limit:
        rows = rows[:args.limit]
    output = ROOT / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    previous = {}
    if output.exists():
        with output.open(encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    result = json.loads(line)
                    previous[result["index"]] = result
    with output.open("a", encoding="utf-8", buffering=1) as f:
        for i, row in enumerate(rows):
            if i in previous:
                continue
            started = time.monotonic()
            try:
                reply, finish = completion(args.url, row["question"], args.max_tokens)
                ok, error = grade(reply, row["answer"])
            except Exception as exc:
                reply, finish, ok, error = "", "error", False, f"{type(exc).__name__}: {str(exc)[:100]}"
            result = {"index": i, "family": row["family"], "expression": row["expression"],
                      "expected": row["answer"], "reply": reply, "correct": ok,
                      "error": error, "finish_reason": finish,
                      "seconds": round(time.monotonic() - started, 2)}
            f.write(json.dumps(result, ensure_ascii=False) + "\n")
            previous[i] = result
            if (i + 1) % 20 == 0:
                print("completed", i + 1, "/", len(rows), "accuracy_so_far",
                      round(sum(v["correct"] for v in previous.values()) / len(previous), 3), flush=True)
    # Grader changes can re-score saved replies without spending more CPU inference.
    for r in previous.values():
        if r["finish_reason"] != "error":
            r["correct"], r["error"] = grade(r["reply"], r["expected"])
    with output.open("w", encoding="utf-8") as f:
        for i in sorted(previous):
            f.write(json.dumps(previous[i], ensure_ascii=False) + "\n")
    groups = defaultdict(list)
    for r in previous.values(): groups[r["family"]].append(r)
    summary = {"split": args.split, "grader_version": 2, "evaluated": len(previous), "requested": len(rows),
               "overall_accuracy": sum(x["correct"] for x in previous.values()) / len(previous),
               "format_or_parse_errors": sum(bool(x["error"]) for x in previous.values()),
               "by_family": {k: {"correct":sum(x["correct"] for x in v),"n":len(v)} for k,v in sorted(groups.items())}}
    (output.parent / (output.stem + "-summary.json")).write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
