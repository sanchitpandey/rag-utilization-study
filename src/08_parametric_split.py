"""
08_parametric_split.py — Classify questions as "known" or "unknown" to each model.

A question is labelled "known" for a given model if the model correctly answers it
with NO retrieval context (parametric-only).  "unknown" otherwise.

This script:
  1. Runs each model in "none" (no-retrieval) mode over the eval set.
  2. Labels each (model, question) pair as known / unknown.
  3. Writes per-model labels to a JSON mapping: {model_name: {qid: "known"|"unknown"}}

Outputs
-------
  <output>  — parametric_splits.json
  <none_preds_dir>/<model_name>.jsonl  — raw none-retrieval predictions
"""

from __future__ import annotations

import argparse
import gc
import json
import os

import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from utils import exact_match, load_jsonl, save_jsonl

MODELS = {
    "SmolLM2-360M": "HuggingFaceTB/SmolLM2-360M-Instruct",
    "Qwen2.5-1.5B": "Qwen/Qwen2.5-1.5B-Instruct",
    "Qwen2.5-3B": "Qwen/Qwen2.5-3B-Instruct",
    "Qwen2.5-7B": "Qwen/Qwen2.5-7B-Instruct",
}

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

BNB_CONFIG = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
) if torch.cuda.is_available() else None

NO_RETRIEVAL_PROMPT = (
    "Answer the following question in as few words as possible. "
    "Give ONLY the answer, nothing else.\n\n"
    "Question: {question}\n"
    "Answer:"
)


def build_no_retrieval_prompt(question: str) -> str:
    return NO_RETRIEVAL_PROMPT.format(question=question)


def run_inference(model, tokenizer, prompt: str, max_new_tokens: int = 50) -> tuple[str, int]:
    inputs = tokenizer(
        prompt, return_tensors="pt", truncation=True, max_length=2048
    ).to(DEVICE)
    input_len = inputs["input_ids"].shape[1]
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
    generated = outputs[0][input_len:]
    answer = tokenizer.decode(generated, skip_special_tokens=True).strip().split("\n")[0].strip()
    return answer, input_len


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Classify questions as known/unknown per model.")
    p.add_argument("--eval_path", default="data/eval.jsonl")
    p.add_argument("--none_preds_dir", default="data/none_predictions")
    p.add_argument("--output", default="data/parametric_splits.json")
    p.add_argument("--datasets", nargs="+", default=["nq", "hotpotqa"])
    p.add_argument(
        "--models",
        nargs="+",
        default=list(MODELS.keys()),
        choices=list(MODELS.keys()),
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()

    eval_data = load_jsonl(args.eval_path)
    eval_data = [ex for ex in eval_data if ex["dataset"] in args.datasets]
    print(f"Classifying {len(eval_data)} questions for {len(args.models)} models.")

    os.makedirs(args.none_preds_dir, exist_ok=True)
    parametric_splits: dict[str, dict[str, str]] = {}

    for model_name in args.models:
        model_id = MODELS[model_name]
        out_path = os.path.join(args.none_preds_dir, f"none__{model_name}.jsonl")

        # ---- load existing predictions if available ----
        if os.path.exists(out_path):
            existing = load_jsonl(out_path)
            existing_ids = {r["id"] for r in existing}
            if all(ex["id"] in existing_ids for ex in eval_data):
                print(f"[{model_name}] Reusing existing none predictions ({len(existing)} rows)")
                preds = {r["id"]: r["predicted"] for r in existing}
                parametric_splits[model_name] = {
                    ex["id"]: "known" if exact_match(str(preds[ex["id"]]), ex["answers"]) else "unknown"
                    for ex in eval_data
                }
                continue

        # ---- run inference ----
        print(f"\n[{model_name}] Loading model…")
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        load_kwargs: dict = {"device_map": "auto"} if DEVICE == "cuda" else {}
        if BNB_CONFIG is not None:
            load_kwargs["quantization_config"] = BNB_CONFIG
        else:
            load_kwargs["torch_dtype"] = torch.float32
        model = AutoModelForCausalLM.from_pretrained(model_id, **load_kwargs)
        model.eval()
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        results = []
        for ex in tqdm(eval_data, desc=f"{model_name} × none"):
            prompt = build_no_retrieval_prompt(ex["question"])
            answer, n_tokens = run_inference(model, tokenizer, prompt)
            results.append(
                {
                    "id": ex["id"],
                    "question": ex["question"],
                    "gold_answers": ex["answers"],
                    "dataset": ex["dataset"],
                    "predicted": answer,
                    "input_tokens": n_tokens,
                }
            )

        save_jsonl(out_path, results)
        print(f"[{model_name}] Saved {len(results)} predictions → {out_path}")

        parametric_splits[model_name] = {
            r["id"]: (
                "known" if exact_match(str(r["predicted"]), r["gold_answers"]) else "unknown"
            )
            for r in results
        }

        known = sum(1 for v in parametric_splits[model_name].values() if v == "known")
        print(f"[{model_name}] Known: {known}/{len(eval_data)} ({known/len(eval_data)*100:.1f}%)")

        del model, tokenizer
        torch.cuda.empty_cache()
        gc.collect()

    with open(args.output, "w") as f:
        json.dump(parametric_splits, f, indent=2)
    print(f"\nParametric splits saved → {args.output}")


if __name__ == "__main__":
    main()