"""
09_full_grid.py — 4 × 3 evaluation grid: models × {none, noisy, oracle}.

For each (model, retrieval_condition) pair, run inference on the eval set
and compute EM / F1 broken down by parametric split (known / unknown).

Models
------
  SmolLM2-360M, Qwen2.5-1.5B, Qwen2.5-3B, Qwen2.5-7B

Retrieval conditions
--------------------
  none   — no retrieved context
  noisy  — dense retrieval (E5-large-v2 + FAISS)
  oracle — gold passage guaranteed to contain the answer

Outputs
-------
  <output_dir>/<condition>__<model>.jsonl     — per-example results
  <output_dir>/full_grid_v2.csv              — combined results (12 000 rows)
  <output_dir>/corpus_only_grid_v2.csv       — filtered to corpus-match oracle only
  <output_dir>/grid_summary_v2.csv           — aggregated statistics
"""

from __future__ import annotations

import argparse
import gc
import glob
import json
import os
import shutil

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from utils import exact_match, f1_score, load_jsonl, save_jsonl

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

MODELS: dict[str, str] = {
    "SmolLM2-360M": "HuggingFaceTB/SmolLM2-360M-Instruct",
    "Qwen2.5-1.5B": "Qwen/Qwen2.5-1.5B-Instruct",
    "Qwen2.5-3B": "Qwen/Qwen2.5-3B-Instruct",
    "Qwen2.5-7B": "Qwen/Qwen2.5-7B-Instruct",
}

BNB_CONFIG = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
) if torch.cuda.is_available() else None


# ---------------------------------------------------------------------------
# Prompt
# ---------------------------------------------------------------------------

def build_prompt(question: str, passages: list[dict], max_passages: int = 5) -> str:
    if not passages:
        return (
            "Answer the following question in as few words as possible. "
            "Give ONLY the answer, nothing else.\n\n"
            f"Question: {question}\nAnswer:"
        )
    ctx = "\n".join(f"[{i+1}] {p['text']}" for i, p in enumerate(passages[:max_passages]))
    return (
        "Answer the following question based on the provided context. "
        "Give ONLY the answer in as few words as possible. "
        "If the context doesn't contain the answer, make your best guess.\n\n"
        f"Context:\n{ctx}\n\nQuestion: {question}\nAnswer:"
    )


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

def run_inference(
    model,
    tokenizer,
    prompt: str,
    max_new_tokens: int = 50,
) -> tuple[str, int]:
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


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Full 4×3 evaluation grid.")
    p.add_argument("--eval_path", default="data/eval.jsonl")
    p.add_argument("--oracle_path", default="data/results/oracle_eval_results.jsonl")
    p.add_argument("--dense_path", default="data/results/dense_eval_results.jsonl")
    p.add_argument("--parametric_path", default="data/parametric_splits.json")
    p.add_argument("--none_preds_dir", default="data/none_predictions")
    p.add_argument("--output_dir", default="data/grid_v2")
    p.add_argument("--datasets", nargs="+", default=["nq", "hotpotqa"])
    p.add_argument(
        "--models",
        nargs="+",
        default=list(MODELS.keys()),
        choices=list(MODELS.keys()),
    )
    return p.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    # Load eval data
    eval_data = load_jsonl(args.eval_path)
    eval_data = [ex for ex in eval_data if ex["dataset"] in args.datasets]
    eval_lookup = {ex["id"]: ex for ex in eval_data}
    eval_ids = set(ex["id"] for ex in eval_data)
    print(f"Eval: {len(eval_data)} questions")

    # Load retrieval results
    oracle_all = load_jsonl(args.oracle_path)
    oracle_lookup = {ex["id"]: ex["retrieved"] for ex in oracle_all if ex["id"] in eval_ids}
    oracle_type_lookup = {ex["id"]: ex.get("oracle_type", "unknown") for ex in oracle_all}

    dense_all = load_jsonl(args.dense_path)
    dense_lookup = {ex["id"]: ex["retrieved"] for ex in dense_all if ex["id"] in eval_ids}

    with open(args.parametric_path) as f:
        parametric_splits: dict[str, dict[str, str]] = json.load(f)

    retrieval_conditions = {
        "none": {qid: [] for qid in eval_ids},
        "noisy": dense_lookup,
        "oracle": oracle_lookup,
    }

    # ------------------------------------------------------------------ #
    # Grid run
    # ------------------------------------------------------------------ #
    for model_name in args.models:
        model_id = MODELS[model_name]
        conditions_todo = []

        for cond_name in ["none", "noisy", "oracle"]:
            outfile = os.path.join(args.output_dir, f"{cond_name}__{model_name}.jsonl")
            if os.path.exists(outfile):
                existing = load_jsonl(outfile)
                if len(existing) >= len(eval_data):
                    print(f"SKIP {cond_name} × {model_name}: complete ({len(existing)})")
                    continue
            conditions_todo.append(cond_name)

        # Reuse existing "none" predictions if available
        if "none" in conditions_todo:
            none_path = os.path.join(args.none_preds_dir, f"none__{model_name}.jsonl")
            if os.path.exists(none_path):
                existing_none = load_jsonl(none_path)
                existing_ids = {r["id"] for r in existing_none}
                if eval_ids.issubset(existing_ids):
                    reformatted = []
                    for r in existing_none:
                        if r["id"] not in eval_ids:
                            continue
                        reformatted.append({
                            "id": r["id"],
                            "question": r["question"],
                            "gold_answers": r["gold_answers"],
                            "dataset": r["dataset"],
                            "retrieval": "none",
                            "model": model_name,
                            "predicted": r["predicted"],
                            "em": exact_match(str(r["predicted"]), r["gold_answers"]),
                            "f1": f1_score(str(r["predicted"]), r["gold_answers"]),
                            "parametric": parametric_splits.get(model_name, {}).get(r["id"], "unknown"),
                            "oracle_type": oracle_type_lookup.get(r["id"], "unknown"),
                            "input_tokens": r.get("input_tokens", 0),
                        })
                    outfile = os.path.join(args.output_dir, f"none__{model_name}.jsonl")
                    save_jsonl(outfile, reformatted)
                    print(f"Reused none predictions for {model_name} ({len(reformatted)})")
                    conditions_todo.remove("none")

        if not conditions_todo:
            continue

        print(f"\n{'='*60}")
        print(f"Loading: {model_name} ({model_id})")
        print(f"Conditions: {conditions_todo}")
        print(f"{'='*60}")

        tokenizer = AutoTokenizer.from_pretrained(model_id)
        load_kwargs: dict = {"device_map": "auto"} if DEVICE == "cuda" else {}
        if BNB_CONFIG is not None:
            load_kwargs["quantization_config"] = BNB_CONFIG
        else:
            load_kwargs["dtype"] = torch.float32
        model = AutoModelForCausalLM.from_pretrained(model_id, **load_kwargs)
        model.eval()
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        for cond_name in conditions_todo:
            outfile = os.path.join(args.output_dir, f"{cond_name}__{model_name}.jsonl")
            retrieval_lookup = retrieval_conditions[cond_name]
            results = []

            for ex in tqdm(eval_data, desc=f"{cond_name} × {model_name}"):
                passages = retrieval_lookup.get(ex["id"], [])
                prompt = build_prompt(ex["question"], passages)
                answer, n_tokens = run_inference(model, tokenizer, prompt)
                em = exact_match(answer, ex["answers"])
                f1 = f1_score(answer, ex["answers"])
                results.append({
                    "id": ex["id"],
                    "question": ex["question"],
                    "gold_answers": ex["answers"],
                    "dataset": ex["dataset"],
                    "retrieval": cond_name,
                    "model": model_name,
                    "predicted": answer,
                    "em": em,
                    "f1": f1,
                    "parametric": parametric_splits.get(model_name, {}).get(ex["id"], "unknown"),
                    "oracle_type": oracle_type_lookup.get(ex["id"], "unknown"),
                    "input_tokens": n_tokens,
                })

            save_jsonl(outfile, results)
            em_mean = np.mean([r["em"] for r in results]) * 100
            print(f"\n  {cond_name} × {model_name}: EM = {em_mean:.1f}%")
            for ds in args.datasets:
                ds_em = [r["em"] for r in results if r["dataset"] == ds]
                if ds_em:
                    print(f"    {ds}: {np.mean(ds_em)*100:.1f}%")

        del model, tokenizer
        torch.cuda.empty_cache()
        gc.collect()
        print(f"Unloaded {model_name}")

    # ------------------------------------------------------------------ #
    # Consolidate
    # ------------------------------------------------------------------ #
    all_results = []
    for f in sorted(glob.glob(os.path.join(args.output_dir, "*.jsonl"))):
        data = load_jsonl(f)
        all_results.extend(data)

    df = pd.DataFrame(all_results)
    df.to_csv(os.path.join(args.output_dir, "full_grid_v2.csv"), index=False)

    corpus_match_ids = {qid for qid, ot in oracle_type_lookup.items() if ot == "corpus_match"}
    df_corpus = df[df["id"].isin(corpus_match_ids)]
    df_corpus.to_csv(os.path.join(args.output_dir, "corpus_only_grid_v2.csv"), index=False)

    summary = (
        df.groupby(["model", "retrieval", "dataset", "parametric"])
        .agg(em=("em", "mean"), f1=("f1", "mean"), count=("em", "count"))
        .reset_index()
    )
    summary.to_csv(os.path.join(args.output_dir, "grid_summary_v2.csv"), index=False)

    print(f"\nFull grid: {len(df)} rows saved to {args.output_dir}/")
    print(f"Corpus-only: {len(df_corpus)} rows")


if __name__ == "__main__":
    main()