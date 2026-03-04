"""
04_scaling_grid.py — Preliminary API-based grid: Groq models × retrieval conditions.

Runs a 4-model × 4-retrieval-condition grid using the Groq API, evaluating 200 NQ
questions per cell. This was used for initial hypothesis validation before the
full local-model grid (09_full_grid.py).

Models (via Groq)
-----------------
  llama-3.1-8b, llama-3.3-70b

Retrieval conditions
--------------------
  none, bm25, dense, hybrid

Outputs
-------
  <output_dir>/<condition>__<model>.jsonl   — per-cell predictions
  <output_dir>/full_grid.csv               — combined results
  <output_dir>/grid_summary.csv            — per-(model, condition, dataset) summary

Requirements
------------
  pip install groq
  export GROQ_API_KEY=<your_key>
"""

from __future__ import annotations

import argparse
import os
import random
import time

import numpy as np
import pandas as pd
from groq import Groq
from tqdm import tqdm

from utils import exact_match, f1_score, load_jsonl, save_jsonl

# ---------------------------------------------------------------------------
# Model registry
# ---------------------------------------------------------------------------

MODELS: dict[str, str] = {
    "llama-3.1-8b": "llama-3.1-8b-instant",
    "llama-3.3-70b": "llama-3.3-70b-versatile",
}

MODEL_DELAYS: dict[str, float] = {
    "llama-3.1-8b": 2.0,
    "llama-3.3-70b": 4.0,
}

EVAL_SUBSET = 200


# ---------------------------------------------------------------------------
# Prompt
# ---------------------------------------------------------------------------

def build_qa_prompt(question: str, passages: list[dict], max_passages: int = 5) -> str:
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
# API call with retry + backoff
# ---------------------------------------------------------------------------

def call_groq(
    client: Groq,
    prompt: str,
    model_id: str,
    max_retries: int = 3,
) -> str:
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=model_id,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=50,
                temperature=0.0,
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            if "rate_limit" in str(e).lower() or "429" in str(e):
                wait = (2 ** attempt) + random.uniform(0, 1)
                print(f"  Rate limited — waiting {wait:.1f}s…")
                time.sleep(wait)
            else:
                print(f"  API error (attempt {attempt+1}): {e}")
                time.sleep(1)
    return ""


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="API-based preliminary scaling grid.")
    p.add_argument("--bm25_results", default="data/results/bm25_eval_results.jsonl")
    p.add_argument("--dense_results", default="data/results/dense_eval_results.jsonl")
    p.add_argument("--hybrid_results", default="data/results/hybrid_eval_results.jsonl")
    p.add_argument("--output_dir", default="data/grid_api")
    p.add_argument("--eval_subset", type=int, default=EVAL_SUBSET)
    p.add_argument(
        "--models",
        nargs="+",
        default=list(MODELS.keys()),
        choices=list(MODELS.keys()),
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()

    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        raise EnvironmentError("GROQ_API_KEY environment variable not set.")
    client = Groq(api_key=api_key)

    # Build retrieval condition lookup
    bm25_data = load_jsonl(args.bm25_results)
    dense_data = load_jsonl(args.dense_results)
    hybrid_data = load_jsonl(args.hybrid_results)

    retrieval_levels: dict[str, list[dict]] = {
        "bm25": bm25_data,
        "dense": dense_data,
        "hybrid": hybrid_data,
        "none": [
            {**{k: ex[k] for k in ("id", "question", "answers", "dataset", "type")}, "retrieved": []}
            for ex in bm25_data
        ],
    }

    os.makedirs(args.output_dir, exist_ok=True)
    all_results: list[dict] = []

    for ret_name, ret_data in retrieval_levels.items():
        for model_name in args.models:
            model_id = MODELS[model_name]
            outfile = os.path.join(args.output_dir, f"{ret_name}__{model_name}.jsonl")

            if os.path.exists(outfile):
                existing = load_jsonl(outfile)
                if len(existing) >= args.eval_subset:
                    print(f"SKIP {ret_name} × {model_name}: already complete ({len(existing)})")
                    all_results.extend(existing)
                    continue

            print(f"\nRunning: {ret_name} × {model_name}")
            batch: list[dict] = []

            for ex in tqdm(ret_data[: args.eval_subset], desc=f"{ret_name}×{model_name}"):
                prompt = build_qa_prompt(ex["question"], ex["retrieved"])
                t0 = time.time()
                answer = call_groq(client, prompt, model_id)
                latency = time.time() - t0

                result = {
                    "id": ex["id"],
                    "question": ex["question"],
                    "gold_answers": ex["answers"],
                    "dataset": ex["dataset"],
                    "type": ex["type"],
                    "retrieval": ret_name,
                    "model": model_name,
                    "predicted": answer,
                    "em": exact_match(answer, ex["answers"]),
                    "f1": f1_score(answer, ex["answers"]),
                    "latency": latency,
                }
                batch.append(result)
                time.sleep(MODEL_DELAYS.get(model_name, 3.0))

            save_jsonl(outfile, batch)
            all_results.extend(batch)
            print(f"  Saved {len(batch)} results → {outfile}")

    # Consolidate
    df = pd.DataFrame(all_results)
    df.to_csv(os.path.join(args.output_dir, "full_grid.csv"), index=False)

    summary = (
        df.groupby(["retrieval", "model", "dataset"])
        .agg(em=("em", "mean"), f1=("f1", "mean"), count=("em", "count"))
        .reset_index()
    )
    summary.to_csv(os.path.join(args.output_dir, "grid_summary.csv"), index=False)

    print(f"\nConsolidated {len(df)} rows → {args.output_dir}/full_grid.csv")

    print("\n===== EM by Retrieval × Model =====")
    pivot = df.pivot_table(values="em", index="retrieval", columns="model", aggfunc="mean")
    print((pivot * 100).round(1).to_string())


if __name__ == "__main__":
    main()