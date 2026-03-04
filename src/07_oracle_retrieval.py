"""
07_oracle_retrieval.py — Construct oracle retrieval results.

For each eval question, find the best corpus passage that contains a gold answer.
Fall back to a synthetic passage (clearly labelled) when no corpus match exists.

Outputs
-------
  <output>  — oracle_eval_results.jsonl
    Each record includes "oracle_type": "corpus_match" | "synthetic"

  <output_dir>/oracle_stats.json  — validation metadata
"""

from __future__ import annotations

import argparse
import json
import os
import re
from collections import defaultdict

import numpy as np
import pandas as pd
from tqdm import tqdm

from utils import compute_hit_rate, load_jsonl, save_jsonl


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def build_answer_index(
    corpus_texts_lower: np.ndarray,
    all_answers: set[str],
) -> dict[str, list[int]]:
    """Map each answer string to the list of passage indices containing it."""
    sorted_answers = sorted(all_answers, key=len, reverse=True)
    answer_pattern = re.compile("|".join(re.escape(a) for a in sorted_answers))

    answer_to_passages: dict[str, list[int]] = defaultdict(list)
    for idx in tqdm(range(len(corpus_texts_lower)), desc="Indexing passages by answer"):
        text = corpus_texts_lower[idx]
        for match in set(answer_pattern.findall(text)):
            answer_to_passages[match].append(idx)

    return answer_to_passages


def select_best_passage(
    passage_indices: list[int],
    answer: str,
    corpus_texts_lower: np.ndarray,
) -> tuple[int | None, float]:
    """
    Score candidate passages and return the best index.

    Criteria (descending priority):
      1. Answer appears as a word-boundary match (not inside another word).
      2. Answer appears early in the passage (higher relevance density).
      3. Passage is focused (~50 words is ideal).
    """
    answer_lower = answer.lower().strip()
    try:
        boundary_pattern = re.compile(r"(?<!\w)" + re.escape(answer_lower) + r"(?!\w)")
    except re.error:
        boundary_pattern = None

    best_idx: int | None = None
    best_score = -1.0

    for pidx in passage_indices:
        text = corpus_texts_lower[pidx]
        pos = text.find(answer_lower)
        if pos < 0:
            continue
        boundary_match = bool(boundary_pattern and boundary_pattern.search(text))
        position_score = 1.0 - (pos / max(len(text), 1))
        focus_score = 1.0 / max(len(text.split()) / 50, 1)
        score = (2.0 if boundary_match else 0.0) + position_score + focus_score
        if score > best_score:
            best_score = score
            best_idx = pidx

    return best_idx, best_score


def make_synthetic_passage(question: str, answer: str) -> dict:
    text = (
        f"{answer} is commonly known in reference to the topic of {question}. "
        f"According to available sources, {answer} is the documented answer. "
        f"This information relates to {question}."
    )
    return {"id": "synthetic", "title": "Synthetic Oracle", "text": text, "score": 100.0}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build oracle retrieval results.")
    p.add_argument("--eval_path", default="data/eval.jsonl")
    p.add_argument("--corpus_path", default="data/corpus/wiki_500k.parquet")
    p.add_argument("--dense_results", default="data/results/dense_eval_results.jsonl")
    p.add_argument("--output", default="data/results/oracle_eval_results.jsonl")
    p.add_argument("--datasets", nargs="+", default=["nq", "hotpotqa"])
    return p.parse_args()


def main() -> None:
    args = parse_args()

    eval_data = load_jsonl(args.eval_path)
    eval_data = [ex for ex in eval_data if ex["dataset"] in args.datasets]
    print(f"Eval: {len(eval_data)} questions")

    dense_results = load_jsonl(args.dense_results)
    dense_lookup = {ex["id"]: ex["retrieved"] for ex in dense_results}

    print(f"Loading corpus from {args.corpus_path}…")
    corpus_df = pd.read_parquet(args.corpus_path)
    corpus_texts_lower: np.ndarray = corpus_df["text"].str.lower().values
    corpus_ids = corpus_df["id"].values
    corpus_titles = corpus_df["title"].values
    corpus_texts = corpus_df["text"].values
    print(f"  {len(corpus_df)} passages loaded.")

    all_answers = {a.lower().strip() for ex in eval_data for a in ex["answers"]}
    print(f"  {len(all_answers)} unique answer strings to locate.")

    answer_to_passages = build_answer_index(corpus_texts_lower, all_answers)
    coverage = len(answer_to_passages) / len(all_answers) * 100
    print(f"  Coverage: {len(answer_to_passages)}/{len(all_answers)} ({coverage:.1f}%)")

    # ------------------------------------------------------------------ #
    # Select best passage per question
    # ------------------------------------------------------------------ #
    oracle_passages: dict[str, dict] = {}
    missing_ids: list[str] = []

    for ex in tqdm(eval_data, desc="Selecting oracle passages"):
        qid = ex["id"]
        best_pidx: int | None = None
        best_score = -1.0
        best_answer: str | None = None

        for answer in ex["answers"]:
            answer_lower = answer.lower().strip()
            if answer_lower not in answer_to_passages:
                continue
            pidx, score = select_best_passage(
                answer_to_passages[answer_lower], answer, corpus_texts_lower
            )
            if pidx is not None and score > best_score:
                best_pidx = pidx
                best_score = score
                best_answer = answer

        if best_pidx is not None:
            oracle_passages[qid] = {
                "passage": {
                    "id": str(corpus_ids[best_pidx]),
                    "title": str(corpus_titles[best_pidx]),
                    "text": str(corpus_texts[best_pidx]),
                    "score": 100.0,
                },
                "oracle_type": "corpus_match",
                "matched_answer": best_answer,
            }
        else:
            missing_ids.append(qid)

    # Fallback: synthetic passages
    for qid in missing_ids:
        ex = next(e for e in eval_data if e["id"] == qid)
        oracle_passages[qid] = {
            "passage": make_synthetic_passage(ex["question"], ex["answers"][0]),
            "oracle_type": "synthetic",
            "matched_answer": ex["answers"][0],
        }

    n_synthetic = len(missing_ids)
    pct_synthetic = n_synthetic / len(eval_data) * 100

    if pct_synthetic > 30:
        print(f"WARNING: {pct_synthetic:.1f}% synthetic passages. Report corpus-only results separately.")
    elif pct_synthetic > 10:
        print(f"CAUTION: {pct_synthetic:.1f}% synthetic. Report both 'all oracle' and 'corpus-only' results.")
    else:
        print(f"Synthetic rate {pct_synthetic:.1f}% — acceptable (<10%).")

    # ------------------------------------------------------------------ #
    # Build final results (oracle passage first, then dense fill)
    # ------------------------------------------------------------------ #
    oracle_results = []
    for ex in eval_data:
        oracle_info = oracle_passages[ex["id"]]
        oracle_passage = oracle_info["passage"]
        dense_retrieved = dense_lookup.get(ex["id"], [])

        oracle_text_prefix = oracle_passage["text"][:80].lower()
        retrieved = [oracle_passage] + [
            doc for doc in dense_retrieved
            if doc["text"][:80].lower() != oracle_text_prefix
        ][:19]

        oracle_results.append(
            {
                "id": ex["id"],
                "question": ex["question"],
                "answers": ex["answers"],
                "dataset": ex["dataset"],
                "type": ex["type"],
                "retrieved": retrieved,
                "oracle_type": oracle_info["oracle_type"],
            }
        )

    # ------------------------------------------------------------------ #
    # Save + validate
    # ------------------------------------------------------------------ #
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    save_jsonl(args.output, oracle_results)
    print(f"Saved oracle results → {args.output}")

    compute_hit_rate(oracle_results, "ORACLE RETRIEVAL (all)")
    corpus_only = [ex for ex in oracle_results if ex["oracle_type"] == "corpus_match"]
    compute_hit_rate(corpus_only, "ORACLE (corpus-match only)")

    # Stats JSON
    missing_by_dataset: dict[str, int] = defaultdict(int)
    for qid in missing_ids:
        ex = next(e for e in eval_data if e["id"] == qid)
        missing_by_dataset[ex["dataset"]] += 1

    stats = {
        "total_questions": len(eval_data),
        "corpus_match": len(eval_data) - n_synthetic,
        "synthetic": n_synthetic,
        "synthetic_pct": round(pct_synthetic, 2),
        "coverage_pct": round(coverage, 2),
        "missing_by_dataset": dict(missing_by_dataset),
        "corpus_size": len(corpus_df),
    }
    stats_path = os.path.join(os.path.dirname(args.output), "oracle_stats.json")
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)
    print(f"Oracle stats → {stats_path}")


if __name__ == "__main__":
    main()