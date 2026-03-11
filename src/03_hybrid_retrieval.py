"""
03_hybrid_retrieval.py — Reciprocal Rank Fusion of BM25 + Dense results.

RRF score: score(d) = Σ_r 1 / (k + rank_r(d))
where k=60 (Cormack et al., 2009).

Outputs
-------
  <output>  — hybrid_eval_results.jsonl
"""

from __future__ import annotations

import argparse
import os
import time
from collections import defaultdict

import bm25s
import faiss
import numpy as np

from utils import compute_hit_rate, load_jsonl, save_jsonl

RRF_K = 60  # Standard RRF constant

# Fusion

def reciprocal_rank_fusion(
    bm25_results: list[dict],
    dense_results: list[dict],
    k: int = RRF_K,
    top_k: int = 20,
) -> list[dict]:
    hybrid_results = []

    for bm25_ex, dense_ex in zip(bm25_results, dense_results):
        assert bm25_ex["id"] == dense_ex["id"], (
            f"ID mismatch: {bm25_ex['id']} vs {dense_ex['id']}"
        )

        doc_scores: dict[str, float] = defaultdict(float)
        doc_data: dict[str, dict] = {}

        for rank, doc in enumerate(bm25_ex["retrieved"]):
            doc_id = doc["id"]
            doc_scores[doc_id] += 1.0 / (k + rank + 1)
            doc_data[doc_id] = doc

        for rank, doc in enumerate(dense_ex["retrieved"]):
            doc_id = doc["id"]
            doc_scores[doc_id] += 1.0 / (k + rank + 1)
            doc_data[doc_id] = doc

        sorted_docs = sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)

        retrieved = []
        for doc_id, score in sorted_docs[:top_k]:
            entry = doc_data[doc_id].copy()
            entry["score"] = float(score)
            retrieved.append(entry)

        hybrid_results.append(
            {
                "id": bm25_ex["id"],
                "question": bm25_ex["question"],
                "answers": bm25_ex["answers"],
                "dataset": bm25_ex["dataset"],
                "type": bm25_ex["type"],
                "retrieved": retrieved,
            }
        )

    return hybrid_results

# Main

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Hybrid RRF fusion of BM25 + Dense.")
    p.add_argument("--bm25_results", default="data/results/bm25_eval_results.jsonl")
    p.add_argument("--dense_results", default="data/results/dense_eval_results.jsonl")
    p.add_argument("--output", default="data/results/hybrid_eval_results.jsonl")
    p.add_argument("--rrf_k", type=int, default=RRF_K)
    p.add_argument("--top_k", type=int, default=20)
    return p.parse_args()


def main() -> None:
    args = parse_args()

    bm25_res = load_jsonl(args.bm25_results)
    dense_res = load_jsonl(args.dense_results)
    print(f"BM25: {len(bm25_res)}, Dense: {len(dense_res)}")

    hybrid_results = reciprocal_rank_fusion(bm25_res, dense_res, k=args.rrf_k, top_k=args.top_k)

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    save_jsonl(args.output, hybrid_results)
    print(f"Saved hybrid results → {args.output}")

    compute_hit_rate(bm25_res, label="BM25")
    compute_hit_rate(dense_res, label="Dense (E5)")
    compute_hit_rate(hybrid_results, label="Hybrid RRF")

if __name__ == "__main__":
    main()