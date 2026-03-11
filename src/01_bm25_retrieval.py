"""
01_bm25_retrieval.py — Build a 500k-passage Wikipedia corpus and run BM25 retrieval.

Steps
-----
1. Sample ~500 000 passages from Wikipedia (2023-11-01, EN) via deterministic hash.
2. Tokenise with bm25s and build an in-memory BM25 index.
3. Retrieve top-20 passages for every eval question.
4. Report hit-rate@{1,5,10,20} per dataset.

Outputs
-------
  <corpus_output>           — wiki_500k.parquet
  <bm25_results>            — bm25_eval_results.jsonl
"""

from __future__ import annotations

import argparse
import hashlib
import os
import time

import bm25s
import numpy as np
import pandas as pd
from datasets import load_dataset
from tqdm import tqdm

from utils import compute_hit_rate, load_jsonl, save_jsonl


TARGET_PASSAGES = 500_000
SAMPLE_RATE = 0.02
MIN_TEXT_LEN = 100
MAX_CHUNK_WORDS = 100


def should_sample_article(title: str, rate: float = SAMPLE_RATE) -> bool:
    """Hash-based deterministic sampling — reproducible without shuffling."""
    h = hashlib.md5(title.encode()).hexdigest()
    return int(h[:8], 16) / 0xFFFFFFFF < rate


def chunk_article(title: str, text: str, max_words: int = MAX_CHUNK_WORDS) -> list[dict]:
    words = text.split()
    chunks = []
    for i in range(0, len(words), max_words):
        chunk_text = " ".join(words[i : i + max_words])
        if len(chunk_text.strip()) > 50:
            chunks.append({"title": title, "text": chunk_text.strip()})
    return chunks


def build_corpus(corpus_output: str) -> list[dict]:
    """Stream Wikipedia, sample passages, save to parquet."""
    if os.path.exists(corpus_output):
        print(f"Corpus already exists at {corpus_output}, loading…")
        df = pd.read_parquet(corpus_output)
        return df.to_dict("records")

    wiki = load_dataset("wikimedia/wikipedia", "20231101.en", split="train", streaming=True)

    passages: list[dict] = []
    article_count = 0

    for article in tqdm(wiki, desc="Scanning Wikipedia"):
        if len(passages) >= TARGET_PASSAGES:
            break
        title = article.get("title") or ""
        text = article.get("text", "")
        if not text or len(text) < MIN_TEXT_LEN:
            continue
        if not should_sample_article(title):
            continue
        for chunk in chunk_article(title, text):
            chunk["id"] = str(len(passages))
            passages.append(chunk)
            if len(passages) >= TARGET_PASSAGES:
                break
        article_count += 1

    print(f"Collected {len(passages)} passages from {article_count} articles.")

    os.makedirs(os.path.dirname(corpus_output) or ".", exist_ok=True)
    df = pd.DataFrame(passages)
    df.to_parquet(corpus_output, index=False)
    print(f"Saved corpus → {corpus_output} ({os.path.getsize(corpus_output)/1e6:.1f} MB)")
    return passages


# BM25 index & retrieval

def build_and_retrieve(
    passages: list[dict],
    eval_data: list[dict],
    top_k: int = 20,
) -> list[dict]:
    corpus_texts = [p["text"] for p in passages]

    print("Tokenising corpus…")
    t0 = time.time()
    corpus_tokens = bm25s.tokenize(corpus_texts, stopwords="en")
    print(f"  done in {time.time() - t0:.1f}s")

    print("Building BM25 index…")
    t0 = time.time()
    retriever = bm25s.BM25()
    retriever.index(corpus_tokens)
    print(f"  done in {time.time() - t0:.1f}s")

    query_texts = [ex["question"] for ex in eval_data]
    query_tokens = bm25s.tokenize(query_texts, stopwords="en")

    print(f"Retrieving top-{top_k} for {len(eval_data)} queries…")
    result_ids, result_scores = retriever.retrieve(query_tokens, k=top_k)

    results = []
    for i, ex in enumerate(eval_data):
        retrieved = [
            {
                "id": passages[int(result_ids[i, j])]["id"],
                "title": passages[int(result_ids[i, j])]["title"],
                "text": passages[int(result_ids[i, j])]["text"],
                "score": float(result_scores[i, j]),
            }
            for j in range(top_k)
        ]
        results.append(
            {
                "id": ex["id"],
                "question": ex["question"],
                "answers": ex["answers"],
                "dataset": ex["dataset"],
                "type": ex["type"],
                "retrieved": retrieved,
            }
        )

    return results


# Main

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="BM25 corpus build + retrieval.")
    p.add_argument("--eval_path", default="data/eval.jsonl")
    p.add_argument("--corpus_output", default="data/corpus/wiki_500k.parquet")
    p.add_argument("--bm25_results", default="data/results/bm25_eval_results.jsonl")
    p.add_argument("--top_k", type=int, default=20)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    eval_data = load_jsonl(args.eval_path)
    print(f"Loaded {len(eval_data)} eval examples.")

    passages = build_corpus(args.corpus_output)
    results = build_and_retrieve(passages, eval_data, top_k=args.top_k)

    os.makedirs(os.path.dirname(args.bm25_results) or ".", exist_ok=True)
    save_jsonl(args.bm25_results, results)
    print(f"Saved BM25 results → {args.bm25_results}")

    compute_hit_rate(results, label="BM25")


if __name__ == "__main__":
    main()