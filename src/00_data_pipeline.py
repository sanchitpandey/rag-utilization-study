"""
00_data_pipeline.py — Build train/eval splits from NQ-Open, HotpotQA, PopQA.

Outputs
-------
  <output_dir>/train.jsonl   — 5 000 examples  (NQ 2k + HotpotQA 2k + PopQA 1k)
  <output_dir>/eval.jsonl    — 1 500 examples  (NQ 500 + HotpotQA 500 + PopQA 500)

Each line is a JSON object:
  {
    "id":       "<dataset>_<index>",
    "question": "...",
    "answers":  ["...", ...],
    "dataset":  "nq" | "hotpotqa" | "popqa",
    "type":     "factoid" | "multi-hop" | "long-tail"
  }
"""

from __future__ import annotations

import argparse
import json
import os
import random

from datasets import load_dataset
from tqdm import tqdm


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_example(
    idx: int,
    question: str,
    answers: list[str],
    dataset: str,
    qtype: str,
) -> dict:
    return {
        "id": f"{dataset}_{idx}",
        "question": question.strip(),
        "answers": list(set(a.strip() for a in answers if a)),
        "dataset": dataset,
        "type": qtype,
    }


def save_jsonl(path: str, data: list[dict]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in data:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


# ---------------------------------------------------------------------------
# Dataset loaders
# ---------------------------------------------------------------------------

def load_nq_open(
    train_target: int = 2000,
    eval_target: int = 500,
) -> tuple[list[dict], list[dict]]:
    ds = load_dataset("nq_open", split="train", streaming=True)
    train, eval_ = [], []
    for i, ex in enumerate(tqdm(ds, desc="NQ-Open")):
        question = ex.get("question")
        answers = ex.get("answer")
        if not question or not answers:
            continue
        example = make_example(i, question, answers, "nq", "factoid")
        if len(train) < train_target:
            train.append(example)
        elif len(eval_) < eval_target:
            eval_.append(example)
        else:
            break
    return train, eval_


def load_hotpot(
    train_target: int = 2000,
    eval_target: int = 500,
) -> tuple[list[dict], list[dict]]:
    ds = load_dataset("hotpot_qa", "fullwiki", split="train", streaming=True)
    train, eval_ = [], []
    for i, ex in enumerate(tqdm(ds, desc="HotpotQA")):
        example = make_example(i, ex["question"], [ex["answer"]], "hotpotqa", "multi-hop")
        if len(train) < train_target:
            train.append(example)
        elif len(eval_) < eval_target:
            eval_.append(example)
        else:
            break
    return train, eval_


def load_popqa(
    train_target: int = 1000,
    eval_target: int = 500,
) -> tuple[list[dict], list[dict]]:
    ds = load_dataset("akariasai/PopQA", split="test", streaming=True)
    train, eval_ = [], []
    for i, ex in enumerate(tqdm(ds, desc="PopQA")):
        question = ex.get("question")
        answers = ex.get("possible_answers")
        if not question or not answers:
            continue
        if isinstance(answers, str):
            answers = [answers]
        example = make_example(i, question, answers, "popqa", "long-tail")
        if len(train) < train_target:
            train.append(example)
        elif len(eval_) < eval_target:
            eval_.append(example)
        else:
            break
    return train, eval_


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build train/eval JSONL splits.")
    p.add_argument("--output_dir", default="data", help="Output directory")
    p.add_argument("--nq_train", type=int, default=2000)
    p.add_argument("--nq_eval", type=int, default=500)
    p.add_argument("--hotpot_train", type=int, default=2000)
    p.add_argument("--hotpot_eval", type=int, default=500)
    p.add_argument("--popqa_train", type=int, default=1000)
    p.add_argument("--popqa_eval", type=int, default=500)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    random.seed(args.seed)

    nq_train, nq_eval = load_nq_open(args.nq_train, args.nq_eval)
    hp_train, hp_eval = load_hotpot(args.hotpot_train, args.hotpot_eval)
    pq_train, pq_eval = load_popqa(args.popqa_train, args.popqa_eval)

    train_data = nq_train + hp_train + pq_train
    eval_data = nq_eval + hp_eval + pq_eval

    os.makedirs(args.output_dir, exist_ok=True)
    save_jsonl(os.path.join(args.output_dir, "train.jsonl"), train_data)
    save_jsonl(os.path.join(args.output_dir, "eval.jsonl"), eval_data)

    print(f"\nSaved {len(train_data)} train / {len(eval_data)} eval examples to {args.output_dir}/")
    for ds in ["nq", "hotpotqa", "popqa"]:
        n_tr = sum(1 for x in train_data if x["dataset"] == ds)
        n_ev = sum(1 for x in eval_data if x["dataset"] == ds)
        print(f"  {ds:<10}  train={n_tr}  eval={n_ev}")


if __name__ == "__main__":
    main()