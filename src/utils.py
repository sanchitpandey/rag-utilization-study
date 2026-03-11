"""
utils.py — Shared utilities for the RAG Utilization Study.

Covers:
  - JSONL I/O
  - Answer normalization
  - Exact match and token-level F1
  - Hit-rate evaluation
  - Bootstrap confidence intervals
"""

from __future__ import annotations

import json
import re
import string
from collections import defaultdict
from typing import Any

import numpy as np

# I/O

def load_jsonl(path: str) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def save_jsonl(path: str, data: list[dict]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for row in data:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def load_json(path: str) -> Any:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def save_json(path: str, data: Any, indent: int = 2) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=indent, ensure_ascii=False)

# Answer normalization

def normalize_answer(s: str) -> str:
    """Lowercase, strip articles/punctuation, collapse whitespace."""
    s = str(s).lower().strip()
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    s = s.translate(str.maketrans("", "", string.punctuation))
    return " ".join(s.split())

# Metrics

def exact_match(prediction: str, gold_answers: list[str]) -> float:
    """Return 1.0 if normalized prediction matches any normalized gold answer."""
    pred_norm = normalize_answer(prediction)
    return float(any(normalize_answer(g) == pred_norm for g in gold_answers))


def f1_score(prediction: str, gold_answers: list[str]) -> float:
    """Token-level F1 between prediction and the best-matching gold answer."""
    pred_tokens = normalize_answer(prediction).split()
    best_f1 = 0.0
    for gold in gold_answers:
        gold_tokens = normalize_answer(gold).split()
        common = set(pred_tokens) & set(gold_tokens)
        if not common:
            continue
        precision = len(common) / len(pred_tokens) if pred_tokens else 0.0
        recall = len(common) / len(gold_tokens) if gold_tokens else 0.0
        f1 = (
            2 * precision * recall / (precision + recall)
            if (precision + recall) > 0
            else 0.0
        )
        best_f1 = max(best_f1, f1)
    return best_f1


def answer_in_passages(answers: list[str], passages: list[dict], k: int = 20) -> bool:
    """Check whether any gold answer string appears in the top-k passage texts."""
    if not passages:
        return False
    text = " ".join(p["text"] for p in passages[:k]).lower()
    return any(a.lower() in text for a in answers)

# Hit-rate table

def compute_hit_rate(
    results: list[dict],
    label: str = "",
    k_values: list[int] = (1, 5, 10, 20),
) -> None:
    """Print hit-rate table broken down by dataset and k."""
    hits: dict[str, dict[int, int]] = defaultdict(lambda: defaultdict(int))
    counts: dict[str, int] = defaultdict(int)

    for ex in results:
        dataset = ex["dataset"]
        counts[dataset] += 1
        counts["all"] += 1
        for k in k_values:
            top_k_text = " ".join(r["text"] for r in ex["retrieved"][:k]).lower()
            if any(a.lower() in top_k_text for a in ex["answers"]):
                hits[dataset][k] += 1
                hits["all"][k] += 1

    if label:
        print(f"\n{'='*55}")
        print(f" {label}")
        print(f"{'='*55}")
    print(f"{'Dataset':<12} " + " ".join(f"{'@'+str(k):>6}" for k in k_values) + f" {'Count':>6}")
    print("-" * 50)
    for dataset in sorted(counts.keys()):
        row = f"{dataset:<12}"
        for k in k_values:
            rate = hits[dataset][k] / counts[dataset] * 100
            row += f" {rate:5.1f}%"
        row += f" {counts[dataset]:>6}"
        print(row)

# Bootstrap CI

def bootstrap_ci(
    scores: list[float] | np.ndarray,
    n_boot: int = 2000,
    ci: float = 0.95,
    seed: int = 42,
) -> tuple[float, float, float]:
    """
    Return (mean, lower_bound, upper_bound) via non-parametric bootstrap.

    Parameters
    ----------
    scores:  Per-example binary/continuous scores.
    n_boot:  Number of bootstrap resamples.
    ci:      Coverage (default 0.95 → 95% CI).
    seed:    Random seed for reproducibility.
    """
    rng = np.random.default_rng(seed)
    scores = np.asarray(scores, dtype=float)
    if len(scores) == 0:
        return 0.0, 0.0, 0.0

    boot_means = np.array([
        np.mean(rng.choice(scores, size=len(scores), replace=True))
        for _ in range(n_boot)
    ])
    lo = np.percentile(boot_means, (1 - ci) / 2 * 100)
    hi = np.percentile(boot_means, (1 + ci) / 2 * 100)
    return float(np.mean(scores)), float(lo), float(hi)