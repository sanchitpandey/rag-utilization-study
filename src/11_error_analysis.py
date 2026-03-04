"""
11_error_analysis.py — Taxonomy of oracle-retrieval failures.

For each failure in the oracle condition (UNKNOWN questions, corpus-match
passages only), classify the error into one of six mutually-exclusive
categories applied in priority order:

  1. Verbose OK   — gold answer is a substring of the prediction (correct
                    but wrapped in extra words; fails EM, would pass a
                    looser evaluation)
  2. Refusal      — model declined, said it couldn't determine the answer,
                    or returned an empty / trivially short response
  3. Partial Match — prediction shares tokens with a gold answer (F1 > 0.3)
                    but is not a substring match
  4. Wrong Entity — prediction text (≥ 2 tokens) appears verbatim in the
                    oracle passage, meaning the model read the passage but
                    extracted the wrong span
  5. Extraction   — prediction tokens overlap significantly with passage
                    content (> 50 % of content words hit) without being
                    verbatim, suggesting a malformed extraction
  6. Irrelevant   — none of the above; model ignored the passage entirely

Outputs
-------
  <output_dir>/figure4_error_heatmap.{png,pdf}
  <output_dir>/figure5_error_stacked.{png,pdf}
  <output_dir>/error_taxonomy.csv          — raw per-question classifications
  <output_dir>/error_taxonomy_summary.csv  — % breakdown per model
"""

from __future__ import annotations

import argparse
import ast
import os
import re
import string

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

matplotlib.rcParams.update(
    {
        "font.size": 11,
        "font.family": "sans-serif",
        "axes.titlesize": 13,
        "axes.labelsize": 12,
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "axes.spines.top": False,
        "axes.spines.right": False,
    }
)

MODEL_ORDER = ["SmolLM2-360M", "Qwen2.5-1.5B", "Qwen2.5-3B", "Qwen2.5-7B"]
MODEL_SHORT = {
    "SmolLM2-360M": "360M",
    "Qwen2.5-1.5B": "1.5B",
    "Qwen2.5-3B": "3B",
    "Qwen2.5-7B": "7B",
}

ERROR_CATEGORIES = [
    "Irrelevant",
    "Extraction",
    "Wrong Entity",
    "Partial Match",
    "Verbose OK",
    "Refusal",
]

CATEGORY_COLORS = {
    "Irrelevant":    "#c0392b",
    "Extraction":    "#e67e22",
    "Wrong Entity":  "#f1c40f",
    "Partial Match": "#1abc9c",
    "Verbose OK":    "#00bcd4",
    "Refusal":       "#95a5a6",
}

REFUSAL_PHRASES = [
    "don't know",
    "do not know",
    "cannot determine",
    "can't determine",
    "not mentioned",
    "no information",
    "cannot find",
    "not provided",
    "not stated",
    "i cannot",
    "i can't",
    "unable to",
    "does not contain",
    "doesn't contain",
    "not enough information",
    "cannot answer",
    "can't answer",
    "not clear",
    "not specified",
    "not given",
    "the context does not",
    "the passage does not",
    "based on the provided context",
    "not available",
    "not sure",
]


# ---------------------------------------------------------------------------
# Text utilities (self-contained — no dependency on utils.py at import time)
# ---------------------------------------------------------------------------

def normalize_answer(s: str) -> str:
    s = str(s).lower().strip()
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    s = s.translate(str.maketrans("", "", string.punctuation))
    return " ".join(s.split())


def f1_score(prediction: str, gold_answers: list[str]) -> float:
    pred_tokens = normalize_answer(prediction).split()
    best_f1 = 0.0
    for gold in gold_answers:
        gold_tokens = normalize_answer(gold).split()
        common = set(pred_tokens) & set(gold_tokens)
        if not common:
            continue
        prec = len(common) / len(pred_tokens) if pred_tokens else 0.0
        rec = len(common) / len(gold_tokens) if gold_tokens else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
        best_f1 = max(best_f1, f1)
    return best_f1


def parse_gold_answers(raw) -> list[str]:
    """Parse gold_answers from CSV (stored as string repr of list)."""
    if isinstance(raw, list):
        return raw
    try:
        parsed = ast.literal_eval(str(raw))
        if isinstance(parsed, list):
            return [str(a) for a in parsed]
    except (ValueError, SyntaxError):
        pass
    return [str(raw)]


# ---------------------------------------------------------------------------
# Error classifier
# ---------------------------------------------------------------------------

def classify_failure(
    pred: str,
    gold_answers: list[str],
    passage_text: str,
) -> str:
    """
    Assign one of the six error categories to a failed prediction.
    Categories are applied in priority order (see module docstring).
    """
    pred_str = str(pred).strip()
    pred_norm = normalize_answer(pred_str)
    pred_lower = pred_str.lower()
    passage_lower = (passage_text or "").lower()

    for gold in gold_answers:
        gold_norm = normalize_answer(gold)
        if gold_norm and gold_norm in pred_norm:
            return "Verbose OK"

    if not pred_norm:
        return "Refusal"
    if any(phrase in pred_lower for phrase in REFUSAL_PHRASES):
        return "Refusal"
    if len(pred_norm.split()) > 25:
        return "Refusal"

    f1 = f1_score(pred_str, gold_answers)
    if f1 > 0.3:
        return "Partial Match"

    content_words = [w for w in pred_norm.split() if len(w) > 3]
    if pred_norm and pred_norm in passage_lower:
        return "Wrong Entity"
    if len(content_words) >= 2 and all(w in passage_lower for w in content_words):
        return "Wrong Entity"

    if content_words and passage_lower:
        hit_rate = sum(1 for w in content_words if w in passage_lower) / len(content_words)
        if hit_rate > 0.5:
            return "Extraction"

    return "Irrelevant"


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_corpus_grid(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["gold_answers_parsed"] = df["gold_answers"].apply(parse_gold_answers)
    return df


def load_oracle_passages(oracle_path: str) -> dict[str, str]:
    """Return {question_id: oracle_passage_text}."""
    import json

    passage_lookup: dict[str, str] = {}
    with open(oracle_path, encoding="utf-8") as f:
        for line in f:
            ex = json.loads(line)
            retrieved = ex.get("retrieved", [])
            if retrieved:
                passage_lookup[ex["id"]] = retrieved[0].get("text", "")
    return passage_lookup


# ---------------------------------------------------------------------------
# Classification loop
# ---------------------------------------------------------------------------

def build_taxonomy(
    df: pd.DataFrame,
    passage_lookup: dict[str, str],
) -> pd.DataFrame:
    """
    Filter to oracle failures on UNKNOWN questions (corpus-match only),
    then classify each failure.
    Returns a DataFrame with one row per failure including its category.
    """
    # Restrict to oracle condition, unknown questions, failures only
    mask = (
        (df["retrieval"] == "oracle")
        & (df["parametric"] == "unknown")
        & (df["em"] == 0)
    )
    failures = df[mask].copy()

    categories = []
    for _, row in failures.iterrows():
        passage = passage_lookup.get(str(row["id"]), "")
        cat = classify_failure(
            row["predicted"],
            row["gold_answers_parsed"],
            passage,
        )
        categories.append(cat)

    failures = failures.copy()
    failures["error_category"] = categories
    return failures


# ---------------------------------------------------------------------------
# Summary table
# ---------------------------------------------------------------------------

def build_summary(failures: pd.DataFrame) -> pd.DataFrame:
    """Compute % breakdown per model (rows=category, cols=model)."""
    rows = []
    for model in MODEL_ORDER:
        model_failures = failures[failures["model"] == model]
        total = len(model_failures)
        if total == 0:
            for cat in ERROR_CATEGORIES:
                rows.append({"model": model, "category": cat, "count": 0, "pct": 0.0})
            continue
        counts = model_failures["error_category"].value_counts()
        for cat in ERROR_CATEGORIES:
            n = counts.get(cat, 0)
            rows.append({
                "model": model,
                "category": cat,
                "count": int(n),
                "pct": round(n / total * 100, 1),
            })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

def figure4_heatmap(summary: pd.DataFrame, output_dir: str) -> None:
    pivot = summary.pivot(index="category", columns="model", values="pct")
    pivot = pivot.reindex(
        index=ERROR_CATEGORIES,
        columns=MODEL_ORDER,
    )
    col_labels = [MODEL_SHORT[m] for m in MODEL_ORDER]

    fig, ax = plt.subplots(figsize=(9, 5.5))
    sns.heatmap(
        pivot,
        ax=ax,
        annot=True,
        fmt=".0f",
        cmap="YlOrRd",
        linewidths=0.5,
        linecolor="white",
        cbar_kws={"label": "% of Oracle Failures"},
        xticklabels=col_labels,
        yticklabels=ERROR_CATEGORIES,
        vmin=0,
        vmax=65,
    )
    ax.set_xlabel("Model")
    ax.set_ylabel("Error Category")
    ax.set_title(
        "Oracle Failure Taxonomy Across Model Scale\n"
        "(% of failures in each category, corpus-match only)",
        fontweight="bold",
        pad=14,
    )
    plt.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(output_dir, f"figure4_error_heatmap.{ext}"))
    plt.close(fig)
    print("Saved figure4_error_heatmap")


def figure5_stacked(summary: pd.DataFrame, output_dir: str) -> None:
    fig, ax = plt.subplots(figsize=(9, 6))

    x = np.arange(len(MODEL_ORDER))
    bottoms = np.zeros(len(MODEL_ORDER))

    for cat in ERROR_CATEGORIES:
        heights = []
        for model in MODEL_ORDER:
            row = summary[(summary["model"] == model) & (summary["category"] == cat)]
            heights.append(float(row["pct"].values[0]) if len(row) > 0 else 0.0)
        ax.bar(
            x,
            heights,
            bottom=bottoms,
            label=cat,
            color=CATEGORY_COLORS[cat],
            width=0.55,
            edgecolor="white",
            linewidth=0.5,
        )
        bottoms += np.array(heights)

    ax.set_xticks(x)
    ax.set_xticklabels([MODEL_SHORT[m] for m in MODEL_ORDER])
    ax.set_xlabel("Model Size")
    ax.set_ylabel("% of Oracle Failures")
    ax.set_ylim(0, 110)
    ax.set_title(
        "How Do Oracle Failures Break Down?\n"
        "(what goes wrong when the answer is in the passage)",
        fontweight="bold",
        pad=14,
    )
    ax.legend(
        loc="upper right",
        ncol=2,
        framealpha=0.9,
        fontsize=9,
    )
    ax.grid(alpha=0.12, axis="y", zorder=0)
    plt.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(output_dir, f"figure5_error_stacked.{ext}"))
    plt.close(fig)
    print("Saved figure5_error_stacked")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Oracle failure taxonomy analysis.")
    p.add_argument(
        "--corpus_only_path",
        default="data/grid_v2/corpus_only_grid_v2.csv",
        help="Corpus-match-only grid CSV (output of 09_full_grid.py)",
    )
    p.add_argument(
        "--oracle_path",
        default="data/results/oracle_eval_results.jsonl",
        help="Oracle retrieval results with passage texts",
    )
    p.add_argument(
        "--output_dir",
        default="results/figures",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    print(f"Loading corpus-only grid from {args.corpus_only_path}…")
    df = load_corpus_grid(args.corpus_only_path)
    print(f"  {len(df)} rows loaded.")

    print(f"Loading oracle passages from {args.oracle_path}…")
    passage_lookup = load_oracle_passages(args.oracle_path)
    print(f"  {len(passage_lookup)} passages loaded.")

    print("Classifying failures…")
    failures = build_taxonomy(df, passage_lookup)
    print(f"  {len(failures)} failure cases classified.")

    summary = build_summary(failures)

    print(f"\n{'='*70}")
    print(" ERROR TAXONOMY (% per model, oracle · UNKNOWN · corpus-match only)")
    print(f"{'='*70}")
    pivot_display = summary.pivot(index="category", columns="model", values="pct")
    pivot_display = pivot_display.reindex(index=ERROR_CATEGORIES, columns=MODEL_ORDER)
    pivot_display.columns = [MODEL_SHORT[m] for m in MODEL_ORDER]
    print(pivot_display.to_string())

    failures.to_csv(os.path.join(args.output_dir, "error_taxonomy.csv"), index=False)
    summary.to_csv(os.path.join(args.output_dir, "error_taxonomy_summary.csv"), index=False)
    print(f"\nSaved CSVs to {args.output_dir}/")

    figure4_heatmap(summary, args.output_dir)
    figure5_stacked(summary, args.output_dir)

    print(f"\nAll outputs saved to {args.output_dir}/")


if __name__ == "__main__":
    main()