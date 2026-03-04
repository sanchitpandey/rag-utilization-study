"""
06_analysis.py — Retrieval scaling curve analysis.

Loads all local-model grid results and produces:
  - EM vs model scale broken down by retrieval condition
  - Context length vs accuracy analysis
  - Scaling curve figure

This is a lightweight analysis step that precedes the full statistical
analysis in 10_statistical_analysis.py.

Outputs
-------
  <output_dir>/retrieval_scaling_curve.png
  <output_dir>/context_length_vs_accuracy.png
"""

from __future__ import annotations

import argparse
import glob
import os
import re
import string

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from utils import exact_match, load_jsonl

MODEL_ORDER = ["SmolLM2-360M", "Qwen2.5-1.5B", "Qwen2.5-3B", "Qwen2.5-7B"]
MODEL_SIZES = {"SmolLM2-360M": 0.36, "Qwen2.5-1.5B": 1.5, "Qwen2.5-3B": 3.0, "Qwen2.5-7B": 7.0}
RETRIEVAL_ORDER = ["none", "bm25", "dense", "hybrid"]
COLORS = {"none": "#636e72", "bm25": "#fdcb6e", "dense": "#0984e3", "hybrid": "#6c5ce7"}


def detect_meta(filename: str) -> tuple[str, str]:
    """Parse retrieval condition and model name from filename."""
    base = os.path.basename(filename).replace(".jsonl", "")
    parts = base.split("__")
    if len(parts) != 2:
        return "unknown", "unknown"
    return parts[0], parts[1]  # retrieval, model


def load_all_results(grid_dir: str) -> pd.DataFrame:
    rows = []
    for f in glob.glob(os.path.join(grid_dir, "*.jsonl")):
        retrieval, model = detect_meta(f)
        if retrieval == "unknown" or model not in MODEL_SIZES:
            continue
        data = load_jsonl(f)
        for r in data:
            pred = r.get("predicted", r.get("pred", ""))
            gold = r.get("gold_answers", r.get("gold", []))
            rows.append({
                "model": model,
                "retrieval": retrieval,
                "pred": pred,
                "gold": gold,
                "input_tokens": r.get("input_tokens"),
            })
    df = pd.DataFrame(rows)
    df["em"] = df.apply(lambda r: exact_match(str(r["pred"]), r["gold"]), axis=1)
    return df


def plot_scaling_curve(df: pd.DataFrame, output_dir: str) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))

    for ret in RETRIEVAL_ORDER:
        sub = df[df["retrieval"] == ret]
        if sub.empty:
            continue
        sizes, means = [], []
        for model in MODEL_ORDER:
            m_sub = sub[sub["model"] == model]
            if m_sub.empty:
                continue
            sizes.append(MODEL_SIZES[model])
            means.append(m_sub["em"].mean() * 100)
        if sizes:
            ax.plot(sizes, means, marker="o", linewidth=2, label=ret, color=COLORS.get(ret))

    ax.axhline(0, linestyle="--", color="#dfe6e9", linewidth=0.8)
    ax.set_xscale("log")
    ax.set_xlabel("Model Size (Billions of Parameters)")
    ax.set_ylabel("Exact Match (%)")
    ax.set_title("Retrieval Utility vs Model Scale", fontweight="bold")
    ax.set_xticks([0.36, 1.5, 3, 7])
    ax.set_xticklabels(["360M", "1.5B", "3B", "7B"])
    ax.legend(framealpha=0.9)
    ax.grid(alpha=0.15)
    plt.tight_layout()
    path = os.path.join(output_dir, "retrieval_scaling_curve.png")
    fig.savefig(path, dpi=300)
    plt.close(fig)
    print(f"Saved → {path}")


def plot_context_length(df: pd.DataFrame, output_dir: str) -> None:
    df = df[df["input_tokens"].notna()].copy()
    df["token_bin"] = pd.cut(
        df["input_tokens"],
        bins=[0, 256, 512, 1024, 2048],
        labels=["0-256", "256-512", "512-1024", "1024-2048"],
    )
    dense_df = df[df["retrieval"] == "dense"]

    fig, ax = plt.subplots(figsize=(8, 5))
    for model in MODEL_ORDER:
        sub = dense_df[dense_df["model"] == model]
        if sub.empty:
            continue
        grouped = sub.groupby("token_bin", observed=True)["em"].mean() * 100
        ax.plot(grouped.index.astype(str), grouped.values, marker="o", label=model, linewidth=2)

    ax.set_xlabel("Input Context Length (tokens)")
    ax.set_ylabel("Exact Match (%)")
    ax.set_title("Context Length vs Retrieval Accuracy (Dense)", fontweight="bold")
    ax.legend(framealpha=0.9)
    ax.grid(alpha=0.15)
    plt.tight_layout()
    path = os.path.join(output_dir, "context_length_vs_accuracy.png")
    fig.savefig(path, dpi=300)
    plt.close(fig)
    print(f"Saved → {path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Retrieval scaling curve analysis.")
    p.add_argument("--grid_dir", default="data/grid_local",
                   help="Directory containing per-(condition, model) JSONL files")
    p.add_argument("--output_dir", default="results/figures")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    df = load_all_results(args.grid_dir)
    if df.empty:
        print("No results found. Run 05_local_scaling_grid.py first.")
        return

    print(f"Loaded {len(df)} rows")
    print(f"  Models:    {sorted(df['model'].unique())}")
    print(f"  Retrieval: {sorted(df['retrieval'].unique())}")

    print("\n===== EM by Retrieval × Model =====")
    pivot = df.pivot_table(values="em", index="retrieval", columns="model", aggfunc="mean")
    print((pivot * 100).round(1).to_string())

    print("\n===== Delta (retrieval − none) =====")
    for model in MODEL_ORDER:
        m_sub = df[df["model"] == model]
        none_em = m_sub[m_sub["retrieval"] == "none"]["em"].mean()
        for ret in ["bm25", "dense", "hybrid"]:
            ret_em = m_sub[m_sub["retrieval"] == ret]["em"].mean()
            delta = (ret_em - none_em) * 100
            sign = "+" if delta >= 0 else ""
            print(f"  {model:<18} {ret:<8} {sign}{delta:.1f}pp")

    plot_scaling_curve(df, args.output_dir)
    if "input_tokens" in df.columns:
        plot_context_length(df, args.output_dir)


if __name__ == "__main__":
    main()