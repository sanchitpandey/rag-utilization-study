"""
10_statistical_analysis.py — Bootstrap CIs, McNemar's tests, and publication figures.

Figures produced
----------------
  figure1_scaling_curves.{png,pdf}   — EM vs model scale (unknown + known, all conditions)
  figure2_utilization_gap.{png,pdf}  — Oracle utilization gap
  figure3_distraction.{png,pdf}      — Distraction effect bar chart

Tables produced
---------------
  table1_main_results.csv    — EM% [95% CI] for unknown + known questions
  ci_results.csv             — Full bootstrap CI table
  significance_tests.csv     — McNemar's test summary
"""

from __future__ import annotations

import argparse
import os

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

from utils import bootstrap_ci

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
MODEL_SIZES = {"SmolLM2-360M": 0.36, "Qwen2.5-1.5B": 1.5, "Qwen2.5-3B": 3.0, "Qwen2.5-7B": 7.0}
MODEL_SHORT = {"SmolLM2-360M": "360M", "Qwen2.5-1.5B": "1.5B", "Qwen2.5-3B": "3B", "Qwen2.5-7B": "7B"}
RETRIEVAL_ORDER = ["none", "noisy", "oracle"]
COLORS = {"none": "#636e72", "noisy": "#d63031", "oracle": "#00b894"}
MARKERS = {"none": "o", "noisy": "s", "oracle": "D"}
LABELS = {"none": "No retrieval", "noisy": "Dense retrieval", "oracle": "Oracle retrieval"}

BONFERRONI_N = 24


# ---------------------------------------------------------------------------
# Bootstrap CI computation
# ---------------------------------------------------------------------------

def compute_all_cis(df_corpus: pd.DataFrame) -> pd.DataFrame:
    np.random.seed(42)
    ci_results = []
    for model in MODEL_ORDER:
        for ret in RETRIEVAL_ORDER:
            for split in ["known", "unknown", "all"]:
                for ds in ["nq", "hotpotqa", "all"]:
                    sub = df_corpus[(df_corpus["model"] == model) & (df_corpus["retrieval"] == ret)]
                    if split != "all":
                        sub = sub[sub["parametric"] == split]
                    if ds != "all":
                        sub = sub[sub["dataset"] == ds]
                    if len(sub) == 0:
                        continue
                    mean_em, lo_em, hi_em = bootstrap_ci(sub["em"].values)
                    mean_f1, lo_f1, hi_f1 = bootstrap_ci(sub["f1"].values)
                    ci_results.append(
                        {
                            "model": model, "retrieval": ret, "split": split,
                            "dataset": ds, "n": len(sub),
                            "em": mean_em, "em_lo": lo_em, "em_hi": hi_em,
                            "f1": mean_f1, "f1_lo": lo_f1, "f1_hi": hi_f1,
                        }
                    )
    return pd.DataFrame(ci_results)


# ---------------------------------------------------------------------------
# McNemar's test
# ---------------------------------------------------------------------------

def mcnemar_test(em_a: np.ndarray, em_b: np.ndarray) -> tuple[float, int, int]:
    b_only = ((em_a == 0) & (em_b == 1)).sum()
    c_only = ((em_a == 1) & (em_b == 0)).sum()
    if b_only + c_only == 0:
        return 1.0, int(b_only), int(c_only)
    chi2 = (abs(b_only - c_only) - 1) ** 2 / (b_only + c_only)
    p_value = 1 - stats.chi2.cdf(chi2, df=1)
    return float(p_value), int(b_only), int(c_only)


def run_significance_tests(df_corpus: pd.DataFrame) -> pd.DataFrame:
    sig_results = []
    alpha_corrected = 0.05 / BONFERRONI_N

    for comp_name, ret_a, ret_b in [
        ("oracle vs none", "none", "oracle"),
        ("noisy vs none", "none", "noisy"),
        ("oracle vs noisy", "noisy", "oracle"),
    ]:
        for split in ["unknown", "known"]:
            for model in MODEL_ORDER:
                sub_a = df_corpus[
                    (df_corpus["model"] == model) & (df_corpus["retrieval"] == ret_a)
                    & (df_corpus["parametric"] == split)
                ]
                sub_b = df_corpus[
                    (df_corpus["model"] == model) & (df_corpus["retrieval"] == ret_b)
                    & (df_corpus["parametric"] == split)
                ]
                merged = sub_a[["id", "em"]].merge(
                    sub_b[["id", "em"]], on="id", suffixes=(f"_{ret_a}", f"_{ret_b}")
                )
                if len(merged) < 10:
                    continue
                p_val, b_only, c_only = mcnemar_test(
                    merged[f"em_{ret_a}"].values, merged[f"em_{ret_b}"].values
                )
                delta = (merged[f"em_{ret_b}"].mean() - merged[f"em_{ret_a}"].mean()) * 100
                sig_label = (
                    "***" if p_val < 0.001
                    else "**" if p_val < 0.01
                    else "*" if p_val < alpha_corrected
                    else "ns"
                )
                sig_results.append(
                    {
                        "comparison": comp_name, "split": split, "model": model,
                        "n": len(merged), "delta_pp": round(delta, 1),
                        "p_value": round(p_val, 4), "significant": sig_label != "ns",
                        "sig_label": sig_label,
                    }
                )
    return pd.DataFrame(sig_results)


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

def figure1_scaling_curves(ci_df: pd.DataFrame, output_dir: str) -> None:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.5))

    for split, ax, title in [
        ("unknown", ax1, "UNKNOWN Questions\n(model cannot answer alone)"),
        ("known", ax2, "KNOWN Questions\n(model answers correctly alone)"),
    ]:
        for ret in RETRIEVAL_ORDER:
            sizes, means, lo_errs, hi_errs = [], [], [], []
            for model in MODEL_ORDER:
                cell = ci_df[
                    (ci_df["model"] == model) & (ci_df["retrieval"] == ret)
                    & (ci_df["split"] == split) & (ci_df["dataset"] == "all")
                ]
                if len(cell) > 0:
                    c = cell.iloc[0]
                    sizes.append(MODEL_SIZES[model])
                    means.append(c["em"] * 100)
                    lo_errs.append((c["em"] - c["em_lo"]) * 100)
                    hi_errs.append((c["em_hi"] - c["em"]) * 100)
            ax.errorbar(
                sizes, means, yerr=[lo_errs, hi_errs],
                marker=MARKERS[ret], color=COLORS[ret],
                linewidth=2.2, markersize=9, capsize=5,
                label=LABELS[ret], alpha=0.9,
                zorder=3 if ret == "oracle" else 2,
            )
        ax.set_xscale("log")
        ax.set_xlabel("Model Size (Parameters)")
        ax.set_ylabel("Exact Match (%)")
        ax.set_title(title, fontweight="bold", pad=12)
        ax.legend(framealpha=0.9, fontsize=9)
        ax.grid(alpha=0.15, zorder=0)
        ax.set_xticks([0.36, 1.5, 3, 7])
        ax.set_xticklabels(["360M", "1.5B", "3B", "7B"])

    fig.suptitle("Retrieval Utilization Across Model Scale", fontsize=15, fontweight="bold", y=1.02)
    plt.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(output_dir, f"figure1_scaling_curves.{ext}"))
    plt.close(fig)
    print("Saved figure1_scaling_curves")


def figure2_utilization_gap(ci_df: pd.DataFrame, output_dir: str) -> None:
    fig, ax = plt.subplots(figsize=(8, 5.5))
    sizes = [MODEL_SIZES[m] for m in MODEL_ORDER]

    oracle_em_unknown, noisy_em_unknown = [], []
    for model in MODEL_ORDER:
        for lst, ret in [(oracle_em_unknown, "oracle"), (noisy_em_unknown, "noisy")]:
            cell = ci_df[
                (ci_df["model"] == model) & (ci_df["retrieval"] == ret)
                & (ci_df["split"] == "unknown") & (ci_df["dataset"] == "all")
            ]
            lst.append(cell.iloc[0]["em"] * 100 if len(cell) > 0 else 0)

    ax.fill_between(sizes, oracle_em_unknown, 100, alpha=0.15, color="#d63031",
                    label="Wasted oracle (answer in passage, model fails)")
    ax.fill_between(sizes, 0, oracle_em_unknown, alpha=0.15, color="#00b894")
    ax.plot(sizes, oracle_em_unknown, marker="D", color="#00b894", linewidth=2.5, markersize=10,
            label="Oracle utilization (UNKNOWN)", zorder=5)
    ax.plot(sizes, noisy_em_unknown, marker="s", color="#d63031", linewidth=2, markersize=8,
            label="Noisy utilization (UNKNOWN)", linestyle="--", zorder=4)

    for i, model in enumerate(MODEL_ORDER):
        if oracle_em_unknown[i] > 0:
            ax.annotate(
                f"{oracle_em_unknown[i]:.0f}%",
                (sizes[i], oracle_em_unknown[i]),
                textcoords="offset points", xytext=(8, 8),
                fontsize=10, fontweight="bold", color="#00b894",
            )

    ax.set_xscale("log")
    ax.set_xlabel("Model Size (Parameters)")
    ax.set_ylabel("% of Questions Answered Correctly")
    ax.set_title(
        "Oracle Retrieval Utilization Gap\n(answer is in passage — does the model extract it?)",
        fontweight="bold", pad=12,
    )
    ax.legend(loc="upper left", fontsize=9, framealpha=0.9)
    ax.grid(alpha=0.15)
    ax.set_xticks([0.36, 1.5, 3, 7])
    ax.set_xticklabels(["360M", "1.5B", "3B", "7B"])
    ax.set_ylim(-2, 105)
    plt.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(output_dir, f"figure2_utilization_gap.{ext}"))
    plt.close(fig)
    print("Saved figure2_utilization_gap")


def figure3_distraction(ci_df: pd.DataFrame, output_dir: str) -> None:
    fig, ax = plt.subplots(figsize=(8, 5.5))

    def get_em(model, ret, split):
        cell = ci_df[
            (ci_df["model"] == model) & (ci_df["retrieval"] == ret)
            & (ci_df["split"] == split) & (ci_df["dataset"] == "all")
        ]
        return cell.iloc[0]["em"] * 100 if len(cell) > 0 else 0

    distraction_noisy = [100 - get_em(m, "noisy", "known") for m in MODEL_ORDER]
    distraction_oracle = [100 - get_em(m, "oracle", "known") for m in MODEL_ORDER]

    x = np.arange(len(MODEL_ORDER))
    width = 0.35
    bars1 = ax.bar(x - width / 2, distraction_noisy, width, label="Noisy retrieval", color="#d63031", alpha=0.8)
    bars2 = ax.bar(x + width / 2, distraction_oracle, width, label="Oracle retrieval", color="#e17055", alpha=0.8)

    for bars in (bars1, bars2):
        for bar in bars:
            h = bar.get_height()
            if h > 0:
                ax.annotate(
                    f"{h:.0f}%",
                    xy=(bar.get_x() + bar.get_width() / 2, h),
                    xytext=(0, 3), textcoords="offset points",
                    ha="center", va="bottom", fontsize=9, fontweight="bold",
                )

    ax.set_xlabel("Model Size")
    ax.set_ylabel("% of Known Answers Destroyed by Retrieval")
    ax.set_title(
        "The Distraction Effect\n(model knew the answer, but retrieval made it forget)",
        fontweight="bold", pad=12,
    )
    ax.set_xticks(x)
    ax.set_xticklabels([MODEL_SHORT[m] for m in MODEL_ORDER])
    ax.legend(framealpha=0.9)
    ax.grid(alpha=0.15, axis="y")
    ax.set_ylim(0, 115)
    plt.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(output_dir, f"figure3_distraction.{ext}"))
    plt.close(fig)
    print("Saved figure3_distraction")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Statistical analysis and figure generation.")
    p.add_argument("--grid_path", default="data/grid_v2/full_grid_v2.csv")
    p.add_argument("--corpus_only_path", default="data/grid_v2/corpus_only_grid_v2.csv")
    p.add_argument("--output_dir", default="results/figures")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    df_corpus = pd.read_csv(args.corpus_only_path)
    print(f"Corpus-only results: {len(df_corpus)} rows")

    print("\nComputing bootstrap CIs…")
    ci_df = compute_all_cis(df_corpus)
    ci_df.to_csv(os.path.join(args.output_dir, "ci_results.csv"), index=False)
    print(f"  Saved ci_results.csv ({len(ci_df)} cells)")

    print("\nRunning McNemar's tests…")
    sig_df = run_significance_tests(df_corpus)
    sig_df.to_csv(os.path.join(args.output_dir, "significance_tests.csv"), index=False)
    print(sig_df[["comparison", "split", "model", "delta_pp", "p_value", "sig_label"]].to_string(index=False))

    print("\nGenerating figures…")
    figure1_scaling_curves(ci_df, args.output_dir)
    figure2_utilization_gap(ci_df, args.output_dir)
    figure3_distraction(ci_df, args.output_dir)

    # Print key numbers
    print(f"\n{'='*70}")
    print(" KEY NUMBERS (for abstract / paper)")
    print(f"{'='*70}")
    for model in MODEL_ORDER:
        cell = ci_df[
            (ci_df["model"] == model) & (ci_df["retrieval"] == "oracle")
            & (ci_df["split"] == "unknown") & (ci_df["dataset"] == "all")
        ]
        if len(cell) > 0:
            em = cell.iloc[0]["em"] * 100
            print(f"  {model}: oracle EM on UNKNOWN = {em:.1f}% → failure rate {100-em:.0f}%")
    print()
    for model in MODEL_ORDER[1:]:
        none_cell = ci_df[(ci_df["model"] == model) & (ci_df["retrieval"] == "none")
                          & (ci_df["split"] == "known") & (ci_df["dataset"] == "all")]
        noisy_cell = ci_df[(ci_df["model"] == model) & (ci_df["retrieval"] == "noisy")
                           & (ci_df["split"] == "known") & (ci_df["dataset"] == "all")]
        if len(none_cell) > 0 and len(noisy_cell) > 0:
            drop = (none_cell.iloc[0]["em"] - noisy_cell.iloc[0]["em"]) * 100
            print(f"  {model}: noisy retrieval destroys {drop:.0f}pp of known answers")

    print(f"\nAll outputs saved to {args.output_dir}/")


if __name__ == "__main__":
    main()