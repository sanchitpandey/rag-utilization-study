#!/usr/bin/env bash
# reproduce.sh
#
# One-command reproduction of all paper results.
#
# Usage:
#   bash reproduce.sh
#
# Pre-requisites:
#   pip install -r requirements.txt
#   export HF_TOKEN=<your_token>       # required for Qwen models
#   export GROQ_API_KEY=<your_key>     # optional — only for step 04 (API grid)
#
# Hardware: 1× A100 40GB recommended. Estimated runtime: ~6h.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SRC="${ROOT}/src"
DATA="${ROOT}/data"

echo "=========================================================="
echo " Can Small Language Models Use What They Retrieve?"
echo " Full Reproduction Pipeline"
echo "=========================================================="
echo "Root : ${ROOT}"
echo "Device: $(python -c 'import torch; print("CUDA" if torch.cuda.is_available() else "CPU")')"
echo ""

# ---------- Step 0: Tests ---------------------------------------------------
echo "[0/9] Running unit tests…"
python -m pytest "${ROOT}/tests/" -q --tb=short
echo "  Tests passed."
echo ""

# ---------- Step 1: Data ----------------------------------------------------
echo "[1/9] Building dataset splits…"
python "${SRC}/00_data_pipeline.py" \
  --output_dir "${DATA}"
echo ""

# ---------- Step 2: BM25 ----------------------------------------------------
echo "[2/9] Building Wikipedia corpus + BM25 retrieval…"
python "${SRC}/01_bm25_retrieval.py" \
  --eval_path "${DATA}/eval.jsonl" \
  --corpus_output "${DATA}/corpus/wiki_500k.parquet" \
  --bm25_results "${DATA}/results/bm25_eval_results.jsonl"
echo ""

# ---------- Step 3: Dense ---------------------------------------------------
echo "[3/9] Dense retrieval (E5-large-v2 + FAISS)…"
python "${SRC}/02_dense_retrieval.py" \
  --corpus_path "${DATA}/corpus/wiki_500k.parquet" \
  --eval_path "${DATA}/eval.jsonl" \
  --embeddings_dir "${DATA}/embeddings" \
  --results_path "${DATA}/results/dense_eval_results.jsonl"
echo ""

# ---------- Step 4: Hybrid --------------------------------------------------
echo "[4/9] Hybrid RRF fusion…"
python "${SRC}/03_hybrid_retrieval.py" \
  --bm25_results "${DATA}/results/bm25_eval_results.jsonl" \
  --dense_results "${DATA}/results/dense_eval_results.jsonl" \
  --output "${DATA}/results/hybrid_eval_results.jsonl"
echo ""

# ---------- Step 5: Oracle --------------------------------------------------
echo "[5/9] Oracle passage construction…"
python "${SRC}/07_oracle_retrieval.py" \
  --eval_path "${DATA}/eval.jsonl" \
  --corpus_path "${DATA}/corpus/wiki_500k.parquet" \
  --dense_results "${DATA}/results/dense_eval_results.jsonl" \
  --output "${DATA}/results/oracle_eval_results.jsonl"
echo ""

# ---------- Step 6: Parametric split ----------------------------------------
echo "[6/9] Classifying questions as known/unknown per model…"
python "${SRC}/08_parametric_split.py" \
  --eval_path "${DATA}/eval.jsonl" \
  --none_preds_dir "${DATA}/none_predictions" \
  --output "${DATA}/parametric_splits.json"
echo ""

# ---------- Step 7: Full grid -----------------------------------------------
echo "[7/9] Running full 4×3 evaluation grid…"
python "${SRC}/09_full_grid.py" \
  --eval_path "${DATA}/eval.jsonl" \
  --oracle_path "${DATA}/results/oracle_eval_results.jsonl" \
  --dense_path "${DATA}/results/dense_eval_results.jsonl" \
  --parametric_path "${DATA}/parametric_splits.json" \
  --none_preds_dir "${DATA}/none_predictions" \
  --output_dir "${DATA}/grid_v2"
echo ""

# ---------- Step 8: Statistical analysis + figures 1–3 ----------------------
echo "[8/9] Statistical analysis + figures 1–3…"
python "${SRC}/10_statistical_analysis.py" \
  --grid_path "${DATA}/grid_v2/full_grid_v2.csv" \
  --corpus_only_path "${DATA}/grid_v2/corpus_only_grid_v2.csv" \
  --output_dir "${ROOT}/results/figures"
echo ""

# ---------- Step 9: Error taxonomy + figures 4–5 ----------------------------
echo "[9/9] Error taxonomy analysis + figures 4–5…"
python "${SRC}/11_error_analysis.py" \
  --corpus_only_path "${DATA}/grid_v2/corpus_only_grid_v2.csv" \
  --oracle_path "${DATA}/results/oracle_eval_results.jsonl" \
  --output_dir "${ROOT}/results/figures"
echo ""

echo "=========================================================="
echo " Reproduction complete."
echo " Figures : ${ROOT}/results/figures/"
echo " Data    : ${ROOT}/data/grid_v2/"
echo "=========================================================="