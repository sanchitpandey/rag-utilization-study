"""
02_dense_retrieval.py — Dense retrieval with E5-large-v2 + FAISS.

Steps
-----
1. Load Wikipedia corpus (parquet).
2. Encode all 500k passages with E5-large-v2 (mean-pool, half precision).
3. Build a FAISS IndexFlatIP over L2-normalised embeddings.
4. Encode eval queries and retrieve top-20 passages.
5. Report hit-rate@{1,5,10,20}.

Outputs
-------
  <embeddings_dir>/e5_embeddings_500k.npy    — (500000, 1024) float32
  <embeddings_dir>/faiss_e5_500k.index       — FAISS flat inner-product index
  <results_path>                              — dense_eval_results.jsonl
"""

from __future__ import annotations

import argparse
import gc
import os
import time

import faiss
import numpy as np
import pandas as pd
import torch
from transformers import AutoModel, AutoTokenizer
from tqdm import tqdm

from utils import compute_hit_rate, load_jsonl, save_jsonl

MODEL_NAME = "intfloat/e5-large-v2"
BATCH_SIZE = 256
MAX_PASSAGE_LEN = 128
MAX_QUERY_LEN = 64
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ---------------------------------------------------------------------------
# Encoding
# ---------------------------------------------------------------------------

def load_encoder(model_name: str = MODEL_NAME):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if DEVICE == "cuda":
        model = AutoModel.from_pretrained(model_name).cuda().half()
    else:
        model = AutoModel.from_pretrained(model_name).float()
    model.eval()
    return tokenizer, model


def mean_pool(outputs, attention_mask):
    token_embs = outputs.last_hidden_state
    mask_expanded = attention_mask.unsqueeze(-1).expand(token_embs.size()).float()
    summed = torch.sum(token_embs * mask_expanded, dim=1)
    counts = torch.clamp(mask_expanded.sum(1), min=1e-9)
    return (summed / counts).cpu().numpy().astype(np.float32)


def encode_batch(
    tokenizer,
    model,
    texts: list[str],
    prefix: str = "passage: ",
    max_length: int = MAX_PASSAGE_LEN,
) -> np.ndarray:
    prefixed = [prefix + t for t in texts]
    inputs = tokenizer(
        prefixed,
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    ).to(DEVICE)
    with torch.no_grad():
        outputs = model(**inputs)
    return mean_pool(outputs, inputs["attention_mask"])


def encode_corpus(
    passages: list[dict],
    tokenizer,
    model,
    embeddings_path: str,
    batch_size: int = BATCH_SIZE,
) -> np.ndarray:
    if os.path.exists(embeddings_path):
        print(f"Loading existing embeddings from {embeddings_path}…")
        return np.load(embeddings_path)

    # Sort by length to minimise padding overhead
    length_sorted_idx = sorted(range(len(passages)), key=lambda i: len(passages[i]["text"]))
    inv_idx = [0] * len(passages)
    for new_pos, old_pos in enumerate(length_sorted_idx):
        inv_idx[old_pos] = new_pos
    sorted_passages = [passages[i] for i in length_sorted_idx]

    all_embs: list[np.ndarray] = []
    t0 = time.time()
    for i in tqdm(range(0, len(sorted_passages), batch_size), desc="Encoding passages"):
        batch_texts = [p["text"] for p in sorted_passages[i : i + batch_size]]
        embs = encode_batch(tokenizer, model, batch_texts, prefix="passage: ", max_length=MAX_PASSAGE_LEN)
        all_embs.append(embs)

    sorted_embeddings = np.concatenate(all_embs)
    embeddings = sorted_embeddings[inv_idx]

    os.makedirs(os.path.dirname(embeddings_path) or ".", exist_ok=True)
    np.save(embeddings_path, embeddings)
    print(f"Embeddings saved → {embeddings_path} ({time.time() - t0:.0f}s, {os.path.getsize(embeddings_path)/1e9:.2f} GB)")
    return embeddings


# ---------------------------------------------------------------------------
# FAISS index
# ---------------------------------------------------------------------------

def build_faiss_index(embeddings: np.ndarray, index_path: str) -> faiss.Index:
    if os.path.exists(index_path):
        print(f"Loading existing FAISS index from {index_path}…")
        return faiss.read_index(index_path)

    embs = np.ascontiguousarray(embeddings.astype("float32", copy=False))
    faiss.normalize_L2(embs)
    dim = embs.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embs)
    faiss.write_index(index, index_path)
    print(f"FAISS index saved → {index_path} ({index.ntotal} vectors, dim={dim})")
    return index


# ---------------------------------------------------------------------------
# Retrieval
# ---------------------------------------------------------------------------

def retrieve_dense(
    eval_data: list[dict],
    passages: list[dict],
    tokenizer,
    model,
    index: faiss.Index,
    top_k: int = 20,
    query_batch: int = 64,
) -> list[dict]:
    query_texts = [ex["question"] for ex in eval_data]
    query_embs_list: list[np.ndarray] = []
    for i in tqdm(range(0, len(query_texts), query_batch), desc="Encoding queries"):
        batch = query_texts[i : i + query_batch]
        embs = encode_batch(tokenizer, model, batch, prefix="query: ", max_length=MAX_QUERY_LEN)
        query_embs_list.append(embs)

    query_embeddings = np.concatenate(query_embs_list)
    faiss.normalize_L2(query_embeddings)

    scores, indices = index.search(query_embeddings, top_k)

    results = []
    for i, ex in enumerate(eval_data):
        retrieved = [
            {
                "id": passages[int(indices[i][j])]["id"],
                "title": passages[int(indices[i][j])]["title"],
                "text": passages[int(indices[i][j])]["text"],
                "score": float(scores[i][j]),
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


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Dense retrieval with E5-large-v2 + FAISS.")
    p.add_argument("--corpus_path", default="data/corpus/wiki_500k.parquet")
    p.add_argument("--eval_path", default="data/eval.jsonl")
    p.add_argument("--embeddings_dir", default="data/embeddings")
    p.add_argument("--results_path", default="data/results/dense_eval_results.jsonl")
    p.add_argument("--top_k", type=int, default=20)
    p.add_argument("--batch_size", type=int, default=256)
    return p.parse_args()


def main() -> None:
    args = parse_args()

    print("Loading corpus…")
    df = pd.read_parquet(args.corpus_path)
    passages = df.to_dict("records")
    print(f"  {len(passages)} passages loaded.")

    eval_data = load_jsonl(args.eval_path)
    print(f"  {len(eval_data)} eval examples loaded.")

    tokenizer, model = load_encoder()

    embeddings_path = os.path.join(args.embeddings_dir, "e5_embeddings_500k.npy")
    embeddings = encode_corpus(passages, tokenizer, model, embeddings_path, batch_size=args.batch_size)

    index_path = os.path.join(args.embeddings_dir, "faiss_e5_500k.index")
    index = build_faiss_index(embeddings, index_path)

    results = retrieve_dense(eval_data, passages, tokenizer, model, index, top_k=args.top_k)

    os.makedirs(os.path.dirname(args.results_path) or ".", exist_ok=True)
    save_jsonl(args.results_path, results)
    print(f"Saved dense results → {args.results_path}")

    compute_hit_rate(results, label="Dense (E5-large-v2)")

    # Free GPU memory
    del model
    torch.cuda.empty_cache()
    gc.collect()


if __name__ == "__main__":
    main()