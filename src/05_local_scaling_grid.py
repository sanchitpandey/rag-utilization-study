"""
05_local_scaling_grid.py — Local model grid: SmolLM2 + Qwen2.5 family.

Runs inference for SmolLM2-360M, Qwen2.5-1.5B, Qwen2.5-3B across all retrieval
conditions on a 200-question NQ+HotpotQA subset. Also includes a prompt ablation
study (v1_forced / v2_permissive / v3_minimal) for Qwen2.5-3B with dense retrieval.

This is a precursor to the full 09_full_grid.py run.

Outputs
-------
  <output_dir>/<condition>__<model>.jsonl   — per-cell predictions
"""

from __future__ import annotations

import argparse
import gc
import os

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from utils import answer_in_passages, exact_match, f1_score, load_jsonl, save_jsonl

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

BNB_CONFIG = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
) if torch.cuda.is_available() else None

MODELS: dict[str, str] = {
    "SmolLM2-360M": "HuggingFaceTB/SmolLM2-360M-Instruct",
    "Qwen2.5-1.5B": "Qwen/Qwen2.5-1.5B-Instruct",
    "Qwen2.5-3B": "Qwen/Qwen2.5-3B-Instruct",
}


# ---------------------------------------------------------------------------
# Prompts (three variants for ablation study)
# ---------------------------------------------------------------------------

def build_prompt_v1(question: str, passages: list[dict], max_passages: int = 5) -> str:
    """Forced: 'answer based on context'."""
    if not passages:
        return (
            "Answer the following question in as few words as possible. "
            "Give ONLY the answer, nothing else.\n\n"
            f"Question: {question}\nAnswer:"
        )
    ctx = "\n".join(f"[{i+1}] {p['text']}" for i, p in enumerate(passages[:max_passages]))
    return (
        "Answer the following question based on the provided context. "
        "Give ONLY the answer in as few words as possible. "
        "If the context doesn't contain the answer, make your best guess.\n\n"
        f"Context:\n{ctx}\n\nQuestion: {question}\nAnswer:"
    )


def build_prompt_v2(question: str, passages: list[dict], max_passages: int = 5) -> str:
    """Permissive: 'use if helpful, else your own knowledge'."""
    if not passages:
        return (
            "Answer the following question in as few words as possible. "
            "Give ONLY the answer, nothing else.\n\n"
            f"Question: {question}\nAnswer:"
        )
    ctx = "\n".join(f"[{i+1}] {p['text']}" for i, p in enumerate(passages[:max_passages]))
    return (
        "Here is some context that may or may not be relevant. "
        "Use it only if it helps answer the question. "
        "Otherwise, rely on your own knowledge. "
        "Give ONLY the answer in as few words as possible.\n\n"
        f"Context:\n{ctx}\n\nQuestion: {question}\nAnswer:"
    )


def build_prompt_v3(question: str, passages: list[dict], max_passages: int = 5) -> str:
    """Minimal: just appends context as 'Reference material'."""
    if not passages:
        return f"Question: {question}\nAnswer in as few words as possible:"
    ctx = "\n".join(f"[{i+1}] {p['text']}" for i, p in enumerate(passages[:max_passages]))
    return f"Question: {question}\n\nReference material:\n{ctx}\n\nAnswer in as few words as possible:"


PROMPT_VARIANTS = {
    "v1_forced": build_prompt_v1,
    "v2_permissive": build_prompt_v2,
    "v3_minimal": build_prompt_v3,
}


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

def run_inference(model, tokenizer, prompt: str, max_new_tokens: int = 50) -> tuple[str, int]:
    inputs = tokenizer(
        prompt, return_tensors="pt", truncation=True, max_length=2048
    ).to(DEVICE)
    input_len = inputs["input_ids"].shape[1]
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
    generated = outputs[0][input_len:]
    answer = tokenizer.decode(generated, skip_special_tokens=True).strip().split("\n")[0].strip()
    return answer, input_len


def load_model(model_id: str):
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    load_kwargs: dict = {"device_map": "auto"} if DEVICE == "cuda" else {}
    if BNB_CONFIG is not None:
        load_kwargs["quantization_config"] = BNB_CONFIG
    else:
        load_kwargs["dtype"] = torch.float32
    model = AutoModelForCausalLM.from_pretrained(model_id, **load_kwargs)
    model.eval()
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return model, tokenizer


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Local model scaling grid (preliminary).")
    p.add_argument("--eval_path", default="data/eval.jsonl")
    p.add_argument("--bm25_results", default="data/results/bm25_eval_results.jsonl")
    p.add_argument("--dense_results", default="data/results/dense_eval_results.jsonl")
    p.add_argument("--hybrid_results", default="data/results/hybrid_eval_results.jsonl")
    p.add_argument("--output_dir", default="data/grid_local")
    p.add_argument("--nq_n", type=int, default=100)
    p.add_argument("--hotpot_n", type=int, default=100)
    p.add_argument(
        "--models",
        nargs="+",
        default=list(MODELS.keys()),
        choices=list(MODELS.keys()),
    )
    p.add_argument("--run_prompt_ablation", action="store_true",
                   help="Run 3-prompt ablation for Qwen2.5-3B × dense")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    eval_data_full = load_jsonl(args.eval_path)
    nq_eval = [ex for ex in eval_data_full if ex["dataset"] == "nq"][: args.nq_n]
    hp_eval = [ex for ex in eval_data_full if ex["dataset"] == "hotpotqa"][: args.hotpot_n]
    eval_subset = nq_eval + hp_eval
    eval_ids = {ex["id"] for ex in eval_subset}
    print(f"Eval subset: {len(eval_subset)} (NQ={len(nq_eval)}, HotpotQA={len(hp_eval)})")

    def filter_by_ids(results: list[dict]) -> list[dict]:
        return [ex for ex in results if ex["id"] in eval_ids]

    retrieval_levels: dict[str, list[dict]] = {
        "bm25": filter_by_ids(load_jsonl(args.bm25_results)),
        "dense": filter_by_ids(load_jsonl(args.dense_results)),
        "hybrid": filter_by_ids(load_jsonl(args.hybrid_results)),
        "none": [
            {**{k: ex[k] for k in ("id", "question", "answers", "dataset", "type")}, "retrieved": []}
            for ex in eval_subset
        ],
    }

    os.makedirs(args.output_dir, exist_ok=True)

    for model_name in args.models:
        model_id = MODELS[model_name]
        print(f"\n{'='*60}\nLoading: {model_name}\n{'='*60}")
        model, tokenizer = load_model(model_id)

        for ret_name, ret_data in retrieval_levels.items():
            outfile = os.path.join(args.output_dir, f"{ret_name}__{model_name}.jsonl")
            if os.path.exists(outfile):
                print(f"SKIP {ret_name} × {model_name}")
                continue

            results = []
            for ex in tqdm(ret_data, desc=f"{ret_name}×{model_name}"):
                prompt = build_prompt_v1(ex["question"], ex["retrieved"])
                answer, n_tokens = run_inference(model, tokenizer, prompt)
                results.append({
                    "id": ex["id"],
                    "question": ex["question"],
                    "gold_answers": ex["answers"],
                    "dataset": ex["dataset"],
                    "retrieval": ret_name,
                    "model": model_name,
                    "predicted": answer,
                    "em": exact_match(answer, ex["answers"]),
                    "f1": f1_score(answer, ex["answers"]),
                    "input_tokens": n_tokens,
                })
            save_jsonl(outfile, results)
            em = np.mean([r["em"] for r in results]) * 100
            print(f"  {ret_name} × {model_name}: EM={em:.1f}%")

        # ---- Optional prompt ablation (Qwen2.5-3B × dense only) ----
        if args.run_prompt_ablation and model_name == "Qwen2.5-3B":
            print("\nRunning prompt ablation (Qwen2.5-3B × dense)…")
            dense_data = retrieval_levels["dense"]
            retrieval_success = {
                ex["id"]: answer_in_passages(ex["answers"], ex["retrieved"])
                for ex in dense_data
            }
            for pname, pfunc in PROMPT_VARIANTS.items():
                preds = []
                for ex in tqdm(dense_data, desc=f"Qwen-3B × dense × {pname}"):
                    prompt = pfunc(ex["question"], ex["retrieved"])
                    answer, _ = run_inference(model, tokenizer, prompt)
                    preds.append({
                        "id": ex["id"],
                        "predicted": answer,
                        "gold_answers": ex["answers"],
                        "prompt_variant": pname,
                        "answer_retrieved": retrieval_success.get(ex["id"], False),
                    })
                em_all = np.mean([exact_match(str(p["predicted"]), p["gold_answers"]) for p in preds]) * 100
                em_hit = np.mean([
                    exact_match(str(p["predicted"]), p["gold_answers"])
                    for p in preds if p["answer_retrieved"]
                ]) * 100
                print(f"  {pname}: overall={em_all:.1f}%  answer-in-passages={em_hit:.1f}%")
                save_jsonl(os.path.join(args.output_dir, f"prompt_ablation__{pname}.jsonl"), preds)

        del model, tokenizer
        torch.cuda.empty_cache()
        gc.collect()
        print(f"Unloaded {model_name}")


if __name__ == "__main__":
    main()