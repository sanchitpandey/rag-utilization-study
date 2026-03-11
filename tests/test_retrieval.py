"""
tests/test_retrieval.py — Unit tests for retrieval utilities.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import pytest
from utils import answer_in_passages, compute_hit_rate
from io import StringIO
import contextlib


SAMPLE_PASSAGES = [
    {"id": "0", "title": "France", "text": "Paris is the capital of France.", "score": 1.0},
    {"id": "1", "title": "Germany", "text": "Berlin is the capital of Germany.", "score": 0.9},
    {"id": "2", "title": "Spain", "text": "Madrid is the capital of Spain.", "score": 0.8},
]

SAMPLE_RESULTS = [
    {
        "id": "nq_0", "question": "What is the capital of France?",
        "answers": ["Paris"], "dataset": "nq", "type": "factoid",
        "retrieved": SAMPLE_PASSAGES,
    },
    {
        "id": "nq_1", "question": "What is the capital of Germany?",
        "answers": ["Berlin"], "dataset": "nq", "type": "factoid",
        "retrieved": SAMPLE_PASSAGES,
    },
    {
        "id": "hp_0", "question": "Who directed Inception?",
        "answers": ["Christopher Nolan"], "dataset": "hotpotqa", "type": "multi-hop",
        "retrieved": SAMPLE_PASSAGES,
    },
]

class TestAnswerInPassages:
    def test_answer_present(self):
        assert answer_in_passages(["Paris"], SAMPLE_PASSAGES, k=20) is True

    def test_answer_absent(self):
        assert answer_in_passages(["London"], SAMPLE_PASSAGES, k=20) is False

    def test_k_limits_search(self):
        assert answer_in_passages(["Berlin"], SAMPLE_PASSAGES, k=1) is False
        assert answer_in_passages(["Berlin"], SAMPLE_PASSAGES, k=2) is True

    def test_multiple_answers(self):
        assert answer_in_passages(["Nonexistent", "Paris"], SAMPLE_PASSAGES, k=20) is True

    def test_empty_passages(self):
        assert answer_in_passages(["Paris"], [], k=20) is False

    def test_case_insensitive(self):
        assert answer_in_passages(["paris"], SAMPLE_PASSAGES, k=20) is True

class TestComputeHitRate:
    def test_runs_without_error(self):
        buf = StringIO()
        with contextlib.redirect_stdout(buf):
            compute_hit_rate(SAMPLE_RESULTS, label="Test")
        output = buf.getvalue()
        assert "nq" in output
        assert "hotpotqa" in output

    def test_hit_rate_values(self):
        buf = StringIO()
        with contextlib.redirect_stdout(buf):
            compute_hit_rate(SAMPLE_RESULTS, k_values=[20])
        output = buf.getvalue()
        # The all-datasets row should show 66.7% (2/3 hits)
        assert "66.7" in output or "67" in output

import importlib.util as _ilu
import sys as _sys

def _load_hybrid_module():
    """Load 03_hybrid_retrieval via importlib, mocking heavy optional deps."""
    import types
    for stub in ("bm25s", "faiss"):
        if stub not in _sys.modules:
            _sys.modules[stub] = types.ModuleType(stub)
    src_dir = os.path.join(os.path.dirname(__file__), "..", "src")
    spec = _ilu.spec_from_file_location(
        "hybrid_retrieval",
        os.path.join(src_dir, "03_hybrid_retrieval.py"),
    )
    mod = _ilu.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

class TestRRFFusion:
    def test_rrf_merges_results(self):
        hybrid = _load_hybrid_module()
        reciprocal_rank_fusion = hybrid.reciprocal_rank_fusion

        fake_bm25 = [
            {"id": "q1", "question": "q", "answers": ["a"], "dataset": "nq", "type": "factoid",
             "retrieved": [{"id": "p1", "title": "T", "text": "T", "score": 1.0},
                           {"id": "p2", "title": "T", "text": "T", "score": 0.5}]},
        ]
        fake_dense = [
            {"id": "q1", "question": "q", "answers": ["a"], "dataset": "nq", "type": "factoid",
             "retrieved": [{"id": "p2", "title": "T", "text": "T", "score": 0.9},
                           {"id": "p3", "title": "T", "text": "T", "score": 0.8}]},
        ]
        result = reciprocal_rank_fusion(fake_bm25, fake_dense)
        assert len(result) == 1
        ids = [r["id"] for r in result[0]["retrieved"]]
        # p2 appears in both → should have highest RRF score
        assert ids[0] == "p2"