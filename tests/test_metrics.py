"""
tests/test_metrics.py — Unit tests for EM, F1, and normalization.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import pytest
from utils import normalize_answer, exact_match, f1_score, bootstrap_ci
import numpy as np


# ---------------------------------------------------------------------------
# normalize_answer
# ---------------------------------------------------------------------------

class TestNormalizeAnswer:
    def test_lowercase(self):
        assert normalize_answer("United States") == "united states"

    def test_strips_articles(self):
        assert normalize_answer("the United States") == "united states"
        assert normalize_answer("a cat") == "cat"
        assert normalize_answer("An apple") == "apple"

    def test_strips_punctuation(self):
        assert normalize_answer("hello, world!") == "hello world"
        assert normalize_answer("27 B.C.") == "27 bc"

    def test_collapses_whitespace(self):
        assert normalize_answer("  hello   world  ") == "hello world"

    def test_empty_string(self):
        assert normalize_answer("") == ""


# ---------------------------------------------------------------------------
# exact_match
# ---------------------------------------------------------------------------

class TestExactMatch:
    def test_exact(self):
        assert exact_match("Paris", ["Paris"]) == 1.0

    def test_case_insensitive(self):
        assert exact_match("paris", ["Paris"]) == 1.0

    def test_article_stripped(self):
        assert exact_match("the Eiffel Tower", ["Eiffel Tower"]) == 1.0

    def test_multiple_gold(self):
        assert exact_match("France", ["Paris", "France"]) == 1.0

    def test_wrong_answer(self):
        assert exact_match("Berlin", ["Paris"]) == 0.0

    def test_partial_no_match(self):
        # partial overlap should not count as EM
        assert exact_match("Jake Lloyd", ["Jake Matthew Lloyd"]) == 0.0

    def test_27_bc(self):
        assert exact_match("27 BC", ["27 BC"]) == 1.0


# ---------------------------------------------------------------------------
# f1_score
# ---------------------------------------------------------------------------

class TestF1Score:
    def test_exact_match_f1(self):
        assert f1_score("Pete Rose", ["Pete Rose"]) == 1.0

    def test_partial_overlap(self):
        # "Jake Lloyd" vs "Jake Matthew Lloyd" → 2 tokens common out of 2 pred, 3 gold
        score = f1_score("Jake Lloyd", ["Jake Matthew Lloyd"])
        assert 0 < score < 1.0

    def test_no_overlap(self):
        assert f1_score("Berlin", ["Paris"]) == 0.0

    def test_multiple_gold(self):
        # Should pick the best-matching gold answer
        score = f1_score("Pete Rose", ["Ty Cobb", "Pete Rose"])
        assert score == 1.0

    def test_empty_prediction(self):
        assert f1_score("", ["Paris"]) == 0.0


# ---------------------------------------------------------------------------
# bootstrap_ci
# ---------------------------------------------------------------------------

class TestBootstrapCI:
    def test_all_zeros(self):
        mean, lo, hi = bootstrap_ci([0.0] * 100)
        assert mean == 0.0
        assert lo == 0.0
        assert hi == 0.0

    def test_all_ones(self):
        mean, lo, hi = bootstrap_ci([1.0] * 100)
        assert mean == 1.0
        assert lo == pytest.approx(1.0, abs=0.05)
        assert hi == pytest.approx(1.0, abs=0.05)

    def test_ci_ordering(self):
        np.random.seed(0)
        scores = np.random.binomial(1, 0.3, size=200).astype(float)
        mean, lo, hi = bootstrap_ci(scores)
        assert lo <= mean <= hi

    def test_ci_coverage(self):
        np.random.seed(42)
        scores = np.random.binomial(1, 0.5, size=500).astype(float)
        mean, lo, hi = bootstrap_ci(scores, n_boot=1000, ci=0.95)
        # True mean is 0.5; CI should include it
        assert lo < 0.5 < hi

    def test_empty_input(self):
        mean, lo, hi = bootstrap_ci([])
        assert mean == 0.0