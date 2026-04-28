"""
tests/test_evaluator.py
=======================
Verify evaluation primitives:

  • Precision/Recall/F1 match reference values
  • MAP, NDCG@10 sane on toy inputs
  • JSD diagnostic returns 0.00 for context-agnostic baseline
  • Paired bootstrap returns sensible p-value
"""

from __future__ import annotations

import numpy as np

from caff.evaluator import (
    average_precision,
    jensen_shannon_bits,
    ndcg_at_k,
    paired_bootstrap,
    precision_recall_f1,
)


def test_precision_recall_f1_basic():
    scores = np.array([0.9, 0.8, 0.4, 0.1])
    labels = np.array([1, 1, 0, 0])
    out = precision_recall_f1(scores, labels, threshold=0.5)
    assert out["precision"] == 1.0
    assert out["recall"] == 1.0
    assert out["f1"] == 1.0


def test_precision_recall_f1_imperfect():
    scores = np.array([0.9, 0.4, 0.6, 0.1])
    labels = np.array([1, 1, 0, 0])
    # Predictions ≥0.5: [1, 0, 1, 0]
    # TP=1 (idx 0), FP=1 (idx 2), FN=1 (idx 1)
    out = precision_recall_f1(scores, labels, threshold=0.5)
    assert np.isclose(out["precision"], 0.5)
    assert np.isclose(out["recall"], 0.5)
    assert np.isclose(out["f1"], 0.5)


def test_average_precision_perfect_ranking():
    """All positives ranked above all negatives → AP = 1.0."""
    scores = np.array([0.9, 0.8, 0.3, 0.1])
    labels = np.array([1, 1, 0, 0])
    assert average_precision(scores, labels) == 1.0


def test_ndcg_at_10_perfect():
    """Perfect ranking → NDCG = 1.0."""
    scores = np.array([0.9, 0.8, 0.7, 0.1, 0.05])
    labels = np.array([1, 1, 1, 0, 0])
    assert np.isclose(ndcg_at_k(scores, labels, k=10), 1.0)


def test_jsd_zero_when_equal():
    """JSD between identical distributions is 0 — the CBE signature
    (paper Table 10: every context-agnostic baseline → 0.00 bits)."""
    assert jensen_shannon_bits(0.5, 0.5) == 0.0
    assert jensen_shannon_bits(0.8, 0.8) == 0.0


def test_jsd_positive_when_different():
    """Different distributions yield positive JSD."""
    j = jensen_shannon_bits(0.9, 0.1)
    assert j > 0.5         # large divergence


def test_jsd_paper_appendix_c_calculation():
    """Reproduce the JSD calculation in paper Appendix C verbatim:
        p=0.872, q=0.238 → 2·JSD ≈ 1.84 bits.
    """
    p, q = 0.872, 0.238
    jsd_pure = jensen_shannon_bits(p, q)
    doubled = 2.0 * jsd_pure
    # Paper rounds to 1.84; allow small tolerance
    assert 1.7 < doubled < 1.95


def test_paired_bootstrap_zero_difference():
    """Identical metrics → delta ≈ 0, p ≈ 1."""
    a = [0.7, 0.8, 0.6, 0.9, 0.5]
    b = list(a)             # identical
    out = paired_bootstrap(a, b, n_resamples=1000, seed=42)
    assert abs(out["delta_mean"]) < 1e-9


def test_paired_bootstrap_clear_difference():
    """Strongly different metrics → small p-value."""
    a = [0.9, 0.85, 0.92, 0.88, 0.91]
    b = [0.6, 0.55, 0.65, 0.58, 0.62]
    out = paired_bootstrap(a, b, n_resamples=1000, seed=42)
    assert out["delta_mean"] > 0.2
    assert out["p_value"] < 0.05