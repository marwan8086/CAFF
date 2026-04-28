"""
tests/test_scorer.py
====================
Verify scorer (Eqs. 18-19) properties:

  â€¢ W^ctx_â„“ = W_0 + A_â„“ B_â„“^T + Î”_â„“                  (Eq. 18)
  â€¢ s âˆˆ (0, 1)                                        (Eq. 19, sigmoid)
  â€¢ W^ctx computed once per hop, reused per candidate (S3)
"""

from __future__ import annotations

import torch

from caff.scorer import DepthCorrection, HopScorer


def test_depth_correction_shape():
    """A_â„“ B_â„“^T has shape (d, d) regardless of Ï."""
    dc = DepthCorrection(d=32, rho=4)
    AB = dc()
    assert AB.shape == (32, 32)


def test_hop_scorer_W_ctx_decomposition():
    """Eq. 18: W^ctx_â„“ = W_0 + A_â„“ B_â„“^T + Î”_â„“"""
    d, rho = 32, 4
    scorer = HopScorer(d=d, rho=rho)
    W0 = torch.randn(d, d)
    z = torch.randn(2, d)

    W_ctx = scorer.compute_W_ctx(W0, z)            # (B, d, d)
    AB = scorer.depth_correction()                  # (d, d)
    delta = scorer.dbm_block(z)                     # (B, d, d)
    expected = W0.unsqueeze(0) + AB.unsqueeze(0) + delta
    assert torch.allclose(W_ctx, expected, atol=1e-5)


def test_score_in_unit_interval():
    """Eq. 19: Ïƒ(...) âˆˆ (0, 1)."""
    d, rho = 32, 4
    scorer = HopScorer(d=d, rho=rho)
    W0 = torch.randn(d, d)
    z = torch.randn(1, d)
    W_ctx = scorer.compute_W_ctx(W0, z).squeeze(0)
    v = torch.randn(d)
    q = torch.randn(d)
    E_r = torch.randn(10, d)
    s = scorer.score_candidates(W_ctx, v, q, E_r)
    assert s.shape == (10,)
    assert (s >= 0).all() and (s <= 1).all()


def test_S3_per_hop_caching_correctness():
    """Smart Engineering S3: scoring N candidates with the same W_ctx
    must equal the per-candidate computation."""
    d, rho = 32, 4
    scorer = HopScorer(d=d, rho=rho)
    W0 = torch.randn(d, d)
    z = torch.randn(1, d)
    W_ctx = scorer.compute_W_ctx(W0, z).squeeze(0)   # (d, d)
    v = torch.randn(d)
    q = torch.randn(d)
    E_r = torch.randn(5, d)

    batched = scorer.score_candidates(W_ctx, v, q, E_r)
    individual = torch.stack([
        scorer.score(
            W_ctx.unsqueeze(0), v,
            q.unsqueeze(0), E_r[i:i+1],
        ).squeeze(0)
        for i in range(5)
    ])
    assert torch.allclose(batched, individual, atol=1e-5)


def test_dbm_can_be_disabled():
    """Ablation Â§10.1: w/o DBM â†’ Î”_â„“ â‰¡ 0."""
    d, rho = 32, 4
    scorer = HopScorer(d=d, rho=rho, use_dbm=False)
    W0 = torch.randn(d, d)
    z = torch.randn(2, d)
    W_ctx = scorer.compute_W_ctx(W0, z)
    # Without DBM, W^ctx_â„“ = W_0 + A_â„“ B_â„“^T (constant across batch)
    assert torch.allclose(W_ctx[0], W_ctx[1], atol=1e-6)
