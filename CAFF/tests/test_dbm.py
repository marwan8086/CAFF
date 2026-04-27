"""
tests/test_dbm.py
=================
Verify DBM (Eqs. 16-17) properties:

  • Gate output ∈ (0, 1)^ρ                      (Eq. 16, sigmoid)
  • Δ shape is (B, d, d)                        (Eq. 17)
  • rank(Δ) ≤ ρ                                  (Proposition 4)
  • z=0 → fixed Δ (graceful degradation)         (paper §6.3)
"""

from __future__ import annotations

import pytest
import torch

from caff.dbm import DBM, ContextGate, DBMBlock


def test_gate_sigmoid_output_range():
    """Eq. 16: g_ℓ ∈ (0, 1)^ρ under sigmoid activation."""
    gate = ContextGate(d=64, rho=8, activation="sigmoid")
    z = torch.randn(4, 64)
    g = gate(z)
    assert g.shape == (4, 8)
    assert (g > 0).all() and (g < 1).all()


def test_gate_relu_ablation():
    """Ablation §10.1: ReLU gate produces non-negative output."""
    gate = ContextGate(d=64, rho=8, activation="relu")
    z = torch.randn(4, 64)
    g = gate(z)
    assert (g >= 0).all()


def test_dbm_shape():
    """Eq. 17: Δ_ℓ ∈ ℝ^{d×d} per batch item."""
    dbm = DBM(d=64, rho=8)
    g = torch.rand(4, 8)
    delta = dbm(g)
    assert delta.shape == (4, 64, 64)


def test_dbm_rank_bound():
    """Proposition 4: rank(Δ_ℓ) ≤ ρ.

    With randomly initialized U, V and a strictly-positive gate
    (sigmoid output is in (0,1)), rank should be exactly ρ
    almost surely.
    """
    rho = 8
    dbm = DBM(d=64, rho=rho)
    g = torch.rand(2, rho) * 0.5 + 0.25       # bounded away from 0
    delta = dbm(g)
    for b in range(delta.shape[0]):
        rank = torch.linalg.matrix_rank(delta[b]).item()
        assert rank <= rho, f"rank(Δ) = {rank} exceeds ρ = {rho}"


def test_dbm_graceful_degradation_at_z_zero():
    """Paper §6.3: when z=0, gate becomes σ(b^g_ℓ) — a constant —
    and Δ_ℓ collapses to a fixed rank-ρ increment, recovering
    DepthBilinear."""
    block = DBMBlock(d=64, rho=8)
    z_zero_a = torch.zeros(2, 64)
    z_zero_b = torch.zeros(2, 64)
    delta_a = block(z_zero_a)
    delta_b = block(z_zero_b)
    # Same input ⟹ same output (deterministic)
    assert torch.allclose(delta_a, delta_b)
    # Both items in the batch produce identical Δ since z is identical
    assert torch.allclose(delta_a[0], delta_a[1])


def test_dbm_context_sensitivity():
    """Different z values produce different Δ matrices."""
    block = DBMBlock(d=64, rho=8)
    z1 = torch.randn(1, 64)
    z2 = torch.randn(1, 64)
    delta1 = block(z1)
    delta2 = block(z2)
    assert not torch.allclose(delta1, delta2)


def test_dbm_implements_u_diag_g_v_t():
    """Verify literal Eq. 17: Δ = U · diag(g) · V^T (not element-wise gating).

    This guards Failure Mode F1 from the build contract.
    """
    d, rho = 16, 4
    dbm = DBM(d=d, rho=rho)
    g = torch.tensor([[0.5, 0.5, 0.5, 0.5]])  # (1, rho)
    delta_module = dbm(g).squeeze(0)            # (d, d)

    # Reference computation: U @ diag(g) @ V^T
    delta_ref = dbm.U @ torch.diag(g.squeeze(0)) @ dbm.V.t()
    assert torch.allclose(delta_module, delta_ref, atol=1e-5)