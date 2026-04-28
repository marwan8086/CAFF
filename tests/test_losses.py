"""
tests/test_losses.py
====================
Verify loss formulations (Eqs. 21-23):

  • BCE matches torch reference
  • DC and HC3 are non-negative hinges
  • HC3 returns 0 when s_pos exceeds s_neg + γ_C
  • Combined L = L_BCE + λ_D L_DC + λ_C L_HC3
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from caff.losses import (
    BCEFilterLoss,
    CAFFCombinedLoss,
    DepthContrastiveLoss,
    HC3Loss,
)


def test_bce_matches_reference():
    bce = BCEFilterLoss()
    logits = torch.tensor([0.5, -1.0, 2.0])
    labels = torch.tensor([1.0, 0.0, 1.0])
    expected = F.binary_cross_entropy_with_logits(logits, labels)
    assert torch.isclose(bce(logits, labels), expected)


def test_dc_zero_when_satisfied():
    """L_DC = 0 when s_correct exceeds s_wrong by ≥ γ_D."""
    dc = DepthContrastiveLoss(margin=0.20)
    s_correct = torch.tensor([0.9, 0.8])
    s_wrong   = torch.tensor([0.5, 0.4])
    loss = dc(s_correct, s_wrong)
    assert loss.item() == 0.0


def test_dc_positive_when_violated():
    """L_DC > 0 when margin is violated."""
    dc = DepthContrastiveLoss(margin=0.20)
    s_correct = torch.tensor([0.5])
    s_wrong   = torch.tensor([0.7])  # s_wrong > s_correct → violation
    loss = dc(s_correct, s_wrong)
    assert loss.item() > 0.0


def test_hc3_zero_when_satisfied():
    """L_HC3 = 0 when s_pos exceeds s_neg by ≥ γ_C."""
    hc3 = HC3Loss(margin=0.25)
    s_pos = torch.tensor([0.9, 0.8])
    s_neg = torch.tensor([0.3, 0.2])
    assert hc3(s_pos, s_neg).item() == 0.0


def test_hc3_positive_when_violated():
    """L_HC3 > 0 when margin γ_C = 0.25 is violated."""
    hc3 = HC3Loss(margin=0.25)
    s_pos = torch.tensor([0.5])
    s_neg = torch.tensor([0.6])     # negative scores higher than positive
    loss = hc3(s_pos, s_neg)
    assert loss.item() > 0.0
    # Specifically: max(0, 0.6 - 0.5 + 0.25) = 0.35
    assert torch.isclose(loss, torch.tensor(0.35))


def test_combined_loss_decomposition():
    """L = L_BCE + λ_D L_DC + λ_C L_HC3 (Eq. 22)."""
    combined = CAFFCombinedLoss(lambda_D=0.40, lambda_C=0.35,
                                gamma_D=0.20, gamma_C=0.25)
    logits = torch.tensor([0.5, -1.0])
    labels = torch.tensor([1.0, 0.0])
    s_corr = torch.tensor([0.4])
    s_wrng = torch.tensor([0.5])
    s_pos  = torch.tensor([0.4])
    s_neg  = torch.tensor([0.5])

    out = combined(
        bce_logits=logits, bce_labels=labels,
        dc_correct=s_corr, dc_wrong=s_wrng,
        hc3_pos=s_pos, hc3_neg=s_neg,
    )
    expected = (
        out["bce"]
        + 0.40 * out["dc"]
        + 0.35 * out["hc3"]
    )
    assert torch.isclose(out["total"], expected)