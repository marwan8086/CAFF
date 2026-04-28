"""
caff/losses.py
==============
Training losses for CAFF — paper §6.4-6.5, Equations 21-23.

Equation 21 (HC3 — Hop-Conditioned Context Contrast):

    L_HC3 = (1 / |T_HC3|) Σ max( 0,
                                  s(Q, r, ℓ, z^(b))
                                - s(Q, r, ℓ, z^(a))
                                + γ_C )

  where (Q, r, ℓ, z^(a), z^(b)) is a context-contrast triplet:
    z^(a) — context where the triple was labeled positive
    z^(b) — context where the SAME triple was labeled negative
    γ_C   — margin (paper: 0.25)

Equation 22 (combined objective):

    L = L_BCE + λ_D · L_DC + λ_C · L_HC3
        (paper: λ_D=0.40, λ_C=0.35)

Equation 23 (DC — Depth Contrastive):

    L_DC = (1 / |T_DC|) Σ max( 0,
                                s_{ℓ-} - s_{ℓ+} + γ_D )

  where the same (Q, r) pair is anchored at a "correct" hop ℓ_+
  vs an "incorrect" hop ℓ_-, with margin γ_D = 0.20.

Theoretical role (paper Proposition 3): minimizing L_HC3 maximizes
a variational lower bound on I(Y; S | z) — the conditional mutual
information that Theorem 1 identifies as the source of the CBE
error floor.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class BCEFilterLoss(nn.Module):
    """Standard binary cross-entropy on triple relevance labels.

    Operates on logits (pre-sigmoid) for numerical stability.
    Optionally weights positive class to counter class imbalance
    (typical in BFS-augmented KG-RAG: ~95% negatives).
    """

    def __init__(self, pos_weight: float | None = None) -> None:
        super().__init__()
        if pos_weight is not None:
            self.register_buffer("pos_weight", torch.tensor([pos_weight]))
        else:
            self.pos_weight = None

    def forward(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        logits : Tensor of shape (N,)   — pre-sigmoid filter scores
        labels : Tensor of shape (N,)   — gold relevance ∈ {0, 1}

        Returns
        -------
        Scalar tensor.
        """
        return F.binary_cross_entropy_with_logits(
            logits, labels.float(),
            pos_weight=self.pos_weight,
        )


class DepthContrastiveLoss(nn.Module):
    """Depth-contrastive hinge loss — Eq. 23.

    Discourages the model from assigning the same score to a
    triple at the "right" hop vs. the "wrong" hop. This addresses
    the Position-Conflation Error (PCE) component of the loss
    landscape.

    Parameters
    ----------
    margin : float
        γ_D in the paper. Default 0.20 (paper §8.4).
    """

    def __init__(self, margin: float = 0.20) -> None:
        super().__init__()
        self.margin = margin

    def forward(
        self,
        s_correct_hop: torch.Tensor,
        s_wrong_hop: torch.Tensor,
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        s_correct_hop : Tensor of shape (N,)
            Scores for (Q, r) at the gold hop ℓ_+.
        s_wrong_hop : Tensor of shape (N,)
            Scores for the SAME (Q, r) at a non-gold hop ℓ_-.

        Returns
        -------
        Scalar tensor.
        """
        # max(0, s_- - s_+ + γ_D)
        violations = F.relu(s_wrong_hop - s_correct_hop + self.margin)
        return violations.mean()


class HC3Loss(nn.Module):
    """Hop-Conditioned Context Contrast — Eq. 21.

    For each anchor (Q, r, ℓ), pulls scores under the positive
    context z^(a) up and pushes scores under the negative context
    z^(b) down. The same (Q, r, ℓ) is scored in both contexts —
    only z differs. This directly trains the model to satisfy
    the formal correctness criterion in §14.1:

        "a filter is context-correct iff it assigns DIFFERENT
         scores to the same triple under different retained
         contexts wherever the optimal posterior differs."

    By Proposition 3, minimizing this loss maximizes a variational
    lower bound on I(Y; S | z).

    Parameters
    ----------
    margin : float
        γ_C in the paper. Default 0.25 (paper §8.4).
    """

    def __init__(self, margin: float = 0.25) -> None:
        super().__init__()
        self.margin = margin

    def forward(
        self,
        s_pos_context: torch.Tensor,
        s_neg_context: torch.Tensor,
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        s_pos_context : Tensor of shape (N,)
            Score s(Q, r, ℓ, z^(a)) where the triple was labeled 1.
        s_neg_context : Tensor of shape (N,)
            Score s(Q, r, ℓ, z^(b)) where the SAME (Q, r, ℓ) was
            labeled 0 (different upstream context).

        Returns
        -------
        Scalar tensor.

        Notes
        -----
        N is the total number of negative contexts across all
        anchors (typically 8 × |anchors|, per §6.4 mining policy).
        s_pos_context is broadcast/repeated to match N.
        """
        # max(0, s^(b) - s^(a) + γ_C)
        violations = F.relu(s_neg_context - s_pos_context + self.margin)
        return violations.mean()


class CAFFCombinedLoss(nn.Module):
    """Combined training objective — Eq. 22.

        L = L_BCE + λ_D · L_DC + λ_C · L_HC3

    Defaults match paper §8.4: λ_D = 0.40, λ_C = 0.35.

    Ablation note (paper §10.1):
      • Setting λ_C = 0 → "w/o L_HC3" variant, -1.4 acc
      • Setting λ_D = 0 → "w/o L_DC" variant,  -0.7 acc
    """

    def __init__(
        self,
        lambda_D: float = 0.40,
        lambda_C: float = 0.35,
        gamma_D: float = 0.20,
        gamma_C: float = 0.25,
        bce_pos_weight: float | None = None,
    ) -> None:
        super().__init__()
        self.bce = BCEFilterLoss(pos_weight=bce_pos_weight)
        self.dc = DepthContrastiveLoss(margin=gamma_D)
        self.hc3 = HC3Loss(margin=gamma_C)
        self.lambda_D = lambda_D
        self.lambda_C = lambda_C

    def forward(
        self,
        bce_logits: torch.Tensor,
        bce_labels: torch.Tensor,
        dc_correct: torch.Tensor | None = None,
        dc_wrong: torch.Tensor | None = None,
        hc3_pos: torch.Tensor | None = None,
        hc3_neg: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """Compute combined loss and individual components.

        Returns
        -------
        Dict with keys: 'total', 'bce', 'dc', 'hc3'.
        Components that have no triplets (None inputs) contribute 0.
        """
        l_bce = self.bce(bce_logits, bce_labels)

        if dc_correct is not None and dc_wrong is not None and len(dc_correct) > 0:
            l_dc = self.dc(dc_correct, dc_wrong)
        else:
            l_dc = torch.zeros((), device=bce_logits.device)

        if hc3_pos is not None and hc3_neg is not None and len(hc3_pos) > 0:
            l_hc3 = self.hc3(hc3_pos, hc3_neg)
        else:
            l_hc3 = torch.zeros((), device=bce_logits.device)

        total = l_bce + self.lambda_D * l_dc + self.lambda_C * l_hc3

        return {
            "total": total,
            "bce": l_bce.detach(),
            "dc": l_dc.detach(),
            "hc3": l_hc3.detach(),
        }