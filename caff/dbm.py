"""
caff/dbm.py
===========
Dynamic Bilinear Modulation (DBM) — paper §6.3, Equations 16-17.

The DBM generates a low-rank, sigmoid-gated, context-dependent
perturbation to the scoring matrix at runtime (not during training).

Equation 16 (gate):
    g_ℓ = σ(P_ℓ z_{ℓ-1} + b^g_ℓ)  ∈ (0,1)^ρ

Equation 17 (DBM matrix):
    Δ_ℓ(g_ℓ) = U_ℓ · diag(g_ℓ) · V_ℓ^T  ∈ ℝ^{d×d}

Critical property (Proposition 4): rank(Δ_ℓ) ≤ ρ.
Critical property (paper §6.3): when z_{ℓ-1} = 0, gate becomes
σ(b^g_ℓ) — a constant — and Δ_ℓ collapses to a fixed rank-ρ
increment, recovering DepthBilinear gracefully.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class ContextGate(nn.Module):
    """Context gate (Eq. 16).

    Maps the CSV z_{ℓ-1} ∈ ℝ^d to a ρ-dimensional gating vector
    g_ℓ ∈ (0,1)^ρ via affine + sigmoid.
    """

    def __init__(self, d: int, rho: int, activation: str = "sigmoid") -> None:
        super().__init__()
        self.P = nn.Linear(d, rho, bias=True)  # P_ℓ z + b^g_ℓ
        # Initialize P with small variance so initial gate ≈ 0.5
        # (neutral perturbation at training start)
        nn.init.normal_(self.P.weight, mean=0.0, std=0.01)
        nn.init.zeros_(self.P.bias)
        assert activation in {"sigmoid", "relu"}, f"unknown activation {activation}"
        self.activation = activation

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        z : Tensor of shape (B, d)  — CSV from previous hop

        Returns
        -------
        g : Tensor of shape (B, rho), values in (0, 1) for sigmoid.
        """
        h = self.P(z)
        if self.activation == "sigmoid":
            return torch.sigmoid(h)
        return F.relu(h)  # ablation §10.1, paper reports -0.3 acc


class DBM(nn.Module):
    """Dynamic Bilinear Modulation matrix Δ_ℓ (Eq. 17).

    Δ_ℓ(g_ℓ) = U_ℓ · diag(g_ℓ) · V_ℓ^T

    where U_ℓ, V_ℓ ∈ ℝ^{d×ρ} are trainable per-hop factor matrices.
    rank(Δ_ℓ) ≤ ρ by construction (Proposition 4).
    """

    def __init__(self, d: int, rho: int) -> None:
        super().__init__()
        # U and V are d×ρ trainable factor matrices (Eq. 17)
        self.U = nn.Parameter(torch.empty(d, rho))
        self.V = nn.Parameter(torch.empty(d, rho))
        # Xavier-uniform init keeps Δ in a stable range relative to W_0
        nn.init.xavier_uniform_(self.U)
        nn.init.xavier_uniform_(self.V)
        self.d = d
        self.rho = rho

    def forward(self, g: torch.Tensor) -> torch.Tensor:
        """Construct Δ_ℓ from a batch of gates.

        Parameters
        ----------
        g : Tensor of shape (B, ρ)  — context gates (output of Eq. 16)

        Returns
        -------
        Δ : Tensor of shape (B, d, d)  — one DBM matrix per item

        Implementation note
        -------------------
        Δ_b = U · diag(g_b) · V^T
            = sum_k g_b[k] · (U[:, k] outer V[:, k])
        We compute this efficiently as
            (U * g_b)  @  V^T          # broadcasting g over rows of U^T
        which avoids materializing the diagonal matrix.
        """
        # (B, d, rho) = U.unsqueeze(0) * g.unsqueeze(1)
        U_scaled = self.U.unsqueeze(0) * g.unsqueeze(1)  # (B, d, rho)
        # (B, d, d) = U_scaled @ V^T
        delta = torch.bmm(U_scaled, self.V.t().unsqueeze(0).expand(g.shape[0], -1, -1))
        return delta


class DBMBlock(nn.Module):
    """Combined gate + DBM matrix generator for a single hop ℓ.

    Wraps Eq. 16 (gate) and Eq. 17 (DBM) into a single module
    per hop. The full per-hop scoring matrix W^ctx_ℓ is assembled
    in caff/scorer.py per Eq. 18.
    """

    def __init__(
        self,
        d: int,
        rho: int,
        gate_activation: str = "sigmoid",
    ) -> None:
        super().__init__()
        self.gate = ContextGate(d, rho, activation=gate_activation)
        self.dbm = DBM(d, rho)
        self.d = d
        self.rho = rho

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        z : Tensor of shape (B, d)  — CSV from previous hop

        Returns
        -------
        Δ : Tensor of shape (B, d, d)  — context-specific perturbation
        """
        g = self.gate(z)              # Eq. 16
        delta = self.dbm(g)           # Eq. 17
        return delta