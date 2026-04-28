"""
caff/scorer.py
==============
Context-aware scoring layer — paper §6.3, Equations 18-19.

Equation 18 (context-aware scoring matrix):

    W^ctx_ℓ  =  W_0  +  A_ℓ B_ℓ^T  +  Δ_ℓ(g_ℓ)
                ───   ──────────     ────────────
                shared depth-specific  context-specific
                base   (PCE correction)(CBE correction)

Equation 19 (CAFF score):

    s_ℓ(Q, h, r, t) = σ( q^T W^ctx_ℓ e_r + v^T(q ⊙ e_r) + β_ℓ )

Where:
  • W_0 ∈ ℝ^{d×d}      — shared base scoring matrix
  • A_ℓ, B_ℓ ∈ ℝ^{d×ρ} — depth-specific PCE-correction factors
  • Δ_ℓ ∈ ℝ^{d×d}     — DBM matrix (from caff/dbm.py)
  • v ∈ ℝ^d           — shared element-wise interaction vector
  • β_ℓ ∈ ℝ           — depth-specific bias
  • ⊙                 — Hadamard (element-wise) product

Computational cost (paper §6.3): W^ctx_ℓ is precomputed ONCE per hop
(not per candidate). This is "Smart Engineering S3" in our build.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .dbm import DBMBlock


class DepthCorrection(nn.Module):
    """Depth-specific low-rank PCE correction A_ℓ B_ℓ^T (Eq. 18, middle term).

    This term corrects the Position-Conflation Error (PCE) — the
    failure mode addressed by depth-stratified baselines like
    SubgraphRAG. CAFF retains it and adds the CBE correction (Δ_ℓ)
    on top.
    """

    def __init__(self, d: int, rho: int) -> None:
        super().__init__()
        self.A = nn.Parameter(torch.empty(d, rho))
        self.B = nn.Parameter(torch.empty(d, rho))
        nn.init.xavier_uniform_(self.A)
        nn.init.xavier_uniform_(self.B)

    def forward(self) -> torch.Tensor:
        """Return A_ℓ B_ℓ^T of shape (d, d). Computed once per hop."""
        return self.A @ self.B.t()


class HopScorer(nn.Module):
    """CAFF scorer for a single hop ℓ — Eqs. 18-19.

    Per-hop trainable parameters (paper §8.4 confines all learnable
    capacity to <12M params):
      • A_ℓ, B_ℓ ∈ ℝ^{d×ρ}     — depth correction
      • U_ℓ, V_ℓ ∈ ℝ^{d×ρ}     — DBM factors (in DBMBlock)
      • P_ℓ ∈ ℝ^{ρ×d}, b^g_ℓ   — gate (in DBMBlock)
      • β_ℓ ∈ ℝ                — depth-specific bias

    Shared parameters (across all hops):
      • W_0 ∈ ℝ^{d×d}
      • v ∈ ℝ^d

    Smart Engineering S3 — per-hop W^ctx caching:
      W^ctx_ℓ depends only on z_{ℓ-1}, NOT on (h, r, t). For a hop
      with N_ℓ ≤ 500 candidates, we compute W^ctx ONCE and reuse
      it across all candidates. This realizes the paper's claim
      of "zero per-candidate overhead after one-time precomputation".
    """

    def __init__(
        self,
        d: int,
        rho: int,
        gate_activation: str = "sigmoid",
        use_dbm: bool = True,
    ) -> None:
        super().__init__()
        self.d = d
        self.rho = rho
        self.use_dbm = use_dbm

        # Per-hop components
        self.depth_correction = DepthCorrection(d, rho)
        self.dbm_block = DBMBlock(d, rho, gate_activation=gate_activation)
        self.beta = nn.Parameter(torch.zeros(1))  # depth-specific bias

    def compute_W_ctx(
        self,
        W0: torch.Tensor,
        z_prev: torch.Tensor,
    ) -> torch.Tensor:
        """Assemble W^ctx_ℓ per Eq. 18 — once per (hop, query).

        Parameters
        ----------
        W0 : Tensor of shape (d, d)
            Shared base matrix (provided by CAFFModel).
        z_prev : Tensor of shape (B, d)
            CSV from previous hop (z_{ℓ-1}).

        Returns
        -------
        W_ctx : Tensor of shape (B, d, d)
            One scoring matrix per item in the batch.
        """
        B = z_prev.shape[0]

        # Shared base term
        W_base = W0.unsqueeze(0).expand(B, -1, -1)               # (B, d, d)

        # Depth-specific PCE correction (constant across batch)
        AB = self.depth_correction()                              # (d, d)
        W_pce = AB.unsqueeze(0).expand(B, -1, -1)                 # (B, d, d)

        # Context-specific CBE correction (per-item)
        if self.use_dbm:
            delta = self.dbm_block(z_prev)                        # (B, d, d)
        else:
            # Ablation §10.1: w/o DBM → Δ_ℓ ≡ 0
            delta = torch.zeros(B, self.d, self.d, device=z_prev.device)

        return W_base + W_pce + delta

    def score(
        self,
        W_ctx: torch.Tensor,
        v: torch.Tensor,
        q: torch.Tensor,
        e_r: torch.Tensor,
    ) -> torch.Tensor:
        """Compute scores s_ℓ for a batch of (q, e_r) pairs (Eq. 19).

        Parameters
        ----------
        W_ctx : Tensor of shape (B, d, d)
            Pre-computed context-aware scoring matrices.
        v : Tensor of shape (d,)
            Shared element-wise interaction vector.
        q : Tensor of shape (B, d)
            ℓ2-normalized query embeddings.
        e_r : Tensor of shape (B, d)
            ℓ2-normalized relation embeddings.

        Returns
        -------
        s : Tensor of shape (B,)
            Sigmoid-activated scores in (0, 1).

        Notes
        -----
        For multiple candidates sharing the same (q, W_ctx) — the
        common case during a single hop — see `score_candidates`
        below for an efficient batched implementation.
        """
        # Bilinear term: q^T W^ctx e_r        — (B,)
        qW = torch.bmm(q.unsqueeze(1), W_ctx).squeeze(1)        # (B, d)
        bilinear = (qW * e_r).sum(dim=-1)                        # (B,)

        # Element-wise interaction: v^T (q ⊙ e_r)  — (B,)
        interaction = (v.unsqueeze(0) * (q * e_r)).sum(dim=-1)   # (B,)

        # Logit + sigmoid
        logit = bilinear + interaction + self.beta
        return torch.sigmoid(logit)

    def score_candidates(
        self,
        W_ctx: torch.Tensor,
        v: torch.Tensor,
        q: torch.Tensor,
        E_r: torch.Tensor,
    ) -> torch.Tensor:
        """Score N candidates against a single (q, W_ctx) — Eq. 19, batched.

        Smart Engineering S3 manifestation: W_ctx is computed once
        per hop, then reused for all N candidates of that hop.

        Parameters
        ----------
        W_ctx : Tensor of shape (d, d)        — single hop matrix
        v : Tensor of shape (d,)
        q : Tensor of shape (d,)              — single query
        E_r : Tensor of shape (N, d)          — N candidate relations

        Returns
        -------
        s : Tensor of shape (N,)              — score per candidate
        """
        # Bilinear: q^T W = (d,)  →  expand to (1, d)  →  matmul with E_r^T
        qW = (q.unsqueeze(0) @ W_ctx).squeeze(0)                 # (d,)
        bilinear = E_r @ qW                                      # (N,)

        # Interaction: v^T (q ⊙ E_r)
        interaction = (v.unsqueeze(0) * q.unsqueeze(0) * E_r).sum(dim=-1)  # (N,)

        logit = bilinear + interaction + self.beta
        return torch.sigmoid(logit)

    def score_logits(
        self,
        W_ctx: torch.Tensor,
        v: torch.Tensor,
        q: torch.Tensor,
        E_r: torch.Tensor,
    ) -> torch.Tensor:
        """Same as score_candidates but returns pre-sigmoid logits.

        Used by BCEWithLogitsLoss for numerical stability.
        """
        qW = (q.unsqueeze(0) @ W_ctx).squeeze(0)
        bilinear = E_r @ qW
        interaction = (v.unsqueeze(0) * q.unsqueeze(0) * E_r).sum(dim=-1)
        return bilinear + interaction + self.beta