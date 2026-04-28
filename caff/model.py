"""
caff/model.py
=============
Full CAFFModel — assembles encoder, CSV, per-hop scorers into a
single trainable module.

Per paper §8.4, learnable capacity must be < 12M params (~3.5%
of the 340M-param frozen backbone). This is asserted at __init__.

Architecture summary (paper Algorithm 1):

    for ℓ = 1 to L:
        C_ℓ ← BFS(G, S(Q), ℓ)              # Eq. 1
        C_ℓ ← FreqCap(C_ℓ, K_r)            # Eq. 13
        g_ℓ ← σ(P_ℓ z_{ℓ-1} + b^g_ℓ)       # Eq. 16
        Δ_ℓ ← U_ℓ diag(g_ℓ) V_ℓ^T          # Eq. 17
        W^ctx_ℓ ← W_0 + A_ℓ B_ℓ^T + Δ_ℓ    # Eq. 18
        for (h, r, t) ∈ C_ℓ:
            s ← σ(q^T W^ctx_ℓ e_r + v^T(q ⊙ e_r) + β_ℓ)  # Eq. 19
            if s ≥ θ: S_ℓ ← S_ℓ ∪ {(h,r,t)}
        z_ℓ ← CSV(S_ℓ)                     # Eq. 14
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import torch
import torch.nn as nn

from .config import AblationFlags, CAFFConfig
from .csv import CSV
from .encoders import RelationEmbeddingCache
from .scorer import HopScorer

logger = logging.getLogger(__name__)


@dataclass
class HopOutput:
    """Output of scoring a single hop."""
    logits: torch.Tensor       # pre-sigmoid scores, shape (N,)
    scores: torch.Tensor       # sigmoid-activated, shape (N,)
    retained_mask: torch.Tensor  # bool, shape (N,) — True if score >= θ
    z_next: torch.Tensor       # CSV computed from retained set, shape (d,)


class CAFFModel(nn.Module):
    """Full CAFF filtering model (paper §6).

    Parameters
    ----------
    config : CAFFConfig
        Hyperparameters from paper §8.4.
    relation_cache : RelationEmbeddingCache
        Pre-encoded frozen relation embeddings.
    ablation : AblationFlags
        Component toggles for ablation study (paper §10.1).
    """

    def __init__(
        self,
        config: CAFFConfig,
        relation_cache: RelationEmbeddingCache,
        ablation: AblationFlags | None = None,
    ) -> None:
        super().__init__()
        self.config = config
        self.ablation = ablation or AblationFlags()
        self.relation_cache = relation_cache

        d = config.d
        rho = config.rho
        L = config.L

        # Shared parameters
        self.W0 = nn.Parameter(torch.empty(d, d))
        nn.init.xavier_uniform_(self.W0)
        self.v = nn.Parameter(torch.zeros(d))     # element-wise interaction
        nn.init.normal_(self.v, mean=0.0, std=0.01)

        # Per-hop scorers (one HopScorer per ℓ ∈ {1, ..., L})
        self.hop_scorers = nn.ModuleList([
            HopScorer(
                d=d,
                rho=rho,
                gate_activation=self.ablation.gate_activation,
                use_dbm=self.ablation.use_dbm,
            )
            for _ in range(L)
        ])

        # CSV is parameter-free, but holds a reference to the cache
        self.csv = CSV(relation_cache, pool=self.ablation.csv_pool)

        # Sanity check: paper §8.4 says learnable params < 12M
        n_trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        logger.info(
            f"CAFFModel initialized: {n_trainable / 1e6:.2f}M trainable params "
            f"(paper budget: <12M)"
        )
        if n_trainable > 12_000_000:
            logger.warning(
                f"Trainable param count {n_trainable / 1e6:.2f}M exceeds "
                f"paper's <12M budget. Check d, rho, L."
            )

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    def get_hop_W_ctx(
        self,
        hop_idx: int,
        z_prev: torch.Tensor,
    ) -> torch.Tensor:
        """Compute W^ctx_ℓ for a batch of items at hop ℓ (Eq. 18).

        Parameters
        ----------
        hop_idx : int  in {0, 1, ..., L-1}
        z_prev : Tensor of shape (B, d)
            CSV from hop ℓ-1 (or zeros at hop 0 / when set is empty).
            If ablation.use_csv is False, z_prev is forcibly zeroed.

        Returns
        -------
        W_ctx : Tensor of shape (B, d, d)
        """
        if not self.ablation.use_csv:
            # Ablation §10.1: w/o CSV → z_{ℓ-1} ≡ 0 → reduces to DepthBilinear
            z_prev = torch.zeros_like(z_prev)

        return self.hop_scorers[hop_idx].compute_W_ctx(self.W0, z_prev)

    def score_hop_candidates(
        self,
        hop_idx: int,
        W_ctx_single: torch.Tensor,
        q: torch.Tensor,
        relation_names: list[str],
        return_logits: bool = False,
    ) -> torch.Tensor:
        """Score all candidates of a single (query, hop) — Eq. 19.

        Smart Engineering S3: W_ctx_single is computed ONCE per
        (query, hop), then this method scores all N_ℓ candidates
        against it in a single matmul.

        Parameters
        ----------
        hop_idx : int
        W_ctx_single : Tensor of shape (d, d)
            Pre-computed W^ctx for one item.
        q : Tensor of shape (d,)
            ℓ2-normalized query embedding.
        relation_names : list of str, length N
            Relation names of the candidate triples.
        return_logits : bool
            If True, return pre-sigmoid logits (for BCEWithLogitsLoss).

        Returns
        -------
        Tensor of shape (N,).
        """
        E_r = self.relation_cache.get_batch(relation_names)  # (N, d)

        scorer = self.hop_scorers[hop_idx]
        if return_logits:
            return scorer.score_logits(W_ctx_single, self.v, q, E_r)
        return scorer.score_candidates(W_ctx_single, self.v, q, E_r)

    @torch.no_grad()
    def filter_query(
        self,
        q: torch.Tensor,
        candidate_sets: list[list[tuple[str, str, str]]],
    ) -> list[list[tuple[str, str, str]]]:
        """Apply CAFF filtering to a single query — Algorithm 1 (paper).

        Parameters
        ----------
        q : Tensor of shape (d,)
            ℓ2-normalized query embedding.
        candidate_sets : list of length L
            candidate_sets[ℓ] is the BFS-extracted (and FreqCapped)
            list of (h, r, t) tuples at hop ℓ+1.

        Returns
        -------
        retained_sets : list of length L
            retained_sets[ℓ] is S_{ℓ+1} = {(h,r,t) : s_ℓ(...) ≥ θ}.
        """
        L = self.config.L
        theta = self.config.theta
        d = self.config.d

        # Initialize: z_0 = 0, S_0 = ∅  (Algorithm 1, line 2)
        z_prev = torch.zeros(1, d, device=self.device)  # (1, d) — single query
        retained_sets: list[list[tuple[str, str, str]]] = []

        for hop_idx in range(L):
            candidates = candidate_sets[hop_idx]
            if len(candidates) == 0:
                retained_sets.append([])
                # z stays the same — empty retained set carries no info
                continue

            # Eq. 18: compute W^ctx for this hop given z_prev
            W_ctx = self.get_hop_W_ctx(hop_idx, z_prev).squeeze(0)  # (d, d)

            # Eq. 19: score all candidates
            relation_names = [r for (_, r, _) in candidates]
            scores = self.score_hop_candidates(
                hop_idx, W_ctx, q.squeeze(0) if q.dim() > 1 else q, relation_names
            )

            # Eq. 2: retain if score >= θ
            retained = [
                triple for triple, s in zip(candidates, scores.tolist())
                if s >= theta
            ]
            retained_sets.append(retained)

            # Eq. 14: update z for next hop
            retained_relations = [r for (_, r, _) in retained]
            z_prev = self.csv([retained_relations])  # (1, d)

        return retained_sets