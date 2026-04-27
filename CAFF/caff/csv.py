"""
caff/csv.py
===========
Contextual Summary Vector (CSV) — paper §6.2, Equation 14.

The CSV is the parameter-free, permutation-invariant encoding of
the previously retained set S_{ℓ-1} that breaks the Context
Blindness Error (Theorem 1).

Equation 14 (verbatim from paper):

    z_{ℓ-1} = ┌─ (1/|S_{ℓ-1}|) Σ_{(h,r,t) ∈ S_{ℓ-1}} e_r,  if S_{ℓ-1} ≠ ∅
              └─ 0_d                                       otherwise

Properties (paper §6.2):
  • Parameter-free (no trainable weights)
  • Permutation-invariant (mean is symmetric)
  • Linear projection of the RRC: z = E^T · Z   (Property 1)
  • Injective on the simplex when rank(E) = |R|  (Lemma 2)
  • Noise-stable: E[||z̃ - z||²] = σ²d / |S_{ℓ-1}|  (Proposition 2)
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .encoders import RelationEmbeddingCache


class CSV(nn.Module):
    """Contextual Summary Vector encoder (Eq. 14).

    Stateless. Looks up frozen relation embeddings from the cache
    and computes the mean (or max-pool, for ablation §10.1) over
    the retained set.

    Parameters
    ----------
    relation_cache : RelationEmbeddingCache
        Pre-encoded frozen relation embeddings, |R| × d.
    pool : {"mean", "max"}
        Pooling strategy. Paper default is "mean" (Eq. 14).
        Ablation §10.1 reports max-pool yields -0.5 acc.
    """

    def __init__(
        self,
        relation_cache: RelationEmbeddingCache,
        pool: str = "mean",
    ) -> None:
        super().__init__()
        assert pool in {"mean", "max"}, f"pool must be mean or max, got {pool}"
        self.relation_cache = relation_cache
        self.pool = pool
        self.d = relation_cache.embeddings.shape[1]

    @property
    def device(self) -> torch.device:
        return self.relation_cache.embeddings.device

    def forward(self, retained_relations: list[list[str]]) -> torch.Tensor:
        """Compute CSV for a batch of retained sets.

        Parameters
        ----------
        retained_relations : list of list of str
            For each item in the batch, the list of relation names
            in S_{ℓ-1}. Empty inner list = empty retained set →
            CSV returns the zero vector (Eq. 14, second case).

        Returns
        -------
        Tensor of shape (B, d).
        """
        batch_size = len(retained_relations)
        z = torch.zeros(batch_size, self.d, device=self.device)

        for b, relations in enumerate(retained_relations):
            if len(relations) == 0:
                # Eq. 14, empty case: z = 0_d
                continue
            embeds = self.relation_cache.get_batch(relations)  # (|S|, d)
            if self.pool == "mean":
                z[b] = embeds.mean(dim=0)
            elif self.pool == "max":
                z[b] = embeds.max(dim=0).values

        return z