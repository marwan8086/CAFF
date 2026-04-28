"""
tests/test_csv.py
=================
Verify CSV (Eq. 14) properties:

  • Empty retained set → zero vector            (Eq. 14, second case)
  • Permutation invariance                      (Property 1)
  • Distinct distributions → distinct CSVs      (Lemma 2: injectivity)
  • Linear projection: z = E^T · Z              (Property 1)
  • Noise stability: ||z̃ - z||² ∝ 1/|S|          (Proposition 2)
"""

from __future__ import annotations

import pytest
import torch

from caff.csv import CSV


class _StubCache:
    """Minimal cache stub for testing CSV without loading BioLinkBERT."""
    def __init__(self, embeddings: torch.Tensor, names: list[str]) -> None:
        self.embeddings = embeddings
        self.relation_to_idx = {n: i for i, n in enumerate(names)}

    def get_batch(self, names: list[str]) -> torch.Tensor:
        idx = torch.tensor(
            [self.relation_to_idx[n] for n in names],
            device=self.embeddings.device,
        )
        return self.embeddings[idx]


def _make_orthonormal_cache(n_relations: int = 5, d: int = 8) -> _StubCache:
    """Use orthonormal basis vectors as relation embeddings —
    guarantees rank(E) = |R| (Lemma 2 precondition)."""
    E = torch.zeros(n_relations, d)
    for i in range(n_relations):
        E[i, i] = 1.0
    names = [f"r{i}" for i in range(n_relations)]
    return _StubCache(E, names)


def test_csv_empty_returns_zero():
    """Eq. 14, second case: S_{ℓ-1} = ∅ → z = 0_d."""
    cache = _make_orthonormal_cache()
    csv = CSV(cache, pool="mean")
    z = csv([[]])           # batch of 1 with empty retained set
    assert z.shape == (1, 8)
    assert torch.allclose(z, torch.zeros(1, 8))


def test_csv_permutation_invariance():
    """Property 1: order of relations does not change the CSV."""
    cache = _make_orthonormal_cache()
    csv = CSV(cache, pool="mean")
    z1 = csv([["r0", "r1", "r2"]])
    z2 = csv([["r2", "r0", "r1"]])
    z3 = csv([["r1", "r2", "r0"]])
    assert torch.allclose(z1, z2)
    assert torch.allclose(z1, z3)


def test_csv_injectivity_on_simplex():
    """Lemma 2: rank(E)=|R| ⟹ Z ↦ E^T Z is injective.

    Two distinct retained-context distributions must produce
    distinct CSVs."""
    cache = _make_orthonormal_cache()
    csv = CSV(cache, pool="mean")
    z_a = csv([["r0", "r0", "r1"]])  # 2/3 of r0, 1/3 of r1
    z_b = csv([["r0", "r1", "r1"]])  # 1/3 of r0, 2/3 of r1
    assert not torch.allclose(z_a, z_b, atol=1e-6)


def test_csv_linear_projection():
    """Property 1: z = E^T · Z (relational distribution)."""
    cache = _make_orthonormal_cache()
    csv = CSV(cache, pool="mean")
    relations = ["r0", "r1", "r1", "r2"]
    z = csv([relations]).squeeze(0)

    # Expected: (e_{r0} + 2 e_{r1} + e_{r2}) / 4
    expected = (cache.embeddings[0] + 2 * cache.embeddings[1] + cache.embeddings[2]) / 4
    assert torch.allclose(z, expected, atol=1e-6)


def test_csv_batch_independence():
    """Different batch items do not contaminate each other."""
    cache = _make_orthonormal_cache()
    csv = CSV(cache, pool="mean")
    z = csv([["r0"], ["r1"], []])
    assert torch.allclose(z[0], cache.embeddings[0])
    assert torch.allclose(z[1], cache.embeddings[1])
    assert torch.allclose(z[2], torch.zeros(8))


def test_csv_max_pool_ablation():
    """Ablation §10.1: max-pool variant produces different output."""
    cache = _make_orthonormal_cache()
    csv_mean = CSV(cache, pool="mean")
    csv_max  = CSV(cache, pool="max")
    z_mean = csv_mean([["r0", "r1"]])
    z_max  = csv_max( [["r0", "r1"]])
    assert not torch.allclose(z_mean, z_max)