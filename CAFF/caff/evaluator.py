"""
caff/evaluator.py
=================
Evaluation suite for CAFF — paper §8.3, §9, §10.4.

Implements:
  • Triple-level: Precision, Recall, F1, MAP, NDCG@10  (paper Table 4)
  • Hop-stratified precision (paper Table 5)
  • Path Survival Rate, PSR (paper Table 11)
  • Context-swap diagnostic, JSD in bits (paper Table 10, App. C)
  • Paired bootstrap significance test, B=10,000 (paper §8.4)

Conventions
-----------
All triple-level metrics are computed at threshold θ=0.50 unless
otherwise noted. NDCG@10 uses the standard graded-relevance form
(binary labels degenerate to NDCG = DCG@10 / IDCG@10).
"""

from __future__ import annotations

import logging
import math
from collections import defaultdict
from dataclasses import dataclass

import numpy as np
import torch

from .config import CAFFConfig
from .data import CAFFTripleDataset, TripleInstance
from .encoders import FrozenBioEncoder
from .model import CAFFModel
from .csv import CSV

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────
# Triple-level metrics
# ─────────────────────────────────────────────────────────────────


def precision_recall_f1(
    scores: np.ndarray,
    labels: np.ndarray,
    threshold: float = 0.5,
) -> dict[str, float]:
    """Standard P/R/F1 at a fixed threshold."""
    preds = (scores >= threshold).astype(int)
    tp = int(((preds == 1) & (labels == 1)).sum())
    fp = int(((preds == 1) & (labels == 0)).sum())
    fn = int(((preds == 0) & (labels == 1)).sum())
    p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
    return {"precision": p, "recall": r, "f1": f}


def average_precision(scores: np.ndarray, labels: np.ndarray) -> float:
    """Standard AP (area under the PR curve, exact)."""
    if labels.sum() == 0:
        return 0.0
    order = np.argsort(-scores)
    sorted_labels = labels[order]
    cum_tp = np.cumsum(sorted_labels)
    precision_at_k = cum_tp / (np.arange(len(sorted_labels)) + 1)
    return float((precision_at_k * sorted_labels).sum() / sorted_labels.sum())


def mean_average_precision(per_query_scores: dict[str, tuple[np.ndarray, np.ndarray]]) -> float:
    """MAP averaged over queries."""
    aps = []
    for qid, (scores, labels) in per_query_scores.items():
        if len(scores) == 0:
            continue
        aps.append(average_precision(scores, labels))
    return float(np.mean(aps)) if aps else 0.0


def ndcg_at_k(scores: np.ndarray, labels: np.ndarray, k: int = 10) -> float:
    """NDCG@k with binary labels."""
    if labels.sum() == 0:
        return 0.0
    order = np.argsort(-scores)[:k]
    dcg = sum(
        labels[idx] / math.log2(rank + 2)
        for rank, idx in enumerate(order)
    )
    ideal_order = np.argsort(-labels)[:k]
    idcg = sum(
        labels[idx] / math.log2(rank + 2)
        for rank, idx in enumerate(ideal_order)
    )
    return dcg / idcg if idcg > 0 else 0.0


def mean_ndcg_at_k(
    per_query_scores: dict[str, tuple[np.ndarray, np.ndarray]],
    k: int = 10,
) -> float:
    vs = []
    for qid, (s, l) in per_query_scores.items():
        if len(s) == 0:
            continue
        vs.append(ndcg_at_k(s, l, k=k))
    return float(np.mean(vs)) if vs else 0.0


# ─────────────────────────────────────────────────────────────────
# Hop-stratified precision (paper Table 5)
# ─────────────────────────────────────────────────────────────────


def hop_stratified_precision(
    instances: list[TripleInstance],
    scores: np.ndarray,
    threshold: float = 0.5,
) -> dict[int, float]:
    """Precision computed separately per hop ℓ ∈ {1, 2, 3}."""
    by_hop: dict[int, list[tuple[float, int]]] = defaultdict(list)
    for inst, s in zip(instances, scores.tolist()):
        by_hop[inst.hop].append((s, inst.label))

    out: dict[int, float] = {}
    for hop, pairs in by_hop.items():
        s_arr = np.array([p[0] for p in pairs])
        l_arr = np.array([p[1] for p in pairs])
        out[hop] = precision_recall_f1(s_arr, l_arr, threshold)["precision"]
    return out


# ─────────────────────────────────────────────────────────────────
# Path Survival Rate (paper §12.4, Table 11)
# ─────────────────────────────────────────────────────────────────


def path_survival_rate(
    gold_paths: dict[str, list[list[tuple[str, str, str]]]],
    retained_per_query: dict[str, set[tuple[str, str, str]]],
) -> float:
    """Fraction of gold reasoning paths fully preserved by the filter.

        PSR(Q) = (1/|P(Q)|) Σ_{p ∈ P(Q)}  𝟙[p ⊆ ⋃_ℓ S_ℓ]
        PSR    = mean over queries

    Parameters
    ----------
    gold_paths : dict
        query_id -> list of paths; each path = list of (h, r, t).
    retained_per_query : dict
        query_id -> set of retained (h, r, t) tuples (across all hops).

    Returns
    -------
    PSR ∈ [0, 1].
    """
    psrs = []
    for qid, paths in gold_paths.items():
        if not paths:
            continue
        retained = retained_per_query.get(qid, set())
        survived = sum(
            1 for path in paths
            if all(edge in retained for edge in path)
        )
        psrs.append(survived / len(paths))
    return float(np.mean(psrs)) if psrs else 0.0


# ─────────────────────────────────────────────────────────────────
# Paired bootstrap significance (paper §8.4)
# ─────────────────────────────────────────────────────────────────


def paired_bootstrap(
    metric_a_per_query: list[float],
    metric_b_per_query: list[float],
    n_resamples: int = 10_000,
    seed: int = 42,
) -> dict[str, float]:
    """Paired bootstrap test for H0: metric_a == metric_b.

    Paper §8.4 uses B=10,000 resamples and reports p<0.01 for all
    main results. A two-sided test is reported.

    Returns
    -------
    Dict with mean delta, 95% CI, and two-sided p-value.
    """
    rng = np.random.default_rng(seed)
    a = np.asarray(metric_a_per_query, dtype=float)
    b = np.asarray(metric_b_per_query, dtype=float)
    assert len(a) == len(b), "Paired test requires same query set"
    n = len(a)
    delta_obs = (a - b).mean()

    deltas = np.empty(n_resamples, dtype=float)
    for i in range(n_resamples):
        idx = rng.integers(0, n, size=n)
        deltas[i] = (a[idx] - b[idx]).mean()

    p = float(np.mean(np.abs(deltas - delta_obs) >= abs(delta_obs)))
    return {
        "delta_mean": float(delta_obs),
        "ci_lo_95":   float(np.quantile(deltas, 0.025)),
        "ci_hi_95":   float(np.quantile(deltas, 0.975)),
        "p_value":    p,
    }


# ─────────────────────────────────────────────────────────────────
# Context-swap JSD diagnostic (paper §10.4, Appendix C)
# ─────────────────────────────────────────────────────────────────


def jensen_shannon_bits(p: float, q: float) -> float:
    """JSD between Bernoulli(p) and Bernoulli(q), in bits.

    Paper Appendix C reports 2·JSD as a symmetric divergence measure.
    Pure JSD is bounded above by 1 bit; 2·JSD can reach 2 bits.
    """
    def _kl(a: float, b: float) -> float:
        if a in (0.0, 1.0):
            return 0.0
        return a * math.log2(a / b) + (1 - a) * math.log2((1 - a) / (1 - b))
    m = 0.5 * (p + q)
    if m in (0.0, 1.0):
        return 0.0
    jsd = 0.5 * _kl(p, m) + 0.5 * _kl(q, m)
    return jsd  # in [0, 1] bits


def context_swap_diagnostic(
    model: CAFFModel,
    encoder: FrozenBioEncoder,
    cfg: CAFFConfig,
) -> dict[str, float]:
    """Synthetic CBE-elimination test (paper Appendix C).

    Setup (verbatim paper):
      • Synthetic 9-entity, 11-triple KG, |R|=5
      • Relation embeddings = orthonormal basis vectors in ℝ^d
      • Two contexts:
          A : S_1 = {⟨Disease, r1, Gene⟩},      z^(A) ≈ e_{r1}, y=1
          B : S_1 = {⟨Disease, r2, Phenotype⟩}, z^(B) ≈ e_{r2}, y=0
      • Target hop-2 triple: ⟨Gene, r3, Pathway⟩
      • Score s^(A), s^(B) under each context, report 2·JSD.

    Expected (paper Table 10):
      • Any context-AGNOSTIC filter → JSD = 0.00 bits  (CBE)
      • CAFF-NoHC3                  → JSD ≈ 1.41 bits
      • CAFF (Full)                 → JSD ≈ 1.84 bits

    Parameters
    ----------
    model : CAFFModel  — must already be trained
    encoder : FrozenBioEncoder
    cfg : CAFFConfig

    Returns
    -------
    {
      "s_A": float,      # score under context A
      "s_B": float,      # score under context B
      "jsd_bits": float, # 2 * JSD(Bern(s_A) || Bern(s_B))
    }
    """
    device = model.device
    d = cfg.d

    # Orthonormal relation embeddings (paper App. C)
    e = torch.zeros(5, d, device=device)
    for i in range(5):
        e[i, i] = 1.0
    # We bypass the relation cache here because this is a synthetic
    # diagnostic — relations don't correspond to real BioLinkBERT
    # surface forms.

    # Synthetic query embedding (also unit norm, orthogonal to e)
    q = torch.zeros(d, device=device)
    q[5] = 1.0

    # Build z^(A) = e_{r1} and z^(B) = e_{r2}
    z_A = e[0].unsqueeze(0)  # (1, d)
    z_B = e[1].unsqueeze(0)

    # Score the same hop-2 candidate (relation r3 = e[2]) under both
    hop_idx = 1  # hop ℓ=2 → 0-indexed = 1
    with torch.no_grad():
        W_ctx_A = model.get_hop_W_ctx(hop_idx, z_A).squeeze(0)
        W_ctx_B = model.get_hop_W_ctx(hop_idx, z_B).squeeze(0)
        scorer = model.hop_scorers[hop_idx]
        s_A = scorer.score_candidates(W_ctx_A, model.v, q, e[2:3]).item()
        s_B = scorer.score_candidates(W_ctx_B, model.v, q, e[2:3]).item()

    jsd_pure = jensen_shannon_bits(s_A, s_B)
    jsd_doubled = 2.0 * jsd_pure  # paper convention (App. C)

    return {"s_A": s_A, "s_B": s_B, "jsd_bits": jsd_doubled}


# ─────────────────────────────────────────────────────────────────
# Main evaluator
# ─────────────────────────────────────────────────────────────────


@dataclass
class EvaluationReport:
    """Full evaluation output suitable for paper Tables 4, 5, 11."""
    precision: float
    recall: float
    f1: float
    map: float
    ndcg_at_10: float
    hop_precision: dict[int, float]
    psr: float | None
    n_instances: int
    threshold: float


class CAFFEvaluator:
    """Run filtering inference + compute all metrics on a dataset.

    Two evaluation modes:
      • 'teacher_forced'  : z_{ℓ-1} computed from gold-positive
                            relations of hop ℓ-1. Used during
                            training-time validation; isolates
                            the scorer from upstream errors.
      • 'autoregressive'  : z_{ℓ-1} computed from MODEL-retained
                            set at hop ℓ-1. Used at final test
                            time — this is the deployment regime.
    """

    def __init__(
        self,
        config: CAFFConfig,
        encoder: FrozenBioEncoder,
        mode: str = "teacher_forced",
        threshold: float | None = None,
        gold_paths: dict[str, list[list[tuple[str, str, str]]]] | None = None,
    ) -> None:
        assert mode in {"teacher_forced", "autoregressive"}
        self.config = config
        self.encoder = encoder
        self.mode = mode
        self.threshold = threshold if threshold is not None else config.theta
        self.gold_paths = gold_paths
        self._q_cache: dict[str, torch.Tensor] = {}

    def _q_emb(self, query_id: str, question: str, device: torch.device) -> torch.Tensor:
        if query_id not in self._q_cache:
            with torch.no_grad():
                self._q_cache[query_id] = self.encoder.encode([question])[0].to(device)
        return self._q_cache[query_id]

    @torch.no_grad()
    def _score_dataset(
        self,
        model: CAFFModel,
        dataset: CAFFTripleDataset,
    ) -> tuple[np.ndarray, list[TripleInstance], dict[str, set]]:
        """Score every instance in the dataset, return aligned arrays.

        Also returns per-query retained sets for PSR computation.
        """
        model.eval()
        device = model.device
        scores_out: list[float] = []
        instances_out: list[TripleInstance] = []
        retained_per_query: dict[str, set[tuple[str, str, str]]] = defaultdict(set)

        # Group by query for sequential hop processing
        per_query: dict[str, dict[int, list[TripleInstance]]] = defaultdict(
            lambda: defaultdict(list)
        )
        question_for_query: dict[str, str] = {}
        for inst in dataset.instances:
            per_query[inst.query_id][inst.hop].append(inst)
            question_for_query.setdefault(inst.query_id, inst.query_id)

        # Need question text — pull from original QA records
        for rec in dataset.qa_records:
            question_for_query[rec.query_id] = rec.question

        for query_id, hops in per_query.items():
            q_emb = self._q_emb(query_id, question_for_query[query_id], device)
            z_prev = torch.zeros(self.config.d, device=device)

            for hop in sorted(hops.keys()):
                hop_idx = hop - 1
                instances = hops[hop]
                W_ctx = model.get_hop_W_ctx(hop_idx, z_prev.unsqueeze(0)).squeeze(0)
                relation_names = [inst.relation for inst in instances]
                hop_scores = model.score_hop_candidates(
                    hop_idx, W_ctx, q_emb, relation_names,
                ).cpu().numpy()

                for inst, sc in zip(instances, hop_scores):
                    scores_out.append(float(sc))
                    instances_out.append(inst)
                    if sc >= self.threshold:
                        retained_per_query[query_id].add(
                            (inst.head, inst.relation, inst.tail)
                        )

                # Update z_prev for next hop
                if self.mode == "teacher_forced":
                    gold_relations = [i.relation for i in instances if i.label == 1]
                else:  # autoregressive
                    gold_relations = [
                        i.relation
                        for i, sc in zip(instances, hop_scores)
                        if sc >= self.threshold
                    ]
                if gold_relations:
                    z_prev = model.csv([gold_relations]).squeeze(0)
                else:
                    z_prev = torch.zeros(self.config.d, device=device)

        return np.array(scores_out), instances_out, dict(retained_per_query)

    def evaluate(
        self,
        model: CAFFModel,
        dataset: CAFFTripleDataset,
    ) -> dict[str, float]:
        """Compute all triple-level + hop-stratified + PSR metrics.

        Returns flat dict suitable for logging or table export.
        """
        scores, instances, retained = self._score_dataset(model, dataset)
        labels = np.array([i.label for i in instances])

        prf = precision_recall_f1(scores, labels, self.threshold)

        per_query: dict[str, tuple[np.ndarray, np.ndarray]] = defaultdict(
            lambda: (np.array([]), np.array([]))
        )
        # Group scores/labels by query for MAP / NDCG
        q_groups: dict[str, list[tuple[float, int]]] = defaultdict(list)
        for inst, sc, lbl in zip(instances, scores.tolist(), labels.tolist()):
            q_groups[inst.query_id].append((sc, lbl))
        for qid, items in q_groups.items():
            s_arr = np.array([x[0] for x in items])
            l_arr = np.array([x[1] for x in items])
            per_query[qid] = (s_arr, l_arr)

        map_val = mean_average_precision(per_query)
        ndcg_val = mean_ndcg_at_k(per_query, k=10)
        hop_prec = hop_stratified_precision(instances, scores, self.threshold)

        psr_val: float | None = None
        if self.gold_paths is not None:
            psr_val = path_survival_rate(self.gold_paths, retained)

        out = {
            "precision":  prf["precision"],
            "recall":     prf["recall"],
            "f1":         prf["f1"],
            "map":        map_val,
            "ndcg@10":    ndcg_val,
            "hop1_prec":  hop_prec.get(1, 0.0),
            "hop2_prec":  hop_prec.get(2, 0.0),
            "hop3_prec":  hop_prec.get(3, 0.0),
            "n_instances": len(instances),
        }
        if psr_val is not None:
            out["psr"] = psr_val
        return out