#!/usr/bin/env python
"""
evaluate.py — Standalone evaluation entry point.

Loads a saved checkpoint and runs the full evaluation suite on a
test split (paper §9, §10, §12.4).

Usage
-----
    python evaluate.py \\
        --checkpoint runs/caff_full/seed_42/best.pt \\
        --test-split data/processed/test.json \\
        --report-bootstrap-vs runs/depthbilinear/seed_42/best.pt \\
        --output-json results/test_metrics_seed_42.json
"""

from __future__ import annotations

import argparse
import json
import logging
from collections import defaultdict
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

from caff import (
    AblationFlags,
    CAFFConfig,
    CAFFEvaluator,
    CAFFModel,
    CAFFTripleDataset,
    CachedBFSExtractor,
    FrozenBioEncoder,
    KnowledgeGraph,
    RelationEmbeddingCache,
    load_qa_split,
)
from caff.evaluator import paired_bootstrap, average_precision
from caff.utils.logging import setup_logging

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate a CAFF checkpoint.")
    p.add_argument("--checkpoint", required=True, help="Path to .pt checkpoint.")
    p.add_argument("--test-split", default=None,
                   help="Test JSON; defaults to config.test_path.")
    p.add_argument("--cache-dir", default="cache")
    p.add_argument("--mode", default="autoregressive",
                   choices=["teacher_forced", "autoregressive"])
    p.add_argument("--threshold", type=float, default=None,
                   help="Override retention threshold θ.")
    p.add_argument("--report-bootstrap-vs", default=None,
                   help="Path to baseline checkpoint for paired bootstrap.")
    p.add_argument("--output-json", default=None,
                   help="Write metrics to this JSON file.")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


def load_checkpoint(
    ckpt_path: str | Path,
    device: str,
    cache_dir: Path,
) -> tuple[CAFFModel, CAFFConfig, AblationFlags, FrozenBioEncoder, KnowledgeGraph]:
    """Restore model + config + KG + encoder from a saved checkpoint."""
    payload = torch.load(ckpt_path, map_location=device)
    config = CAFFConfig(**payload["config"])
    ablation = AblationFlags()  # ablation isn't saved in checkpoint —
                                # caller sets it externally if needed

    logger.info(f"Loading KG from {config.kg_path}...")
    kg = KnowledgeGraph.from_tsv(config.kg_path, min_relation_freq=50)

    encoder = FrozenBioEncoder(config.encoder_name, device=device)
    rel_cache = RelationEmbeddingCache(
        encoder, kg.relations,
        cache_path=cache_dir / "relation_embeddings.pt",
    )

    model = CAFFModel(config, rel_cache, ablation=ablation).to(device)
    model.load_state_dict(payload["model"])
    model.eval()

    logger.info(f"Restored checkpoint from {ckpt_path}")
    return model, config, ablation, encoder, kg


def per_query_average_precision(
    model: CAFFModel,
    dataset: CAFFTripleDataset,
    evaluator: CAFFEvaluator,
) -> dict[str, float]:
    """Per-query AP for paired bootstrap testing."""
    scores, instances, _retained = evaluator._score_dataset(model, dataset)
    by_query: dict[str, list[tuple[float, int]]] = defaultdict(list)
    for inst, sc in zip(instances, scores.tolist()):
        by_query[inst.query_id].append((sc, inst.label))
    aps: dict[str, float] = {}
    for qid, items in by_query.items():
        s = np.array([x[0] for x in items])
        l = np.array([x[1] for x in items])
        if l.sum() == 0:
            continue
        aps[qid] = average_precision(s, l)
    return aps


def main() -> None:
    args = parse_args()
    setup_logging(level="INFO")
    cache_dir = Path(args.cache_dir)

    # ─── Load primary checkpoint ────────────────────────────────
    model, config, ablation, encoder, kg = load_checkpoint(
        args.checkpoint, args.device, cache_dir
    )

    # ─── Test dataset ───────────────────────────────────────────
    test_path = args.test_split or config.test_path
    test_recs = load_qa_split(test_path)
    bfs = CachedBFSExtractor(kg, L=config.L, K_r=config.K_r,
                             cache_dir=cache_dir / "bfs")
    test_ds = CAFFTripleDataset(test_recs, bfs, require_gold=True)

    # ─── Primary evaluation ─────────────────────────────────────
    evaluator = CAFFEvaluator(
        config=config,
        encoder=encoder,
        mode=args.mode,
        threshold=args.threshold or config.theta,
    )
    metrics = evaluator.evaluate(model, test_ds)

    logger.info("─" * 60)
    logger.info(f"Test metrics  (mode={args.mode}, θ={evaluator.threshold})")
    logger.info("─" * 60)
    for k, v in metrics.items():
        if isinstance(v, float):
            logger.info(f"  {k:14s} = {v:.4f}")
        else:
            logger.info(f"  {k:14s} = {v}")

    # ─── Paired bootstrap vs. baseline ──────────────────────────
    bootstrap_result = None
    if args.report_bootstrap_vs is not None:
        logger.info(f"\nLoading baseline: {args.report_bootstrap_vs}")
        baseline_model, _, _, _, _ = load_checkpoint(
            args.report_bootstrap_vs, args.device, cache_dir
        )
        baseline_eval = CAFFEvaluator(
            config=config, encoder=encoder, mode=args.mode,
            threshold=args.threshold or config.theta,
        )
        ap_a = per_query_average_precision(model,          test_ds, evaluator)
        ap_b = per_query_average_precision(baseline_model, test_ds, baseline_eval)
        common = sorted(set(ap_a) & set(ap_b))
        bootstrap_result = paired_bootstrap(
            [ap_a[q] for q in common],
            [ap_b[q] for q in common],
            n_resamples=10_000,
            seed=config.seed,
        )
        logger.info("─" * 60)
        logger.info(f"Paired bootstrap (CAFF vs baseline, B=10,000)")
        logger.info("─" * 60)
        logger.info(f"  Δ_AP (mean)   = {bootstrap_result['delta_mean']:+.4f}")
        logger.info(f"  95% CI        = [{bootstrap_result['ci_lo_95']:+.4f}, "
                    f"{bootstrap_result['ci_hi_95']:+.4f}]")
        logger.info(f"  p-value       = {bootstrap_result['p_value']:.4f}")
        if bootstrap_result['p_value'] < 0.01:
            logger.info("  → Significant at p < 0.01 (paper §8.4 threshold)")

    # ─── Persist ────────────────────────────────────────────────
    if args.output_json:
        out = {
            "metrics": metrics,
            "bootstrap": bootstrap_result,
            "checkpoint": str(args.checkpoint),
            "test_split": str(test_path),
            "mode": args.mode,
            "threshold": evaluator.threshold,
        }
        out_path = Path(args.output_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w", encoding="utf-8") as f:
            json.dump(out, f, indent=2)
        logger.info(f"\nMetrics written to {out_path}")


if __name__ == "__main__":
    main()