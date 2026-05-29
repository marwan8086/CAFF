#!/usr/bin/env python
"""
per_relation_f1.py -- Per-relation F1 breakdown on the test set.

Loads a trained checkpoint, scores the test set, then aggregates per
relation type. Reports precision, recall, F1, and support (number of
positive and negative instances) per relation. This localizes which
relations CAFF handles well and which remain difficult.

Usage
-----
    python scripts/per_relation_f1.py \
        --checkpoint runs/no_dc/seed_42/best.pt \
        --threshold 0.80 \
        --mode autoregressive \
        --output-json results/per_relation_seed42.json

Output
------
- JSON with per-relation metrics
- Pretty table printed to stdout, sorted by support (most common first)
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

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
from caff.evaluator import precision_recall_f1
from caff.utils.logging import setup_logging

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Per-relation F1 breakdown.")
    p.add_argument("--checkpoint", required=True, help="Path to .pt checkpoint.")
    p.add_argument("--test-split", default=None,
                   help="Test JSON; defaults to config.test_path.")
    p.add_argument("--cache-dir", default="cache")
    p.add_argument("--mode", default="autoregressive",
                   choices=["teacher_forced", "autoregressive"])
    p.add_argument("--threshold", type=float, default=None,
                   help="Retention threshold theta (default: config.theta).")
    p.add_argument("--output-json", default=None,
                   help="Write per-relation metrics to this JSON file.")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


def load_checkpoint(
    ckpt_path: str,
    device: str,
    cache_dir: Path,
) -> tuple[CAFFModel, CAFFConfig, FrozenBioEncoder, KnowledgeGraph]:
    """Restore model + config + KG + encoder from a saved checkpoint."""
    payload = torch.load(ckpt_path, map_location=device)
    config = CAFFConfig(**payload["config"])
    ablation = AblationFlags()
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
    return model, config, encoder, kg


def main() -> None:
    args = parse_args()
    setup_logging(level="INFO")
    cache_dir = Path(args.cache_dir)

    # Load checkpoint
    model, config, encoder, kg = load_checkpoint(args.checkpoint, args.device, cache_dir)

    # Test dataset
    test_path = args.test_split or config.test_path
    test_recs = load_qa_split(test_path)
    bfs = CachedBFSExtractor(kg, L=config.L, K_r=config.K_r,
                             cache_dir=cache_dir / "bfs")
    test_ds = CAFFTripleDataset(test_recs, bfs, require_gold=True)

    # Score the test set
    threshold = args.threshold if args.threshold is not None else config.theta
    evaluator = CAFFEvaluator(
        config=config, encoder=encoder, mode=args.mode, threshold=threshold,
    )
    logger.info(f"Scoring test set (mode={args.mode}, theta={threshold})...")
    scores, instances, _retained = evaluator._score_dataset(model, test_ds)
    scores_np = scores.cpu().numpy() if torch.is_tensor(scores) else np.asarray(scores)

    # Aggregate per relation
    by_rel_scores: dict[str, list[float]] = defaultdict(list)
    by_rel_labels: dict[str, list[int]] = defaultdict(list)
    for inst, sc in zip(instances, scores_np.tolist()):
        rel = inst.relation
        by_rel_scores[rel].append(sc)
        by_rel_labels[rel].append(inst.label)

    # Compute per-relation metrics
    rows = []
    for rel in sorted(by_rel_scores.keys()):
        s = np.asarray(by_rel_scores[rel])
        l = np.asarray(by_rel_labels[rel])
        n_total = len(l)
        n_pos = int(l.sum())
        n_neg = n_total - n_pos
        pos_rate = n_pos / n_total if n_total > 0 else 0.0
        metrics = precision_recall_f1(s, l, threshold=threshold)
        rows.append({
            "relation": rel,
            "n_total": n_total,
            "n_pos": n_pos,
            "n_neg": n_neg,
            "pos_rate": pos_rate,
            "precision": metrics["precision"],
            "recall": metrics["recall"],
            "f1": metrics["f1"],
        })

    # Sort by support (descending)
    rows.sort(key=lambda r: -r["n_total"])

    # Print table
    print()
    print("=" * 108)
    print(f"Per-relation F1 breakdown  (mode={args.mode}, theta={threshold})")
    print(f"Checkpoint: {args.checkpoint}")
    print("=" * 108)
    print(f"{'relation':<55} | {'n_total':>8} | {'n_pos':>6} | {'pos%':>6} | "
          f"{'prec':>6} | {'recall':>6} | {'F1':>6}")
    print("-" * 108)
    for row in rows:
        rel_short = row["relation"][:55]
        print(f"{rel_short:<55} | {row['n_total']:>8} | {row['n_pos']:>6} | "
              f"{row['pos_rate']*100:>5.1f}% | "
              f"{row['precision']:>6.4f} | {row['recall']:>6.4f} | {row['f1']:>6.4f}")
    print("=" * 108)

    # Overall sanity check
    all_scores = np.concatenate([np.asarray(by_rel_scores[r]) for r in by_rel_scores])
    all_labels = np.concatenate([np.asarray(by_rel_labels[r]) for r in by_rel_labels])
    overall = precision_recall_f1(all_scores, all_labels, threshold=threshold)
    print(f"{'OVERALL':<55} | {len(all_labels):>8} | {int(all_labels.sum()):>6} | "
          f"{all_labels.mean()*100:>5.1f}% | "
          f"{overall['precision']:>6.4f} | {overall['recall']:>6.4f} | {overall['f1']:>6.4f}")
    print()

    # Save JSON
    if args.output_json:
        out = {
            "checkpoint": str(args.checkpoint),
            "mode": args.mode,
            "threshold": threshold,
            "overall": overall,
            "per_relation": rows,
        }
        out_path = Path(args.output_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w", encoding="utf-8") as f:
            json.dump(out, f, indent=2)
        logger.info(f"Per-relation metrics written to {out_path}")


if __name__ == "__main__":
    main()
