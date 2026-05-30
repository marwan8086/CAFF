#!/usr/bin/env python
"""
split_aware_analysis.py -- Stratify test F1 by seed novelty.

Splits the test set into two groups based on whether each query's
seeds were seen during training:

  - seen_seeds:   at least one of the query's seeds appears in train
  - unseen_seeds: none of the query's seeds appear in train

Then computes F1 (and per-hop, per-relation) separately on each group.
If F1_unseen is close to F1_seen, the model generalizes to novel
seed entities; if F1_unseen is much lower, the model is memorizing.

This is the closest "external-like" validation available without
introducing a separate KG or dataset.

Usage:
    python scripts/split_aware_analysis.py \
        --checkpoint runs/no_dc/seed_42/best.pt \
        --train-split data/processed/train.json \
        --test-split data/processed/test.json \
        --threshold 0.80 \
        --mode autoregressive \
        --output-json results/split_aware_seed42.json
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
    p = argparse.ArgumentParser(description="Split-aware F1 stratification.")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--train-split", required=True,
                   help="Train JSON, used to compute the set of seen seeds.")
    p.add_argument("--test-split", default=None)
    p.add_argument("--cache-dir", default="cache")
    p.add_argument("--mode", default="autoregressive",
                   choices=["teacher_forced", "autoregressive"])
    p.add_argument("--threshold", type=float, default=None)
    p.add_argument("--output-json", default=None)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


def load_checkpoint(ckpt_path: str, device: str, cache_dir: Path):
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


def compute_seen_seeds(train_path: str) -> set[str]:
    """Collect the set of all seeds that appear in the training split."""
    train_recs = load_qa_split(train_path)
    seen: set[str] = set()
    for rec in train_recs:
        # rec.seeds is a list of seed strings; rec might be a dict or dataclass.
        if hasattr(rec, "seeds"):
            seeds = rec.seeds
        else:
            seeds = rec.get("seeds", [])
        for s in seeds:
            seen.add(s)
    return seen


def classify_query(seeds: list[str], seen_set: set[str]) -> str:
    """Classify a test query as 'seen' if any of its seeds are in train, else 'unseen'."""
    for s in seeds:
        if s in seen_set:
            return "seen"
    return "unseen"


def main() -> None:
    args = parse_args()
    setup_logging(level="INFO")
    cache_dir = Path(args.cache_dir)

    # Step 1: build the set of seen seeds from train
    logger.info(f"Building seen-seeds set from {args.train_split}...")
    seen_seeds = compute_seen_seeds(args.train_split)
    logger.info(f"  {len(seen_seeds):,} unique training seeds")

    # Step 2: load checkpoint
    model, config, encoder, kg = load_checkpoint(args.checkpoint, args.device, cache_dir)

    # Step 3: load test set and classify each query
    test_path = args.test_split or config.test_path
    test_recs = load_qa_split(test_path)

    # Build query_id -> group mapping
    qid_to_group: dict[str, str] = {}
    n_seen, n_unseen = 0, 0
    for rec in test_recs:
        if hasattr(rec, "query_id"):
            qid = rec.query_id
            seeds = rec.seeds
        else:
            qid = rec.get("query_id")
            seeds = rec.get("seeds", [])
        group = classify_query(seeds, seen_seeds)
        qid_to_group[qid] = group
        if group == "seen":
            n_seen += 1
        else:
            n_unseen += 1
    logger.info(f"  Test queries: {len(test_recs):,}")
    logger.info(f"    seen-seed group:   {n_seen:,} ({n_seen/len(test_recs)*100:.1f}%)")
    logger.info(f"    unseen-seed group: {n_unseen:,} ({n_unseen/len(test_recs)*100:.1f}%)")

    # Step 4: score the test set
    bfs = CachedBFSExtractor(kg, L=config.L, K_r=config.K_r,
                             cache_dir=cache_dir / "bfs")
    test_ds = CAFFTripleDataset(test_recs, bfs, require_gold=True)
    threshold = args.threshold if args.threshold is not None else config.theta
    evaluator = CAFFEvaluator(
        config=config, encoder=encoder, mode=args.mode, threshold=threshold,
    )
    logger.info(f"Scoring test set (mode={args.mode}, theta={threshold})...")
    scores, instances, _retained = evaluator._score_dataset(model, test_ds)
    scores_np = scores.cpu().numpy() if torch.is_tensor(scores) else np.asarray(scores)

    # Step 5: aggregate by group
    by_group_scores: dict[str, list[float]] = defaultdict(list)
    by_group_labels: dict[str, list[int]] = defaultdict(list)
    # Also stratify by (group, hop) for a finer view
    by_group_hop_scores: dict[tuple[str, int], list[float]] = defaultdict(list)
    by_group_hop_labels: dict[tuple[str, int], list[int]] = defaultdict(list)

    for inst, sc in zip(instances, scores_np.tolist()):
        group = qid_to_group.get(inst.query_id, "unknown")
        by_group_scores[group].append(sc)
        by_group_labels[group].append(inst.label)
        by_group_hop_scores[(group, inst.hop)].append(sc)
        by_group_hop_labels[(group, inst.hop)].append(inst.label)

    # Step 6: print results
    print()
    print("=" * 88)
    print(f"Split-aware F1 stratification  (mode={args.mode}, theta={threshold})")
    print(f"Checkpoint: {args.checkpoint}")
    print("=" * 88)
    print(f"Seen-seed group:    {n_seen:>5} queries  (seed appears in train)")
    print(f"Unseen-seed group:  {n_unseen:>5} queries  (no seed appears in train)")
    print()
    print(f"{'group':<10} | {'n_total':>8} | {'n_pos':>6} | {'pos%':>6} | "
          f"{'prec':>6} | {'recall':>6} | {'F1':>6}")
    print("-" * 66)

    group_rows = []
    for group in ["seen", "unseen"]:
        if group not in by_group_scores:
            continue
        s = np.asarray(by_group_scores[group])
        l = np.asarray(by_group_labels[group])
        n_total = len(l)
        n_pos = int(l.sum())
        pos_rate = n_pos / n_total if n_total > 0 else 0.0
        m = precision_recall_f1(s, l, threshold=threshold)
        group_rows.append({
            "group": group,
            "n_total": n_total,
            "n_pos": n_pos,
            "pos_rate": pos_rate,
            "precision": m["precision"],
            "recall": m["recall"],
            "f1": m["f1"],
        })
        print(f"{group:<10} | {n_total:>8} | {n_pos:>6} | {pos_rate*100:>5.1f}% | "
              f"{m['precision']:>6.4f} | {m['recall']:>6.4f} | {m['f1']:>6.4f}")

    # Compute the generalization gap
    if len(group_rows) == 2:
        f1_seen = group_rows[0]["f1"]
        f1_unseen = group_rows[1]["f1"]
        gap = f1_seen - f1_unseen
        rel_gap = (gap / f1_seen * 100) if f1_seen > 0 else 0.0
        print()
        print(f"Generalization gap: F1_seen - F1_unseen = {gap:+.4f} ({rel_gap:+.1f}%)")
        if abs(gap) < 0.02:
            print(f"  ==> Small gap; model generalizes well to novel seeds.")
        elif gap > 0:
            print(f"  ==> Larger F1 on seen seeds; some memorization effect.")
        else:
            print(f"  ==> Larger F1 on unseen; unusual.")

    # Per (group, hop) breakdown
    print()
    print("=" * 88)
    print(f"Per (group, hop) breakdown")
    print("=" * 88)
    print(f"{'group':<10} | {'hop':>4} | {'n_total':>8} | {'n_pos':>6} | "
          f"{'prec':>6} | {'recall':>6} | {'F1':>6}")
    print("-" * 66)
    grouphop_rows = []
    for group in ["seen", "unseen"]:
        for hop in [1, 2, 3]:
            key = (group, hop)
            if key not in by_group_hop_scores:
                continue
            s = np.asarray(by_group_hop_scores[key])
            l = np.asarray(by_group_hop_labels[key])
            n_total = len(l)
            n_pos = int(l.sum())
            if n_total == 0:
                continue
            m = precision_recall_f1(s, l, threshold=threshold)
            grouphop_rows.append({
                "group": group, "hop": hop,
                "n_total": n_total, "n_pos": n_pos,
                "precision": m["precision"], "recall": m["recall"], "f1": m["f1"],
            })
            print(f"{group:<10} | {hop:>4} | {n_total:>8} | {n_pos:>6} | "
                  f"{m['precision']:>6.4f} | {m['recall']:>6.4f} | {m['f1']:>6.4f}")
    print("=" * 88)

    # Save JSON
    if args.output_json:
        out = {
            "checkpoint": str(args.checkpoint),
            "mode": args.mode,
            "threshold": threshold,
            "n_seen_seeds_in_train": len(seen_seeds),
            "n_queries_seen_group": n_seen,
            "n_queries_unseen_group": n_unseen,
            "by_group": group_rows,
            "by_group_hop": grouphop_rows,
        }
        out_path = Path(args.output_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w", encoding="utf-8") as f:
            json.dump(out, f, indent=2)
        logger.info(f"Results written to {out_path}")


if __name__ == "__main__":
    main()
