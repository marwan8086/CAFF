#!/usr/bin/env python
"""
hop_stratified_analysis.py -- Cross-tabulate hop and relation.

Loads a trained checkpoint, scores the test set, then aggregates by
(hop, relation) pairs. This answers:

- Which relations appear at which hops? (the hop x relation count matrix)
- What is per-hop, per-relation F1?
- Is the hop3 precision drop driven by has_phenotype, or by all relations?
- How do score distributions differ across hops?

Usage:
    python scripts/hop_stratified_analysis.py \
        --checkpoint runs/no_dc/seed_42/best.pt \
        --threshold 0.80 \
        --mode autoregressive \
        --output-json results/hop_stratified_seed42.json
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
    p = argparse.ArgumentParser(description="Hop x relation cross-tabulation.")
    p.add_argument("--checkpoint", required=True)
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


def main() -> None:
    args = parse_args()
    setup_logging(level="INFO")
    cache_dir = Path(args.cache_dir)

    model, config, encoder, kg = load_checkpoint(args.checkpoint, args.device, cache_dir)

    test_path = args.test_split or config.test_path
    test_recs = load_qa_split(test_path)
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

    # Aggregate by (hop, relation)
    by_key_scores: dict[tuple[int, str], list[float]] = defaultdict(list)
    by_key_labels: dict[tuple[int, str], list[int]] = defaultdict(list)
    # Also aggregate by hop only
    by_hop_scores: dict[int, list[float]] = defaultdict(list)
    by_hop_labels: dict[int, list[int]] = defaultdict(list)

    for inst, sc in zip(instances, scores_np.tolist()):
        key = (inst.hop, inst.relation)
        by_key_scores[key].append(sc)
        by_key_labels[key].append(inst.label)
        by_hop_scores[inst.hop].append(sc)
        by_hop_labels[inst.hop].append(inst.label)

    # Per-hop metrics
    hop_rows = []
    print()
    print("=" * 96)
    print(f"Hop-stratified summary  (mode={args.mode}, theta={threshold})")
    print(f"Checkpoint: {args.checkpoint}")
    print("=" * 96)
    print(f"{'hop':>4} | {'n_total':>8} | {'n_pos':>6} | {'pos%':>6} | "
          f"{'prec':>6} | {'recall':>6} | {'F1':>6} | "
          f"{'score_mean':>10} | {'score_std':>9}")
    print("-" * 96)
    for hop in sorted(by_hop_scores.keys()):
        s = np.asarray(by_hop_scores[hop])
        l = np.asarray(by_hop_labels[hop])
        n_total = len(l)
        n_pos = int(l.sum())
        pos_rate = n_pos / n_total if n_total > 0 else 0.0
        m = precision_recall_f1(s, l, threshold=threshold)
        hop_rows.append({
            "hop": hop,
            "n_total": n_total,
            "n_pos": n_pos,
            "pos_rate": pos_rate,
            "precision": m["precision"],
            "recall": m["recall"],
            "f1": m["f1"],
            "score_mean": float(s.mean()),
            "score_std": float(s.std()),
            "score_mean_pos": float(s[l == 1].mean()) if n_pos > 0 else None,
            "score_mean_neg": float(s[l == 0].mean()) if (n_total - n_pos) > 0 else None,
        })
        print(f"{hop:>4} | {n_total:>8} | {n_pos:>6} | {pos_rate*100:>5.1f}% | "
              f"{m['precision']:>6.4f} | {m['recall']:>6.4f} | {m['f1']:>6.4f} | "
              f"{s.mean():>10.4f} | {s.std():>9.4f}")

    # Hop x relation cross-tab (counts)
    print()
    print("=" * 96)
    print(f"Hop x relation counts (n_total, n_positive)")
    print("=" * 96)
    relations_sorted = sorted({rel for (_, rel) in by_key_scores.keys()},
                              key=lambda r: -sum(len(by_key_labels[(h, r)])
                                                 for h in [1, 2, 3]))
    print(f"{'relation':<55} | {'hop 1':>14} | {'hop 2':>14} | {'hop 3':>14}")
    print("-" * 96)
    cross_rows = []
    for rel in relations_sorted:
        cells = []
        rel_row = {"relation": rel}
        for hop in [1, 2, 3]:
            key = (hop, rel)
            n = len(by_key_labels.get(key, []))
            npos = int(sum(by_key_labels.get(key, [])))
            cells.append(f"{n:>6}/{npos:<6}")
            rel_row[f"hop{hop}_n_total"] = n
            rel_row[f"hop{hop}_n_pos"] = npos
        cross_rows.append(rel_row)
        rel_short = rel[:55]
        print(f"{rel_short:<55} | {cells[0]:>14} | {cells[1]:>14} | {cells[2]:>14}")

    # Per (hop, relation) F1 for top-2 relations
    print()
    print("=" * 96)
    print(f"Per (hop, relation) F1 for the top-2 relations by support")
    print("=" * 96)
    top_relations = relations_sorted[:2]
    f1_rows = []
    print(f"{'relation':<25} | {'hop':>4} | {'n_total':>8} | {'n_pos':>6} | "
          f"{'prec':>6} | {'recall':>6} | {'F1':>6}")
    print("-" * 80)
    for rel in top_relations:
        for hop in [1, 2, 3]:
            key = (hop, rel)
            if key not in by_key_scores:
                continue
            s = np.asarray(by_key_scores[key])
            l = np.asarray(by_key_labels[key])
            if len(l) == 0:
                continue
            n_pos = int(l.sum())
            m = precision_recall_f1(s, l, threshold=threshold)
            f1_rows.append({
                "relation": rel, "hop": hop,
                "n_total": len(l), "n_pos": n_pos,
                "precision": m["precision"], "recall": m["recall"], "f1": m["f1"],
            })
            print(f"{rel[:25]:<25} | {hop:>4} | {len(l):>8} | {n_pos:>6} | "
                  f"{m['precision']:>6.4f} | {m['recall']:>6.4f} | {m['f1']:>6.4f}")
    print("=" * 96)

    # Save JSON
    if args.output_json:
        out = {
            "checkpoint": str(args.checkpoint),
            "mode": args.mode,
            "threshold": threshold,
            "per_hop": hop_rows,
            "hop_x_relation_counts": cross_rows,
            "per_hop_relation_f1_top2": f1_rows,
        }
        out_path = Path(args.output_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w", encoding="utf-8") as f:
            json.dump(out, f, indent=2)
        logger.info(f"Results written to {out_path}")


if __name__ == "__main__":
    main()
