"""
scripts/per_hop_temperature_sweep.py - Joint per-hop temperature and
threshold sweep.

Section 18 found that the F1 ceiling under per-hop fine-step
thresholding is set by calibration stability, not parameter count.
Higher rho smooths the score distribution, breaking the per-hop dev
threshold search. The natural fix is temperature scaling: post-hoc,
no retraining.

This script mirrors scripts/per_hop_threshold_sweep.py exactly,
except the per-hop sweep loop now jointly searches over (T, theta)
instead of theta alone.

For each hop l in {1, 2, 3}:
  1. Score dev set (post-sigmoid as the evaluator returns it).
  2. Recover logits via inverse-sigmoid: logit = log(s / (1 - s)).
  3. For each T in temperature grid:
       For each theta in threshold grid:
         new_scores = sigmoid(logits / T)
         compute F1(new_scores >= theta)
       Track best (T*, theta*).
  4. Apply (T*_l, theta*_l) on test set.

If Section 18's hypothesis is right, T < 1.0 (sharpening) should
recover per-hop F1; at rho=16, T = 1.0 should remain optimal (or
near it) and per-hop F1 should match the Section 12 headline.

Usage:
    python scripts/per_hop_temperature_sweep.py
        --config configs/caff_orphanet.yaml
        --checkpoint runs/caff_orphanet/seed_42/best.pt
        --device cuda
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import torch
import yaml

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
from caff.utils import set_global_seed

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("per_hop_temp_sweep")


# Sweep grids
TEMPERATURE_GRID = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.4, 1.6, 2.0, 3.0, 5.0]
THRESHOLD_GRID = np.arange(0.30, 0.91, 0.01)


def load_config(yaml_path: Path):
    with yaml_path.open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    cfg_dict = raw.get("config", {})
    abl_dict = raw.get("ablation", {})
    config = CAFFConfig(**cfg_dict)
    ablation = AblationFlags(**abl_dict) if abl_dict else AblationFlags()
    return config, ablation


def score_dataset(config, ablation, checkpoint_path, qa_path, cache_dir, device):
    """Returns (scores, labels, hops) for every triple instance in qa_path.

    scores are post-sigmoid (in [0, 1]).
    """
    set_global_seed(config.seed, deterministic=config.deterministic)
    kg = KnowledgeGraph.from_tsv(
        config.kg_path, min_relation_freq=config.min_relation_freq
    )
    encoder = FrozenBioEncoder(config.encoder_name, device=device)
    rel_cache = RelationEmbeddingCache(
        encoder=encoder,
        relations=kg.relations,
        cache_path=cache_dir / "relation_embeddings.pt",
    )
    bfs = CachedBFSExtractor(
        kg, L=config.L, K_r=config.K_r, cache_dir=cache_dir / "bfs"
    )
    recs = load_qa_split(qa_path)
    ds = CAFFTripleDataset(recs, bfs, require_gold=True)
    logger.info(f"  Dataset: {len(ds):,} triple instances from {qa_path}")

    model = CAFFModel(config, rel_cache, ablation=ablation).to(device)
    payload = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(payload["model"], strict=False)
    model.eval()

    evaluator = CAFFEvaluator(
        config=config, encoder=encoder,
        mode="teacher_forced", threshold=0.5,
    )
    scores, instances, _ = evaluator._score_dataset(model, ds)
    labels = np.array([i.label for i in instances])
    hops = np.array([i.hop for i in instances])
    return scores, labels, hops


def scores_to_logits(scores: np.ndarray) -> np.ndarray:
    """Inverse-sigmoid: convert post-sigmoid scores back to logits.

    logit = log(s / (1 - s))

    We clip away from 0 and 1 to avoid +/-inf. After clipping, the
    extreme values of logits are bounded by log((1-eps)/eps).
    """
    eps = 1e-7
    s = np.clip(scores, eps, 1.0 - eps)
    return np.log(s / (1.0 - s))


def find_optimal_per_hop_T_theta(scores, labels, hops):
    """For each hop, find best (T, theta) jointly on dev data."""
    logits = scores_to_logits(scores)

    optimal = {}
    for hop in sorted(set(int(h) for h in hops)):
        mask = hops == hop
        l_hop = logits[mask]
        y_hop = labels[mask]
        n_pos = int(y_hop.sum())
        n_total = len(y_hop)

        best_f1 = -1.0
        best_T = 1.0
        best_t = 0.5
        for T in TEMPERATURE_GRID:
            scaled = np.clip(l_hop / T, -60.0, 60.0)
            new_scores = 1.0 / (1.0 + np.exp(-scaled))
            for theta in THRESHOLD_GRID:
                prf = precision_recall_f1(new_scores, y_hop, float(theta))
                if prf["f1"] > best_f1:
                    best_f1 = prf["f1"]
                    best_T = T
                    best_t = float(theta)

        logger.info(
            f"  hop={hop}: optimal T={best_T:.2f} theta={best_t:.2f} "
            f"F1={best_f1:.4f} ({n_pos:,}/{n_total:,} positives, "
            f"{100*n_pos/max(n_total,1):.2f}%)"
        )
        optimal[hop] = (best_T, best_t)
    return optimal


def apply_per_hop_T_theta(scores, labels, hops, per_hop_Tt):
    """Apply per-hop (T, theta) to compute precision/recall/F1."""
    logits = scores_to_logits(scores)
    preds = np.zeros_like(labels, dtype=int)
    for hop, (T, t) in per_hop_Tt.items():
        mask = hops == hop
        scaled = np.clip(logits[mask] / T, -60.0, 60.0)
        new_scores = 1.0 / (1.0 + np.exp(-scaled))
        preds[mask] = (new_scores >= t).astype(int)
    tp = int(((preds == 1) & (labels == 1)).sum())
    fp = int(((preds == 1) & (labels == 0)).sum())
    fn = int(((preds == 0) & (labels == 1)).sum())
    p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
    return {"precision": p, "recall": r, "f1": f}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/caff_orphanet.yaml")
    parser.add_argument(
        "--checkpoint", default="runs/caff_orphanet/seed_42/best.pt"
    )
    parser.add_argument(
        "--dev-path", default="data/processed/dev.json",
        help="QA file used to TUNE per-hop (T, theta).",
    )
    parser.add_argument(
        "--test-path", default="data/processed/test.json",
        help="Held-out file for the final reported number.",
    )
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--cache-dir", default="cache")
    args = parser.parse_args()

    config_path = Path(args.config)
    ckpt_path = Path(args.checkpoint)
    if not config_path.exists() or not ckpt_path.exists():
        logger.error("Config or checkpoint not found.")
        return 1

    config, ablation = load_config(config_path)

    logger.info("Step 1/3: Score DEV set")
    dev_scores, dev_labels, dev_hops = score_dataset(
        config, ablation, ckpt_path, args.dev_path,
        Path(args.cache_dir), args.device,
    )

    logger.info("Step 2/3: Find optimal per-hop (T, theta) on DEV")
    per_hop_Tt = find_optimal_per_hop_T_theta(dev_scores, dev_labels, dev_hops)

    logger.info("Step 3/3: Score TEST set and apply per-hop (T, theta)")
    test_scores, test_labels, test_hops = score_dataset(
        config, ablation, ckpt_path, args.test_path,
        Path(args.cache_dir), args.device,
    )

    g50 = precision_recall_f1(test_scores, test_labels, 0.50)
    g80 = precision_recall_f1(test_scores, test_labels, 0.80)
    per_hop_TT = apply_per_hop_T_theta(test_scores, test_labels, test_hops, per_hop_Tt)

    print()
    print("=" * 75)
    print("RESULTS ON HELD-OUT TEST SET")
    print("=" * 75)
    print(f"Per-hop (T, theta) tuned on dev:")
    for hop, (T, t) in per_hop_Tt.items():
        print(f"  hop={hop}: T={T:.2f}  theta={t:.2f}")
    print()
    print(f"{'method':>35} | {'precision':>9} | {'recall':>7} | {'F1':>7}")
    print("-" * 75)
    print(f"{'global theta=0.50 (T=1)':>35} | {g50['precision']:>9.4f} | "
          f"{g50['recall']:>7.4f} | {g50['f1']:>7.4f}")
    print(f"{'global theta=0.80 (T=1, current)':>35} | {g80['precision']:>9.4f} | "
          f"{g80['recall']:>7.4f} | {g80['f1']:>7.4f}")
    print(f"{'per-hop (T, theta) (NEW)':>35} | {per_hop_TT['precision']:>9.4f} | "
          f"{per_hop_TT['recall']:>7.4f} | {per_hop_TT['f1']:>7.4f}")
    print("=" * 75)
    delta = per_hop_TT["f1"] - g80["f1"]
    pct = 100 * delta / max(g80["f1"], 1e-9)
    print(f"Per-hop (T, theta) F1 vs global theta=0.80:  {delta:+.4f}  ({pct:+.1f}%)")

    return 0


if __name__ == "__main__":
    sys.exit(main())
