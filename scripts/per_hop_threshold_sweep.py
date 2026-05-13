"""
scripts/per_hop_threshold_sweep.py â€” Find optimal threshold per hop.

The trade-off is that high thresholds (e.g. 0.80) work well for hop-1
where the model is confident, but they cut too much for hop-2 and
hop-3 where positive signal is weaker. Using a single global threshold
optimises the average at the cost of multi-hop recall.

Strategy:
    - Score the dev set once.
    - For each hop in {1, 2, 3}, sweep thresholds on its slice and
      pick the one that maximises F1 within that hop.
    - Apply the chosen per-hop thresholds to the test set and report
      the resulting global F1, precision, recall.

Usage:
    python scripts/per_hop_threshold_sweep.py
        --config configs/caff_orphanet.yaml
        --checkpoint runs/caff_orphanet/seed_42/best.pt
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
logger = logging.getLogger("per_hop_sweep")


def load_config(yaml_path: Path) -> tuple[CAFFConfig, AblationFlags]:
    with yaml_path.open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    cfg_dict = raw.get("config", {})
    abl_dict = raw.get("ablation", {})
    config = CAFFConfig(**cfg_dict)
    ablation = AblationFlags(**abl_dict) if abl_dict else AblationFlags()
    return config, ablation


def score_dataset(
    config: CAFFConfig,
    ablation: AblationFlags,
    checkpoint_path: Path,
    qa_path: str,
    cache_dir: Path,
    device: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Returns (scores, labels, hops) for every triple instance in qa_path."""
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


def find_optimal_per_hop(
    scores: np.ndarray,
    labels: np.ndarray,
    hops: np.ndarray,
    candidate_thresholds: np.ndarray,
) -> dict[int, float]:
    """For each unique hop, pick the threshold that maximises F1 on
    that hop only."""
    optimal: dict[int, float] = {}
    for hop in sorted(set(int(h) for h in hops)):
        mask = hops == hop
        s_hop = scores[mask]
        l_hop = labels[mask]
        best_f1 = -1.0
        best_t = 0.5
        for t in candidate_thresholds:
            prf = precision_recall_f1(s_hop, l_hop, float(t))
            if prf["f1"] > best_f1:
                best_f1 = prf["f1"]
                best_t = float(t)
        n_pos = int(l_hop.sum())
        n_total = len(l_hop)
        logger.info(
            f"  hop={hop}: optimal theta={best_t:.2f}, F1={best_f1:.4f} "
            f"({n_pos:,}/{n_total:,} positives, "
            f"{100*n_pos/max(n_total,1):.2f}%)"
        )
        optimal[hop] = best_t
    return optimal


def apply_per_hop(
    scores: np.ndarray,
    labels: np.ndarray,
    hops: np.ndarray,
    per_hop_thresholds: dict[int, float],
) -> dict[str, float]:
    """Apply hop-specific thresholds to compute global precision/recall/F1."""
    preds = np.zeros_like(labels, dtype=int)
    for hop, t in per_hop_thresholds.items():
        mask = hops == hop
        preds[mask] = (scores[mask] >= t).astype(int)
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
        help="QA file used to TUNE per-hop thresholds.",
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
    candidate_thresholds = np.arange(0.30, 0.91, 0.01)

    # â”€â”€â”€ Tune on DEV â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    logger.info("Step 1/3: Score DEV set")
    dev_scores, dev_labels, dev_hops = score_dataset(
        config, ablation, ckpt_path, args.dev_path,
        Path(args.cache_dir), args.device,
    )

    logger.info("Step 2/3: Find optimal per-hop thresholds on DEV")
    per_hop = find_optimal_per_hop(
        dev_scores, dev_labels, dev_hops, candidate_thresholds
    )

    # â”€â”€â”€ Apply to TEST â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    logger.info("Step 3/3: Score TEST set and apply per-hop thresholds")
    test_scores, test_labels, test_hops = score_dataset(
        config, ablation, ckpt_path, args.test_path,
        Path(args.cache_dir), args.device,
    )

    # Global theta (single best for comparison)
    global_results = {}
    for t in [0.50, 0.80]:
        global_results[t] = precision_recall_f1(test_scores, test_labels, t)

    per_hop_global = apply_per_hop(
        test_scores, test_labels, test_hops, per_hop
    )

    # Per-hop breakdown on test
    print()
    print("=" * 75)
    print("RESULTS ON HELD-OUT TEST SET")
    print("=" * 75)
    print(f"Per-hop thresholds tuned on dev:")
    for hop, t in per_hop.items():
        print(f"  hop={hop}: theta={t:.2f}")
    print()
    print(
        f"{'method':>30} | {'precision':>9} | {'recall':>7} | {'F1':>7}"
    )
    print("-" * 75)
    g50 = global_results[0.50]
    print(
        f"{'global theta=0.50':>30} | {g50['precision']:>9.4f} | "
        f"{g50['recall']:>7.4f} | {g50['f1']:>7.4f}"
    )
    g80 = global_results[0.80]
    print(
        f"{'global theta=0.80 (current)':>30} | {g80['precision']:>9.4f} | "
        f"{g80['recall']:>7.4f} | {g80['f1']:>7.4f}"
    )
    ph = per_hop_global
    print(
        f"{'per-hop theta (NEW)':>30} | {ph['precision']:>9.4f} | "
        f"{ph['recall']:>7.4f} | {ph['f1']:>7.4f}"
    )
    print("=" * 75)

    delta = per_hop_global["f1"] - global_results[0.80]["f1"]
    pct = 100 * delta / max(global_results[0.80]["f1"], 1e-9)
    print(f"Per-hop F1 vs global theta=0.80:  {delta:+.4f}  ({pct:+.1f}%)")

    return 0


if __name__ == "__main__":
    sys.exit(main())

