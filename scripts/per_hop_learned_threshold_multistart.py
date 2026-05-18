"""
scripts/per_hop_learned_threshold_multistart.py - Multi-start
gradient-based per-hop threshold learning.

Section 20 finding: gradient descent on the soft-F1 surrogate is
multi-modal on CAFF logit distributions. Seeds 1337 and 2024
converged to the same useful basin (0.74, 0.82, 0.88), but seed 42
got stuck in a high-precision local optimum (0.85, 0.88, 0.91) and
regressed F1 by 0.029.

This script tests the natural fix: multi-start optimization.

For each hop, we run K independent gradient descents from K
different starting thresholds, then pick whichever final theta
maximizes hard F1 on dev. The starting grid is sampled from the
threshold range that grid search (Section 12) sweeps over, so we
include a warm start at every coarse grid point.

Algorithm:
    starting_thetas = [0.30, 0.40, ..., 0.90]   # 7 starts
    for each hop:
        results = []
        for theta_init in starting_thetas:
            theta = nn.Parameter(logit(theta_init))
            optimize for N steps with Adam
            track best hard_F1_dev
            results.append((best_theta, best_hard_F1))
        winner = max(results, key=lambda r: r[1])
    Apply winner thresholds on test.

If Section 20's diagnosis is right, this should fix the seed 42
failure: at least one of the 7 starts will land in the (0.74,
0.82, 0.88) basin, and the max-over-starts selection picks it.

Usage:
    python scripts/per_hop_learned_threshold_multistart.py
        --config configs/caff_orphanet.yaml
        --checkpoint runs/caff_orphanet/seed_42/best.pt
        --num-starts 7
        --device cuda
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
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
logger = logging.getLogger("per_hop_multistart")


DEFAULT_LR = 0.05
DEFAULT_STEPS = 1000
DEFAULT_TAU = 1.0
# Starting thresholds from coarse grid (in score space, will be
# converted to logit space). Matches Section 12's coarse range.
DEFAULT_STARTING_THETAS = [0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90]


def load_config(yaml_path: Path):
    with yaml_path.open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    cfg_dict = raw.get("config", {})
    abl_dict = raw.get("ablation", {})
    config = CAFFConfig(**cfg_dict)
    ablation = AblationFlags(**abl_dict) if abl_dict else AblationFlags()
    return config, ablation


def score_dataset(config, ablation, checkpoint_path, qa_path, cache_dir, device):
    """Returns (scores, labels, hops). scores are post-sigmoid."""
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
    eps = 1e-7
    s = np.clip(scores, eps, 1.0 - eps)
    return np.log(s / (1.0 - s))


def score_to_logit(score: float) -> float:
    eps = 1e-7
    s = max(min(score, 1.0 - eps), eps)
    return float(np.log(s / (1.0 - s)))


def soft_f1_loss(logits: torch.Tensor, labels: torch.Tensor,
                 theta: torch.Tensor, tau: float = 1.0) -> torch.Tensor:
    p = torch.sigmoid((logits - theta) / tau)
    tp = (p * labels).sum()
    fp = (p * (1.0 - labels)).sum()
    fn = ((1.0 - p) * labels).sum()
    f1 = 2.0 * tp / (2.0 * tp + fp + fn + 1e-12)
    return 1.0 - f1


def hard_f1_at_logit(logits: np.ndarray, labels: np.ndarray, theta_logit: float) -> float:
    preds = (logits >= theta_logit).astype(int)
    tp = int(((preds == 1) & (labels == 1)).sum())
    fp = int(((preds == 1) & (labels == 0)).sum())
    fn = int(((preds == 0) & (labels == 1)).sum())
    p = tp / max(tp + fp, 1)
    r = tp / max(tp + fn, 1)
    return 2 * p * r / max(p + r, 1e-12)


def learn_single_start(logits_np, labels_np, theta_init_logit, lr, steps, tau, device="cpu"):
    """Run one gradient descent from a single starting theta.

    Returns the best (theta_logit, hard_f1_dev) seen along the trajectory.
    """
    l_hop = torch.tensor(logits_np, dtype=torch.float32, device=device)
    y_hop = torch.tensor(labels_np.astype(np.float32), device=device)
    theta = nn.Parameter(torch.tensor([theta_init_logit], dtype=torch.float32, device=device))
    opt = torch.optim.Adam([theta], lr=lr)

    best_hard_f1 = hard_f1_at_logit(logits_np, labels_np, theta_init_logit)
    best_theta_logit = theta_init_logit

    for step in range(steps):
        opt.zero_grad()
        loss = soft_f1_loss(l_hop, y_hop, theta, tau=tau)
        loss.backward()
        opt.step()

        if step % 10 == 0 or step == steps - 1:
            cur = theta.item()
            hard_f1 = hard_f1_at_logit(logits_np, labels_np, cur)
            if hard_f1 > best_hard_f1:
                best_hard_f1 = hard_f1
                best_theta_logit = cur

    return best_theta_logit, best_hard_f1


def learn_per_hop_multistart(scores, labels, hops, num_hops,
                              starting_thetas, lr, steps, tau, device="cpu"):
    """For each hop, run K starts and pick the best by hard_F1_dev."""
    logits = scores_to_logits(scores)

    learned = {}
    for hop in sorted(set(int(h) for h in hops)):
        mask = hops == hop
        logits_np = logits[mask]
        labels_np = labels[mask]

        n_pos = int(labels_np.sum())
        n_total = int(labels_np.size)

        results = []
        for s_theta in starting_thetas:
            init_logit = score_to_logit(s_theta)
            best_theta_logit, best_hard_f1 = learn_single_start(
                logits_np, labels_np, init_logit, lr, steps, tau, device,
            )
            best_score = float(1.0 / (1.0 + np.exp(-best_theta_logit)))
            results.append({
                "start": s_theta,
                "best_theta_logit": best_theta_logit,
                "best_theta_score": best_score,
                "best_hard_f1": best_hard_f1,
            })

        # Pick the winner across starts
        winner = max(results, key=lambda r: r["best_hard_f1"])
        logger.info(
            f"  hop={hop}: WINNER theta={winner['best_theta_score']:.4f} "
            f"(logit={winner['best_theta_logit']:+.4f}, "
            f"hard_F1_dev={winner['best_hard_f1']:.4f}, "
            f"from start={winner['start']:.2f}) "
            f"[{n_pos:,}/{n_total:,} positives]"
        )
        # Log all starts for transparency
        for r in results:
            marker = " <-- winner" if r is winner else ""
            logger.info(
                f"    start={r['start']:.2f} -> theta={r['best_theta_score']:.4f} "
                f"hard_F1={r['best_hard_f1']:.4f}{marker}"
            )

        learned[hop] = {
            "theta_logit": winner["best_theta_logit"],
            "theta_score": winner["best_theta_score"],
            "hard_f1_dev": winner["best_hard_f1"],
            "winning_start": winner["start"],
            "all_starts": results,
            "n_pos": n_pos,
            "n_total": n_total,
        }
    return learned


def apply_per_hop_learned(scores, labels, hops, learned):
    preds = np.zeros_like(labels, dtype=int)
    for hop, info in learned.items():
        mask = hops == hop
        t = info["theta_score"]
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
    parser.add_argument("--dev-path", default="data/processed/dev.json")
    parser.add_argument("--test-path", default="data/processed/test.json")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--cache-dir", default="cache")
    parser.add_argument("--tau", type=float, default=DEFAULT_TAU)
    parser.add_argument("--lr", type=float, default=DEFAULT_LR)
    parser.add_argument("--steps", type=int, default=DEFAULT_STEPS)
    parser.add_argument(
        "--num-starts", type=int, default=7,
        help="Number of starting thresholds. Default 7 = {0.3, 0.4, ..., 0.9}."
    )
    args = parser.parse_args()

    if args.num_starts == 7:
        starting_thetas = DEFAULT_STARTING_THETAS
    else:
        starting_thetas = list(np.linspace(0.30, 0.90, args.num_starts))

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

    logger.info(f"Step 2/3: Multi-start gradient descent "
                f"({args.num_starts} starts, lr={args.lr}, "
                f"steps={args.steps}, tau={args.tau})")
    logger.info(f"  Starting thetas: {[f'{t:.2f}' for t in starting_thetas]}")
    learned = learn_per_hop_multistart(
        dev_scores, dev_labels, dev_hops, num_hops=config.L,
        starting_thetas=starting_thetas,
        lr=args.lr, steps=args.steps, tau=args.tau,
        device="cpu",
    )

    logger.info("Step 3/3: Score TEST set and apply learned thresholds")
    test_scores, test_labels, test_hops = score_dataset(
        config, ablation, ckpt_path, args.test_path,
        Path(args.cache_dir), args.device,
    )

    g50 = precision_recall_f1(test_scores, test_labels, 0.50)
    g80 = precision_recall_f1(test_scores, test_labels, 0.80)
    learned_test = apply_per_hop_learned(test_scores, test_labels, test_hops, learned)

    print()
    print("=" * 80)
    print(f"RESULTS ON HELD-OUT TEST SET  "
          f"(multi-start, n={args.num_starts}, tau={args.tau})")
    print("=" * 80)
    print(f"Per-hop thresholds LEARNED via multi-start soft-F1 on dev:")
    for hop, info in learned.items():
        print(f"  hop={hop}: theta={info['theta_score']:.4f} "
              f"(logit={info['theta_logit']:+.4f}, "
              f"hard_F1_dev={info['hard_f1_dev']:.4f}, "
              f"won from start={info['winning_start']:.2f})")
    print()
    print(f"{'method':>40} | {'precision':>9} | {'recall':>7} | {'F1':>7}")
    print("-" * 80)
    print(f"{'global theta=0.50':>40} | {g50['precision']:>9.4f} | "
          f"{g50['recall']:>7.4f} | {g50['f1']:>7.4f}")
    print(f"{'global theta=0.80 (current baseline)':>40} | {g80['precision']:>9.4f} | "
          f"{g80['recall']:>7.4f} | {g80['f1']:>7.4f}")
    print(f"{'multi-start learned per-hop (NEW)':>40} | {learned_test['precision']:>9.4f} | "
          f"{learned_test['recall']:>7.4f} | {learned_test['f1']:>7.4f}")
    print("=" * 80)
    delta = learned_test["f1"] - g80["f1"]
    pct = 100 * delta / max(g80["f1"], 1e-9)
    print(f"Multi-start learned F1 vs global theta=0.80:  {delta:+.4f}  ({pct:+.1f}%)")

    return 0


if __name__ == "__main__":
    sys.exit(main())
