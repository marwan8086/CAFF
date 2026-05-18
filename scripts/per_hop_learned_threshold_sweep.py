"""
scripts/per_hop_learned_threshold_sweep.py - Gradient-based per-hop
threshold learning via soft F1 surrogate (v2 with tau argument).

Section 19 ends: "Closing the remaining gap requires changing the
calibration mechanism itself - e.g. learned per-hop thresholds
trained jointly with the classification loss, not just rescaling
its inputs."

This script tests exactly that. Instead of grid-searching per-hop
theta on dev (Section 12 method), it *learns* per-hop thresholds
by gradient descent on a differentiable F1 surrogate.

Key insight (no retraining needed):
  Per-hop thresholds are *post-hoc* parameters; they don't change
  the model's logit outputs. So we can:
    1. Score dev set once with the existing checkpoint.
    2. Initialize 3 learnable thresholds (one per hop).
    3. Optimize them with Adam on the soft-F1 loss.
    4. Apply final thresholds on test.

The soft-F1 loss is differentiable in the thresholds because
predictions are kept as continuous sigmoid outputs:

    p_l = sigmoid((logit - theta_l) / tau)

    TP_soft = sum_{i: y_i = 1}    p_l(i)
    FP_soft = sum_{i: y_i = 0}    p_l(i)
    FN_soft = sum_{i: y_i = 1} (1 - p_l(i))

    F1_soft = 2 * TP_soft / (2 * TP_soft + FP_soft + FN_soft)
    L = 1 - F1_soft

Tau (temperature) controls the sharpness:
  - tau = 1.0: sharp sigmoid, vanishing gradients near boundary,
    optimizer may converge to high-precision/low-recall regime.
  - tau > 1.0: softer sigmoid, better gradients, soft-F1 closer
    to hard F1 in the interior but smoother boundary.

The script accepts --tau to sweep this hyperparameter.

Usage:
    python scripts/per_hop_learned_threshold_sweep.py
        --config configs/caff_orphanet.yaml
        --checkpoint runs/caff_orphanet/seed_42/best.pt
        --tau 3.0
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
logger = logging.getLogger("per_hop_learn")


# Learning hyperparameters
DEFAULT_LEARN_LR = 0.05
DEFAULT_LEARN_STEPS = 1000
DEFAULT_TAU = 1.0


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
    """Inverse-sigmoid: convert post-sigmoid scores back to logits."""
    eps = 1e-7
    s = np.clip(scores, eps, 1.0 - eps)
    return np.log(s / (1.0 - s))


def soft_f1_loss(logits: torch.Tensor, labels: torch.Tensor,
                 theta: torch.Tensor, tau: float = 1.0) -> torch.Tensor:
    """Differentiable F1 surrogate.

    p = sigmoid((logit - theta) / tau)
    L = 1 - 2*TP / (2*TP + FP + FN)
    """
    p = torch.sigmoid((logits - theta) / tau)
    tp = (p * labels).sum()
    fp = (p * (1.0 - labels)).sum()
    fn = ((1.0 - p) * labels).sum()
    f1 = 2.0 * tp / (2.0 * tp + fp + fn + 1e-12)
    return 1.0 - f1


def hard_f1_at_theta(logits: np.ndarray, labels: np.ndarray, theta_logit: float) -> float:
    """Compute the actual hard F1 at a given threshold (in logit space)."""
    preds = (logits >= theta_logit).astype(int)
    tp = int(((preds == 1) & (labels == 1)).sum())
    fp = int(((preds == 1) & (labels == 0)).sum())
    fn = int(((preds == 0) & (labels == 1)).sum())
    p = tp / max(tp + fp, 1)
    r = tp / max(tp + fn, 1)
    return 2 * p * r / max(p + r, 1e-12)


def learn_per_hop_thresholds(scores, labels, hops, num_hops,
                              lr: float, steps: int, tau: float,
                              device="cpu"):
    """Learn per-hop thresholds via gradient descent on soft F1.

    Returns a dict: hop -> {theta_logit, theta_score, soft_f1, hard_f1, ...}
    """
    logits = scores_to_logits(scores)

    learned = {}
    for hop in sorted(set(int(h) for h in hops)):
        mask = hops == hop
        logits_np = logits[mask]
        labels_np = labels[mask]

        l_hop = torch.tensor(logits_np, dtype=torch.float32, device=device)
        y_hop = torch.tensor(labels_np.astype(np.float32), device=device)

        n_pos = int(y_hop.sum().item())
        n_total = int(y_hop.numel())

        # Initialize theta in logit space at 0.0 (= sigmoid 0.5)
        theta = nn.Parameter(torch.zeros(1, device=device))
        opt = torch.optim.Adam([theta], lr=lr)

        # Track best hard F1 (not soft) for principled selection
        best_hard_f1 = -1.0
        best_theta_logit = 0.0
        best_soft_f1 = 0.0
        for step in range(steps):
            opt.zero_grad()
            loss = soft_f1_loss(l_hop, y_hop, theta, tau=tau)
            loss.backward()
            opt.step()

            # Compute hard F1 at current theta on dev (every 10 steps)
            if step % 10 == 0 or step == steps - 1:
                cur_theta = theta.item()
                hard_f1 = hard_f1_at_theta(logits_np, labels_np, cur_theta)
                if hard_f1 > best_hard_f1:
                    best_hard_f1 = hard_f1
                    best_theta_logit = cur_theta
                    best_soft_f1 = 1.0 - loss.item()

        theta_score = float(1.0 / (1.0 + np.exp(-best_theta_logit)))
        learned[hop] = {
            "theta_logit": best_theta_logit,
            "theta_score": theta_score,
            "soft_f1": best_soft_f1,
            "hard_f1_dev": best_hard_f1,
            "n_pos": n_pos,
            "n_total": n_total,
        }
        logger.info(
            f"  hop={hop}: learned theta_logit={best_theta_logit:+.4f} "
            f"(score={theta_score:.4f}) "
            f"soft_F1={best_soft_f1:.4f} hard_F1_dev={best_hard_f1:.4f} "
            f"({n_pos:,}/{n_total:,} positives)"
        )
    return learned


def apply_per_hop_learned(scores, labels, hops, learned):
    """Apply learned per-hop thresholds. Use HARD predictions."""
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
    parser.add_argument(
        "--dev-path", default="data/processed/dev.json",
    )
    parser.add_argument(
        "--test-path", default="data/processed/test.json",
    )
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--cache-dir", default="cache")
    parser.add_argument("--tau", type=float, default=DEFAULT_TAU,
                        help="Soft sigmoid temperature. Higher = smoother.")
    parser.add_argument("--lr", type=float, default=DEFAULT_LEARN_LR,
                        help="Adam learning rate for threshold gradient descent.")
    parser.add_argument("--steps", type=int, default=DEFAULT_LEARN_STEPS,
                        help="Number of gradient steps.")
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

    logger.info("Step 2/3: Learn per-hop thresholds on DEV via soft-F1")
    logger.info(f"  (LR={args.lr}, steps={args.steps}, tau={args.tau})")
    learned = learn_per_hop_thresholds(
        dev_scores, dev_labels, dev_hops, num_hops=config.L,
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
    print(f"RESULTS ON HELD-OUT TEST SET  (tau={args.tau}, lr={args.lr}, steps={args.steps})")
    print("=" * 80)
    print(f"Per-hop thresholds LEARNED via soft-F1 on dev:")
    for hop, info in learned.items():
        print(f"  hop={hop}: theta={info['theta_score']:.4f} "
              f"(logit={info['theta_logit']:+.4f}, "
              f"soft_F1={info['soft_f1']:.4f}, hard_F1_dev={info['hard_f1_dev']:.4f})")
    print()
    print(f"{'method':>40} | {'precision':>9} | {'recall':>7} | {'F1':>7}")
    print("-" * 80)
    print(f"{'global theta=0.50':>40} | {g50['precision']:>9.4f} | "
          f"{g50['recall']:>7.4f} | {g50['f1']:>7.4f}")
    print(f"{'global theta=0.80 (current baseline)':>40} | {g80['precision']:>9.4f} | "
          f"{g80['recall']:>7.4f} | {g80['f1']:>7.4f}")
    print(f"{'learned per-hop theta (NEW)':>40} | {learned_test['precision']:>9.4f} | "
          f"{learned_test['recall']:>7.4f} | {learned_test['f1']:>7.4f}")
    print("=" * 80)
    delta = learned_test["f1"] - g80["f1"]
    pct = 100 * delta / max(g80["f1"], 1e-9)
    print(f"Learned per-hop F1 vs global theta=0.80:  {delta:+.4f}  ({pct:+.1f}%)")

    return 0


if __name__ == "__main__":
    sys.exit(main())
