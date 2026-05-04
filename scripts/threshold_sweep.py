"""
scripts/threshold_sweep.py — Sweep decision thresholds against the
dev set using a saved best.pt checkpoint, without retraining.

Why this is fast:
    The model emits per-triple scores once. The threshold only
    enters precision_recall_f1 (and hop_stratified_precision).
    MAP and NDCG are threshold-independent. We score the dev set
    once, then re-evaluate F1 across many thresholds.

Usage:
    python scripts/threshold_sweep.py
        [--config configs/caff_orphanet.yaml]
        [--checkpoint runs/caff_orphanet/seed_42/best.pt]
        [--thresholds 0.20,0.25,0.30,0.35,0.40,0.45,0.50,0.55,0.60]
"""
from __future__ import annotations

import argparse
import logging
import sys
from collections import defaultdict
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
from caff.evaluator import (
    hop_stratified_precision,
    mean_average_precision,
    mean_ndcg_at_k,
    precision_recall_f1,
)
from caff.utils import set_global_seed

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("threshold_sweep")


def load_config(yaml_path: Path) -> tuple[CAFFConfig, AblationFlags]:
    """Load YAML config (same logic as train.py::load_config)."""
    with yaml_path.open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    cfg_dict = raw.get("config", {})
    abl_dict = raw.get("ablation", {})
    config = CAFFConfig(**cfg_dict)
    ablation = AblationFlags(**abl_dict) if abl_dict else AblationFlags()
    return config, ablation


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", default="configs/caff_orphanet.yaml",
    )
    parser.add_argument(
        "--checkpoint",
        default="runs/caff_orphanet/seed_42/best.pt",
    )
    parser.add_argument(
        "--thresholds",
        default="0.20,0.25,0.30,0.35,0.40,0.45,0.50,0.55,0.60",
    )
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--cache-dir", default="cache")
    args = parser.parse_args()

    config_path = Path(args.config)
    ckpt_path = Path(args.checkpoint)
    if not config_path.exists():
        logger.error(f"Config not found: {config_path}")
        return 1
    if not ckpt_path.exists():
        logger.error(f"Checkpoint not found: {ckpt_path}")
        return 1

    thresholds = [float(t.strip()) for t in args.thresholds.split(",") if t.strip()]
    if not thresholds:
        logger.error("No thresholds provided.")
        return 1

    # ─── Load config ───────────────────────────────────────
    config, ablation = load_config(config_path)
    set_global_seed(config.seed, deterministic=config.deterministic)
    logger.info(f"Loaded config: {config_path.name}")
    logger.info(f"  KG path:  {config.kg_path}")
    logger.info(f"  Dev path: {config.dev_path}")
    logger.info(f"  Encoder:  {config.encoder_name}")

    # ─── Load KG ───────────────────────────────────────────
    logger.info("Loading KG ...")
    kg = KnowledgeGraph.from_tsv(
        config.kg_path,
        min_relation_freq=config.min_relation_freq,
    )

    # ─── Load encoder + relation cache ─────────────────────
    logger.info(f"Loading encoder: {config.encoder_name}")
    encoder = FrozenBioEncoder(config.encoder_name, device=args.device)
    rel_cache_path = Path(args.cache_dir) / "relation_embeddings.pt"
    relation_cache = RelationEmbeddingCache(
        encoder=encoder,
        relations=kg.relations,
        cache_path=rel_cache_path,
    )

    # ─── BFS extractor (reuses on-disk cache) ──────────────
    bfs = CachedBFSExtractor(
        kg, L=config.L, K_r=config.K_r,
        cache_dir=Path(args.cache_dir) / "bfs",
    )

    # ─── Dev dataset ───────────────────────────────────────
    dev_recs = load_qa_split(config.dev_path)
    dev_ds = CAFFTripleDataset(dev_recs, bfs, require_gold=True)
    logger.info(f"Dev dataset: {len(dev_ds):,} triple instances")

    # ─── Build model and load checkpoint ───────────────────
    model = CAFFModel(config, relation_cache, ablation=ablation).to(args.device)
    payload = torch.load(ckpt_path, map_location=args.device, weights_only=False)
    if not isinstance(payload, dict) or "model" not in payload:
        logger.error(
            f"Unexpected checkpoint format. Keys: "
            f"{list(payload.keys()) if isinstance(payload, dict) else type(payload)}"
        )
        return 1
    missing, unexpected = model.load_state_dict(payload["model"], strict=False)
    if missing:
        logger.warning(
            f"Missing keys when loading: "
            f"{missing[:5]}{'...' if len(missing)>5 else ''}"
        )
    if unexpected:
        logger.warning(
            f"Unexpected keys when loading: "
            f"{unexpected[:5]}{'...' if len(unexpected)>5 else ''}"
        )
    model.eval()

    if "metrics" in payload:
        m = payload["metrics"]
        logger.info(
            f"Checkpoint training metrics: "
            f"epoch={m.get('epoch')}, "
            f"dev_f1={m.get('dev_f1')}, dev_map={m.get('dev_map')}"
        )

    # ─── Score dev once ────────────────────────────────────
    evaluator = CAFFEvaluator(
        config=config,
        encoder=encoder,
        mode="teacher_forced",
        threshold=thresholds[0],
    )
    logger.info("Scoring dev set once (this is the slow part)...")
    scores, instances, _retained = evaluator._score_dataset(model, dev_ds)
    labels = np.array([i.label for i in instances])
    logger.info(
        f"  Done. {len(scores):,} candidate scores; "
        f"{int(labels.sum()):,} positives "
        f"({100*labels.mean():.2f}% positive rate)"
    )

    # ─── Compute MAP / NDCG once (threshold-independent) ───
    q_groups: dict[str, list[tuple[float, int]]] = defaultdict(list)
    for inst, sc, lbl in zip(instances, scores.tolist(), labels.tolist()):
        q_groups[inst.query_id].append((sc, lbl))
    per_query: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for qid, items in q_groups.items():
        per_query[qid] = (
            np.array([x[0] for x in items]),
            np.array([x[1] for x in items]),
        )
    map_val = mean_average_precision(per_query)
    ndcg_val = mean_ndcg_at_k(per_query, k=10)
    logger.info(f"MAP (threshold-independent):     {map_val:.4f}")
    logger.info(f"NDCG@10 (threshold-independent): {ndcg_val:.4f}")

    # ─── Sweep thresholds ──────────────────────────────────
    print()
    print("=" * 80)
    print(
        f"{'theta':>6} | {'precision':>9} | {'recall':>7} | "
        f"{'F1':>7} | {'hop1':>6} | {'hop2':>6} | {'hop3':>6}"
    )
    print("=" * 80)
    best_f1 = -1.0
    best_t = None
    for t in thresholds:
        prf = precision_recall_f1(scores, labels, t)
        hop_prec = hop_stratified_precision(instances, scores, t)
        f1 = prf["f1"]
        if f1 > best_f1:
            best_f1 = f1
            best_t = t
        print(
            f"{t:>6.2f} | {prf['precision']:>9.4f} | {prf['recall']:>7.4f} | "
            f"{f1:>7.4f} | {hop_prec.get(1, 0.0):>6.4f} | "
            f"{hop_prec.get(2, 0.0):>6.4f} | {hop_prec.get(3, 0.0):>6.4f}"
        )
    print("=" * 80)
    print(f"Best threshold: {best_t:.2f}  (F1 = {best_f1:.4f})")
    print(f"MAP = {map_val:.4f}   NDCG@10 = {ndcg_val:.4f}   (constant across thresholds)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
