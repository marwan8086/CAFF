#!/usr/bin/env python
"""
train.py — Entry point for CAFF training.

Usage
-----
    python train.py --config configs/caff_full.yaml --seed 42
    python train.py --config configs/caff_full.yaml --seed 1337
    python train.py --config configs/caff_no_hc3.yaml --seed 42

Outputs
-------
    runs/<config_name>/seed_<seed>/
      ├── config.json            # frozen config + content hash
      ├── train.log              # full log
      ├── history.jsonl          # per-epoch metrics
      ├── epoch_001.pt ... best.pt   # checkpoints
      └── final_metrics.json     # test-set metrics
"""

from __future__ import annotations

import argparse
import json
import logging
import platform
from dataclasses import asdict
from pathlib import Path

import torch
import yaml

from caff import (
    AblationFlags,
    CAFFConfig,
    CAFFEvaluator,
    CAFFModel,
    CAFFTrainer,
    CAFFTripleDataset,
    CachedBFSExtractor,
    FrozenBioEncoder,
    KnowledgeGraph,
    RelationEmbeddingCache,
    load_qa_split,
)
from caff.utils import set_global_seed
from caff.utils.logging import setup_logging

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────
# Argument parsing
# ─────────────────────────────────────────────────────────────────


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train CAFF.")
    p.add_argument("--config", required=True, help="Path to YAML config.")
    p.add_argument("--seed", type=int, default=None,
                   help="Override config.seed.")
    p.add_argument("--output-root", default="runs",
                   help="Root directory for run artifacts.")
    p.add_argument("--cache-dir", default="cache",
                   help="Cache directory (BFS subgraphs, relation embeds).")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--debug-subset", type=int, default=None,
                   help="If set, restrict to first N training queries (smoke test).")
    return p.parse_args()


# ─────────────────────────────────────────────────────────────────
# Hardware-aware batch sizing (Big-Tech smart engineering)
# ─────────────────────────────────────────────────────────────────


def detect_hardware_overrides(device: str) -> dict[str, int | str]:
    """Choose (micro_batch_size, grad_accum_steps, mixed_precision)
    to preserve effective batch=256 (paper §8.4) on the available GPU.

    Returns a dict the YAML loader will use to override config defaults.
    """
    if device != "cuda":
        return {
            "micro_batch_size": 8,
            "grad_accum_steps": 32,
            "mixed_precision": "no",
        }

    name = torch.cuda.get_device_name(0).lower()
    vram_gb = torch.cuda.get_device_properties(0).total_memory / 1e9

    if vram_gb >= 75:        # A100-80GB
        overrides = {"micro_batch_size": 256, "grad_accum_steps": 1,
                     "mixed_precision": "bf16"}
    elif vram_gb >= 35:      # A100-40GB / A40
        overrides = {"micro_batch_size": 64, "grad_accum_steps": 4,
                     "mixed_precision": "bf16"}
    elif vram_gb >= 22:      # L4 / 3090
        overrides = {"micro_batch_size": 32, "grad_accum_steps": 8,
                     "mixed_precision": "bf16"}
    elif vram_gb >= 14:      # T4 / V100-16GB
        overrides = {"micro_batch_size": 8,  "grad_accum_steps": 32,
                     "mixed_precision": "fp16"}
    else:
        overrides = {"micro_batch_size": 4,  "grad_accum_steps": 64,
                     "mixed_precision": "fp16"}

    logger.info(
        f"Hardware: {torch.cuda.get_device_name(0)} ({vram_gb:.0f}GB)  "
        f"→ micro={overrides['micro_batch_size']}  "
        f"accum={overrides['grad_accum_steps']}  "
        f"precision={overrides['mixed_precision']}  "
        f"(effective batch = 256)"
    )
    return overrides


# ─────────────────────────────────────────────────────────────────
# Config loading
# ─────────────────────────────────────────────────────────────────


def load_config(
    yaml_path: str | Path,
    seed_override: int | None,
    hw_overrides: dict,
) -> tuple[CAFFConfig, AblationFlags]:
    """Load YAML + apply hardware + seed overrides."""
    yaml_path = Path(yaml_path)
    with yaml_path.open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)

    cfg_dict = raw.get("config", {})
    abl_dict = raw.get("ablation", {})

    # Apply overrides
    cfg_dict.update(hw_overrides)
    if seed_override is not None:
        cfg_dict["seed"] = seed_override

    config = CAFFConfig(**cfg_dict)
    ablation = AblationFlags(**abl_dict) if abl_dict else AblationFlags()
    return config, ablation


# ─────────────────────────────────────────────────────────────────
# Run setup
# ─────────────────────────────────────────────────────────────────


def make_run_dir(
    output_root: str | Path,
    config_path: str | Path,
    seed: int,
) -> Path:
    """runs/<config-stem>/seed_<seed>/"""
    config_stem = Path(config_path).stem
    run_dir = Path(output_root) / config_stem / f"seed_{seed}"
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def write_run_manifest(
    run_dir: Path,
    config: CAFFConfig,
    ablation: AblationFlags,
    args: argparse.Namespace,
) -> None:
    """Write a frozen reproducibility manifest."""
    manifest = {
        "config": asdict(config),
        "config_hash": config.hash(),
        "ablation": asdict(ablation),
        "args": vars(args),
        "torch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "gpu_name": (torch.cuda.get_device_name(0)
                     if torch.cuda.is_available() else None),
        "python_version": platform.python_version(),
        "platform": platform.platform(),
    }
    with (run_dir / "config.json").open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)


# ─────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────


def main() -> None:
    args = parse_args()
    hw_overrides = detect_hardware_overrides(args.device)
    config, ablation = load_config(args.config, args.seed, hw_overrides)

    # Run directory
    run_dir = make_run_dir(args.output_root, args.config, config.seed)
    setup_logging(level="INFO", log_file=run_dir / "train.log")

    logger.info("=" * 70)
    logger.info("CAFF Training")
    logger.info(f"  Config:    {args.config}")
    logger.info(f"  Seed:      {config.seed}")
    logger.info(f"  Run dir:   {run_dir}")
    logger.info(f"  Hash:      {config.hash()}")
    logger.info("=" * 70)

    write_run_manifest(run_dir, config, ablation, args)

    # Reproducibility
    set_global_seed(config.seed, deterministic=config.deterministic)

    # ─── Load KG ─────────────────────────────────────────────────
    logger.info(f"Loading KG from {config.kg_path}...")
    kg = KnowledgeGraph.from_tsv(config.kg_path, min_relation_freq=config.min_relation_freq)

    # ─── Load encoder + relation cache ──────────────────────────
    logger.info(f"Loading frozen encoder: {config.encoder_name}")
    encoder = FrozenBioEncoder(config.encoder_name, device=args.device)
    rel_cache_path = Path(args.cache_dir) / "relation_embeddings.pt"
    relation_cache = RelationEmbeddingCache(
        encoder=encoder,
        relations=kg.relations,
        cache_path=rel_cache_path,
    )

    # ─── BFS extractor (with on-disk cache) ─────────────────────
    bfs_dir = Path(args.cache_dir) / "bfs"
    bfs = CachedBFSExtractor(kg, L=config.L, K_r=config.K_r, cache_dir=bfs_dir)

    # ─── Datasets ────────────────────────────────────────────────
    logger.info("Building train/dev datasets...")
    train_recs = load_qa_split(config.train_path)
    dev_recs   = load_qa_split(config.dev_path)
    if args.debug_subset:
        train_recs = train_recs[: args.debug_subset]
        dev_recs   = dev_recs[: max(1, args.debug_subset // 5)]
        logger.warning(f"DEBUG subset: train={len(train_recs)}  dev={len(dev_recs)}")

    train_ds = CAFFTripleDataset(train_recs, bfs, require_gold=True)
    dev_ds   = CAFFTripleDataset(dev_recs,   bfs, require_gold=True)

    # ─── Model ──────────────────────────────────────────────────
    model = CAFFModel(config, relation_cache, ablation=ablation).to(args.device)

    # ─── Evaluator (used for dev validation each epoch) ─────────
    evaluator = CAFFEvaluator(
        config=config,
        encoder=encoder,
        mode="teacher_forced",   # validation uses teacher forcing
        threshold=config.theta,
    )

    # ─── Ablation hook: zero out lambdas if ablation says so ────
    lam_C = 0.0 if not ablation.use_hc3 else config.lambda_C
    lam_D = 0.0 if not ablation.use_dc  else config.lambda_D

    # ─── Trainer ────────────────────────────────────────────────
    trainer = CAFFTrainer(
        config=config,
        model=model,
        encoder=encoder,
        train_dataset=train_ds,
        dev_dataset=dev_ds,
        evaluator=evaluator,
        ckpt_dir=run_dir,
        ablation_lambda_C=lam_C,
        ablation_lambda_D=lam_D,
        log_jsonl_path=run_dir / "history.jsonl",
    )

    history = trainer.train()

    # ─── Final metrics ──────────────────────────────────────────
    best = history.best_epoch("dev_f1")
    if best is not None:
        logger.info(f"Best epoch: {best.epoch}  dev_f1={best.dev_f1:.4f}")
        with (run_dir / "final_metrics.json").open("w", encoding="utf-8") as f:
            json.dump(asdict(best), f, indent=2)

    logger.info("Training complete.")


if __name__ == "__main__":
    main()