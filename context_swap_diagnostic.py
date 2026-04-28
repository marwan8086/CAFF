#!/usr/bin/env python
"""
context_swap_diagnostic.py — Reproduce paper Appendix C / Table 10.

The context-swap diagnostic is the empirical signature that
distinguishes CAFF from any context-agnostic baseline. By
construction, every f ∈ F_agn yields JSD = 0.00 bits on this
synthetic test (Definition 5 + Lemma 1). CAFF is expected to
produce ~1.84 bits.

Usage
-----
    python context_swap_diagnostic.py \\
        --checkpoint runs/caff_full/seed_42/best.pt \\
        --report-bits

Outputs (to stdout, plus optional JSON):
    s^(A) under causal context   : 0.792
    s^(B) under phenotypic context: 0.238
    JSD (bits, 2·JSD form)        : 1.84

Paper Table 10 expected ranges:
    BM25, DPR-Bio, BioRAG, SubgraphRAG, DepthBilinear → 0.00 bits
    CAFF-NoHC3                                         → 1.41 bits
    CAFF (Full)                                        → 1.84 bits
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import torch

from caff import (
    AblationFlags,
    CAFFConfig,
    CAFFModel,
    FrozenBioEncoder,
    KnowledgeGraph,
    RelationEmbeddingCache,
)
from caff.evaluator import context_swap_diagnostic
from caff.utils.logging import setup_logging

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run the context-swap diagnostic.")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--cache-dir", default="cache")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--report-bits", action="store_true",
                   help="Print JSD in bits (paper convention).")
    p.add_argument("--output-json", default=None)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    setup_logging(level="INFO")
    cache_dir = Path(args.cache_dir)

    payload = torch.load(args.checkpoint, map_location=args.device)
    config = CAFFConfig(**payload["config"])
    ablation = AblationFlags()

    logger.info(f"Loading KG to recover relation set...")
    kg = KnowledgeGraph.from_tsv(config.kg_path, min_relation_freq=50)
    encoder = FrozenBioEncoder(config.encoder_name, device=args.device)
    rel_cache = RelationEmbeddingCache(
        encoder, kg.relations,
        cache_path=cache_dir / "relation_embeddings.pt",
    )

    model = CAFFModel(config, rel_cache, ablation=ablation).to(args.device)
    model.load_state_dict(payload["model"])
    model.eval()

    logger.info("Running context-swap diagnostic (paper App. C)...")
    result = context_swap_diagnostic(model, encoder, config)

    print("─" * 60)
    print(f"Context-Swap Diagnostic — paper Appendix C / Table 10")
    print("─" * 60)
    print(f"  s^(A)  (causal-grounded context)     = {result['s_A']:.4f}")
    print(f"  s^(B)  (phenotypic-grounded context) = {result['s_B']:.4f}")
    print(f"  2·JSD(Bern(s_A) ∥ Bern(s_B))         = {result['jsd_bits']:.4f} bits")
    print("─" * 60)
    print("Reference values (paper Table 10):")
    print("  Any context-agnostic baseline → 0.00 bits  (CBE signature)")
    print("  CAFF-NoHC3                    → 1.41 bits")
    print("  CAFF (Full)                   → 1.84 bits")
    print("─" * 60)

    if args.output_json:
        out_path = Path(args.output_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w", encoding="utf-8") as f:
            json.dump(result, f, indent=2)
        logger.info(f"Wrote diagnostic to {out_path}")


if __name__ == "__main__":
    main()