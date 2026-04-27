#!/usr/bin/env python
"""
scripts/annotate_triples.py — Annotate gold-relevance labels for QA records.

Implements the labeling protocol described in paper §8.1:

    "Gold relevance labels y_ℓ are assigned by shortest-path
     reachability: triples on any shortest path from seed to gold
     answer entity receive y=1; all others y=0. This yields 3.7M
     labeled training instances across hop depths 1–3."

This script takes raw QA records (with seeds and gold answers) and
emits a JSON file ready to be consumed by `caff.data.load_qa_split`.

Input format
------------
A JSON list of records:
    {
      "query_id":     "pubmedqa_train_001",
      "question":     "What drug targets ...?",
      "seeds":        ["C1234567", "C7654321"],     # UMLS CUIs
      "gold_answer":  "C8888888",                    # UMLS CUI
      "answer_label": "yes" | "no" | "maybe" | text  # optional
    }

Output is the same format, plus this script pre-computes BFS
subgraphs (writing them to the cache directory) and verifies that
the gold-answer entity is reachable within L hops for at least
SOME seeds. Records with no reachable gold answer are flagged.

Usage
-----
    python scripts/annotate_triples.py \
        --kg          data/processed/merged_kg.tsv \
        --qa-input    data/raw/pubmedqa_with_seeds.json \
        --qa-output   data/processed/train.json \
        --cache-dir   cache/bfs/ \
        --L 3 --K-r 20
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

from caff.data import (
    CachedBFSExtractor,
    KnowledgeGraph,
    annotate_gold_relevance,
)
from caff.utils.logging import setup_logging

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Annotate gold relevance.")
    p.add_argument("--kg",         required=True, help="merged_kg.tsv")
    p.add_argument("--qa-input",   required=True, help="Input QA JSON.")
    p.add_argument("--qa-output",  required=True, help="Output annotated JSON.")
    p.add_argument("--cache-dir",  default="cache/bfs",
                   help="BFS subgraph cache directory.")
    p.add_argument("--L",          type=int, default=3, help="Max BFS depth.")
    p.add_argument("--K-r",        type=int, default=20,
                   help="Frequency cap per (head, relation).")
    p.add_argument("--min-relation-freq", type=int, default=50)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    setup_logging(level="INFO")

    # ─── Load KG ─────────────────────────────────────────────────
    kg = KnowledgeGraph.from_tsv(
        args.kg, min_relation_freq=args.min_relation_freq
    )

    # ─── Load raw QA records ────────────────────────────────────
    qa_path = Path(args.qa_input)
    with qa_path.open("r", encoding="utf-8") as f:
        records = json.load(f)
    logger.info(f"Loaded {len(records):,} QA records from {qa_path}")

    # ─── Pre-compute BFS subgraphs (with cache) ─────────────────
    bfs = CachedBFSExtractor(
        kg, L=args.L, K_r=args.K_r, cache_dir=args.cache_dir
    )

    # ─── Annotate each record ───────────────────────────────────
    n_reachable = 0
    n_total_pos = 0
    n_total_cand = 0
    annotated_records: list[dict] = []

    for i, rec in enumerate(records):
        qid    = rec["query_id"]
        seeds  = rec.get("seeds", [])
        gold   = rec.get("gold_answer")
        if not seeds:
            logger.warning(f"  [{qid}] no seeds — skipping")
            continue

        bfs_data = bfs.extract(qid, seeds, gold_answer=gold)
        n_candidates = sum(len(C) for C in bfs_data["candidate_sets_cap"])
        n_pos = len(bfs_data["gold_positives"] or set())
        n_total_cand += n_candidates
        n_total_pos  += n_pos
        if n_pos > 0:
            n_reachable += 1

        out = dict(rec)
        out["n_candidates_per_hop"] = [
            len(C) for C in bfs_data["candidate_sets_cap"]
        ]
        out["n_gold_positives"] = n_pos
        annotated_records.append(out)

        if (i + 1) % 200 == 0:
            logger.info(
                f"  ... {i + 1:,}/{len(records):,}  "
                f"reachable={n_reachable:,}  "
                f"avg_cands={n_total_cand / (i + 1):.0f}  "
                f"avg_pos={n_total_pos / max(1, i + 1):.1f}"
            )

    # ─── Stats ──────────────────────────────────────────────────
    logger.info("─" * 60)
    logger.info(f"Annotation summary:")
    logger.info(f"  Records: {len(annotated_records):,}")
    logger.info(f"  With ≥1 gold-positive triple: {n_reachable:,} "
                f"({100 * n_reachable / max(1, len(annotated_records)):.1f}%)")
    logger.info(f"  Total candidate triples (post-FreqCap): {n_total_cand:,}")
    logger.info(f"  Total gold-positive triples:            {n_total_pos:,}")
    logger.info(f"  Paper §8.1 reports ≈3.7M labeled instances total")
    logger.info("─" * 60)

    # ─── Write output ───────────────────────────────────────────
    out_path = Path(args.qa_output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(annotated_records, f, indent=2)
    logger.info(f"Wrote {len(annotated_records):,} records to {out_path}")


if __name__ == "__main__":
    main()