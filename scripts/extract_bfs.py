"""
scripts/extract_bfs.py
======================
Stand-alone BFS dumper.
Pre-computes BFS reachability from query seeds for an entire QA split
and saves the result as a cache file. This avoids re-running BFS for
every training epoch.
Usage:
    python scripts/extract_bfs.py \
        --kg          data/processed/merged_kg.tsv \
        --qa-input    data/processed/train.json \
        --output      cache/bfs/train_bfs.pt \
        --max-hops    3 \
        --min-relation-freq 50
The output file is a torch.save'd dict mapping query_id -> {
    "hops":           dict[int, list[(h, r, t)]],
    "gold_reachable": bool,
    "gold_hop":       int | None,
    "n_triples":      int,
}
"""
from __future__ import annotations
import argparse
import json
import logging
import sys
import time
from collections import deque
from pathlib import Path
import torch
# Allow running this script from anywhere
sys.path.insert(0, str(Path(__file__).parent.parent))
from caff.data import KnowledgeGraph
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
)
logger = logging.getLogger("extract_bfs")
def bfs_extract(
    kg: KnowledgeGraph,
    seeds: list,
    max_hops: int,
    gold_answer: str | None = None,
) -> dict:
    """Run BFS from `seeds` up to `max_hops` and return triples per hop."""
    visited = set(s for s in seeds if s in kg.entity_to_idx)
    frontier = deque((s, 0) for s in visited)
    hops = {h: [] for h in range(1, max_hops + 1)}
    gold_hop = None
    while frontier:
        node, depth = frontier.popleft()
        if depth >= max_hops:
            continue
        next_depth = depth + 1
        for relation, tail in kg.adj.get(node, []):
            hops[next_depth].append((node, relation, tail))
            if gold_answer is not None and tail == gold_answer and gold_hop is None:
                gold_hop = next_depth
            if tail not in visited:
                visited.add(tail)
                frontier.append((tail, next_depth))
    n_triples = sum(len(v) for v in hops.values())
    return {
        "hops": hops,
        "gold_reachable": gold_hop is not None,
        "gold_hop": gold_hop,
        "n_triples": n_triples,
    }
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract BFS reachability for a QA split.",
    )
    parser.add_argument("--kg", required=True, help="Path to KG TSV file.")
    parser.add_argument("--qa-input", required=True, help="Path to QA JSON file.")
    parser.add_argument("--output", required=True, help="Path to save cache (.pt).")
    parser.add_argument("--max-hops", type=int, default=3, help="Max BFS depth.")
    parser.add_argument(
        "--min-relation-freq",
        type=int,
        default=50,
        help="Filter relations with fewer than this many triples (paper section 8.1).",
    )
    args = parser.parse_args()
    logger.info("Loading KG from %s ...", args.kg)
    kg = KnowledgeGraph.from_tsv(args.kg, min_relation_freq=args.min_relation_freq)
    logger.info("Loading QA records from %s ...", args.qa_input)
    with open(args.qa_input, encoding="utf-8") as f:
        records = json.load(f)
    logger.info("Loaded %d QA records", len(records))
    cache = {}
    n_reachable = 0
    n_skipped = 0
    t0 = time.time()
    for i, rec in enumerate(records):
        qid = rec["query_id"]
        seeds = rec.get("seeds", [])
        gold = rec.get("gold_answer")
        if not seeds:
            n_skipped += 1
            continue
        result = bfs_extract(kg, seeds, args.max_hops, gold_answer=gold)
        cache[qid] = result
        if result["gold_reachable"]:
            n_reachable += 1
        if (i + 1) % 100 == 0:
            elapsed = time.time() - t0
            rate = (i + 1) / max(elapsed, 1e-6)
            logger.info(
                "  processed %d/%d records (%.1f rec/s)",
                i + 1, len(records), rate,
            )
    elapsed = time.time() - t0
    logger.info(
        "BFS extraction complete: %d records in %.1fs",
        len(cache), elapsed,
    )
    logger.info(
        "  Gold-reachable: %d/%d (%.1f%%)",
        n_reachable, len(cache), 100.0 * n_reachable / max(len(cache), 1),
    )
    logger.info("  Skipped (no seeds): %d", n_skipped)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(cache, output_path)
    logger.info(
        "Saved cache to %s (%.1f MB)",
        output_path, output_path.stat().st_size / 1024 / 1024,
    )
if __name__ == "__main__":
    main()
