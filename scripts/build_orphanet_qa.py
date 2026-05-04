"""
scripts/build_orphanet_qa.py — Generate QA records from the real
Orphanet KG (data/processed/merged_kg.tsv).

Each record looks like:
    {
      "query_id":    "orph_train_0001",
      "question":    "What gene is associated with <Disease X>?",
      "seeds":       ["<Disease X>"],
      "gold_answer": "<GeneSymbol>",
      "answer_label": "yes"
    }

Sampling protocol:
  - Walk the directed KG (matches trainer kg.adj BFS)
  - Sample (seed, gold) pairs at distances in {1, 2, 3}
  - Distribute hops uniformly: ~33% each of 1-hop, 2-hop, 3-hop
  - Generate question template based on relation type

Output: data/processed/{train,dev,test}.json
        70/15/15 split (deterministic via seed=42)

Usage:
    python scripts/build_orphanet_qa.py
        [--kg data/processed/merged_kg.tsv]
        [--out-dir data/processed]
        [--n 5000]
"""
from __future__ import annotations

import argparse
import json
import logging
import random
import sys
from collections import defaultdict
from pathlib import Path

import networkx as nx

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("build_orphanet_qa")


# Question templates per relation. Falls back to a generic template
# for unknown relations.
QUESTION_TEMPLATES: dict[str, str] = {
    "Disease-causing germline mutation(s) in":
        "What gene has disease-causing germline mutations associated with {seed}?",
    "Disease-causing somatic mutation(s) in":
        "What gene has disease-causing somatic mutations associated with {seed}?",
    "Major susceptibility factor in":
        "What gene is a major susceptibility factor for {seed}?",
    "Modifying germline mutation in":
        "What gene contains modifying germline mutations affecting {seed}?",
    "Part of a fusion gene in":
        "What gene participates in a fusion gene linked to {seed}?",
    "Role in the phenotype of":
        "What gene plays a role in the phenotype of {seed}?",
    "Candidate gene tested in":
        "What gene is a candidate tested for involvement in {seed}?",
    "associated_with":
        "What entity is associated with {seed}?",
    "has_phenotype":
        "What phenotype is associated with {seed}?",
}


def question_for(relation: str, seed_name: str) -> str:
    template = QUESTION_TEMPLATES.get(
        relation, "What is associated with {seed}?"
    )
    return template.format(seed=seed_name)


def load_kg_tsv(path: Path) -> tuple[nx.DiGraph, set[str], set[str]]:
    """Load KG and return (DiGraph, all_entities, head_entities).

    head_entities are nodes that have outgoing edges -> usable as seeds.
    """
    logger.info(f"Loading KG from {path} ...")
    g = nx.DiGraph()
    all_entities: set[str] = set()
    head_entities: set[str] = set()

    with path.open("r", encoding="utf-8") as f:
        header = f.readline().strip().split("\t")
        # Expected columns: head, relation, tail, head_cui, tail_cui, source
        # We need at least head, relation, tail.
        try:
            i_head = header.index("head")
            i_rel = header.index("relation")
            i_tail = header.index("tail")
        except ValueError:
            logger.error(f"KG TSV header missing required columns: {header}")
            raise

        n_lines = 0
        for line in f:
            cols = line.rstrip("\n").split("\t")
            if len(cols) <= max(i_head, i_rel, i_tail):
                continue
            h, r, t = cols[i_head], cols[i_rel], cols[i_tail]
            if not h or not r or not t:
                continue
            g.add_edge(h, t, relation=r)
            all_entities.add(h)
            all_entities.add(t)
            head_entities.add(h)
            n_lines += 1

    logger.info(
        f"  KG loaded: |V|={g.number_of_nodes():,}  "
        f"|E|={g.number_of_edges():,}  "
        f"(read {n_lines:,} edge rows)"
    )
    return g, all_entities, head_entities


def sample_qa_records(
    g: nx.DiGraph,
    head_entities: set[str],
    n_target: int,
    rng: random.Random,
    hop_distribution: tuple[int, int, int] = (1, 2, 3),
) -> list[dict]:
    """Sample (seed, gold) pairs spread across hops 1..3.

    Strategy:
      - Pick a random head entity.
      - Pick a target hop in {1, 2, 3} round-robin.
      - BFS up to that hop; collect candidate tails at exactly that depth.
      - Pick one tail uniformly. Record the relation of the LAST edge.
    """
    head_list = sorted(head_entities)
    if not head_list:
        return []

    records: list[dict] = []
    attempts = 0
    max_attempts = n_target * 50

    while len(records) < n_target and attempts < max_attempts:
        attempts += 1
        seed = rng.choice(head_list)
        target_hop = hop_distribution[len(records) % len(hop_distribution)]

        # BFS from seed, capped at target_hop
        # frontier[d] = set of nodes reachable at exactly depth d
        frontier: list[set[str]] = [{seed}]
        visited = {seed}
        for d in range(1, target_hop + 1):
            next_layer: set[str] = set()
            for node in frontier[d - 1]:
                for tail in g.successors(node):
                    if tail not in visited:
                        next_layer.add(tail)
            visited.update(next_layer)
            frontier.append(next_layer)
            if not next_layer:
                break

        if len(frontier) <= target_hop or not frontier[target_hop]:
            continue

        gold = rng.choice(sorted(frontier[target_hop]))

        # Get the relation of the FINAL edge (predecessor of gold at depth target_hop-1)
        # Find a parent of gold that is in frontier[target_hop - 1].
        parents_in_layer = [
            p for p in g.predecessors(gold)
            if p in frontier[target_hop - 1]
        ]
        if not parents_in_layer:
            continue
        parent = rng.choice(parents_in_layer)
        relation = g.edges[parent, gold].get("relation", "associated_with")

        records.append({
            "query_id": f"orph_{len(records):05d}",
            "question": question_for(relation, seed),
            "seeds": [seed],
            "gold_answer": gold,
            "answer_label": "yes",
            "_meta": {
                "hop": target_hop,
                "final_relation": relation,
            },
        })

    logger.info(
        f"  Sampled {len(records):,} QA records "
        f"(after {attempts:,} attempts)"
    )
    if len(records) < n_target:
        logger.warning(
            f"  Reached only {len(records):,} of {n_target:,} target."
        )
    return records


def split_records(
    records: list[dict],
    rng: random.Random,
    train_frac: float = 0.70,
    dev_frac: float = 0.15,
) -> tuple[list[dict], list[dict], list[dict]]:
    """Deterministic 70/15/15 split."""
    shuffled = records.copy()
    rng.shuffle(shuffled)
    n = len(shuffled)
    n_train = int(train_frac * n)
    n_dev = int(dev_frac * n)
    train = shuffled[:n_train]
    dev = shuffled[n_train:n_train + n_dev]
    test = shuffled[n_train + n_dev:]
    # Re-id within each split for clarity
    for split_name, split in (("train", train), ("dev", dev), ("test", test)):
        for i, rec in enumerate(split):
            rec["query_id"] = f"orph_{split_name}_{i:05d}"
    return train, dev, test


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--kg", default="data/processed/merged_kg.tsv")
    p.add_argument("--out-dir", default="data/processed")
    p.add_argument("--n", type=int, default=5000,
                   help="Total QA records to generate (across all splits).")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    kg_path = Path(args.kg)
    out_dir = Path(args.out_dir)
    if not kg_path.exists():
        logger.error(f"KG file not found: {kg_path}")
        return 1
    out_dir.mkdir(parents=True, exist_ok=True)

    rng = random.Random(args.seed)

    g, _, head_entities = load_kg_tsv(kg_path)
    if not head_entities:
        logger.error("KG has no entities with outgoing edges; cannot sample.")
        return 1

    records = sample_qa_records(
        g, head_entities, n_target=args.n, rng=rng
    )
    if not records:
        logger.error("No QA records generated.")
        return 1

    train, dev, test = split_records(records, rng)

    for name, split in (("train", train), ("dev", dev), ("test", test)):
        out_path = out_dir / f"{name}.json"
        with out_path.open("w", encoding="utf-8") as f:
            json.dump(split, f, indent=2, ensure_ascii=False)
        logger.info(f"  Wrote {out_path} ({len(split):,} records)")

    # Print hop distribution for sanity
    hop_counts = defaultdict(int)
    for rec in records:
        hop_counts[rec["_meta"]["hop"]] += 1
    logger.info(
        f"Hop distribution: " +
        ", ".join(f"hop{h}={hop_counts[h]:,}" for h in sorted(hop_counts))
    )

    rel_counts = defaultdict(int)
    for rec in records:
        rel_counts[rec["_meta"]["final_relation"]] += 1
    top_rels = sorted(rel_counts.items(), key=lambda kv: -kv[1])[:5]
    logger.info(
        f"Top final relations: " +
        ", ".join(f"{r}={c:,}" for r, c in top_rels)
    )

    return 0


if __name__ == "__main__":
    sys.exit(main())
