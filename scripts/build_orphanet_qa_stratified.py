"""
scripts/build_orphanet_qa_stratified.py - Stratified QA sampler.

Unlike the uniform BFS sampler in build_orphanet_qa.py (which picks a
random head and lets the relation distribution emerge naturally), this
sampler enforces a target relation-bucket ratio:

  - 50% gene-disease relations (the relations a clinician cares about)
  - 35% has_phenotype  (HPO phenotype links)
  - 15% is_a           (ontology backbone)

Within the gene-disease bucket, it enforces a sub-ratio:

  - 30% OTG-added relation (gene_associated_with_disease_otg)
  - 70% original Orphanet gene-disease relations

This is the experiment proposed at the end of Section 16: lift the
OTG share of QA records from 0.23% (in the natural sample) to ~15%
(50% gene-disease share x 30% OTG share). With that lift, OTG-related
records become a meaningful slice of the test set and KG enrichment
becomes evaluable.

Algorithm:
  - Pre-bucket KG edges by relation family.
  - Pick a target bucket per record by round-robin.
  - Within a target bucket, pick a (head, relation, tail) triple.
  - Then build a hop-1/2/3 path that ends on that triple's tail.
    - hop=1: the triple itself.
    - hop=2: any predecessor of the head -> head -> tail.
    - hop=3: predecessor of predecessor -> ... -> tail.

The output JSON keeps the same schema as build_orphanet_qa.py so
train.py loads it without changes.
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
logger = logging.getLogger("build_qa_stratified")


# Relation buckets
GENE_DISEASE_RELATIONS = {
    "disease_causing_germline_mutation_s_in",
    "disease_causing_germline_mutation_s_loss_of_function_in",
    "major_susceptibility_factor_in",
    "candidate_gene_tested_in",
    "role_in_the_phenotype_of",
    "part_of_a_fusion_gene_in",
    "disease_causing_somatic_mutation_s_in",
    "disease_causing_germline_mutation_s_gain_of_function_in",
    "modifying_germline_mutation_in",
}
OTG_RELATION = "gene_associated_with_disease_otg"
PHENOTYPE_RELATION = "has_phenotype"
ISA_RELATION = "is_a"


# Same templates as the original sampler so wording is identical.
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


def load_kg_tsv(path: Path):
    """Load KG and return (DiGraph, edges_by_relation_bucket)."""
    logger.info(f"Loading KG from {path} ...")
    g = nx.DiGraph()

    # edges_by_bucket[bucket] = list of (head, relation, tail)
    edges_by_bucket: dict[str, list[tuple[str, str, str]]] = {
        "otg": [],
        "gene_disease_orphanet": [],
        "has_phenotype": [],
        "is_a": [],
        "other": [],
    }

    with path.open("r", encoding="utf-8") as f:
        header = f.readline().strip().split("\t")
        try:
            i_head = header.index("head")
            i_rel = header.index("relation")
            i_tail = header.index("tail")
        except ValueError:
            logger.error(f"KG TSV header missing required columns: {header}")
            raise

        for line in f:
            cols = line.rstrip("\n").split("\t")
            if len(cols) <= max(i_head, i_rel, i_tail):
                continue
            h, r, t = cols[i_head], cols[i_rel], cols[i_tail]
            if not h or not r or not t:
                continue
            g.add_edge(h, t, relation=r)

            if r == OTG_RELATION:
                bucket = "otg"
            elif r in GENE_DISEASE_RELATIONS:
                bucket = "gene_disease_orphanet"
            elif r == PHENOTYPE_RELATION:
                bucket = "has_phenotype"
            elif r == ISA_RELATION:
                bucket = "is_a"
            else:
                bucket = "other"
            edges_by_bucket[bucket].append((h, r, t))

    logger.info(
        f"  KG loaded: |V|={g.number_of_nodes():,}  "
        f"|E|={g.number_of_edges():,}"
    )
    logger.info(f"  Edges by bucket:")
    for b, edges in edges_by_bucket.items():
        logger.info(f"    {b}: {len(edges):,}")

    return g, edges_by_bucket


def find_path_ending_at(g, head, tail, target_hop, rng, max_tries=20):
    """Find a path of exactly target_hop edges from some seed to tail,
    such that the LAST edge is (head, _, tail).

    Returns (seed, relation_of_last_edge) or None if no path found.
    """
    if target_hop == 1:
        rel = g.edges[head, tail].get("relation", "associated_with")
        return head, rel

    if target_hop == 2:
        # Pick a predecessor of head
        preds = list(g.predecessors(head))
        if not preds:
            return None
        seed = rng.choice(preds)
        rel = g.edges[head, tail].get("relation", "associated_with")
        return seed, rel

    # target_hop == 3: pick a pred-of-pred
    for _ in range(max_tries):
        preds1 = list(g.predecessors(head))
        if not preds1:
            return None
        p1 = rng.choice(preds1)
        preds2 = list(g.predecessors(p1))
        if not preds2:
            continue
        seed = rng.choice(preds2)
        rel = g.edges[head, tail].get("relation", "associated_with")
        return seed, rel

    return None


def sample_one_record(
    g, edges_by_bucket, target_bucket, target_hop, rng, record_idx
):
    """Sample one QA record for a target bucket and hop."""
    bucket_edges = edges_by_bucket.get(target_bucket, [])
    if not bucket_edges:
        return None

    # Try a few edges until one gives us a valid path
    for _ in range(50):
        h, r, t = rng.choice(bucket_edges)
        result = find_path_ending_at(g, h, t, target_hop, rng)
        if result is None:
            continue
        seed, rel = result
        return {
            "query_id": f"orph_{record_idx:05d}",
            "question": question_for(rel, seed),
            "seeds": [seed],
            "gold_answer": t,
            "answer_label": "yes",
            "_meta": {
                "hop": target_hop,
                "final_relation": rel,
                "bucket": target_bucket,
            },
        }
    return None


def sample_stratified(
    g, edges_by_bucket, n_target, rng,
    bucket_targets, hop_targets
):
    """Sample QA records honoring a target bucket distribution.

    bucket_targets: dict mapping bucket name to share (sums to 1.0)
    hop_targets:    dict mapping hop (1,2,3) to share (sums to 1.0)
    """
    # Compute target counts per (bucket, hop)
    counts = {}
    for bname, bshare in bucket_targets.items():
        for hop, hshare in hop_targets.items():
            counts[(bname, hop)] = int(round(n_target * bshare * hshare))

    # Adjust for rounding (make sure total == n_target)
    actual_total = sum(counts.values())
    if actual_total < n_target:
        # Add remainder to the largest cell
        key = max(counts, key=counts.get)
        counts[key] += n_target - actual_total

    logger.info(f"  Target counts per (bucket, hop):")
    for (bname, hop), cnt in sorted(counts.items()):
        logger.info(f"    {bname:<25} hop={hop} -> {cnt}")

    records = []
    attempts = 0

    for (target_bucket, target_hop), need in counts.items():
        produced = 0
        local_attempts = 0
        while produced < need and local_attempts < need * 100:
            local_attempts += 1
            attempts += 1
            rec = sample_one_record(
                g, edges_by_bucket, target_bucket, target_hop,
                rng, len(records)
            )
            if rec is None:
                continue
            records.append(rec)
            produced += 1

        if produced < need:
            logger.warning(
                f"  Bucket {target_bucket} hop={target_hop}: "
                f"got {produced}/{need}"
            )

    logger.info(
        f"  Sampled {len(records):,}/{n_target:,} records "
        f"(after {attempts:,} attempts)"
    )
    return records


def split_records(records, rng, train_frac=0.70, dev_frac=0.15):
    """Same 70/15/15 split as build_orphanet_qa.py."""
    shuffled = records.copy()
    rng.shuffle(shuffled)
    n = len(shuffled)
    n_train = int(train_frac * n)
    n_dev = int(dev_frac * n)
    train = shuffled[:n_train]
    dev = shuffled[n_train:n_train + n_dev]
    test = shuffled[n_train + n_dev:]
    for split_name, split in (("train", train), ("dev", dev), ("test", test)):
        for i, rec in enumerate(split):
            rec["query_id"] = f"orph_{split_name}_{i:05d}"
    return train, dev, test


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--kg", default="data/processed/merged_kg_v3.tsv")
    p.add_argument("--out-dir", default="data/processed")
    p.add_argument("--n", type=int, default=20000)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    kg_path = Path(args.kg)
    out_dir = Path(args.out_dir)
    if not kg_path.exists():
        logger.error(f"KG file not found: {kg_path}")
        return 1
    out_dir.mkdir(parents=True, exist_ok=True)

    rng = random.Random(args.seed)

    g, edges_by_bucket = load_kg_tsv(kg_path)

    # Bucket targets: 50% gene-disease (of which 30% OTG, 70% Orphanet),
    #                 35% has_phenotype, 15% is_a
    bucket_targets = {
        "otg":                   0.50 * 0.30,  # 15%
        "gene_disease_orphanet": 0.50 * 0.70,  # 35%
        "has_phenotype":         0.35,
        "is_a":                  0.15,
    }
    assert abs(sum(bucket_targets.values()) - 1.0) < 1e-6

    hop_targets = {1: 1/3, 2: 1/3, 3: 1/3}

    logger.info("Bucket targets:")
    for b, s in bucket_targets.items():
        logger.info(f"  {b}: {s*100:.1f}%")

    records = sample_stratified(
        g, edges_by_bucket, args.n, rng, bucket_targets, hop_targets
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

    # Sanity checks
    hop_counts = defaultdict(int)
    bucket_counts = defaultdict(int)
    rel_counts = defaultdict(int)
    for rec in records:
        hop_counts[rec["_meta"]["hop"]] += 1
        bucket_counts[rec["_meta"]["bucket"]] += 1
        rel_counts[rec["_meta"]["final_relation"]] += 1

    logger.info(
        f"Hop distribution: " +
        ", ".join(f"hop{h}={hop_counts[h]:,}" for h in sorted(hop_counts))
    )
    logger.info(f"Bucket distribution:")
    for b, c in sorted(bucket_counts.items(), key=lambda kv: -kv[1]):
        logger.info(f"  {b}: {c:,} ({100*c/len(records):.1f}%)")

    logger.info(f"Top final relations:")
    top_rels = sorted(rel_counts.items(), key=lambda kv: -kv[1])[:8]
    for r, c in top_rels:
        marker = "  <- OTG" if r == OTG_RELATION else ""
        logger.info(f"  {r}: {c:,} ({100*c/len(records):.2f}%){marker}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
