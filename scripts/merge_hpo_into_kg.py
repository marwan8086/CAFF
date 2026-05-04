"""
scripts/merge_hpo_into_kg.py — Merge HPO data into the existing KG.

Inputs:
    data/processed/merged_kg.tsv          (Orphanet KG, ~124K edges)
    data/raw/hpo/hp_hierarchy.tsv         (HPO is_a relations)
    data/raw/hpo/hpoa_omim.tsv            (OMIM disease -> HPO)

Output:
    data/processed/merged_kg_v2.tsv       (expanded KG)

The output schema matches build_kg.py output:
    head    relation    tail    head_cui    tail_cui    source

Where head_cui / tail_cui are left empty for HPO additions
(no UMLS normalization in this pass — paper plans Phase 4 for that).

Usage:
    python scripts/merge_hpo_into_kg.py
"""
from __future__ import annotations

import logging
import sys
from collections import Counter, defaultdict
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("merge_hpo")


ROOT = Path(__file__).parent.parent
KG_IN = ROOT / "data" / "processed" / "merged_kg.tsv"
HPO_DIR = ROOT / "data" / "raw" / "hpo"
HIERARCHY = HPO_DIR / "hp_hierarchy.tsv"
OMIM_TSV = HPO_DIR / "hpoa_omim.tsv"
KG_OUT = ROOT / "data" / "processed" / "merged_kg_v2.tsv"

# Column convention from build_kg.py output:
HEADER = "head\trelation\ttail\thead_cui\ttail_cui\tsource"


def load_existing_kg(path: Path) -> tuple[list[str], set[tuple[str, str, str]]]:
    """Return (raw_lines_minus_header, set_of_existing_(h,r,t))."""
    if not path.exists():
        logger.error(f"Existing KG missing: {path}")
        sys.exit(1)
    lines: list[str] = []
    seen: set[tuple[str, str, str]] = set()
    with path.open("r", encoding="utf-8") as f:
        header = f.readline().rstrip("\n")
        for line in f:
            line = line.rstrip("\n")
            if not line:
                continue
            cols = line.split("\t")
            if len(cols) < 3:
                continue
            h, r, t = cols[0], cols[1], cols[2]
            seen.add((h, r, t))
            lines.append(line)
    logger.info(
        f"Loaded existing KG: {len(lines):,} edges "
        f"(header: {header})"
    )
    return lines, seen


def load_hpo_terms_map() -> dict[str, str]:
    """Build hpo_id -> name map from hp_terms.tsv."""
    terms_path = HPO_DIR / "hp_terms.tsv"
    mp: dict[str, str] = {}
    if not terms_path.exists():
        return mp
    with terms_path.open("r", encoding="utf-8") as f:
        f.readline()  # header
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) >= 2:
                mp[parts[0]] = parts[1]
    return mp


def emit_line(h: str, r: str, t: str, source: str) -> str:
    # head_cui and tail_cui are empty (no UMLS resolution here)
    return f"{h}\t{r}\t{t}\t\t\t{source}"


def main() -> int:
    existing_lines, existing_set = load_existing_kg(KG_IN)
    hpo_names = load_hpo_terms_map()
    logger.info(f"Loaded {len(hpo_names):,} HPO term names")

    # ─── Merge HPO hierarchy ───────────────────────────────
    new_lines: list[str] = []
    n_skip_dup = 0

    if HIERARCHY.exists():
        logger.info("Merging HPO hierarchy (is_a) ...")
        n_seen = 0
        with HIERARCHY.open("r", encoding="utf-8") as f:
            f.readline()  # header
            for line in f:
                parts = line.rstrip("\n").split("\t")
                if len(parts) < 4:
                    continue
                child_id, child_name, parent_id, parent_name = parts[:4]
                # Use names as nodes (consistent with Orphanet style:
                # disease names are nodes, not codes).
                head = child_name or child_id
                tail = parent_name or parent_id
                relation = "is_a"
                if (head, relation, tail) in existing_set:
                    n_skip_dup += 1
                    continue
                existing_set.add((head, relation, tail))
                new_lines.append(emit_line(head, relation, tail, "hpo"))
                n_seen += 1
        logger.info(f"  Added {n_seen:,} is_a edges (skipped {n_skip_dup:,} dup)")

    # ─── Merge HPOA OMIM annotations ───────────────────────
    if OMIM_TSV.exists():
        logger.info("Merging HPOA OMIM annotations (has_phenotype) ...")
        n_added = 0
        n_dup = 0
        with OMIM_TSV.open("r", encoding="utf-8") as f:
            f.readline()  # header
            for line in f:
                parts = line.rstrip("\n").split("\t")
                if len(parts) < 4:
                    continue
                disease_id, disease_name, hpo_id, hpo_term = parts[:4]
                # Use disease_name as head (text), hpo_term as tail (text)
                head = disease_name or disease_id
                tail = hpo_term or hpo_id
                relation = "has_phenotype"
                if (head, relation, tail) in existing_set:
                    n_dup += 1
                    continue
                existing_set.add((head, relation, tail))
                new_lines.append(emit_line(head, relation, tail, "hpoa_omim"))
                n_added += 1
        logger.info(f"  Added {n_added:,} has_phenotype edges (skipped {n_dup:,} dup)")

    # ─── Write merged_kg_v2.tsv ────────────────────────────
    KG_OUT.parent.mkdir(parents=True, exist_ok=True)
    with KG_OUT.open("w", encoding="utf-8") as f:
        f.write(HEADER + "\n")
        for line in existing_lines:
            f.write(line + "\n")
        for line in new_lines:
            f.write(line + "\n")

    # Quick stats on output
    n_old = len(existing_lines)
    n_new = len(new_lines)
    n_total = n_old + n_new

    # Count entities and relations in the merged file
    nodes: set[str] = set()
    rel_counts: Counter[str] = Counter()
    for line in existing_lines:
        cols = line.split("\t")
        if len(cols) >= 3:
            nodes.add(cols[0]); nodes.add(cols[2]); rel_counts[cols[1]] += 1
    for line in new_lines:
        cols = line.split("\t")
        if len(cols) >= 3:
            nodes.add(cols[0]); nodes.add(cols[2]); rel_counts[cols[1]] += 1

    logger.info("=" * 60)
    logger.info(" Merge complete")
    logger.info("=" * 60)
    logger.info(f"  Output: {KG_OUT}")
    logger.info(f"  Edges:    {n_old:,} (Orphanet) + {n_new:,} (HPO) = {n_total:,}")
    logger.info(f"  Entities: {len(nodes):,}")
    logger.info(f"  Relations: {len(rel_counts)}")
    top = rel_counts.most_common(10)
    logger.info(f"  Top relations:")
    for r, c in top:
        logger.info(f"    {r:50s}  {c:>10,}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
