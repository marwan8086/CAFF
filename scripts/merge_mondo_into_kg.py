"""
scripts/merge_mondo_into_kg.py — Merge MONDO into the existing KG v2.

Inputs:
    data/processed/merged_kg_v2.tsv          (Orphanet+HPO KG, ~291K edges)
    data/raw/mondo/mondo_terms.tsv           (MONDO id -> name)
    data/raw/mondo/mondo_hierarchy.tsv       (MONDO is_a)
    data/raw/mondo/mondo_xrefs.tsv           (MONDO -> Orphanet/OMIM/etc.)
    data/raw/orphanet/ORDO_names_en_2025.tsv (Orphanet code -> name)
    data/raw/hpo/hpoa_omim.tsv               (OMIM:code -> disease_name)

Output:
    data/processed/merged_kg_v3.tsv          (KG with MONDO hierarchy + xrefs)

What this adds to the KG:
    1. MONDO is_a edges (~40K): hub structure for disease hierarchy
    2. equivalent_to edges from MONDO terms to Orphanet diseases
       and to OMIM diseases (uses NAME on both sides for KG consistency)
    3. New nodes: MONDO disease names that did not appear in v2

Output schema (matches build_kg.py):
    head    relation    tail    head_cui    tail_cui    source

Usage:
    python scripts/merge_mondo_into_kg.py
"""
from __future__ import annotations

import logging
import sys
from collections import Counter
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("merge_mondo")


ROOT = Path(__file__).parent.parent
KG_IN = ROOT / "data" / "processed" / "merged_kg_v2.tsv"
KG_OUT = ROOT / "data" / "processed" / "merged_kg_v3.tsv"

MONDO_DIR = ROOT / "data" / "raw" / "mondo"
MONDO_TERMS = MONDO_DIR / "mondo_terms.tsv"
MONDO_HIERARCHY = MONDO_DIR / "mondo_hierarchy.tsv"
MONDO_XREFS = MONDO_DIR / "mondo_xrefs.tsv"

ORPHA_NAMES = ROOT / "data" / "raw" / "orphanet" / "ORDO_names_en_2025.tsv"
HPOA_OMIM = ROOT / "data" / "raw" / "hpo" / "hpoa_omim.tsv"

HEADER = "head\trelation\ttail\thead_cui\ttail_cui\tsource"


def load_existing_kg(path: Path) -> tuple[list[str], set[tuple[str, str, str]], set[str]]:
    """Returns (raw_lines, existing_(h,r,t)_set, existing_node_names)."""
    if not path.exists():
        logger.error(f"Existing KG missing: {path}")
        sys.exit(1)
    lines: list[str] = []
    seen: set[tuple[str, str, str]] = set()
    nodes: set[str] = set()
    with path.open("r", encoding="utf-8") as f:
        f.readline()  # header
        for line in f:
            line = line.rstrip("\n")
            if not line:
                continue
            cols = line.split("\t")
            if len(cols) < 3:
                continue
            h, r, t = cols[0], cols[1], cols[2]
            seen.add((h, r, t))
            nodes.add(h)
            nodes.add(t)
            lines.append(line)
    logger.info(
        f"Loaded existing KG: {len(lines):,} edges, "
        f"{len(nodes):,} unique nodes"
    )
    return lines, seen, nodes


def load_two_col_tsv(path: Path, sep: str = "\t") -> dict[str, str]:
    """Load a TSV with header into {col1: col2}."""
    out: dict[str, str] = {}
    if not path.exists():
        return out
    with path.open("r", encoding="utf-8") as f:
        f.readline()  # header
        for line in f:
            parts = line.rstrip("\n").split(sep)
            if len(parts) >= 2 and parts[0] and parts[1]:
                out[parts[0]] = parts[1]
    return out


def load_mondo_xrefs() -> tuple[dict[str, list[tuple[str, str]]], int]:
    """Returns {mondo_id: [(source, code), ...]} and total count."""
    out: dict[str, list[tuple[str, str]]] = {}
    n = 0
    if not MONDO_XREFS.exists():
        return out, 0
    with MONDO_XREFS.open("r", encoding="utf-8") as f:
        f.readline()  # header
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 4:
                continue
            mondo_id, _mondo_name, source, code = parts[0], parts[1], parts[2], parts[3]
            out.setdefault(mondo_id, []).append((source, code))
            n += 1
    return out, n


def load_omim_names() -> dict[str, str]:
    """Build {OMIM:code -> disease_name} from hpoa_omim.tsv."""
    mp: dict[str, str] = {}
    if not HPOA_OMIM.exists():
        return mp
    with HPOA_OMIM.open("r", encoding="utf-8") as f:
        f.readline()  # header
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) >= 2 and parts[0].startswith("OMIM:"):
                # parts: disease_id  disease_name  hpo_id  hpo_term
                mp[parts[0]] = parts[1]
    return mp


def emit_line(h: str, r: str, t: str, source: str) -> str:
    return f"{h}\t{r}\t{t}\t\t\t{source}"


def main() -> int:
    existing_lines, existing_set, existing_nodes = load_existing_kg(KG_IN)

    mondo_id_to_name = load_two_col_tsv(MONDO_TERMS)
    orpha_code_to_name = load_two_col_tsv(ORPHA_NAMES)
    omim_id_to_name = load_omim_names()
    mondo_xrefs, n_xrefs = load_mondo_xrefs()

    logger.info(f"MONDO terms loaded:        {len(mondo_id_to_name):,}")
    logger.info(f"Orphanet code->name:       {len(orpha_code_to_name):,}")
    logger.info(f"OMIM:code->disease_name:   {len(omim_id_to_name):,}")
    logger.info(f"MONDO xref entries:        {n_xrefs:,}")

    new_lines: list[str] = []

    # ─── 1. MONDO is_a edges ──────────────────────────────
    n_added_isa = 0
    n_dup_isa = 0
    if MONDO_HIERARCHY.exists():
        logger.info("Adding MONDO is_a edges ...")
        with MONDO_HIERARCHY.open("r", encoding="utf-8") as f:
            f.readline()  # header
            for line in f:
                parts = line.rstrip("\n").split("\t")
                if len(parts) < 4:
                    continue
                _child_id, child_name, _parent_id, parent_name = parts[:4]
                if not child_name or not parent_name:
                    continue
                head = child_name
                tail = parent_name
                relation = "is_a"
                if (head, relation, tail) in existing_set:
                    n_dup_isa += 1
                    continue
                existing_set.add((head, relation, tail))
                new_lines.append(emit_line(head, relation, tail, "mondo"))
                n_added_isa += 1
        logger.info(
            f"  Added {n_added_isa:,} MONDO is_a edges "
            f"(skipped {n_dup_isa:,} duplicates already in KG v2)"
        )

    # ─── 2. equivalent_to edges (MONDO -> Orphanet/OMIM) ──
    n_added_eq_orpha = 0
    n_added_eq_omim = 0
    n_skip_eq_dup = 0
    n_skip_eq_no_orpha_name = 0
    n_skip_eq_no_omim_name = 0

    logger.info("Adding equivalent_to edges (MONDO <-> Orphanet/OMIM) ...")
    for mondo_id, refs in mondo_xrefs.items():
        mondo_name = mondo_id_to_name.get(mondo_id)
        if not mondo_name:
            continue
        for source, code in refs:
            if source == "Orphanet":
                orpha_name = orpha_code_to_name.get(code)
                if not orpha_name:
                    n_skip_eq_no_orpha_name += 1
                    continue
                head = mondo_name
                tail = orpha_name
                relation = "equivalent_to"
                if (head, relation, tail) in existing_set:
                    n_skip_eq_dup += 1
                    continue
                existing_set.add((head, relation, tail))
                new_lines.append(emit_line(head, relation, tail, "mondo_orphanet"))
                n_added_eq_orpha += 1
            elif source == "OMIM":
                omim_full_id = f"OMIM:{code}"
                omim_name = omim_id_to_name.get(omim_full_id)
                if not omim_name:
                    n_skip_eq_no_omim_name += 1
                    continue
                head = mondo_name
                tail = omim_name
                relation = "equivalent_to"
                if (head, relation, tail) in existing_set:
                    n_skip_eq_dup += 1
                    continue
                existing_set.add((head, relation, tail))
                new_lines.append(emit_line(head, relation, tail, "mondo_omim"))
                n_added_eq_omim += 1

    logger.info(
        f"  Added {n_added_eq_orpha:,} MONDO->Orphanet equivalent_to edges "
        f"(skipped {n_skip_eq_no_orpha_name:,} missing Orphanet names)"
    )
    logger.info(
        f"  Added {n_added_eq_omim:,} MONDO->OMIM equivalent_to edges "
        f"(skipped {n_skip_eq_no_omim_name:,} missing OMIM names)"
    )
    logger.info(
        f"  Skipped {n_skip_eq_dup:,} duplicate equivalent_to edges"
    )

    # ─── 3. Write merged_kg_v3.tsv ────────────────────────
    KG_OUT.parent.mkdir(parents=True, exist_ok=True)
    with KG_OUT.open("w", encoding="utf-8") as f:
        f.write(HEADER + "\n")
        for line in existing_lines:
            f.write(line + "\n")
        for line in new_lines:
            f.write(line + "\n")

    n_old = len(existing_lines)
    n_new = len(new_lines)
    n_total = n_old + n_new

    nodes: set[str] = set()
    rel_counts: Counter[str] = Counter()
    for line in existing_lines:
        cols = line.split("\t")
        if len(cols) >= 3:
            nodes.add(cols[0])
            nodes.add(cols[2])
            rel_counts[cols[1]] += 1
    for line in new_lines:
        cols = line.split("\t")
        if len(cols) >= 3:
            nodes.add(cols[0])
            nodes.add(cols[2])
            rel_counts[cols[1]] += 1

    logger.info("=" * 60)
    logger.info(" Merge complete")
    logger.info("=" * 60)
    logger.info(f"  Output: {KG_OUT}")
    logger.info(
        f"  Edges:    {n_old:,} (v2) + {n_new:,} (MONDO) = {n_total:,}"
    )
    logger.info(f"  Entities: {len(nodes):,}")
    logger.info(f"  Relations: {len(rel_counts)}")
    logger.info(f"  Top relations:")
    for r, c in rel_counts.most_common(10):
        logger.info(f"    {r:50s}  {c:>10,}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
