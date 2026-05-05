"""
scripts/convert_mondo_to_tsv.py — Convert MONDO ontology to TSV files.

Reads:
    data/raw/mondo/mondo.obo

Writes:
    data/raw/mondo/mondo_terms.tsv      (MONDO id -> name)
    data/raw/mondo/mondo_hierarchy.tsv  (child MONDO -> parent MONDO is_a)
    data/raw/mondo/mondo_xrefs.tsv      (MONDO id -> Orphanet/OMIM/DOID code)

Filtering:
    - Only [Term] entries with id starting with "MONDO:" (skip BFO, CHEBI, etc.)
    - Skip obsolete terms
    - is_a only kept if parent is also MONDO:
    - xrefs only kept for Orphanet:, OMIM:, DOID: (the ones we can use to merge)

Usage:
    python scripts/convert_mondo_to_tsv.py
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("mondo_to_tsv")


ROOT = Path(__file__).parent.parent
MONDO_DIR = ROOT / "data" / "raw" / "mondo"
MONDO_OBO = MONDO_DIR / "mondo.obo"

OUT_TERMS = MONDO_DIR / "mondo_terms.tsv"
OUT_HIERARCHY = MONDO_DIR / "mondo_hierarchy.tsv"
OUT_XREFS = MONDO_DIR / "mondo_xrefs.tsv"

# Cross-reference sources we care about (used to merge with existing KG).
ALLOWED_XREF_SOURCES = ("Orphanet", "OMIM", "DOID", "MESH", "ICD10CM", "UMLS")


def parse_obo(path: Path) -> tuple[
    dict[str, str],
    list[tuple[str, str]],
    list[tuple[str, str, str]],
]:
    """Parse mondo.obo.

    Returns:
        id_to_name: {MONDO:id -> name}
        is_a_pairs: [(child_id, parent_id), ...] (only MONDO->MONDO)
        xrefs:     [(mondo_id, source, code), ...]
    """
    if not path.exists():
        logger.error(f"Missing file: {path}")
        return {}, [], []

    logger.info(f"Parsing {path.name} ({path.stat().st_size / 1024 / 1024:.1f} MB) ...")

    id_to_name: dict[str, str] = {}
    is_a_pairs: list[tuple[str, str]] = []
    xrefs: list[tuple[str, str, str]] = []

    current_id: str | None = None
    current_name: str | None = None
    current_is_a: list[str] = []
    current_xrefs: list[tuple[str, str]] = []
    current_obsolete = False
    in_term = False
    is_mondo_term = False

    n_total_terms = 0
    n_mondo_terms = 0
    n_obsolete = 0
    n_other_terms = 0

    def flush() -> None:
        nonlocal n_mondo_terms, n_obsolete, n_other_terms, n_total_terms
        n_total_terms += 1
        if current_obsolete:
            n_obsolete += 1
            return
        if not is_mondo_term:
            n_other_terms += 1
            return
        if current_id and current_name:
            id_to_name[current_id] = current_name
            for parent in current_is_a:
                # Only keep MONDO -> MONDO is_a edges
                if parent.startswith("MONDO:"):
                    is_a_pairs.append((current_id, parent))
            for source, code in current_xrefs:
                xrefs.append((current_id, source, code))
            n_mondo_terms += 1

    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if line == "[Term]":
                if in_term:
                    flush()
                # Reset
                current_id = None
                current_name = None
                current_is_a = []
                current_xrefs = []
                current_obsolete = False
                is_mondo_term = False
                in_term = True
                continue
            if line.startswith("[") and line.endswith("]"):
                # Other stanza ([Typedef] etc) - flush and stop
                if in_term:
                    flush()
                in_term = False
                continue
            if not in_term:
                continue

            if line.startswith("id: "):
                current_id = line[4:].strip()
                if current_id.startswith("MONDO:"):
                    is_mondo_term = True
            elif line.startswith("name: "):
                current_name = line[6:].strip()
            elif line.startswith("is_a: "):
                # "is_a: MONDO:XXX ! parent name"
                rest = line[6:].strip()
                parent_id = rest.split(" ")[0].split("!")[0].strip()
                current_is_a.append(parent_id)
            elif line.startswith("xref: "):
                # "xref: Orphanet:377788 {source=...}"
                rest = line[6:].strip()
                # Strip annotation in {}
                code_part = rest.split("{")[0].strip()
                if ":" not in code_part:
                    continue
                source, code = code_part.split(":", 1)
                source = source.strip()
                code = code.strip()
                if source in ALLOWED_XREF_SOURCES and code:
                    current_xrefs.append((source, code))
            elif line.startswith("is_obsolete: true"):
                current_obsolete = True

    if in_term:
        flush()

    logger.info(
        f"  Total [Term] stanzas: {n_total_terms:,}"
    )
    logger.info(
        f"  MONDO terms kept:    {n_mondo_terms:,}"
    )
    logger.info(
        f"  Obsolete dropped:    {n_obsolete:,}"
    )
    logger.info(
        f"  Non-MONDO dropped:   {n_other_terms:,} (BFO, CHEBI, GO, etc.)"
    )
    logger.info(
        f"  is_a (MONDO->MONDO): {len(is_a_pairs):,}"
    )
    logger.info(
        f"  xrefs kept:          {len(xrefs):,}"
    )

    return id_to_name, is_a_pairs, xrefs


def write_outputs(
    id_to_name: dict[str, str],
    is_a_pairs: list[tuple[str, str]],
    xrefs: list[tuple[str, str, str]],
) -> None:
    # ─── mondo_terms.tsv ─────────────────────────────────────
    with OUT_TERMS.open("w", encoding="utf-8") as f:
        f.write("mondo_id\tname\n")
        for mid, name in sorted(id_to_name.items()):
            sn = name.replace("\t", " ").replace("\n", " ").replace("\r", " ")
            f.write(f"{mid}\t{sn}\n")
    logger.info(f"  Wrote {OUT_TERMS.name} ({len(id_to_name):,} rows)")

    # ─── mondo_hierarchy.tsv ─────────────────────────────────
    with OUT_HIERARCHY.open("w", encoding="utf-8") as f:
        f.write("child_mondo_id\tchild_name\tparent_mondo_id\tparent_name\n")
        for child, parent in is_a_pairs:
            cn = id_to_name.get(child, child)
            pn = id_to_name.get(parent, parent)
            cn = cn.replace("\t", " ").replace("\n", " ").replace("\r", " ")
            pn = pn.replace("\t", " ").replace("\n", " ").replace("\r", " ")
            f.write(f"{child}\t{cn}\t{parent}\t{pn}\n")
    logger.info(f"  Wrote {OUT_HIERARCHY.name} ({len(is_a_pairs):,} rows)")

    # ─── mondo_xrefs.tsv ─────────────────────────────────────
    with OUT_XREFS.open("w", encoding="utf-8") as f:
        f.write("mondo_id\tmondo_name\tsource\tcode\n")
        for mid, source, code in xrefs:
            mn = id_to_name.get(mid, mid)
            mn = mn.replace("\t", " ").replace("\n", " ").replace("\r", " ")
            f.write(f"{mid}\t{mn}\t{source}\t{code}\n")
    logger.info(f"  Wrote {OUT_XREFS.name} ({len(xrefs):,} rows)")


def main() -> int:
    if not MONDO_DIR.exists():
        logger.error(f"MONDO directory missing: {MONDO_DIR}")
        return 1

    id_to_name, is_a_pairs, xrefs = parse_obo(MONDO_OBO)
    if not id_to_name:
        logger.error("No MONDO terms parsed; cannot continue.")
        return 1

    write_outputs(id_to_name, is_a_pairs, xrefs)

    # Stats: distribution of xref sources
    from collections import Counter
    source_counts = Counter(s for _, s, _ in xrefs)
    logger.info("=" * 60)
    logger.info(" MONDO conversion complete")
    logger.info("=" * 60)
    logger.info(f"  MONDO terms:         {len(id_to_name):,}")
    logger.info(f"  is_a relations:      {len(is_a_pairs):,}")
    logger.info(f"  Total xrefs:         {len(xrefs):,}")
    logger.info("  xref breakdown:")
    for s, c in source_counts.most_common():
        logger.info(f"    {s:12s}  {c:>8,}")
    logger.info(f"  Output: {MONDO_DIR}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
