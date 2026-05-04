"""
scripts/convert_hpo_to_tsv.py — Convert HPO ontology + annotations
to TSV files mergeable with the existing Orphanet KG.

Reads:
    data/raw/hpo/hp.obo
    data/raw/hpo/phenotype.hpoa

Writes:
    data/raw/hpo/hp_hierarchy.tsv      (HPO is_a relations)
    data/raw/hpo/hp_terms.tsv          (HPO id -> name)
    data/raw/hpo/hpoa_omim.tsv         (OMIM disease -> HPO term)

Filtering:
    - hpoa_omim.tsv contains only OMIM rows (skip ORPHA to avoid
      duplication with Orphanet data already in the KG).
    - aspect must be 'P' (Phenotypic) — skip Course/Inheritance/Modifier.
    - obsolete HPO terms are dropped.

Usage:
    python scripts/convert_hpo_to_tsv.py
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
logger = logging.getLogger("hpo_to_tsv")


ROOT = Path(__file__).parent.parent
HPO_DIR = ROOT / "data" / "raw" / "hpo"

HP_OBO = HPO_DIR / "hp.obo"
HPOA = HPO_DIR / "phenotype.hpoa"

OUT_HIERARCHY = HPO_DIR / "hp_hierarchy.tsv"
OUT_TERMS = HPO_DIR / "hp_terms.tsv"
OUT_OMIM = HPO_DIR / "hpoa_omim.tsv"


def parse_obo(path: Path) -> tuple[dict[str, str], list[tuple[str, str]]]:
    """Parse hp.obo -> (id_to_name, is_a_pairs).

    is_a_pairs is a list of (child_id, parent_id) tuples.
    Obsolete terms are dropped.
    """
    if not path.exists():
        logger.error(f"Missing file: {path}")
        return {}, []

    logger.info(f"Parsing {path.name} ...")
    id_to_name: dict[str, str] = {}
    is_a_pairs: list[tuple[str, str]] = []

    current_id: str | None = None
    current_name: str | None = None
    current_is_a: list[str] = []
    current_obsolete = False
    in_term = False

    n_terms = 0
    n_obsolete = 0

    def flush() -> None:
        nonlocal n_terms, n_obsolete
        if current_id and not current_obsolete:
            if current_name:
                id_to_name[current_id] = current_name
            for parent in current_is_a:
                is_a_pairs.append((current_id, parent))
            n_terms += 1
        elif current_obsolete:
            n_obsolete += 1

    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if line == "[Term]":
                # Save previous term
                if in_term:
                    flush()
                # Start new
                current_id = None
                current_name = None
                current_is_a = []
                current_obsolete = False
                in_term = True
                continue
            if line.startswith("[") and line.endswith("]"):
                # Some other stanza ([Typedef], etc) - flush and stop
                if in_term:
                    flush()
                in_term = False
                continue
            if not in_term:
                continue
            if line.startswith("id: "):
                current_id = line[4:].strip()
            elif line.startswith("name: "):
                current_name = line[6:].strip()
            elif line.startswith("is_a: "):
                # Format: "is_a: HP:0001507 ! Growth abnormality"
                rest = line[6:].strip()
                parent_id = rest.split(" ")[0].split("!")[0].strip()
                if parent_id.startswith("HP:"):
                    current_is_a.append(parent_id)
            elif line.startswith("is_obsolete: true"):
                current_obsolete = True

    # Flush the last term
    if in_term:
        flush()

    logger.info(f"  Parsed {n_terms:,} HPO terms ({n_obsolete:,} obsolete dropped)")
    logger.info(f"  Extracted {len(is_a_pairs):,} is_a relations")

    # Write hierarchy TSV
    with OUT_HIERARCHY.open("w", encoding="utf-8") as f:
        f.write("child_hpo_id\tchild_name\tparent_hpo_id\tparent_name\n")
        for child_id, parent_id in is_a_pairs:
            child_name = id_to_name.get(child_id, child_id)
            parent_name = id_to_name.get(parent_id, parent_id)
            # Defensive escaping
            cn = child_name.replace("\t", " ").replace("\n", " ").replace("\r", " ")
            pn = parent_name.replace("\t", " ").replace("\n", " ").replace("\r", " ")
            f.write(f"{child_id}\t{cn}\t{parent_id}\t{pn}\n")
    logger.info(f"  Wrote {OUT_HIERARCHY.name} ({len(is_a_pairs):,} rows)")

    # Write terms TSV
    with OUT_TERMS.open("w", encoding="utf-8") as f:
        f.write("hpo_id\tname\n")
        for hpo_id, name in sorted(id_to_name.items()):
            sn = name.replace("\t", " ").replace("\n", " ").replace("\r", " ")
            f.write(f"{hpo_id}\t{sn}\n")
    logger.info(f"  Wrote {OUT_TERMS.name} ({len(id_to_name):,} rows)")

    return id_to_name, is_a_pairs


def parse_hpoa(path: Path, id_to_name: dict[str, str]) -> int:
    """Parse phenotype.hpoa -> hpoa_omim.tsv (OMIM-only).

    Schema (from header):
      database_id  disease_name  qualifier  hpo_id  reference
      evidence  onset  frequency  sex  modifier  aspect  biocuration
    """
    if not path.exists():
        logger.error(f"Missing file: {path}")
        return 0

    logger.info(f"Parsing {path.name} (OMIM-only, aspect=P) ...")
    rows: list[tuple[str, str, str]] = []  # (omim_id, disease_name, hpo_id)
    seen: set[tuple[str, str]] = set()

    n_total = 0
    n_omim = 0
    n_phenotypic = 0
    n_kept = 0
    n_skipped_qualifier = 0

    header_indices: dict[str, int] | None = None

    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line:
                continue
            if line.startswith("#"):
                continue
            cols = line.split("\t")
            if header_indices is None:
                # First non-comment line is the header
                header_indices = {name: i for i, name in enumerate(cols)}
                missing = [
                    k for k in ("database_id", "hpo_id", "aspect", "qualifier", "disease_name")
                    if k not in header_indices
                ]
                if missing:
                    logger.error(f"  Missing required HPOA columns: {missing}")
                    return 0
                logger.info(f"  HPOA columns: {list(header_indices.keys())}")
                continue

            n_total += 1
            try:
                db_id = cols[header_indices["database_id"]]
                disease_name = cols[header_indices["disease_name"]]
                qualifier = cols[header_indices["qualifier"]]
                hpo_id = cols[header_indices["hpo_id"]]
                aspect = cols[header_indices["aspect"]]
            except IndexError:
                continue

            # Skip non-OMIM (Orphanet already covered by Orphanet pipeline)
            if not db_id.startswith("OMIM:"):
                continue
            n_omim += 1

            # Phenotypic aspect only
            if aspect != "P":
                continue
            n_phenotypic += 1

            # Skip negated annotations (qualifier "NOT")
            if qualifier and qualifier.strip().upper() == "NOT":
                n_skipped_qualifier += 1
                continue

            if not hpo_id.startswith("HP:"):
                continue

            key = (db_id, hpo_id)
            if key in seen:
                continue
            seen.add(key)

            rows.append((db_id, disease_name.strip(), hpo_id))
            n_kept += 1

    logger.info(
        f"  Total rows: {n_total:,}, OMIM: {n_omim:,}, "
        f"OMIM+P: {n_phenotypic:,}, NOT-skipped: {n_skipped_qualifier:,}, "
        f"Kept (deduped): {n_kept:,}"
    )

    with OUT_OMIM.open("w", encoding="utf-8") as f:
        f.write("disease_id\tdisease_name\thpo_id\thpo_term\n")
        for db_id, dn, hpo_id in rows:
            hpo_term = id_to_name.get(hpo_id, hpo_id)
            sdn = dn.replace("\t", " ").replace("\n", " ").replace("\r", " ")
            sht = hpo_term.replace("\t", " ").replace("\n", " ").replace("\r", " ")
            f.write(f"{db_id}\t{sdn}\t{hpo_id}\t{sht}\n")
    logger.info(f"  Wrote {OUT_OMIM.name} ({len(rows):,} rows)")
    return len(rows)


def main() -> int:
    if not HPO_DIR.exists():
        logger.error(f"HPO directory missing: {HPO_DIR}")
        return 1

    id_to_name, is_a_pairs = parse_obo(HP_OBO)
    if not id_to_name:
        logger.error("No HPO terms parsed; cannot continue.")
        return 1

    n_hpoa = parse_hpoa(HPOA, id_to_name)

    logger.info("=" * 60)
    logger.info(" HPO conversion complete")
    logger.info("=" * 60)
    logger.info(f"  HPO terms:           {len(id_to_name):,}")
    logger.info(f"  is_a relations:      {len(is_a_pairs):,}")
    logger.info(f"  OMIM annotations:    {n_hpoa:,}")
    logger.info(f"  Output: {HPO_DIR}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
