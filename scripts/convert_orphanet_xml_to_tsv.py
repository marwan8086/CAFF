"""
scripts/convert_orphanet_xml_to_tsv.py — Convert Orphanet XML dumps
to the TSV format expected by scripts/build_kg.py::load_orphanet.

Reads:
    data/raw/orphanet/en_product6.xml    (disease-gene)
    data/raw/orphanet/en_product4.xml    (disease-phenotype)
    data/raw/orphanet/en_product1.xml    (disease nomenclature)

Writes:
    data/raw/orphanet/genes_to_diseases_en_2025.tsv
    data/raw/orphanet/phenotypes_en_2025.tsv
    data/raw/orphanet/ORDO_names_en_2025.tsv      (ORPHAcode  preferred_term)

The build_kg.py script reads ORDO from XLSX; we ship a TSV
companion that build_kg.py can be patched to read instead, OR
we install openpyxl to read XLSX directly. This converter emits
TSV (simpler and Git-friendly).

Usage:
    python scripts/convert_orphanet_xml_to_tsv.py
"""
from __future__ import annotations

import logging
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("orphanet_xml_to_tsv")


ROOT = Path(__file__).parent.parent
ORPH = ROOT / "data" / "raw" / "orphanet"

PRODUCT1 = ORPH / "en_product1.xml"
PRODUCT4 = ORPH / "en_product4.xml"
PRODUCT6 = ORPH / "en_product6.xml"

OUT_NAMES = ORPH / "ORDO_names_en_2025.tsv"
OUT_GENES = ORPH / "genes_to_diseases_en_2025.tsv"
OUT_PHENOS = ORPH / "phenotypes_en_2025.tsv"


def _text(element: ET.Element | None, default: str = "") -> str:
    """Safe extraction of element text."""
    if element is None or element.text is None:
        return default
    return element.text.strip()


def parse_nomenclature(path: Path) -> dict[str, str]:
    """Parse en_product1.xml -> {OrphaCode: preferred_name}.

    Returns the mapping AND writes it to OUT_NAMES as a TSV.
    """
    if not path.exists():
        logger.error(f"Missing file: {path}")
        return {}

    logger.info(f"Parsing {path.name} ...")
    tree = ET.parse(path)
    root = tree.getroot()

    names: dict[str, str] = {}
    n_disorders = 0

    # JDBOR > DisorderList > Disorder
    for disorder in root.iter("Disorder"):
        orpha_code = _text(disorder.find("OrphaCode"))
        name = _text(disorder.find("Name"))
        if orpha_code and name:
            names[orpha_code] = name
            n_disorders += 1

    logger.info(f"  Parsed {n_disorders:,} disorders from nomenclature")

    # Write TSV
    with OUT_NAMES.open("w", encoding="utf-8") as f:
        f.write("ORPHAcode\tPreferred term\n")
        for code, name in sorted(names.items(), key=lambda kv: int(kv[0]) if kv[0].isdigit() else 1_000_000):
            # Escape any tabs/newlines defensively
            safe_name = name.replace("\t", " ").replace("\n", " ").replace("\r", " ")
            f.write(f"{code}\t{safe_name}\n")
    logger.info(f"  Wrote {OUT_NAMES.name} ({len(names):,} rows)")
    return names


def parse_genes(path: Path, disease_names: dict[str, str]) -> int:
    """Parse en_product6.xml -> genes_to_diseases TSV."""
    if not path.exists():
        logger.error(f"Missing file: {path}")
        return 0

    logger.info(f"Parsing {path.name} ...")
    tree = ET.parse(path)
    root = tree.getroot()

    rows: list[tuple[str, str, str]] = []  # (orpha_code, gene_symbol, association_type)
    n_disorders = 0

    for disorder in root.iter("Disorder"):
        orpha_code = _text(disorder.find("OrphaCode"))
        if not orpha_code:
            continue
        n_disorders += 1

        # DisorderGeneAssociationList > DisorderGeneAssociation
        assoc_list = disorder.find("DisorderGeneAssociationList")
        if assoc_list is None:
            continue

        for assoc in assoc_list.findall("DisorderGeneAssociation"):
            gene = assoc.find("Gene")
            if gene is None:
                continue
            symbol = _text(gene.find("Symbol"))
            if not symbol:
                continue

            # Association type is optional in some entries.
            # Path: DisorderGeneAssociation > DisorderGeneAssociationType > Name
            assoc_type_elem = assoc.find("DisorderGeneAssociationType/Name")
            if assoc_type_elem is not None:
                assoc_type = _text(assoc_type_elem) or "associated_with"
            else:
                assoc_type = "associated_with"

            rows.append((orpha_code, symbol, assoc_type))

    logger.info(
        f"  Parsed {len(rows):,} disease-gene associations "
        f"from {n_disorders:,} disorders"
    )

    with OUT_GENES.open("w", encoding="utf-8") as f:
        f.write("orpha_code\tgene_symbol\tassociation_type\n")
        for orpha_code, symbol, assoc_type in rows:
            safe_type = assoc_type.replace("\t", " ").replace("\n", " ").replace("\r", " ")
            f.write(f"{orpha_code}\t{symbol}\t{safe_type}\n")
    logger.info(f"  Wrote {OUT_GENES.name} ({len(rows):,} rows)")
    return len(rows)


def parse_phenotypes(path: Path, disease_names: dict[str, str]) -> int:
    """Parse en_product4.xml -> phenotypes TSV."""
    if not path.exists():
        logger.error(f"Missing file: {path}")
        return 0

    logger.info(f"Parsing {path.name} ...")
    tree = ET.parse(path)
    root = tree.getroot()

    rows: list[tuple[str, str, str]] = []  # (orpha_code, hpo_id, hpo_term)
    n_disorders = 0

    # JDBOR > HPODisorderSetStatusList > HPODisorderSetStatus > Disorder
    for status in root.iter("HPODisorderSetStatus"):
        disorder = status.find("Disorder")
        if disorder is None:
            continue
        orpha_code = _text(disorder.find("OrphaCode"))
        if not orpha_code:
            continue
        n_disorders += 1

        assoc_list = disorder.find("HPODisorderAssociationList")
        if assoc_list is None:
            continue

        for assoc in assoc_list.findall("HPODisorderAssociation"):
            hpo = assoc.find("HPO")
            if hpo is None:
                continue
            hpo_id = _text(hpo.find("HPOId"))
            hpo_term = _text(hpo.find("HPOTerm"))
            if hpo_id and hpo_term:
                rows.append((orpha_code, hpo_id, hpo_term))

    logger.info(
        f"  Parsed {len(rows):,} disease-phenotype associations "
        f"from {n_disorders:,} disorders"
    )

    with OUT_PHENOS.open("w", encoding="utf-8") as f:
        f.write("orpha_code\thpo_id\thpo_term\n")
        for orpha_code, hpo_id, hpo_term in rows:
            safe_term = hpo_term.replace("\t", " ").replace("\n", " ").replace("\r", " ")
            f.write(f"{orpha_code}\t{hpo_id}\t{safe_term}\n")
    logger.info(f"  Wrote {OUT_PHENOS.name} ({len(rows):,} rows)")
    return len(rows)


def main() -> int:
    if not ORPH.exists():
        logger.error(f"Orphanet directory missing: {ORPH}")
        logger.error("Run downloads first (see Phase 3 plan).")
        return 1

    # 1. Names (ordo)
    names = parse_nomenclature(PRODUCT1)
    if not names:
        logger.error("No disease names parsed; cannot continue.")
        return 1

    # 2. Genes
    n_genes = parse_genes(PRODUCT6, names)

    # 3. Phenotypes
    n_phenos = parse_phenotypes(PRODUCT4, names)

    logger.info("=" * 60)
    logger.info(" Conversion complete")
    logger.info("=" * 60)
    logger.info(f"  Disorders (names):     {len(names):,}")
    logger.info(f"  Disease-gene rows:     {n_genes:,}")
    logger.info(f"  Disease-phenotype rows: {n_phenos:,}")
    logger.info(f"  Output: {ORPH}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
