#!/usr/bin/env python
"""
scripts/build_kg.py â€” Merge Orphanet + DisGeNET + OMIM into a single
biomedical KG, joined on shared UMLS Concept Unique Identifiers (CUIs).

Implements the data construction pipeline described in paper Â§8.1:

    "We merge Orphanet, DisGeNET, and OMIM on shared UMLS concept
     identifiers. The merged KG contains |V|=148,423 entities,
     |E|=2,318,941 triples, and |R|=47 relation types. Singleton
     relations (<50 triples) are removed, retaining |R|=42."

Usage
-----
    python scripts/build_kg.py \
        --orphanet  data/raw/orphanet/ \
        --disgenet  data/raw/disgenet/all_gene_disease_associations.tsv \
        --omim      data/raw/omim/ \
        --umls      data/raw/umls/MRCONSO.RRF \
        --out       data/processed/merged_kg.tsv \
        --min-relation-freq 50

Important â€” Licensing
---------------------
This script does NOT redistribute source data. You must obtain
each dataset directly from its provider:
    Orphanet:  https://www.orphadata.com         (free, no registration)
    DisGeNET:  https://www.disgenet.org          (academic license)
    OMIM:      https://www.omim.org              (license required)
    UMLS:      https://uts.nlm.nih.gov           (UTS account required)

Output format
-------------
A tab-separated file with the columns:
    head    relation    tail    head_cui    tail_cui    source

ready for KnowledgeGraph.from_tsv() in caff/data.py.
"""

from __future__ import annotations

import argparse
import logging
import re
import xml.etree.ElementTree as ET
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from caff.utils.logging import setup_logging

logger = logging.getLogger(__name__)


# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# UMLS CUI mapping
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€


@dataclass
class UMLSMapper:
    """Map source-specific identifiers (gene symbols, OMIM IDs,
    Orphanet codes) to canonical UMLS CUIs.

    Loaded from MRCONSO.RRF (UMLS metathesaurus core file).
    Format documented at:
        https://www.ncbi.nlm.nih.gov/books/NBK9685/
    """

    # source vocab â†’ identifier in source vocab â†’ CUI
    by_source: dict[str, dict[str, str]]
    # name (lowercased) â†’ CUI for fuzzy fallback
    by_name: dict[str, str]

    @classmethod
    def from_mrconso(cls, mrconso_path: str | Path) -> "UMLSMapper":
        """Build a mapper from MRCONSO.RRF.

        We extract mappings for source vocabularies relevant to CAFF:
            HGNC      â€” gene symbols
            OMIM      â€” OMIM IDs
            ORPHANET  â€” Orphanet codes
            MSH       â€” MeSH terms (fallback)
            SNOMEDCT_US â€” clinical terms (fallback)

        MRCONSO columns (pipe-separated):
            CUI|LAT|TS|LUI|STT|SUI|ISPREF|AUI|SAUI|SCUI|SDUI|SAB|TTY|CODE|...
            0   1   2  3   4   5   6      7   8    9    10   11  12  13
        """
        path = Path(mrconso_path)
        logger.info(f"Loading UMLS MRCONSO from {path}...")
        relevant_sabs = {"HGNC", "OMIM", "ORPHANET", "MSH", "SNOMEDCT_US"}
        by_source: dict[str, dict[str, str]] = defaultdict(dict)
        by_name: dict[str, str] = {}

        with path.open("r", encoding="utf-8", errors="ignore") as f:
            for line_no, line in enumerate(f, start=1):
                parts = line.rstrip("\n").split("|")
                if len(parts) < 15:
                    continue
                if parts[1] != "ENG":          # English only
                    continue
                cui, sab, code, name = parts[0], parts[11], parts[13], parts[14]
                if sab in relevant_sabs and code:
                    by_source[sab][code] = cui
                # First English name we see for a CUI = canonical
                key = name.strip().lower()
                if key and key not in by_name:
                    by_name[key] = cui
                if line_no % 1_000_000 == 0:
                    logger.info(f"  ... {line_no:,} MRCONSO rows processed")

        logger.info(
            f"UMLS mapper built: "
            + ", ".join(f"{sab}={len(by_source[sab]):,}" for sab in relevant_sabs)
            + f", names={len(by_name):,}"
        )
        return cls(by_source=dict(by_source), by_name=by_name)

    def cui_for(self, source: str, code: str) -> str | None:
        """Look up a CUI by source vocabulary + identifier."""
        return self.by_source.get(source, {}).get(code)

    def cui_for_name(self, name: str) -> str | None:
        """Fuzzy fallback: lookup by canonical English name."""
        return self.by_name.get(name.strip().lower())


# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# Source-specific loaders
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€


@dataclass
class RawTriple:
    """An (h, r, t) triple before CUI normalization."""
    head: str
    head_source: str            # vocab tag for UMLS lookup
    head_code: str
    relation: str
    tail: str
    tail_source: str
    tail_code: str
    origin: str                 # 'orphanet' | 'disgenet' | 'omim'


def load_orphanet(orphanet_dir: str | Path) -> list[RawTriple]:
    """Load Orphanet triples from TSV dumps (2025 format).

    Orphanet ships several TSV files; we use:
        genes_to_diseases_en_2025.tsv   â€” disease-gene relationships
        phenotypes_en_2025.tsv         â€” disease-phenotype links
        ORDO_en_2025.xlsx              â€” disease ontology for names

    Falls back gracefully: missing files are warned and skipped.
    """
    orph_dir = Path(orphanet_dir)
    triples: list[RawTriple] = []

    # Load disease names from ORDO ontology
    ordo_path = orph_dir / "ORDO_names_en_2025.tsv"
    disease_names = {}
    if ordo_path.exists():
        logger.info(f"Loading Orphanet disease names from {ordo_path}")
        ordo_df = pd.read_csv(ordo_path, sep="\t")
        # Assuming columns include 'ORPHAcode' and 'Preferred term'
        if 'ORPHAcode' in ordo_df.columns and 'Preferred term' in ordo_df.columns:
            disease_names = dict(zip(ordo_df['ORPHAcode'], ordo_df['Preferred term']))
        logger.info(f"  Loaded {len(disease_names):,} disease names")
    else:
        logger.warning(f"Orphanet ORDO file missing: {ordo_path}")

    # â”€â”€ Disease â†” Gene (genes_to_diseases_en_2025.tsv) â”€â”€â”€â”€â”€â”€â”€
    genes_path = orph_dir / "genes_to_diseases_en_2025.tsv"
    if genes_path.exists():
        logger.info(f"Parsing Orphanet disease-gene file: {genes_path}")
        genes_df = pd.read_csv(genes_path, sep='\t')
        for _, row in genes_df.iterrows():
            orpha_code = str(row.get('orpha_code', '')).strip()
            gene_symbol = str(row.get('gene_symbol', '')).strip()
            association_type = str(row.get('association_type', 'associated_with')).strip().lower()
            association_type = re.sub(r"\W+", "_", association_type).strip("_")

            disease_name = disease_names.get(int(orpha_code) if orpha_code.isdigit() else orpha_code, f"ORPHA:{orpha_code}")

            if gene_symbol:
                triples.append(
                    RawTriple(
                        head=disease_name,
                        head_source="ORPHANET",
                        head_code=orpha_code,
                        relation=association_type,
                        tail=gene_symbol,
                        tail_source="HGNC",
                        tail_code=gene_symbol,
                        origin="orphanet",
                    )
                )
        logger.info(f"  Orphanet disease-gene: {len(triples):,} triples so far")
    else:
        logger.warning(f"Orphanet genes file missing: {genes_path}")

    # â”€â”€ Disease â†” Phenotype (phenotypes_en_2025.tsv) â”€â”€â”€â”€â”€â”€â”€â”€â”€
    pheno_path = orph_dir / "phenotypes_en_2025.tsv"
    if pheno_path.exists():
        logger.info(f"Parsing Orphanet disease-phenotype file: {pheno_path}")
        before = len(triples)
        pheno_df = pd.read_csv(pheno_path, sep='\t')
        for _, row in pheno_df.iterrows():
            orpha_code = str(row.get('orpha_code', '')).strip()
            hpo_id = str(row.get('hpo_id', '')).strip()
            hpo_term = str(row.get('hpo_term', '')).strip()

            disease_name = disease_names.get(int(orpha_code) if orpha_code.isdigit() else orpha_code, f"ORPHA:{orpha_code}")

            if hpo_term:
                triples.append(
                    RawTriple(
                        head=disease_name,
                        head_source="ORPHANET",
                        head_code=orpha_code,
                        relation="has_phenotype",
                        tail=hpo_term,
                        tail_source="HPO",
                        tail_code=hpo_id,
                        origin="orphanet",
                    )
                )
        logger.info(f"  Orphanet disease-phenotype: +{len(triples) - before:,}")

    return triples


def load_disgenet(disgenet_path: str | Path, score_threshold: float = 0.3) -> list[RawTriple]:
    """Load DisGeNET gene-disease associations.

    File: all_gene_disease_associations.tsv
        Columns: geneId, geneSymbol, ..., diseaseId, diseaseName,
                 diseaseType, ..., score, ...
    """
    path = Path(disgenet_path)
    if not path.exists():
        logger.warning(f"DisGeNET file missing: {path}")
        return []

    logger.info(f"Loading DisGeNET from {path} (score >= {score_threshold})...")
    df = pd.read_csv(path, sep="\t", low_memory=False)
    df = df[df["score"] >= score_threshold]

    triples: list[RawTriple] = []
    for _, row in df.iterrows():
        gene_symbol = str(row.get("geneSymbol", "")).strip()
        disease_id  = str(row.get("diseaseId",  "")).strip()  # e.g. "C0024796"
        disease_name = str(row.get("diseaseName", "")).strip()
        if not gene_symbol or not disease_id:
            continue
        triples.append(
            RawTriple(
                head=gene_symbol,
                head_source="HGNC",
                head_code=gene_symbol,
                relation="associated_with_disease",
                tail=disease_name,
                tail_source="UMLS_CUI_DIRECT",
                tail_code=disease_id,                # already a CUI
                origin="disgenet",
            )
        )
    logger.info(f"  DisGeNET: {len(triples):,} triples")
    return triples


def load_omim(omim_dir: str | Path) -> list[RawTriple]:
    """Load OMIM gene-phenotype relationships from genemap2.txt.

    File format (tab-separated, '#' comments):
        Chromosome | Genomic Position Start | ... | Approved Gene Symbol
                  | Entrez Gene ID | Ensembl Gene ID | Comments
                  | Phenotypes | Mouse Gene Symbol/ID
    """
    omim_dir = Path(omim_dir)
    triples: list[RawTriple] = []

    genemap = omim_dir / "genemap2.txt"
    if not genemap.exists():
        logger.warning(f"OMIM genemap2.txt missing: {genemap}")
        return []

    logger.info(f"Parsing OMIM genemap2.txt: {genemap}")
    pheno_re = re.compile(r"\s*(?P<pheno>[^,;]+?)\s*,\s*(?P<mim>\d{6})\s*\((?P<key>\d+)\)")

    with genemap.open("r", encoding="utf-8") as f:
        for line in f:
            if line.startswith("#") or not line.strip():
                continue
            cols = line.rstrip("\n").split("\t")
            if len(cols) < 13:
                continue
            gene_symbol  = cols[8].strip()
            phenotypes_s = cols[12].strip()
            if not gene_symbol or not phenotypes_s:
                continue
            for m in pheno_re.finditer(phenotypes_s):
                pheno_name = m.group("pheno").strip()
                pheno_mim  = m.group("mim").strip()
                triples.append(
                    RawTriple(
                        head=gene_symbol,
                        head_source="HGNC",
                        head_code=gene_symbol,
                        relation="causes_phenotype",
                        tail=pheno_name,
                        tail_source="OMIM",
                        tail_code=pheno_mim,
                        origin="omim",
                    )
                )
    logger.info(f"  OMIM gene-phenotype: {len(triples):,}")
    return triples


# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# CUI normalization + merge
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€


def normalize_to_cuis(
    triples: list[RawTriple],
    umls: UMLSMapper,
) -> list[dict]:
    """Resolve each (head, tail) to its UMLS CUI.

    Returns a list of dicts ready for tabular output. Triples that
    fail CUI resolution on EITHER endpoint are dropped (logged).
    """
    out: list[dict] = []
    n_drop_head = 0
    n_drop_tail = 0

    for t in triples:
        # Head CUI
        if t.head_source == "UMLS_CUI_DIRECT":
            head_cui = t.head_code
        else:
            head_cui = umls.cui_for(t.head_source, t.head_code) \
                       or umls.cui_for_name(t.head)
        if not head_cui:
            n_drop_head += 1
            continue

        # Tail CUI
        if t.tail_source == "UMLS_CUI_DIRECT":
            tail_cui = t.tail_code
        else:
            tail_cui = umls.cui_for(t.tail_source, t.tail_code) \
                       or umls.cui_for_name(t.tail)
        if not tail_cui:
            n_drop_tail += 1
            continue

        out.append({
            "head":     t.head,
            "relation": t.relation,
            "tail":     t.tail,
            "head_cui": head_cui,
            "tail_cui": tail_cui,
            "source":   t.origin,
        })

    logger.info(
        f"CUI normalization: kept {len(out):,} / {len(triples):,}  "
        f"(dropped {n_drop_head:,} unresolved heads, "
        f"{n_drop_tail:,} unresolved tails)"
    )
    return out


def deduplicate_and_filter(
    rows: list[dict],
    min_relation_freq: int = 50,
) -> list[dict]:
    """Remove exact duplicates and filter singleton relations.

    Per paper Â§8.1, relations with <50 triples are dropped, retaining
    |R|=42 from the original 47.
    """
    seen: set[tuple[str, str, str]] = set()
    deduped: list[dict] = []
    for row in rows:
        key = (row["head_cui"], row["relation"], row["tail_cui"])
        if key in seen:
            continue
        seen.add(key)
        deduped.append(row)
    logger.info(f"Deduplication: {len(rows):,} â†’ {len(deduped):,}")

    if min_relation_freq > 0:
        rel_counts: dict[str, int] = defaultdict(int)
        for r in deduped:
            rel_counts[r["relation"]] += 1
        kept = {r for r, c in rel_counts.items() if c >= min_relation_freq}
        n_before = len(deduped)
        deduped = [r for r in deduped if r["relation"] in kept]
        logger.info(
            f"Singleton-relation removal (min_freq={min_relation_freq}): "
            f"{n_before:,} â†’ {len(deduped):,}  "
            f"(kept {len(kept)} of {len(rel_counts)} relations)"
        )

    return deduped


# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# Main
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build merged biomedical KG.")
    p.add_argument("--orphanet", help="Orphanet TSV directory.")
    p.add_argument("--disgenet",
                   help="Path to all_gene_disease_associations.tsv")
    p.add_argument("--omim",     help="OMIM data directory.")
    p.add_argument("--umls",     help="Path to MRCONSO.RRF")
    p.add_argument("--out",      required=True, help="Output TSV path.")
    p.add_argument("--min-relation-freq", type=int, default=50,
                   help="Drop relations with fewer than this many triples.")
    p.add_argument("--disgenet-score-threshold", type=float, default=0.3,
                   help="DisGeNET confidence-score cutoff.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    setup_logging(level="INFO")

    # â”€â”€â”€ Load each source â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    umls = None
    if args.umls:
        umls = UMLSMapper.from_mrconso(args.umls)
    else:
        logger.warning("No UMLS provided - will skip CUI normalization")

    raw: list[RawTriple] = []
    if args.orphanet:
        raw.extend(load_orphanet(args.orphanet))
    if args.disgenet:
        raw.extend(load_disgenet(args.disgenet,
                                 score_threshold=args.disgenet_score_threshold))
    if args.omim:
        raw.extend(load_omim(args.omim))
    logger.info(f"Total raw triples loaded: {len(raw):,}")

    # â”€â”€â”€ Normalize to UMLS CUIs â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    if umls:
        rows = normalize_to_cuis(raw, umls)
    else:
        # Skip CUI normalization, use raw names
        rows = []
        for t in raw:
            rows.append({
                "head":     t.head,
                "relation": t.relation,
                "tail":     t.tail,
                "head_cui": t.head_code or t.head,
                "tail_cui": t.tail_code or t.tail,
                "source":   t.origin,
            })
        logger.info(f"Skipped CUI normalization: kept {len(rows):,} triples")

    # â”€â”€â”€ Deduplicate + drop singleton relations â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    rows = deduplicate_and_filter(rows, min_relation_freq=args.min_relation_freq)

    # â”€â”€â”€ Stats â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    entities = {r["head_cui"] for r in rows} | {r["tail_cui"] for r in rows}
    relations = {r["relation"] for r in rows}
    logger.info("â”€" * 60)
    logger.info(f"Final KG:  |V|={len(entities):,}  "
                f"|E|={len(rows):,}  |R|={len(relations)}")
    logger.info(f"Paper Â§8.1 reports |V|=148,423  |E|=2,318,941  |R|=42")
    logger.info("â”€" * 60)

    # â”€â”€â”€ Write TSV â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows, columns=[
        "head", "relation", "tail", "head_cui", "tail_cui", "source",
    ])
    df.to_csv(out_path, sep="\t", index=False)
    logger.info(f"Wrote {len(df):,} rows to {out_path}")


if __name__ == "__main__":
    main()
