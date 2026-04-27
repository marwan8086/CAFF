#!/usr/bin/env python
"""
scripts/build_kg.py — Merge Orphanet + DisGeNET + OMIM into a single
biomedical KG, joined on shared UMLS Concept Unique Identifiers (CUIs).

Implements the data construction pipeline described in paper §8.1:

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

Important — Licensing
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


# ─────────────────────────────────────────────────────────────────
# UMLS CUI mapping
# ─────────────────────────────────────────────────────────────────


@dataclass
class UMLSMapper:
    """Map source-specific identifiers (gene symbols, OMIM IDs,
    Orphanet codes) to canonical UMLS CUIs.

    Loaded from MRCONSO.RRF (UMLS metathesaurus core file).
    Format documented at:
        https://www.ncbi.nlm.nih.gov/books/NBK9685/
    """

    # source vocab → identifier in source vocab → CUI
    by_source: dict[str, dict[str, str]]
    # name (lowercased) → CUI for fuzzy fallback
    by_name: dict[str, str]

    @classmethod
    def from_mrconso(cls, mrconso_path: str | Path) -> "UMLSMapper":
        """Build a mapper from MRCONSO.RRF.

        We extract mappings for source vocabularies relevant to CAFF:
            HGNC      — gene symbols
            OMIM      — OMIM IDs
            ORPHANET  — Orphanet codes
            MSH       — MeSH terms (fallback)
            SNOMEDCT_US — clinical terms (fallback)

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


# ─────────────────────────────────────────────────────────────────
# Source-specific loaders
# ─────────────────────────────────────────────────────────────────


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
    """Load Orphanet triples from XML dumps.

    Orphanet ships several XML files; we use:
        en_product1.xml   — disease metadata
        en_product6.xml   — disease-gene relationships
        en_product4.xml   — disease-phenotype links

    Falls back gracefully: missing files are warned and skipped.
    """
    orph_dir = Path(orphanet_dir)
    triples: list[RawTriple] = []

    # ── Disease ↔ Gene (en_product6.xml) ─────────────────────
    p6 = orph_dir / "en_product6.xml"
    if p6.exists():
        logger.info(f"Parsing Orphanet disease-gene file: {p6}")
        tree = ET.parse(p6)
        for disorder in tree.iterfind(".//Disorder"):
            orpha_code_el = disorder.find("OrphaCode")
            disease_name_el = disorder.find("Name")
            if orpha_code_el is None or disease_name_el is None:
                continue
            orpha_code = orpha_code_el.text or ""
            disease_name = disease_name_el.text or ""

            for assoc in disorder.iterfind(".//DisorderGeneAssociation"):
                gene_el = assoc.find(".//Gene")
                if gene_el is None:
                    continue
                gene_symbol_el = gene_el.find("Symbol")
                gene_name_el = gene_el.find("Name")
                rel_el = assoc.find(".//DisorderGeneAssociationType/Name")
                if gene_symbol_el is None:
                    continue
                gene_symbol = gene_symbol_el.text or ""
                rel_text = (rel_el.text or "associated_with").strip().lower()
                rel_text = re.sub(r"\W+", "_", rel_text).strip("_")

                triples.append(
                    RawTriple(
                        head=disease_name,
                        head_source="ORPHANET",
                        head_code=orpha_code,
                        relation=rel_text,
                        tail=gene_symbol,
                        tail_source="HGNC",
                        tail_code=gene_symbol,
                        origin="orphanet",
                    )
                )
        logger.info(f"  Orphanet disease-gene: {len(triples):,} triples so far")
    else:
        logger.warning(f"Orphanet file missing: {p6}")

    # ── Disease ↔ Phenotype (en_product4.xml) ────────────────
    p4 = orph_dir / "en_product4.xml"
    if p4.exists():
        logger.info(f"Parsing Orphanet disease-phenotype file: {p4}")
        tree = ET.parse(p4)
        before = len(triples)
        for disorder in tree.iterfind(".//Disorder"):
            orpha_code = (disorder.findtext("OrphaCode") or "").strip()
            disease_name = (disorder.findtext("Name") or "").strip()
            for hp in disorder.iterfind(".//HPODisorderAssociation"):
                hp_term_el = hp.find(".//HPO")
                if hp_term_el is None:
                    continue
                hp_id = (hp_term_el.findtext("HPOId") or "").strip()
                hp_name = (hp_term_el.findtext("HPOTerm") or "").strip()
                if not hp_name:
                    continue
                triples.append(
                    RawTriple(
                        head=disease_name,
                        head_source="ORPHANET",
                        head_code=orpha_code,
                        relation="has_phenotype",
                        tail=hp_name,
                        tail_source="HPO",      # not in MRCONSO; fall back to name
                        tail_code=hp_id,
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


# ─────────────────────────────────────────────────────────────────
# CUI normalization + merge
# ─────────────────────────────────────────────────────────────────


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

    Per paper §8.1, relations with <50 triples are dropped, retaining
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
    logger.info(f"Deduplication: {len(rows):,} → {len(deduped):,}")

    if min_relation_freq > 0:
        rel_counts: dict[str, int] = defaultdict(int)
        for r in deduped:
            rel_counts[r["relation"]] += 1
        kept = {r for r, c in rel_counts.items() if c >= min_relation_freq}
        n_before = len(deduped)
        deduped = [r for r in deduped if r["relation"] in kept]
        logger.info(
            f"Singleton-relation removal (min_freq={min_relation_freq}): "
            f"{n_before:,} → {len(deduped):,}  "
            f"(kept {len(kept)} of {len(rel_counts)} relations)"
        )

    return deduped


# ─────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build merged biomedical KG.")
    p.add_argument("--orphanet", required=True, help="Orphanet XML directory.")
    p.add_argument("--disgenet", required=True,
                   help="Path to all_gene_disease_associations.tsv")
    p.add_argument("--omim",     required=True, help="OMIM data directory.")
    p.add_argument("--umls",     required=True, help="Path to MRCONSO.RRF")
    p.add_argument("--out",      required=True, help="Output TSV path.")
    p.add_argument("--min-relation-freq", type=int, default=50,
                   help="Drop relations with fewer than this many triples.")
    p.add_argument("--disgenet-score-threshold", type=float, default=0.3,
                   help="DisGeNET confidence-score cutoff.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    setup_logging(level="INFO")

    # ─── Load each source ───────────────────────────────────────
    umls    = UMLSMapper.from_mrconso(args.umls)
    raw: list[RawTriple] = []
    raw.extend(load_orphanet(args.orphanet))
    raw.extend(load_disgenet(args.disgenet,
                             score_threshold=args.disgenet_score_threshold))
    raw.extend(load_omim(args.omim))
    logger.info(f"Total raw triples loaded: {len(raw):,}")

    # ─── Normalize to UMLS CUIs ─────────────────────────────────
    rows = normalize_to_cuis(raw, umls)

    # ─── Deduplicate + drop singleton relations ─────────────────
    rows = deduplicate_and_filter(rows, min_relation_freq=args.min_relation_freq)

    # ─── Stats ──────────────────────────────────────────────────
    entities = {r["head_cui"] for r in rows} | {r["tail_cui"] for r in rows}
    relations = {r["relation"] for r in rows}
    logger.info("─" * 60)
    logger.info(f"Final KG:  |V|={len(entities):,}  "
                f"|E|={len(rows):,}  |R|={len(relations)}")
    logger.info(f"Paper §8.1 reports |V|=148,423  |E|=2,318,941  |R|=42")
    logger.info("─" * 60)

    # ─── Write TSV ──────────────────────────────────────────────
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows, columns=[
        "head", "relation", "tail", "head_cui", "tail_cui", "source",
    ])
    df.to_csv(out_path, sep="\t", index=False)
    logger.info(f"Wrote {len(df):,} rows to {out_path}")


if __name__ == "__main__":
    main()