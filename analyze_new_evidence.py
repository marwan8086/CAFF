"""
analyze_new_evidence.py - Analyze 3 non-Orphanet evidence sources from
Open Targets and identify NEW gene-disease pairs not already in KG v2.

Sources:
- evidence_clingen           (clinical genetics curation)
- evidence_gene2phenotype    (developmental disorders)
- evidence_genomics_england  (UK rare disease panel)

Workflow:
1. Load target metadata for Ensembl -> HGNC mapping
2. Load disease metadata for MONDO -> Orphanet mapping
3. Load each evidence source
4. Map IDs and convert to (orphanet_id, gene_symbol) pairs
5. Compare with KG v2 to identify NEW pairs
6. Report counts + samples

Output: prints stats, saves new_gene_disease_pairs.tsv
"""
from pathlib import Path
import pyarrow.parquet as pq
import pandas as pd

ROOT = Path("data/raw/opentargets")

# ============================================================
# Step 1: Build Ensembl -> HGNC mapping from target.parquet
# ============================================================
print("=" * 75)
print("  Step 1: Load target metadata (Ensembl -> HGNC mapping)")
print("=" * 75)

target_parts = sorted((ROOT / "target").glob("*.parquet"))`nprint(f"  Reading {len(target_parts)} target parts...")`ntarget = pd.concat([pq.read_table(p).to_pandas() for p in target_parts], ignore_index=True)
print(f"  Loaded {len(target):,} target rows")
print(f"  Columns: {list(target.columns)[:10]}...")

# Find the right columns for mapping
ens_col = 'id' if 'id' in target.columns else 'targetId'
sym_col = None
for c in ['approvedSymbol', 'symbol', 'geneSymbol', 'hgncSymbol']:
    if c in target.columns:
        sym_col = c
        break

print(f"  Ensembl column: {ens_col}")
print(f"  Symbol column:  {sym_col}")

if sym_col is None:
    print("  ERROR: no symbol column found. Available columns:")
    for c in target.columns:
        print(f"    {c}")
    raise SystemExit(1)

ens_to_sym = dict(zip(target[ens_col], target[sym_col]))
print(f"  Mappings built: {len(ens_to_sym):,}")
print(f"  Sample: ENSG -> Symbol")
for ens, sym in list(ens_to_sym.items())[:5]:
    print(f"    {ens} -> {sym}")

# ============================================================
# Step 2: Build MONDO -> Orphanet mapping from disease.parquet
# ============================================================
print()
print("=" * 75)
print("  Step 2: Load disease metadata (MONDO -> Orphanet mapping)")
print("=" * 75)

disease_path = ROOT / "disease.parquet"
disease = pq.read_table(disease_path).to_pandas()
print(f"  Loaded {len(disease):,} disease rows")
print(f"  Columns: {list(disease.columns)[:10]}...")

# disease.id can be EFO/MONDO/Orphanet/OMIM
# dbXRefs contains cross-references
# Build a mapping: any disease ID -> Orphanet numeric ID
disease_to_orph = {}  # any_id -> orph_num

for _, row in disease.iterrows():
    did = row.get('id', None)
    if pd.isna(did):
        continue
    xrefs = row.get('dbXRefs', [])
    if xrefs is None or (hasattr(xrefs, '__len__') and len(xrefs) == 0):
        continue

    # The disease.id might itself be Orphanet_NNN
    if str(did).startswith('Orphanet_'):
        num = str(did).replace('Orphanet_', '')
        disease_to_orph[did] = num
        continue

    # Otherwise check xrefs for Orphanet
    for xref in xrefs:
        xref_str = str(xref)
        if 'Orphanet:' in xref_str or 'Orphanet_' in xref_str:
            num = xref_str.replace('Orphanet:', '').replace('Orphanet_', '')
            disease_to_orph[did] = num
            break

print(f"  Disease -> Orphanet mappings: {len(disease_to_orph):,}")
sample_items = list(disease_to_orph.items())[:5]
for did, orph in sample_items:
    print(f"    {did} -> Orphanet:{orph}")

# ============================================================
# Step 3: Load each evidence source
# ============================================================
print()
print("=" * 75)
print("  Step 3: Process evidence sources")
print("=" * 75)

evidence_sources = {
    'clingen': ROOT / "evidence_clingen",
    'gene2phenotype': ROOT / "evidence_gene2phenotype",
    'genomics_england': ROOT / "evidence_genomics_england",
}

all_pairs = []  # (orphanet_id, gene_symbol, source, score)

for source_name, source_dir in evidence_sources.items():
    print(f"\n--- {source_name} ---")
    parts = sorted(source_dir.glob("*.parquet"))
    print(f"  Parts: {len(parts)}")

    dfs = [pq.read_table(p).to_pandas() for p in parts]
    df = pd.concat(dfs, ignore_index=True)
    print(f"  Total rows: {len(df):,}")
    print(f"  Columns (first 10): {list(df.columns)[:10]}")

    # Try to extract gene + disease columns
    # Standard OTG evidence schema:
    # - targetId (Ensembl)
    # - diseaseId or diseaseFromSourceMappedId (MONDO/EFO)
    # - diseaseFromSourceId (raw, e.g. Orphanet_XXX or other)
    # - targetFromSource (gene name)
    # - score (0-1)

    if 'targetId' not in df.columns:
        print(f"  SKIP: no targetId column")
        continue

    # Try multiple disease columns
    disease_col = None
    for c in ['diseaseId', 'diseaseFromSourceMappedId', 'diseaseFromSourceId']:
        if c in df.columns:
            disease_col = c
            break

    if disease_col is None:
        print(f"  SKIP: no disease column found")
        continue

    print(f"  Using disease column: {disease_col}")
    print(f"  Sample disease IDs: {df[disease_col].head(3).tolist()}")

    # Filter to rows we can resolve
    score_col = 'score' if 'score' in df.columns else None

    # Build pairs
    new_pairs_count = 0
    skipped_no_gene = 0
    skipped_no_disease = 0
    skipped_no_orph = 0

    for _, row in df.iterrows():
        ens = row[disease_col]
        target_id = row['targetId']

        # Resolve Ensembl -> HGNC symbol
        sym = ens_to_sym.get(target_id)
        if sym is None or pd.isna(sym):
            skipped_no_gene += 1
            continue

        # Resolve disease -> Orphanet
        disease_id = row[disease_col]
        if pd.isna(disease_id):
            skipped_no_disease += 1
            continue

        orph_num = disease_to_orph.get(disease_id)
        if orph_num is None:
            skipped_no_orph += 1
            continue

        score = row[score_col] if score_col else 1.0
        all_pairs.append((orph_num, sym, source_name, score))
        new_pairs_count += 1

    print(f"  Pairs extracted: {new_pairs_count:,}")
    print(f"  Skipped no gene mapping:    {skipped_no_gene:,}")
    print(f"  Skipped no disease ID:      {skipped_no_disease:,}")
    print(f"  Skipped no Orphanet mapping: {skipped_no_orph:,}")

# ============================================================
# Step 4: Combine + dedupe + compare with KG v2
# ============================================================
print()
print("=" * 75)
print("  Step 4: Combine and compare with KG v2")
print("=" * 75)

if not all_pairs:
    print("  No pairs extracted. Cannot proceed.")
    raise SystemExit(1)

pairs_df = pd.DataFrame(all_pairs, columns=['orphanet_id', 'gene_symbol', 'source', 'score'])
print(f"  Total pairs from all sources: {len(pairs_df):,}")

# Dedupe (orphanet_id, gene_symbol) - keep highest score
pairs_dedup = (
    pairs_df.groupby(['orphanet_id', 'gene_symbol'])
    .agg({'score': 'max', 'source': lambda s: '|'.join(sorted(set(s)))})
    .reset_index()
)
print(f"  Unique (disease, gene) pairs: {len(pairs_dedup):,}")
print(f"  Sources distribution:")
print(pairs_dedup['source'].value_counts().head(10).to_string())

# Load KG v2 and extract existing gene-disease pairs
print()
print("  Loading KG v2 for comparison...")
kg = pd.read_csv("data/processed/merged_kg_v2.tsv", sep='\t', low_memory=False)

gene_disease_relations = [
    'disease_causing_germline_mutation_s_in',
    'disease_causing_germline_mutation_s_loss_of_function_in',
    'major_susceptibility_factor_in',
    'candidate_gene_tested_in',
    'role_in_the_phenotype_of',
    'part_of_a_fusion_gene_in',
    'disease_causing_somatic_mutation_s_in',
    'disease_causing_germline_mutation_s_gain_of_function_in',
    'modifying_germline_mutation_in',
]
kg_gd = kg[
    (kg['source'] == 'orphanet') &
    (kg['relation'].isin(gene_disease_relations))
].copy()
kg_gd['orph_num'] = kg_gd['head_cui'].astype(str).str.replace('.0', '', regex=False)

kg_pairs = set(zip(kg_gd['orph_num'], kg_gd['tail']))
print(f"  KG v2 existing pairs: {len(kg_pairs):,}")

# Find NEW pairs
otg_pair_keys = set(zip(pairs_dedup['orphanet_id'], pairs_dedup['gene_symbol']))
new_keys = otg_pair_keys - kg_pairs
existing_keys = otg_pair_keys & kg_pairs

print(f"  OTG total pairs: {len(otg_pair_keys):,}")
print(f"  Already in KG v2: {len(existing_keys):,} ({100*len(existing_keys)/max(len(otg_pair_keys),1):.1f}%)")
print(f"  NEW pairs (not in KG v2): {len(new_keys):,} ({100*len(new_keys)/max(len(otg_pair_keys),1):.1f}%)")

# Filter pairs_dedup to NEW only
new_pairs = pairs_dedup[
    pairs_dedup.apply(lambda r: (r['orphanet_id'], r['gene_symbol']) in new_keys, axis=1)
].copy()

print(f"\n  New pairs by source:")
print(new_pairs['source'].value_counts().head(10).to_string())

# Sample
print(f"\n  Sample new pairs (first 10):")
for _, r in new_pairs.head(10).iterrows():
    print(f"    Orphanet:{r['orphanet_id']:>8} -- {r['gene_symbol']:<10} ({r['source']}, score={r['score']:.3f})")

# ============================================================
# Step 5: Save new pairs to TSV for KG v3 building
# ============================================================
print()
print("=" * 75)
print("  Step 5: Save new pairs for KG v3 integration")
print("=" * 75)

out_path = Path("data/processed/otg_new_gene_disease_pairs.tsv")
new_pairs.to_csv(out_path, sep='\t', index=False)
print(f"  Saved: {out_path}")
print(f"  Rows: {len(new_pairs):,}")
print(f"  File size: {out_path.stat().st_size:,} bytes")

print()
print("=" * 75)
print("DONE")
print("=" * 75)
print()
print(f"Summary:")
print(f"  KG v2 current gene-disease pairs:  {len(kg_pairs):,}")
print(f"  OTG-extracted pairs (3 sources):   {len(otg_pair_keys):,}")
print(f"  NEW pairs to add to KG v3:         {len(new_keys):,}")
print(f"  Expected KG v3 edge growth:        +{len(new_keys)*2:,} (bidirectional)")

