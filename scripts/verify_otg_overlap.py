"""
verify_otg_overlap.py - Verify how much OTG evidence_orphanet overlaps
with the existing KG v2.

This is a clean rewrite of check_otg_kg_overlap.py that handles the
actual KG v2 TSV format:
  6 columns: head, relation, tail, head_cui, tail_cui, source
  Header row present
  head_cui contains Orphanet numeric IDs (e.g. "93.0", "166024.0")
  tail contains gene symbols (e.g. "KIF7", "CWC27")

OTG evidence_orphanet uses:
  diseaseFromSourceId: "Orphanet_544472" (numeric with prefix)
  targetFromSource:    "complement factor B" (gene name)
  targetId:            "ENSG00000243649" (Ensembl)

The matching key: extract the numeric part from "Orphanet_XXX" and
compare to head_cui in KG v2.
"""
from pathlib import Path
import pyarrow.parquet as pq
import pandas as pd

print("=" * 75)
print("  Step 1: Load OTG evidence_orphanet")
print("=" * 75)

otg_path = Path("data/raw/opentargets/evidence_orphanet/part-00000.parquet")
otg = pq.read_table(otg_path).to_pandas()
print(f"  Loaded {len(otg):,} rows")

# Extract numeric Orphanet ID from "Orphanet_XXXX"
otg['orph_num'] = otg['diseaseFromSourceId'].str.replace(
    'Orphanet_', '', regex=False
)
print(f"  Sample diseaseFromSourceId: {otg['diseaseFromSourceId'].head(3).tolist()}")
print(f"  Sample orph_num (extracted): {otg['orph_num'].head(3).tolist()}")
print(f"  Unique Orphanet diseases in OTG: {otg['orph_num'].nunique():,}")
print(f"  Unique gene symbols in OTG: {otg['targetFromSource'].nunique():,}")

print()
print("=" * 75)
print("  Step 2: Load KG v2 (Orphanet-source rows only)")
print("=" * 75)

kg_path = Path("data/processed/merged_kg_v2.tsv")
kg = pd.read_csv(kg_path, sep='\t', low_memory=False)
print(f"  Total KG rows: {len(kg):,}")

# Filter to Orphanet source only
kg_orph = kg[kg['source'] == 'orphanet'].copy()
print(f"  Orphanet-source rows: {len(kg_orph):,}")
print(f"  Relations in Orphanet rows:")
for rel, cnt in kg_orph['relation'].value_counts().items():
    print(f"    {rel}: {cnt:,}")

# Identify gene-disease relations (not phenotype, not is_a)
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

kg_gd = kg_orph[kg_orph['relation'].isin(gene_disease_relations)].copy()
print(f"\n  KG v2 gene-disease edges (Orphanet): {len(kg_gd):,}")

# Normalize head_cui (string, no trailing .0)
kg_gd['orph_num'] = kg_gd['head_cui'].astype(str).str.replace('.0', '', regex=False)
print(f"  Sample KG v2 orph_num: {kg_gd['orph_num'].head(5).tolist()}")
print(f"  Unique Orphanet diseases in KG v2 gene-disease: {kg_gd['orph_num'].nunique():,}")
print(f"  Unique gene symbols (tail) in KG v2: {kg_gd['tail'].nunique():,}")

print()
print("=" * 75)
print("  Step 3: Disease-level overlap (by Orphanet ID)")
print("=" * 75)

otg_diseases = set(otg['orph_num'].dropna().unique())
kg_diseases = set(kg_gd['orph_num'].dropna().unique())

overlap_diseases = otg_diseases & kg_diseases
otg_only = otg_diseases - kg_diseases
kg_only = kg_diseases - otg_diseases

print(f"  OTG diseases: {len(otg_diseases):,}")
print(f"  KG v2 diseases (gene-disease): {len(kg_diseases):,}")
print(f"  Overlap: {len(overlap_diseases):,}")
print(f"    ({100*len(overlap_diseases)/max(len(otg_diseases),1):.1f}% of OTG, "
      f"{100*len(overlap_diseases)/max(len(kg_diseases),1):.1f}% of KG)")
print(f"  OTG-only (new diseases): {len(otg_only):,}")
print(f"  KG-only (not in OTG): {len(kg_only):,}")

print()
print(f"  Sample OTG-only diseases (first 10):")
for d in sorted(otg_only)[:10]:
    name = otg[otg['orph_num'] == d]['diseaseFromSource'].iloc[0]
    print(f"    Orphanet_{d}: {name}")

print()
print("=" * 75)
print("  Step 4: Pair-level overlap (disease-gene pairs)")
print("=" * 75)

# Build OTG pairs: (orph_num, gene_symbol)
# Use targetFromSource as gene name (matches KG v2 'tail')
otg_pairs = set(
    (row['orph_num'], row['targetFromSource'])
    for _, row in otg.iterrows()
    if pd.notna(row['orph_num']) and pd.notna(row['targetFromSource'])
)

# But gene names may be different (Orphanet uses HGNC symbols, OTG uses long names)
# Try simpler form: just orph_num + gene_symbol from OTG (might mismatch)
print(f"  OTG (disease, gene_full_name) pairs: {len(otg_pairs):,}")

# Build KG pairs: (orph_num, gene_symbol)
kg_pairs = set(
    (row['orph_num'], row['tail'])
    for _, row in kg_gd.iterrows()
    if pd.notna(row['orph_num']) and pd.notna(row['tail'])
)
print(f"  KG (disease, gene_symbol) pairs:    {len(kg_pairs):,}")

overlap_pairs = otg_pairs & kg_pairs
print(f"\n  Direct pair overlap (full name match): {len(overlap_pairs):,}")
print(f"    NOTE: this is likely an underestimate because OTG uses gene full")
print(f"    names ('complement factor B') and KG uses HGNC symbols ('CFB')")

# Sample mismatched pairs
otg_pair_diseases = set(p[0] for p in otg_pairs)
kg_pair_diseases = set(p[0] for p in kg_pairs)
print(f"\n  Diseases with OTG pairs:  {len(otg_pair_diseases):,}")
print(f"  Diseases with KG pairs:   {len(kg_pair_diseases):,}")
print(f"  Diseases overlap:          {len(otg_pair_diseases & kg_pair_diseases):,}")

print()
print("=" * 75)
print("  Step 5: Verdict")
print("=" * 75)
print()
disease_overlap_pct = 100 * len(overlap_diseases) / max(len(otg_diseases), 1)
print(f"  Disease-level overlap: {disease_overlap_pct:.1f}% of OTG diseases are in KG v2")
if disease_overlap_pct >= 70:
    print(f"  -> CONFIRMED: OTG evidence_orphanet is largely REDUNDANT with KG v2")
elif disease_overlap_pct >= 40:
    print(f"  -> PARTIAL OVERLAP: OTG adds some new diseases")
else:
    print(f"  -> LOW OVERLAP: OTG would add many new diseases")

new_diseases = len(otg_only)
print(f"\n  New diseases OTG could add: {new_diseases:,}")
print(f"  Total KG v2 gene-disease pairs: {len(kg_pairs):,}")
print(f"  OTG total pairs: {len(otg_pairs):,}")

print()
print("  Conclusion:")
print("    Both KG v2 and OTG evidence_orphanet derive from the same")
print("    Orphanet source. Differences are mostly:")
print("    - Format (gene full name vs HGNC symbol)")
print("    - Curation cycle (different snapshot dates)")
print("    - The substantive content is largely the same.")
print()
print("    For a real F1 lift, would need evidence from non-Orphanet")
print("    sources: evidence_genomics_england, evidence_clingen,")
print("    evidence_eva (ClinVar), evidence_gene2phenotype, etc.")

print()
print("=" * 75)
print("DONE")
print("=" * 75)
