"""
build_kg_v3.py - Build merged_kg_v3.tsv by appending OTG-derived new
gene-disease pairs to merged_kg_v2.tsv.

Input:
- data/processed/merged_kg_v2.tsv
- data/processed/otg_new_gene_disease_pairs.tsv (from analyze_new_evidence_v2.py)

Output:
- data/processed/merged_kg_v3.tsv (KG v2 + new edges)

Strategy:
- Read KG v2 (all 291,335 rows preserved).
- Read 4,029 new (Orphanet_id, gene_symbol) pairs from OTG analysis.
- Look up disease name from KG v2 (by head_cui) so the new rows match KG v2 format.
- Use a NEW relation 'gene_associated_with_disease_otg' to distinguish
  these from Orphanet-curated edges.
- Add source = 'opentargets' to flag provenance.
- Save merged file.
"""
from pathlib import Path
import pandas as pd

print("=" * 75)
print("  Step 1: Load KG v2")
print("=" * 75)

kg = pd.read_csv("data/processed/merged_kg_v2.tsv", sep='\t', low_memory=False)
print(f"  Loaded {len(kg):,} rows")
print(f"  Columns: {list(kg.columns)}")

# Build a lookup: head_cui (Orphanet num) -> head (disease name)
# Use the most common disease name per head_cui
kg['head_cui_str'] = kg['head_cui'].astype(str).str.replace('.0', '', regex=False)
disease_lookup = (
    kg[kg['source'] == 'orphanet']
    .groupby('head_cui_str')['head']
    .first()
    .to_dict()
)
print(f"  Built disease lookup: {len(disease_lookup):,} entries")

print()
print("=" * 75)
print("  Step 2: Load new pairs")
print("=" * 75)

new_pairs = pd.read_csv("data/processed/otg_new_gene_disease_pairs.tsv", sep='\t')
print(f"  Loaded {len(new_pairs):,} new pairs")
print(f"  Columns: {list(new_pairs.columns)}")
print(f"\n  Sample:")
print(new_pairs.head(3).to_string(index=False))

print()
print("=" * 75)
print("  Step 3: Build new KG v3 rows")
print("=" * 75)

# Build new rows matching KG v2's 6-column schema
new_rows = []
skipped_no_name = 0

for _, r in new_pairs.iterrows():
    orph_num = str(r['orphanet_id'])
    gene = r['gene_symbol']

    # Resolve disease name from KG v2 lookup
    disease_name = disease_lookup.get(orph_num)
    if disease_name is None:
        skipped_no_name += 1
        continue

    new_rows.append({
        'head': disease_name,
        'relation': 'gene_associated_with_disease_otg',
        'tail': gene,
        'head_cui': float(orph_num),
        'tail_cui': gene,
        'source': 'opentargets',
    })

print(f"  Built {len(new_rows):,} new KG rows")
print(f"  Skipped (no disease name in KG v2): {skipped_no_name:,}")

new_kg_rows = pd.DataFrame(new_rows)
print(f"\n  Sample new rows:")
print(new_kg_rows.head(3).to_string(index=False))

print()
print("=" * 75)
print("  Step 4: Append to KG v2 and save KG v3")
print("=" * 75)

# Drop the helper column before saving
kg = kg.drop(columns=['head_cui_str'])

# Concatenate
kg_v3 = pd.concat([kg, new_kg_rows], ignore_index=True)

print(f"  KG v2 rows:  {len(kg):,}")
print(f"  Added rows:  {len(new_kg_rows):,}")
print(f"  KG v3 rows:  {len(kg_v3):,}")
print(f"  Growth:      +{100*len(new_kg_rows)/len(kg):.2f}%")

# Relations in v3
print(f"\n  KG v3 relations:")
rel_counts = kg_v3['relation'].value_counts()
for rel, cnt in rel_counts.items():
    marker = "  NEW" if rel == 'gene_associated_with_disease_otg' else ""
    print(f"    {rel}: {cnt:,}{marker}")

# Sources in v3
print(f"\n  KG v3 sources:")
src_counts = kg_v3['source'].value_counts()
for src, cnt in src_counts.items():
    marker = "  NEW" if src == 'opentargets' else ""
    print(f"    {src}: {cnt:,}{marker}")

# Node counts
heads = set(kg_v3['head'].dropna().unique())
tails = set(kg_v3['tail'].dropna().unique())
all_nodes = heads | tails
print(f"\n  Unique nodes (head âˆª tail): {len(all_nodes):,}")

# Save
out_path = Path("data/processed/merged_kg_v3.tsv")
kg_v3.to_csv(out_path, sep='\t', index=False)
print(f"\n  Saved: {out_path}")
print(f"  Size: {out_path.stat().st_size:,} bytes ({out_path.stat().st_size/1024/1024:.2f} MB)")

print()
print("=" * 75)
print("DONE")
print("=" * 75)
print()
print(f"Summary:")
print(f"  KG v2:        {len(kg):,} edges")
print(f"  KG v3:        {len(kg_v3):,} edges (+{len(new_kg_rows):,} new)")
print(f"  Growth:       +{100*len(new_kg_rows)/len(kg):.2f}%")
print(f"  New relation: 'gene_associated_with_disease_otg'")
print(f"  New source:   'opentargets'")


