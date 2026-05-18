"""
check_otg_kg_overlap.py - Check overlap between OpenTargets evidence_orphanet
and the current KG v2.

Reports:
- How many Open Targets Orphanet diseases match existing KG v2 nodes
- How many new gene nodes would be added
- Estimated edge growth
- Sample matched / unmatched pairs
"""
from pathlib import Path
from collections import Counter
import pyarrow.parquet as pq
import pandas as pd

print("=" * 70)
print("  Step 1: Load OpenTargets evidence_orphanet")
print("=" * 70)

otg_path = Path("data/raw/opentargets/evidence_orphanet/part-00000.parquet")
otg = pq.read_table(otg_path).to_pandas()
print(f"  Loaded {len(otg):,} rows")

# Pick the columns we care about
otg = otg[['diseaseFromSourceId', 'targetId', 'diseaseFromSource',
           'targetFromSource', 'score', 'confidence']].copy()
print(f"  Columns: {list(otg.columns)}")

# Normalize Orphanet IDs - KG v2 uses "Orphanet:XXXX" format usually
print()
print("  Sample Orphanet IDs from OpenTargets:")
print(f"    {otg['diseaseFromSourceId'].head(5).tolist()}")
# Already format Orphanet_XXX, will need to convert to Orphanet:XXX for KG v2

print()
print("=" * 70)
print("  Step 2: Load KG v2 (merged_kg_v2.tsv)")
print("=" * 70)

kg_path = Path("data/processed/merged_kg_v2.tsv")
kg = pd.read_csv(kg_path, sep='\t', header=None,
                 names=['head', 'relation', 'tail'])
print(f"  Loaded {len(kg):,} edges")

# All unique nodes in KG v2
kg_nodes = set(kg['head'].unique()) | set(kg['tail'].unique())
print(f"  Unique nodes: {len(kg_nodes):,}")

# Sample nodes to see format
print()
print("  Sample KG v2 nodes (first 20):")
sample_nodes = sorted(kg_nodes)[:20]
for n in sample_nodes:
    print(f"    {n}")

# Sample relations
print()
print(f"  Unique relations: {kg['relation'].nunique()}")
rel_counts = kg['relation'].value_counts()
print(f"  Relations:")
for rel, cnt in rel_counts.items():
    print(f"    {rel}: {cnt:,}")

print()
print("=" * 70)
print("  Step 3: Try matching Orphanet IDs")
print("=" * 70)

# Convert OTG Orphanet_XXX -> Orphanet:XXX (KG v2 format)
otg['orphanet_kg'] = otg['diseaseFromSourceId'].str.replace(
    'Orphanet_', 'Orphanet:', regex=False
)

# Also try the raw underscore form in case
otg_orph_underscore = set(otg['diseaseFromSourceId'].unique())
otg_orph_colon = set(otg['orphanet_kg'].unique())

# Find overlap with KG nodes
match_colon = otg_orph_colon & kg_nodes
match_underscore = otg_orph_underscore & kg_nodes

print(f"  OTG diseases (with colon, 'Orphanet:XXX'): {len(otg_orph_colon):,}")
print(f"  OTG diseases (with underscore, 'Orphanet_XXX'): {len(otg_orph_underscore):,}")
print(f"  Match with KG (colon):     {len(match_colon):,}")
print(f"  Match with KG (underscore): {len(match_underscore):,}")

# Better match wins
if len(match_colon) >= len(match_underscore):
    chosen_format = 'colon'
    matched_diseases = match_colon
    otg['disease_kg_id'] = otg['orphanet_kg']
else:
    chosen_format = 'underscore'
    matched_diseases = match_underscore
    otg['disease_kg_id'] = otg['diseaseFromSourceId']

print()
print(f"  -> Using format: '{chosen_format}'")
print(f"  -> {len(matched_diseases):,} Orphanet diseases match KG v2 nodes")
print(f"  -> {len(otg_orph_underscore) - len(matched_diseases):,} are NEW (not in KG)")

# Sample matched
sample_matched = sorted(matched_diseases)[:10]
print()
print(f"  Sample MATCHED diseases:")
for d in sample_matched:
    name = otg[otg['disease_kg_id'] == d]['diseaseFromSource'].iloc[0]
    print(f"    {d} -> {name}")

# Sample unmatched
unmatched = (otg_orph_underscore if chosen_format == 'underscore' else otg_orph_colon) - matched_diseases
sample_unmatched = sorted(unmatched)[:10]
print()
print(f"  Sample UNMATCHED diseases (first 10):")
for d in sample_unmatched:
    name = otg[otg['disease_kg_id'] == d]['diseaseFromSource'].iloc[0] if (otg['disease_kg_id'] == d).any() else '?'
    print(f"    {d} -> {name}")

print()
print("=" * 70)
print("  Step 4: Edge addition projection")
print("=" * 70)

# Filter OTG to only matched diseases (gene-disease edges we CAN add)
otg_matched = otg[otg['disease_kg_id'].isin(matched_diseases)]
print(f"  OTG associations on matched diseases: {len(otg_matched):,}")
print(f"  Unique matched diseases: {otg_matched['disease_kg_id'].nunique():,}")
print(f"  Unique genes (ENSG): {otg_matched['targetId'].nunique():,}")

# How many genes are already in KG v2 (none should be, KG v2 doesn't have genes)
ensg_in_kg = sum(1 for g in otg_matched['targetId'].unique() if g in kg_nodes)
print(f"  Genes already in KG v2: {ensg_in_kg:,}")
print(f"  New gene nodes to add:  {otg_matched['targetId'].nunique() - ensg_in_kg:,}")

# Edge count
print()
print(f"  New edges to add (with inverse): {len(otg_matched) * 2:,}")
print(f"    (one per direction: disease_associated_gene + gene_associated_with_disease)")

print()
print("=" * 70)
print("  Step 5: KG v3 estimated size")
print("=" * 70)
new_nodes = otg_matched['targetId'].nunique() - ensg_in_kg
new_edges = len(otg_matched) * 2  # bidirectional

print(f"  Current KG v2:")
print(f"    |V| = {len(kg_nodes):,}")
print(f"    |E| = {len(kg):,}")
print(f"    |R| = {kg['relation'].nunique()}")
print()
print(f"  Projected KG v3:")
print(f"    |V| = {len(kg_nodes) + new_nodes:,} (+{new_nodes:,} genes)")
print(f"    |E| = {len(kg) + new_edges:,} (+{new_edges:,} associations)")
print(f"    |R| = {kg['relation'].nunique() + 2} (+2 new relations)")

print()
print("=" * 70)
print("DONE")
print("=" * 70)
