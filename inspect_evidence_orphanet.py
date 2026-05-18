"""
inspect_evidence_orphanet.py - Inspect the evidence_orphanet parquet file.

Shows:
- Schema (columns, types)
- Row count
- Unique diseases (Orphanet IDs)
- Unique genes (Ensembl IDs)
- Score distribution
- Sample rows
"""
from pathlib import Path
import pyarrow.parquet as pq
import pandas as pd
import numpy as np

path = Path("data/raw/opentargets/evidence_orphanet/part-00000.parquet")
print(f"File: {path}")
print(f"Size: {path.stat().st_size / 1024:.1f} KB")
print()

table = pq.read_table(path)
print(f"Row count: {table.num_rows:,}")
print(f"Column count: {table.num_columns}")
print()

# Print schema
print("Columns:")
print("-" * 60)
for i, field in enumerate(table.schema):
    type_str = str(field.type)
    if len(type_str) > 70:
        type_str = type_str[:67] + "..."
    print(f"  [{i:2d}] {field.name:<40} :: {type_str}")
print()

# Convert to pandas for analysis
df = table.to_pandas()

# Look at first 3 rows
print("First 3 rows (selected fields):")
print("-" * 60)
# Show key fields likely to exist
for i in range(min(3, len(df))):
    print(f"\n--- Row {i} ---")
    for col in df.columns[:15]:  # first 15 columns
        val = df.iloc[i][col]
        if isinstance(val, (list, np.ndarray)):
            preview = str(val)[:80]
        else:
            preview = str(val)[:120]
        print(f"  {col}: {preview}")

# Identify unique counts for key fields
print()
print("=" * 60)
print("KEY STATISTICS")
print("=" * 60)

# Disease IDs
disease_cols = [c for c in df.columns if 'disease' in c.lower() and 'id' in c.lower()]
for col in disease_cols:
    unique_count = df[col].nunique()
    sample = df[col].dropna().unique()[:5]
    print(f"\n{col}: {unique_count:,} unique values")
    print(f"  Sample: {list(sample)}")

# Target/gene IDs
target_cols = [c for c in df.columns if 'target' in c.lower() and 'id' in c.lower()]
for col in target_cols:
    unique_count = df[col].nunique()
    sample = df[col].dropna().unique()[:5]
    print(f"\n{col}: {unique_count:,} unique values")
    print(f"  Sample: {list(sample)}")

# Score-like columns
score_cols = [c for c in df.columns if 'score' in c.lower() or 'confidence' in c.lower()]
for col in score_cols:
    if df[col].dtype in ['float64', 'float32', 'int64', 'int32']:
        print(f"\n{col}:")
        print(f"  min={df[col].min():.4f}, max={df[col].max():.4f}")
        print(f"  mean={df[col].mean():.4f}, median={df[col].median():.4f}")

# Source columns
source_cols = [c for c in df.columns if 'source' in c.lower() or 'datatypeid' in c.lower() or 'datasourceid' in c.lower()]
for col in source_cols:
    if df[col].dtype == 'object':
        try:
            top = df[col].value_counts().head(5)
            print(f"\n{col}: top values:")
            for v, c in top.items():
                print(f"  {v}: {c:,}")
        except Exception:
            pass

print()
print("=" * 60)
print("DONE")
print("=" * 60)
