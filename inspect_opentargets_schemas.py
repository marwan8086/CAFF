"""
inspect_opentargets_schemas.py - Examine the schemas and sample data
from the Open Targets parquet files we just downloaded.

Three files to inspect:
1. disease.parquet              - disease metadata + cross-references
2. target/part-00000.parquet    - gene metadata sample
3. association/part-00000.parquet - gene-disease associations sample

Output: schema, row count, columns, sample rows, ID coverage.
"""
import sys
from pathlib import Path

import pyarrow.parquet as pq

ROOT = Path("data/raw/opentargets")


def inspect(path, name, max_cols_shown=30, max_sample_rows=2):
    print()
    print("=" * 75)
    print(f"  {name}")
    print(f"  Path: {path}")
    print("=" * 75)

    # Load schema only first (efficient)
    pf = pq.ParquetFile(path)
    schema = pf.schema_arrow
    n_rows = pf.metadata.num_rows
    file_size_mb = path.stat().st_size / (1024 * 1024)

    print(f"\nFile size: {file_size_mb:.2f} MB")
    print(f"Row count: {n_rows:,}")
    print(f"Column count: {len(schema.names)}")

    print(f"\nColumns:")
    print("-" * 60)
    for i, (col_name, col_type) in enumerate(zip(schema.names, schema.types)):
        type_str = str(col_type)
        if len(type_str) > 50:
            type_str = type_str[:50] + "..."
        print(f"  [{i:2}] {col_name:<35} :: {type_str}")
        if i >= max_cols_shown:
            print(f"  ... ({len(schema.names) - max_cols_shown} more)")
            break

    # Read just first few rows as sample
    print(f"\nFirst {max_sample_rows} rows (raw):")
    print("-" * 60)
    table = pf.read_row_group(0)
    df_sample = table.slice(0, max_sample_rows).to_pandas()

    for idx, row in df_sample.iterrows():
        print(f"\n--- Row {idx} ---")
        for col in df_sample.columns:
            val = row[col]
            # Truncate long values
            val_str = str(val)
            if len(val_str) > 120:
                val_str = val_str[:120] + "..."
            print(f"  {col}: {val_str}")
        if idx >= max_sample_rows - 1:
            break


def main():
    files = [
        (ROOT / "disease.parquet", "DISEASE (full file, 7 MB)"),
        (ROOT / "target" / "part-00000.parquet", "TARGET (part 0/32, 8 MB)"),
        (ROOT / "association" / "part-00000.parquet", "ASSOCIATION_OVERALL_DIRECT (part 0/N, 14 MB)"),
    ]
    for path, name in files:
        if not path.exists():
            print(f"MISSING: {path}")
            continue
        try:
            inspect(path, name)
        except Exception as e:
            print(f"ERROR inspecting {path}: {e}")

    print()
    print("=" * 75)
    print("  DONE - schemas printed above")
    print("=" * 75)


if __name__ == "__main__":
    main()
