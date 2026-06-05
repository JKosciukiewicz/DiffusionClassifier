"""
Creates a filtered version of gigadb_top_30_moas.csv:
rows sorted by number of missing labels (0 values) ascending, top 5000 rows.

Label encoding: 1 = true, -1 = false, 0 = missing
"""

import polars as pl
from pathlib import Path

DATA_DIR = Path(__file__).parent.parent / "_data" / "gigadb"
INPUT = DATA_DIR / "gigadb_top_30_moas.csv"
OUTPUT = DATA_DIR / "gigadb_top5000_least_missing.csv"

df = pl.read_csv(INPUT)

moa_cols = [c for c in df.columns if c.startswith("moa_")]

result = (
    df.with_columns(
        pl.sum_horizontal([(pl.col(c) == 0).cast(pl.Int32) for c in moa_cols])
        .alias("missing_count")
    )
    .sort("missing_count")
    .head(5000)
)

print(f"Missing count distribution in selected rows:")
print(result["missing_count"].value_counts().sort("missing_count"))

result = result.drop("missing_count")
result.write_csv(OUTPUT)
print(f"\nSaved {len(result)} rows to {OUTPUT}")
