"""
Build Bray DINOv2 .npz datasets from bray_dino_complete.csv.

Pipeline:
  1. Parse plate/well/site from image_id (format: {plate}#{well}_{site}).
  2. Aggregate site-level → well-level (median per well).
  3. Save .npz for top_3, top_5, and all 30 MoA columns.

No per-plate normalization — DINOv2 features are already bounded in [0, 1].

Usage:
    uv run python scripts/preprocess_bray_dino.py \
        --data_file _data/bray_dino/bray_dino_complete.csv \
        --output_dir _data/bray_dino
"""

import argparse

import numpy as np
import polars as pl

FEATURE_COLS = [str(i) for i in range(384)]

MOA_SETS = [
    (["moa_Phosphodiesterase", "moa_Transcription factor", "moa_Dehydrogenase"], "bray_dino_top_3_moas.npz"),
    (["moa_Phosphodiesterase", "moa_Transcription factor", "moa_Dehydrogenase", "moa_Endogenous peptide", "moa_Methyl-lysine/arginine binding protein"], "bray_dino_top_5_moas.npz"),
    (None, "bray_dino_top_30_moas.npz"),  # None = all moa_* columns
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_file", default="_data/bray_dino/bray_dino_complete.csv")
    parser.add_argument("--output_dir", default="_data/bray_dino")
    args = parser.parse_args()

    print("Loading schema...")
    all_cols = list(pl.scan_csv(args.data_file).collect_schema().keys())
    all_moa_cols = [c for c in all_cols if c.startswith("moa_")]
    hier_split_cols = [c for c in all_cols if c.startswith("hier_split_")]
    print(f"  {len(all_moa_cols)} MoA columns, {len(hier_split_cols)} split columns, {len(FEATURE_COLS)} features")

    print("Loading data...")
    df = pl.read_csv(args.data_file)

    print("Parsing plate/well/site from image_id...")
    df = df.with_columns([
        pl.col("image_id").str.split("#").list.get(0).alias("_plate"),
        pl.col("image_id").str.split("#").list.get(1).str.split("_").list.get(0).alias("_well"),
    ])

    print("Aggregating site-level → well-level (median per well)...")
    well_df = df.group_by(["_plate", "_well"]).agg(
        [pl.col(c).median().alias(c) for c in FEATURE_COLS]
        + [pl.col(c).first().alias(c) for c in all_moa_cols + hier_split_cols]
    )
    print(f"  Well-level rows: {len(well_df)}")

    features_np = well_df.select(FEATURE_COLS).to_numpy().astype(np.float32)
    # Save all 5 hier_split columns as (N, 5) array for cross-validation.
    splits_np = np.stack(
        [well_df[f"hier_split_{k}"].to_numpy() for k in range(5)], axis=1
    )

    print(f"  Feature stats: mean={features_np.mean():.3f} std={features_np.std():.3f} "
          f"min={features_np.min():.3f} max={features_np.max():.3f}")
    for k in range(5):
        vals, counts = np.unique(splits_np[:, k], return_counts=True)
        print(f"  hier_split_{k}: " + ", ".join(f"{v}={c}" for v, c in zip(vals, counts)))

    for moa_cols, output_file in MOA_SETS:
        if moa_cols is None:
            moa_cols = all_moa_cols
        output_path = f"{args.output_dir}/{output_file}"
        labels_np = well_df.select(moa_cols).to_numpy().astype(np.float32)
        np.savez_compressed(
            output_path,
            features=features_np,
            labels=labels_np,
            split=splits_np[:, 0],   # backward-compat: fold 0
            splits=splits_np,         # all 5 folds, shape (N, 5)
            moa_columns=np.array(moa_cols),
        )
        print(f"  Saved {output_file} ({len(moa_cols)} MoAs, {len(features_np)} wells)")

    print("\nDone.")


if __name__ == "__main__":
    main()
