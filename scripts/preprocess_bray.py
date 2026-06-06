"""
Build Bray .npz datasets from raw gigadb.csv.

Pipeline:
  1. Aggregate site-level → well-level (median per well).
  2. Per-plate MAD normalization to DMSO negcon (Metadata_cpd_name == "DMSO").
  3. Join with each label file and save .npz — no clipping, no re-normalization.

Usage:
    uv run python scripts/preprocess_bray.py \
        --data_file _data/gigadb/gigadb.csv \
        --output_dir _data/gigadb
"""

import argparse

import numpy as np
import polars as pl

METADATA_KEYWORDS = {
    "Location",
    "Number",
    "Object",
    "Count",
    "EulerNumber",
    "FileName",
    "PathName",
    "URL",
    "Parent",
    "Children",
}

LABEL_EXCLUDE = {
    "image_id",
    "plate_id",
    "well_id",
    "site_id",
    "BROAD_ID",
    "canonical_smiles",
    "chembl_id",
    "hier_split",
}

LABEL_SETS = [
    ("gigadb_top_3_moas.csv", "bray_top_3_moas.npz"),
    ("gigadb_top_5_moas.csv", "bray_top_5_moas.npz"),
    ("gigadb_top_30_moas.csv", "bray_top_30_moas.npz"),
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_file", default="_data/gigadb/gigadb.csv")
    parser.add_argument("--output_dir", default="_data/gigadb")
    args = parser.parse_args()

    print("Loading schema...")
    all_cols = list(pl.scan_csv(args.data_file).collect_schema().keys())
    feature_columns = [
        col
        for col in all_cols
        if col.startswith(("Image_", "Cells_", "Cytoplasm_", "Nuclei_"))
        and not any(kw in col for kw in METADATA_KEYWORDS)
    ]
    well_keys = ["Metadata_Plate", "Metadata_Well"]
    other_metadata = [
        col
        for col in all_cols
        if col not in feature_columns
        and col not in well_keys
        and col != "Metadata_Site"
    ]
    print(
        f"  {len(feature_columns)} feature columns, {len(other_metadata)} other metadata columns"
    )

    schema_overrides = {col: pl.Float32 for col in feature_columns}

    print("Loading data...")
    data = pl.read_csv(args.data_file, schema_overrides=schema_overrides)

    print("Aggregating site-level → well-level (median per well)...")
    well_df = data.group_by(well_keys).agg(
        [pl.col(col).median().alias(col) for col in feature_columns]
        + [pl.col(col).first().alias(col) for col in other_metadata]
    )
    print(f"  Well-level rows: {len(well_df)}")

    print("Per-plate MAD normalization to DMSO negcon...")
    features_np = well_df.select(feature_columns).to_numpy().astype(np.float32)
    plate_ids = well_df["Metadata_Plate"].to_numpy()
    is_dmso = (
        (well_df["Metadata_cpd_name"].fill_null("") == "DMSO").to_numpy().astype(bool)
    )

    print(
        f"  Before: mean={features_np.mean():.3f} std={features_np.std():.3f} "
        f"min={features_np.min():.3f} max={features_np.max():.3f}"
    )

    normalized = features_np.copy()
    for plate in np.unique(plate_ids):
        plate_mask = plate_ids == plate
        dmso_mask = plate_mask & is_dmso
        if dmso_mask.sum() == 0:
            continue
        dmso_feats = features_np[dmso_mask]
        median = np.median(dmso_feats, axis=0)
        mad = np.median(np.abs(dmso_feats - median), axis=0) * 1.4826
        degenerate = mad < 1e-4
        safe_mad = np.where(degenerate, 1.0, mad)
        norm_plate = (features_np[plate_mask] - median) / safe_mad
        norm_plate[:, degenerate] = 0.0
        normalized[plate_mask] = norm_plate

    normalized = np.clip(normalized, -20, 20)

    print(
        f"  Normalized: mean={normalized.mean():.3f} std={normalized.std():.3f} "
        f"min={normalized.min():.3f} max={normalized.max():.3f}"
    )

    broad_ids = well_df["Metadata_broad_sample"].to_numpy()

    for labels_file, output_file in LABEL_SETS:
        labels_path = f"{args.output_dir}/{labels_file}"
        output_path = f"{args.output_dir}/{output_file}"
        print(f"\n{labels_file} → {output_file}")

        labels_schema = pl.scan_csv(labels_path).collect_schema()
        moa_columns = [
            col
            for col in labels_schema.keys()
            if col not in LABEL_EXCLUDE
            and not col.startswith("Unnamed:")
            and not col.startswith("hier_split_")
        ]
        print(
            f"  MoA columns ({len(moa_columns)}): {moa_columns[:5]}{'...' if len(moa_columns) > 5 else ''}"
        )

        labels_df = (
            pl.scan_csv(labels_path)
            .select(["BROAD_ID", "hier_split"] + moa_columns)
            .unique(subset=["BROAD_ID"])
            .fill_null(0)
            .collect()
        )
        label_broad = labels_df["BROAD_ID"].to_numpy()
        label_split = labels_df["hier_split"].to_numpy()
        label_moas = labels_df.select(moa_columns).to_numpy().astype(np.float32)

        # Index join: build lookup from BROAD_ID → label row index.
        label_idx = {bid: i for i, bid in enumerate(label_broad)}
        rows = [
            (i, label_idx[bid]) for i, bid in enumerate(broad_ids) if bid in label_idx
        ]
        if not rows:
            print("  WARNING: no matching BROAD_IDs, skipping.")
            continue
        data_rows, lbl_rows = zip(*rows)
        data_rows = np.array(data_rows)
        lbl_rows = np.array(lbl_rows)

        features_out = normalized[data_rows]
        labels_out = label_moas[lbl_rows]
        split_out = label_split[lbl_rows]

        print(f"  Samples: {len(features_out)}")
        splits, counts = np.unique(split_out, return_counts=True)
        for s, c in zip(splits, counts):
            print(f"    {s}: {c}")

        np.savez_compressed(
            output_path,
            features=features_out,
            labels=labels_out,
            split=split_out,
            moa_columns=np.array(moa_columns),
        )
        print(f"  Saved → {output_path}")

    print("\nDone.")


if __name__ == "__main__":
    main()
