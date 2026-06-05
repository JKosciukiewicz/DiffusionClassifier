"""
Preprocess Bray data: join features + labels once and save to .npz.

Usage:
    uv run python scripts/preprocess_bray.py \
        --data_file _data/gigadb/gigadb_well_level.csv \
        --labels_file _data/gigadb/gigadb_top_5_moas.csv \
        --output _data/gigadb/bray_top_5_moas.npz
"""
import argparse

import numpy as np
import polars as pl

LABEL_EXCLUDE = {
    "image_id", "plate_id", "well_id", "site_id",
    "BROAD_ID", "canonical_smiles", "chembl_id", "hier_split",
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_file", required=True)
    parser.add_argument("--labels_file", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--normalize",
        choices=["none", "robust", "zscore"],
        default="robust",
        help="Per-feature re-normalization fit on the train split. "
        "'robust' = median/IQR (outlier-resistant), 'zscore' = mean/std.",
    )
    parser.add_argument(
        "--clip",
        type=float,
        default=10.0,
        help="Clip normalized features to [-clip, clip] to bound residual "
        "outliers; set <= 0 to disable.",
    )
    args = parser.parse_args()

    print("Scanning schemas...")
    labels_schema = pl.scan_csv(args.labels_file).collect_schema()
    moa_columns = [
        col for col in labels_schema.keys()
        if col not in LABEL_EXCLUDE
        and not col.startswith("Unnamed:")
        and not col.startswith("hier_split_")
    ]
    print(f"  MoA columns ({len(moa_columns)}): {moa_columns}")

    data_schema = pl.scan_csv(args.data_file).collect_schema()
    feature_columns = [col for col in data_schema.keys() if col.startswith("feat_")]
    print(f"  Feature columns: {len(feature_columns)}")

    schema_overrides = {col: pl.Float32 for col in feature_columns}

    print("Joining data + labels (no split filter — keeping all rows)...")
    labels_df = (
        pl.scan_csv(args.labels_file)
        .select(["BROAD_ID", "hier_split"] + moa_columns)
        .unique(subset=["BROAD_ID"])
        .fill_null(0)
    )
    data_df = (
        pl.scan_csv(args.data_file, schema_overrides=schema_overrides, low_memory=True)
        .select(["Metadata_broad_sample"] + feature_columns)
        .fill_null(0)
        .rename({"Metadata_broad_sample": "BROAD_ID"})
    )

    merged = (
        data_df.join(labels_df, on="BROAD_ID", how="inner")
        .select(["hier_split"] + feature_columns + moa_columns)
        .collect(streaming=True)
    )
    print(f"Joined: {len(merged)} samples")

    features = merged.select(feature_columns).to_numpy().astype(np.float32)
    labels = merged.select(moa_columns).to_numpy().astype(np.float32)
    split = np.array(merged["hier_split"].to_list())

    # Re-normalize features per feature using TRAIN-split statistics (applied to
    # all splits — no leakage). The source data is already z-scored but carries
    # extreme outliers (~ -514..4670) from near-degenerate / artefact features;
    # robust median/IQR scaling is outlier-resistant, and the clip bounds spikes.
    if args.normalize != "none":
        train_mask = split == "train"
        if train_mask.sum() == 0:
            raise ValueError("No rows with hier_split == 'train' to fit normalization.")
        train_feats = features[train_mask]
        if args.normalize == "robust":
            center = np.median(train_feats, axis=0)
            q25, q75 = np.percentile(train_feats, [25, 75], axis=0)
            scale = q75 - q25
        else:  # zscore
            center = train_feats.mean(axis=0)
            scale = train_feats.std(axis=0)
        scale = np.where(scale < 1e-6, 1.0, scale)  # leave dead features alone
        features = (features - center) / scale
        if args.clip > 0:
            features = np.clip(features, -args.clip, args.clip)
        features = features.astype(np.float32)
        print(
            f"Normalized ({args.normalize}, clip={args.clip}): "
            f"mean={features.mean():.3f} std={features.std():.3f} "
            f"min={features.min():.3f} max={features.max():.3f}"
        )

    print(f"Saving to {args.output} ...")
    np.savez_compressed(
        args.output,
        features=features,
        labels=labels,
        split=split,
        moa_columns=np.array(moa_columns),
    )

    splits, counts = np.unique(split, return_counts=True)
    for s, c in zip(splits, counts):
        print(f"  {s}: {c} samples")
    print("Done.")


if __name__ == "__main__":
    main()
