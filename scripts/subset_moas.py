"""
Generate a top-N MoA .npz from the existing top-30 .npz without reprocessing raw data.

The top-30 columns are already sorted by label frequency (most → least), so
taking the first N columns gives the same top-N as running the full pipeline.

Usage:
    uv run python scripts/subset_moas.py --top_n 3
    uv run python scripts/subset_moas.py --top_n 3 --src _data/gigadb/bray_top_30_moas.npz
"""

import argparse

import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--top_n", type=int, required=True)
    parser.add_argument("--src", default="_data/gigadb/bray_top_30_moas.npz")
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    src = np.load(args.src, allow_pickle=True)
    moa_columns = src["moa_columns"]

    if args.top_n > len(moa_columns):
        raise ValueError(f"Requested top_{args.top_n} but source only has {len(moa_columns)} MoAs")

    cols = moa_columns[: args.top_n]
    labels = src["labels"][:, : args.top_n]

    out_path = args.out or f"_data/gigadb/bray_top_{args.top_n}_moas.npz"
    np.savez_compressed(
        out_path,
        features=src["features"],
        labels=labels,
        split=src["split"],
        moa_columns=cols,
    )

    splits, counts = np.unique(src["split"], return_counts=True)
    print(f"Saved top-{args.top_n} MoAs → {out_path}")
    print(f"  Columns: {list(cols)}")
    print(f"  Samples: {len(src['features'])}  {dict(zip(splits, counts))}")


if __name__ == "__main__":
    main()
