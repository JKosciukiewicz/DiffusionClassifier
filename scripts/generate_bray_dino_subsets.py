import polars as pl
import os

def generate_subsets(input_file, output_dir):
    print(f"Reading data from {input_file}...")
    df = pl.read_csv(input_file)
    
    moa_cols = [col for col in df.columns if col.startswith("moa_")]
    print(f"Found {len(moa_cols)} MoA columns.")

    # Count (1 and -1) for each MoA
    counts = {}
    for col in moa_cols:
        # 1 is active, -1 is inactive
        total = (df[col] == 1).sum() + (df[col] == -1).sum()
        counts[col] = total

    # Sort by count
    sorted_moas = sorted(counts.items(), key=lambda x: x[1], reverse=True)

    print("\nTop 10 MoAs by (active + inactive) count:")
    for i, (moa, count) in enumerate(sorted_moas[:10], 1):
        active = (df[moa] == 1).sum()
        inactive = (df[moa] == -1).sum()
        print(f"{i}. {moa}: {count} total ({active} active, {inactive} inactive)")

    feature_cols = [col for col in df.columns if not col.startswith("moa_") and "split" not in col and col != "image_id"]
    split_cols = [col for col in df.columns if "split" in col]

    # Rename 'cal' to 'val' in all split columns
    for col in split_cols:
        df = df.with_columns(
            pl.col(col).replace("cal", "val")
        )
    print("Renamed 'cal' split to 'val' in all split columns.")

    def save_subset(top_n, filename):
        subset_moas = [moa for moa, count in sorted_moas[:top_n]]
        selected_cols = ["image_id"] + subset_moas + split_cols + feature_cols
        df_subset = df.select(selected_cols)
        
        output_path = os.path.join(output_dir, filename)
        df_subset.write_csv(output_path)
        print(f"\nSaved top {top_n} MoA subset to {output_path}")
        print(f"Selected MoAs: {subset_moas}")

    save_subset(3, "bray_dino_top_3.csv")
    save_subset(5, "bray_dino_top_5.csv")

if __name__ == "__main__":
    input_file = "_data/bray_dino/bray_dino_complete.csv"
    output_dir = "_data/bray_dino"
    generate_subsets(input_file, output_dir)
