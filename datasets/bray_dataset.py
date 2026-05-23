import numpy as np
import polars as pl
import torch
from torch.utils.data import Dataset


class BrayDataset(Dataset):
    def __init__(
        self,
        data_file="_data/gigadb/gigadb.csv",
        labels_file="_data/gigadb/gigadb_MoA_with_images_filtered_25.csv",
        split_column: str = "hier_split",
        split_value: str = "train",
        transform=None,
        mask_uncertain: bool = True,
        treat_uncertain_as_negative: bool = False,
        noise_std: float = 0.0,  # <-- Added parameter for Gaussian noise
    ):
        """
        Memory-optimized dataset for cell morphology features to predict Mechanism of Action (MoA).
        Uses Polars with lazy frames and Float32 schema overrides for efficient loading.

        Args:
            data_file (str): Path to the CSV file with morphology features
            labels_file (str): Path to the CSV file with MoA labels
            split_column (str): Which hier_split column to use for train/test splitting
            split_value (str): Value to select from the split column ('train' or 'test')
            transform (callable, optional): Optional transform to be applied on features
            mask_uncertain (bool): If True, mask uncertain values (0) in targets during loss calculation
            treat_uncertain_as_negative (bool): If True, treat unknown/uncertain labels (0) as negative (0)
            noise_std (float): Standard deviation of Gaussian noise to add to features. Set to 0.0 for no noise.
        """
        self.transform = transform
        self.mask_uncertain = mask_uncertain
        self.treat_uncertain_as_negative = treat_uncertain_as_negative
        self.noise_std = noise_std  # <-- Store noise standard deviation

        print(
            f"Loading dataset using Polars (treat_uncertain_as_negative={treat_uncertain_as_negative})..."
        )

        # First, identify MoA columns from labels file using lazy loading
        print("Identifying MoA columns...")
        labels_schema = pl.scan_csv(labels_file).collect_schema()

        # Find all mechanism of action columns (binary targets)
        self.moa_columns = [
            col
            for col in labels_schema.keys()
            if not col.startswith("Unnamed:")
            and not col
            in [
                "image_id",
                "plate_id",
                "well_id",
                "site_id",
                "BROAD_ID",
                "canonical_smiles",
                "chembl_id",
                "hier_split",
            ]
            and not col.startswith("hier_split_")
        ]

        # Identify feature columns from data file using lazy loading
        print("Identifying feature columns...")
        data_schema = pl.scan_csv(data_file).collect_schema()

        # Strictly use only 'feat_' prefix as per user design
        self.feature_columns = [
            col for col in data_schema.keys() if col.startswith("feat_")
        ]

        # Optimization: use schema overrides to load features as Float32 directly
        # This significantly reduces memory usage during collect()
        schema_overrides = {col: pl.Float32 for col in self.feature_columns}
        # Also load MoA columns as Float32 if they exist in the schema
        for col in self.moa_columns:
            if col in data_schema:
                schema_overrides[col] = pl.Float32

        # Build the complete query using lazy frames
        print("Building lazy query for efficient data loading...")

        # Load and filter labels
        labels_query = (
            pl.scan_csv(labels_file)
            .select(["BROAD_ID", split_column] + self.moa_columns)
            .filter(pl.col(split_column) == split_value)
            .unique(subset=["BROAD_ID"])
            .fill_null(0)
        )

        # Load and process data with schema overrides and low_memory mode
        data_query = (
            pl.scan_csv(data_file, schema_overrides=schema_overrides, low_memory=True)
            .select(["Metadata_broad_sample"] + self.feature_columns)
            .fill_null(0)
            .rename({"Metadata_broad_sample": "BROAD_ID"})
        )

        # Join data with labels
        merged_query = data_query.join(labels_query, on="BROAD_ID", how="inner").select(
            ["BROAD_ID"] + self.feature_columns + self.moa_columns
        )

        # Execute the query and collect results using streaming if possible
        print("Executing query and loading data (using streaming)...")
        merged_df = merged_query.collect(streaming=True)

        print(f"Loaded {len(merged_df)} samples")

        # Convert to numpy arrays and immediately free memory
        print("Converting to numpy arrays...")
        features_array = merged_df.select(self.feature_columns).to_numpy()
        targets_raw_array = merged_df.select(self.moa_columns).to_numpy()

        # Explicitly delete the DataFrame to free memory before calculating statistics
        del merged_df

        # Create masks for uncertain values (0)
        if self.treat_uncertain_as_negative:
            # If we treat uncertain as negative, we don't need to mask them out
            masks_array = np.ones_like(targets_raw_array, dtype=np.float32)
        elif self.mask_uncertain:
            masks_array = (targets_raw_array != 0).astype(np.float32)
        else:
            masks_array = np.ones_like(targets_raw_array, dtype=np.float32)

        # Convert -1 to 0 for target values (binary classification)
        # 0 (uncertain) also stays 0, which is treated as negative if not masked
        targets_array = np.where(targets_raw_array < 0, 0, targets_raw_array)

        # Store data as numpy arrays
        self.features_data = features_array
        self.targets_data = targets_array
        self.masks_data = masks_array

        # Calculate target distribution statistics
        # self._calculate_target_statistics(targets_raw_array)

        # print(f"Dataset loaded with {len(self.features_data)} samples")
        # print(f"Feature dimension: {len(self.feature_columns)}")
        # print(f"Target dimension: {len(self.moa_columns)}")

    def __len__(self) -> int:
        return len(self.features_data)

    def _calculate_target_statistics(self, targets_raw_array):
        """Calculate target distribution statistics from raw targets array."""
        self.target_counts = {
            "ones": np.zeros(len(self.moa_columns), dtype=np.int32),
            "zeros": np.zeros(len(self.moa_columns), dtype=np.int32),
            "uncertain": np.zeros(len(self.moa_columns), dtype=np.int32),
        }

        self.moa_target_counts = {
            moa: {"ones": 0, "zeros": 0, "uncertain": 0} for moa in self.moa_columns
        }

        for i, moa_name in enumerate(self.moa_columns):
            col_data = targets_raw_array[:, i]
            ones_count = np.sum(col_data == 1)
            zeros_count = np.sum(col_data == -1)
            uncertain_count = np.sum(col_data == 0)

            self.target_counts["ones"][i] = ones_count
            self.target_counts["zeros"][i] = zeros_count
            self.target_counts["uncertain"][i] = uncertain_count

            self.moa_target_counts[moa_name]["ones"] = ones_count
            self.moa_target_counts[moa_name]["zeros"] = zeros_count
            self.moa_target_counts[moa_name]["uncertain"] = uncertain_count

        total_samples = len(targets_raw_array)
        self.total_targets = total_samples * len(self.moa_columns)
        self.total_ones = np.sum(self.target_counts["ones"])
        self.total_zeros = np.sum(self.target_counts["zeros"])
        self.total_uncertain = np.sum(self.target_counts["uncertain"])

        print("Target Distribution Summary:")
        print(f"Total target values: {self.total_targets}")
        print(
            f"Total 1s (positive): {self.total_ones} ({(self.total_ones / self.total_targets) * 100:.2f}%)"
        )
        print(
            f"Total 0s (negative): {self.total_zeros} ({(self.total_zeros / self.total_targets) * 100:.2f}%)"
        )
        print(
            f"Total uncertain: {self.total_uncertain} ({(self.total_uncertain / self.total_targets) * 100:.2f}%)"
        )

        if self.total_zeros > 0:
            imbalance_ratio = self.total_ones / self.total_zeros
            print(f"Positive-to-Negative ratio: {imbalance_ratio:.4f}")

    def __getitem__(self, idx: int) -> tuple:
        """Get a sample from the dataset."""
        features = self.features_data[idx]
        targets = self.targets_data[idx]
        mask = self.masks_data[idx]

        # Convert to tensors
        features = torch.tensor(features, dtype=torch.float32)
        targets = torch.tensor(targets, dtype=torch.float32)
        mask = torch.tensor(mask, dtype=torch.float32)

        # <-- Added: Inject Gaussian Noise if noise_std is greater than 0
        if self.noise_std > 0:
            # Adds noise sampled from N(0, noise_std^2) matching the shape of the features
            features = features + torch.randn_like(features) * self.noise_std

        # Apply transform if specified
        if self.transform:
            features = self.transform(features)

        return features, targets, mask

    def get_feature_names(self):
        return self.feature_columns

    def get_target_names(self):
        return self.moa_columns

    def get_stats(self):
        positive_counts = {}
        for i, moa in enumerate(self.moa_columns):
            positive_counts[moa] = np.sum(self.targets_data[:, i] > 0)

        stats = {
            "num_samples": len(self.features_data),
            "num_features": len(self.feature_columns),
            "num_targets": len(self.moa_columns),
            "feature_prefixes": {
                "Image_": sum(
                    1 for col in self.feature_columns if col.startswith("Image_")
                ),
                "Cells_": sum(
                    1 for col in self.feature_columns if col.startswith("Cells_")
                ),
                "Cytoplasm_": sum(
                    1 for col in self.feature_columns if col.startswith("Cytoplasm_")
                ),
                "Nuclei_": sum(
                    1 for col in self.feature_columns if col.startswith("Nuclei_")
                ),
            },
            "positive_examples_per_moa": positive_counts,
            "target_distribution": {
                "ones": self.total_ones,
                "zeros": self.total_zeros,
                "uncertain": self.total_uncertain,
                "imbalance_ratio": (
                    self.total_ones / self.total_zeros
                    if self.total_zeros > 0
                    else float("inf")
                ),
            },
        }
        return stats

    def get_target_distribution(self):
        moa_imbalance = {}
        for moa in self.moa_columns:
            ones = self.moa_target_counts[moa]["ones"]
            zeros = self.moa_target_counts[moa]["zeros"]
            imbalance_ratio = ones / zeros if zeros > 0 else float("inf")
            moa_imbalance[moa] = {
                "ones": ones,
                "zeros": zeros,
                "uncertain": self.moa_target_counts[moa]["uncertain"],
                "ratio": imbalance_ratio,
                "percent_positive": (
                    ones / (ones + zeros) * 100 if (ones + zeros) > 0 else 0
                ),
            }

        sorted_moa = sorted(
            moa_imbalance.items(), key=lambda x: x[1]["ratio"], reverse=True
        )

        return {
            "overall": {
                "total_targets": self.total_targets,
                "ones": self.total_ones,
                "zeros": self.total_zeros,
                "uncertain": self.total_uncertain,
                "percent_positive": (
                    self.total_ones / (self.total_ones + self.total_zeros) * 100
                    if (self.total_ones + self.total_zeros) > 0
                    else 0
                ),
                "imbalance_ratio": (
                    self.total_ones / self.total_zeros
                    if self.total_zeros > 0
                    else float("inf")
                ),
            },
            "per_moa": dict(sorted_moa),
        }
