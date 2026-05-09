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
    ):
        """
        Memory-optimized dataset for cell morphology features to predict Mechanism of Action (MoA).
        Uses Polars with lazy frames for efficient loading.

        Args:
            data_file (str): Path to the CSV file with morphology features
            labels_file (str): Path to the CSV file with MoA labels
            split_column (str): Which hier_split column to use for train/test splitting
            split_value (str): Value to select from the split column ('train' or 'test')
            transform (callable, optional): Optional transform to be applied on features
            mask_uncertain (bool): If True, mask uncertain values (0) in targets during loss calculation
        """
        self.transform = transform
        self.mask_uncertain = mask_uncertain

        print("Loading dataset using Polars lazy frames...")

        # First, identify MoA columns from labels file using lazy loading
        print("Identifying MoA columns...")
        labels_lazy = pl.scan_csv(labels_file)
        labels_schema = labels_lazy.collect_schema()

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
        data_lazy = pl.scan_csv(data_file)
        data_schema = data_lazy.collect_schema()

        self.feature_columns = [
            col
            for col in data_schema.keys()
            if col.startswith(("Image_", "Cells_", "Cytoplasm_", "Nuclei_"))
        ]

        # Build the complete query using lazy frames
        print("Building lazy query for efficient data loading...")

        # Load and filter labels
        labels_query = (
            pl.scan_csv(labels_file)
            .select(["BROAD_ID", split_column] + self.moa_columns)
            .filter(pl.col(split_column) == split_value)
            .unique(subset=["BROAD_ID"])
        )

        # Load and process data
        data_query = (
            pl.scan_csv(data_file)
            .select(["Metadata_broad_sample"] + self.feature_columns)
            .rename({"Metadata_broad_sample": "BROAD_ID"})
        )

        # Join data with labels
        merged_query = data_query.join(labels_query, on="BROAD_ID", how="inner").select(
            ["BROAD_ID"] + self.feature_columns + self.moa_columns
        )

        # Execute the query and collect results
        print("Executing query and loading data...")
        merged_df = merged_query.collect()

        print(f"Loaded {len(merged_df)} samples")

        # Convert to numpy arrays for efficient storage
        print("Converting to numpy arrays...")
        features_array = (
            merged_df.select(self.feature_columns).to_numpy().astype(np.float32)
        )
        targets_raw_array = (
            merged_df.select(self.moa_columns).to_numpy().astype(np.float32)
        )

        # Create masks for uncertain values (0)
        if self.mask_uncertain:
            masks_array = (targets_raw_array != 0).astype(np.float32)
        else:
            masks_array = np.ones_like(targets_raw_array, dtype=np.float32)

        # Convert -1 to 0 for target values (binary classification)
        targets_array = np.where(targets_raw_array < 0, 0, targets_raw_array)

        # Store data as numpy arrays
        self.features_data = features_array
        self.targets_data = targets_array
        self.masks_data = masks_array

        # Calculate target distribution statistics
        self._calculate_target_statistics(targets_raw_array)

        print(f"Dataset loaded with {len(self.features_data)} samples")
        print(f"Feature dimension: {len(self.feature_columns)}")
        print(f"Target dimension: {len(self.moa_columns)}")

    def __len__(self) -> int:
        return len(self.features_data)

    def _calculate_target_statistics(self, targets_raw_array):
        """Calculate target distribution statistics from raw targets array."""
        # Initialize counters for target values
        self.target_counts = {
            "ones": np.zeros(len(self.moa_columns), dtype=np.int32),
            "zeros": np.zeros(len(self.moa_columns), dtype=np.int32),
            "uncertain": np.zeros(len(self.moa_columns), dtype=np.int32),
        }

        # Dictionary to store counts by MoA name
        self.moa_target_counts = {
            moa: {"ones": 0, "zeros": 0, "uncertain": 0} for moa in self.moa_columns
        }

        # Count occurrences of each value in the raw targets
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

        # Calculate total counts
        total_samples = len(targets_raw_array)
        self.total_targets = total_samples * len(self.moa_columns)
        self.total_ones = np.sum(self.target_counts["ones"])
        self.total_zeros = np.sum(self.target_counts["zeros"])
        self.total_uncertain = np.sum(self.target_counts["uncertain"])

        # Print target distribution summary
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

        # Calculate overall class imbalance ratio
        if self.total_zeros > 0:
            imbalance_ratio = self.total_ones / self.total_zeros
            print(f"Positive-to-Negative ratio: {imbalance_ratio:.4f}")

    def __getitem__(self, idx: int) -> tuple:
        """Get a sample from the dataset."""
        # Get precomputed features, targets and mask
        features = self.features_data[idx]
        targets = self.targets_data[idx]
        mask = self.masks_data[idx]

        # Convert to tensors
        features = torch.tensor(features, dtype=torch.float32)
        targets = torch.tensor(targets, dtype=torch.float32)
        mask = torch.tensor(mask, dtype=torch.float32)

        # Apply transform if specified
        if self.transform:
            features = self.transform(features)

        return features, targets, mask

    def get_feature_names(self):
        """Returns a list of all feature column names"""
        return self.feature_columns

    def get_target_names(self):
        """Returns a list of all MoA target names"""
        return self.moa_columns

    def get_stats(self):
        """Returns basic statistics about the dataset"""
        # Count positive examples per MoA
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
        """Returns detailed target distribution statistics"""
        # Calculate imbalance ratios for each MoA
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

        # Sort by imbalance ratio (most imbalanced first)
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


# if __name__ == "__main__":
#     # Create training dataset
#     train_dataset = BrayDataset(
#         data_file="_data/gigadb/gigadb.csv",
#         labels_file="_data/gigadb/gigadb_MoA_with_images_filtered_25.csv",
#         split_column="hier_split",
#         split_value="train",
#     )

#     # Create test dataset
#     test_dataset = BrayDataset(
#         data_file="_data/gigadb/gigadb.csv",
#         labels_file="_data/gigadb/gigadb_MoA_with_images_filtered_25.csv",
#         split_column="hier_split",
#         split_value="test",
#     )

#     # Print the lengths of both datasets
#     print("Dataset Sizes:")
#     print(f"Training set size: {len(train_dataset)} samples")
#     print(f"Test set size: {len(test_dataset)} samples")

#     # Print more detailed statistics if needed
#     train_stats = train_dataset.get_stats()
#     test_stats = test_dataset.get_stats()

#     print("\nTraining Set Features:")
#     print(f"Number of features: {train_stats['num_features']}")
#     print(f"Number of target MoAs: {train_stats['num_targets']}")

#     print("\nTest Set Features:")
#     print(f"Number of features: {test_stats['num_features']}")
#     print(f"Number of target MoAs: {test_stats['num_targets']}")
