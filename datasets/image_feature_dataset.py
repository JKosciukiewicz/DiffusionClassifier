import polars as pl
import numpy as np
import torch
from torch.utils.data import Dataset


class ImageFeatureDataset(Dataset):
    """PyTorch Dataset for cell image features obtained using dino4cells and cloome models with MOA targets.

    Loads pre-extracted features from CSV and handles split-based data filtering,
    uncertain value masking, and feature/target separation for ML training.
    """

    def __init__(
        self,
        data_file: str,
        split: str,
        split_col: str,
        transform=None,
        mask_uncertain: bool = True,
    ):
        """Initialize dataset from CSV file with specified split using Polars lazy frames.

        Args:
            data_file (str): Path to CSV containing features and MOA targets
            split (str): Data split to load ('train', 'val', 'test', 'cal')
            split_col (str): Column name containing split assignments
            transform: Optional transform to apply to features
            mask_uncertain (bool): Whether to mask uncertain (0) target values
        """
        self.transform = transform
        self.mask_uncertain = mask_uncertain

        print(
            f"Loading ImageFeatureDataset using Polars lazy frames for split='{split}'..."
        )

        # Build efficient query using single lazy frame - identify columns and filter data in one pass
        print("Building efficient query for data loading...")
        data_lazy = pl.scan_csv(data_file)
        
        # Get schema from the lazy frame without extra collection
        data_schema = data_lazy.collect_schema()

        # Identify MoA and feature columns
        self.moa_columns = [col for col in data_schema.keys() if col.startswith("moa_")]
        self.feature_columns = [
            col
            for col in data_schema.keys()
            if col not in self.moa_columns and "split" not in col and "image" not in col
        ]

        print(
            f"Found {len(self.feature_columns)} feature columns and {len(self.moa_columns)} MoA columns"
        )

        # Reuse the same lazy frame for filtered data loading
        query = (
            data_lazy
            .filter(pl.col(split_col) == split)
            .drop_nulls()
            .select(self.feature_columns + self.moa_columns)
        )

        # Execute query and collect results
        print("Executing query and loading data...")
        data_df = query.collect()

        print(f"Loaded {len(data_df)} samples for split='{split}'")

        # Convert to numpy arrays for efficient storage
        print("Converting to numpy arrays...")
        self.features_data = (
            data_df.select(self.feature_columns).to_numpy().astype(np.float32)
        )
        raw_targets = data_df.select(self.moa_columns).to_numpy().astype(np.float32)

        # Create masks for uncertain labels if needed
        if self.mask_uncertain:
            self.masks_data = (raw_targets != 0).astype(np.float32)
        else:
            self.masks_data = np.ones_like(raw_targets, dtype=np.float32)

        # Convert -1 to 0 for binary classification
        self.targets_data = np.where(raw_targets < 0, 0, raw_targets)

        print(f"Feature dimension: {len(self.feature_columns)}")
        print(f"Target dimension: {len(self.moa_columns)}")

    def __len__(self):
        """Return total number of samples in the dataset.

        Returns:
            int: Number of samples
        """
        return len(self.features_data)

    def __getitem__(self, idx):
        """Get a single sample by index.

        Args:
            idx (int): Sample index

        Returns:
            tuple: (features, targets, mask) as PyTorch tensors
        """
        features = torch.tensor(self.features_data[idx], dtype=torch.float32)
        targets = torch.tensor(self.targets_data[idx], dtype=torch.float32)
        mask = torch.tensor(self.masks_data[idx], dtype=torch.float32)

        if self.transform:
            features = self.transform(features)

        return features, targets, mask

    def get_feature_names(self):
        """Get list of feature column names.

        Returns:
            list: Feature column names from the original CSV
        """
        return self.feature_columns

    def get_target_names(self):
        """Get list of MOA target column names.

        Returns:
            list: MOA column names from the original CSV
        """
        return self.moa_columns


if __name__ == "__main__":
    # Test the implementation with actual data files
    try:
        # Try to load from one of the available data files
        test_dataset = ImageFeatureDataset(
            data_file="./data/bbbc_cloome/bbbc_cloome_dataset.csv",
            split="train",
            split_col="split",
        )
        print(f"Successfully loaded dataset with {len(test_dataset)} samples")
        print(f"Features: {len(test_dataset.get_feature_names())}")
        print(f"Targets: {len(test_dataset.get_target_names())}")

        # Test both train and test splits
        test_dataset_test = ImageFeatureDataset(
            data_file="./data/bbbc_cloome/bbbc_cloome_dataset.csv",
            split="test",
            split_col="split",
        )
        print(f"Test split: {len(test_dataset_test)} samples")

    except Exception as e:
        print(f"Test failed: {e}")
        print(
            "ImageFeatureDataset implementation is ready for use with proper data files."
        )
