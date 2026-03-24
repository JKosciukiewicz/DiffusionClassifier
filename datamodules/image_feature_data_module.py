from torch.utils.data import DataLoader
from datasets.image_feature_dataset import (
    ImageFeatureDataset,
)  # Assuming your dataset is in this file
from datamodules.base_data_module import BaseDataModule


class ImageFeatureDataModule(BaseDataModule):
    """Lightning DataModule for cell image features from dino4cells and cloome models.

    Handles data loading and preprocessing for cell feature datasets with support for
    train/validation/test/calibration splits and uncertain value masking.
    """

    def __init__(
        self,
        data_csv_path: str,
        split_col: str,
        train_split_val: str = "train",
        test_split_val: str = "test",
        calibration_split_val: str = "cal",
        validation_split_val: str = "val",
        batch_size: int = 64,
        mask_uncertain: bool = True,
    ):
        """
        Lightning DataModule for datasets obtained using dino4cells and cloome models

        Args:
            batch_size (int): The batch size for the dataloaders.
            cal_split (float): The proportion of the training data to use for the calibration set.
            test_split (float): The proportion of the total data to use for the test set.
            mask_uncertain (bool): Whether to mask uncertain values (0) during training.
        """
        super().__init__()
        self.batch_size = batch_size
        self.mask_uncertain = mask_uncertain
        self.split_col = split_col

        # Define file paths
        self.data_file = data_csv_path

        self.train_split_val = train_split_val
        self.test_split_val = test_split_val
        self.calibration_split_val = calibration_split_val
        self.validation_split_val = validation_split_val

        # Define transforms (can be customized)
        self.train_transform = None
        self.test_transform = None

        # Placeholders for datasets
        self.train_dataset = None
        self.cal_dataset = None
        self.test_dataset = None

    def setup(self, stage: str):
        """Set up dataset splits for the specified training stage.

        Creates appropriate dataset instances based on the stage. For 'fit' stage,
        sets up training and validation datasets. For 'test' stage, sets up test
        and calibration datasets. For 'val' stage, sets up validation dataset only.

        Args:
            stage (str): Training stage - 'fit', 'test', or 'val'
        """
        print(f"--- Setting up data for stage: {stage} ---")
        if stage == "fit":
            self.train_dataset = ImageFeatureDataset(
                data_file=self.data_file,
                split_col=self.split_col,
                split=self.train_split_val,
                transform=self.train_transform,
                mask_uncertain=self.mask_uncertain,
            )
            self.val_dataset = ImageFeatureDataset(
                data_file=self.data_file,
                split_col=self.split_col,
                split=self.validation_split_val,
                transform=self.test_transform,
                mask_uncertain=self.mask_uncertain,
            )
            print(f"Training set size: {len(self.train_dataset)} samples")
            print(f"Val set size: {len(self.val_dataset)} samples")

        if stage == "test":
            self.test_dataset = ImageFeatureDataset(
                data_file=self.data_file,
                split_col=self.split_col,
                split=self.test_split_val,
                transform=self.test_transform,
                mask_uncertain=self.mask_uncertain,
            )
            print(f"Test set size: {len(self.test_dataset)} samples")

            self.cal_dataset = ImageFeatureDataset(
                data_file=self.data_file,
                split_col=self.split_col,
                split=self.calibration_split_val,
                transform=self.test_transform,
                mask_uncertain=self.mask_uncertain,
            )
            print(f"Cal set size: {len(self.cal_dataset)} samples")

        if stage == "val":
            self.val_dataset = ImageFeatureDataset(
                data_file=self.data_file,
                split_col=self.split_col,
                split="val",
                transform=self.test_transform,
                mask_uncertain=self.mask_uncertain,
            )
            print(f"Val set size: {len(self.val_dataset)} samples")

        print("--- Data setup complete ---")

    def train_dataloader(self):
        """Create and return DataLoader for training data.

        Returns:
            DataLoader: Training data loader with shuffling enabled and
                       configured batch size, workers, and memory pinning
        """
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=4,  # Adjust based on your system
            pin_memory=True,
        )

    def val_dataloader(self):
        """Create and return DataLoader for validation data.

        Returns:
            DataLoader: Validation data loader without shuffling and
                       configured batch size, workers, and memory pinning
        """
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=0,  # Adjust based on your system
            pin_memory=True,
        )

    def calibration_dataloader(self):
        """Create and return DataLoader for calibration data.

        Returns:
            DataLoader: Calibration data loader without shuffling and
                       configured batch size, workers, and memory pinning
        """
        return DataLoader(
            self.cal_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=0,  # Adjust based on your system
            pin_memory=True,
        )

    def test_dataloader(self):
        """Create and return DataLoader for test data.

        Returns:
            DataLoader: Test data loader without shuffling and
                       configured batch size, workers, and memory pinning
        """
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=0,  # Adjust based on your system
            pin_memory=True,
        )


if __name__ == "__main__":
    data_module = ImageFeatureDataModule(
        data_csv_path="./data/bray_dino/bray_dino_complete.csv",
        batch_size=32,
        split_col="hier_split_0",
    )

    print("Setting up for training...")
    data_module.setup(stage="fit")
    train_loader = data_module.train_dataloader()
    features, targets, mask = next(iter(train_loader))
    print(f"Feature batch shape: {features.shape}")
    print(f"Target batch shape: {targets.shape}")
    print(f"Mask batch shape: {mask.shape}")
    print("---------------------------------")

    print("\nSetting up for testing...")
    data_module.setup(stage="test")
    test_loader = data_module.test_dataloader()
    features, targets, mask = next(iter(test_loader))
    print(f"Feature batch shape: {features.shape}")
    print(f"Target batch shape: {targets.shape}")
    print(f"Mask batch shape: {mask.shape}")
    print("-------------------------------")
    print("\nSetting up for calibration...")
    cal_loader = data_module.calibration_dataloader()
    features, targets, mask = next(iter(cal_loader))
    print(f"Feature batch shape: {features.shape}")
    print(f"Target batch shape: {targets.shape}")
    print(f"Mask batch shape: {mask.shape}")
    print("-------------------------------")
    print("\nSetting up for validation...")
    data_module.setup(stage="val")
    val_loader = data_module.val_dataloader()
    features, targets, mask = next(iter(val_loader))
    print(f"Feature batch shape: {features.shape}")
    print(f"Target batch shape: {targets.shape}")
    print(f"Mask batch shape: {mask.shape}")
    print("-------------------------------")
