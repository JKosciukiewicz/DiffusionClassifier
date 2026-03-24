import torch
from torch.utils.data import DataLoader, random_split
from datasets.bbbc021_dataset import (
    BBBC021Dataset,
)  # Assuming your dataset is in this file
from datamodules.base_data_module import BaseDataModule


class BBBC021DataModule(BaseDataModule):
    def __init__(
        self,
        batch_size: int = 64,
        data_dir: str = "./data/BBBC021",
        cal_split: float = 0.2,
        test_split: float = 0.2,
        mask_uncertain: bool = True,
        seed: int = 42,
    ):
        """
        Lightning DataModule for the BBBC021Dataset.

        Args:
            batch_size (int): The batch size for the dataloaders.
            data_dir (str): The base directory for the data files.
            cal_split (float): The proportion of the training data to use for the calibration set.
            test_split (float): The proportion of the total data to use for the test set.
            mask_uncertain (bool): Whether to mask uncertain values (0) during training.
            seed (int): The random seed for reproducibility of splits.
        """
        super().__init__()
        self.batch_size = batch_size
        self.data_dir = data_dir
        self.cal_split = cal_split
        self.test_split = test_split
        self.mask_uncertain = mask_uncertain
        self.seed = seed

        # Define file paths
        self.data_file = f"{self.data_dir}/BBBC021_dataset_complete_one_fold.csv"

        # Define transforms (can be customized)
        self.train_transform = None
        self.test_transform = None

        # Placeholders for datasets
        self.train_dataset = None
        self.cal_dataset = None
        self.test_dataset = None

    def setup(self, stage: str):
        """
        Set up the dataset splits for the given stage ('fit' or 'test').

        Args:
            stage (str): Either 'fit' (for training/calibration) or 'test'.
        """
        print(f"--- Setting up data for stage: {stage} ---")
        if stage == "fit":
            # Create the full training dataset using the 'train' split from BBBC021Dataset
            self.train_dataset = BBBC021Dataset(
                data_file=self.data_file,
                split="train",
                transform=self.train_transform,
                mask_uncertain=self.mask_uncertain,
            )
            print(f"Training set size: {len(self.train_dataset)} samples")
            self.val_dataset = BBBC021Dataset(
                data_file=self.data_file,
                split="val",
                transform=self.test_transform,
                mask_uncertain=self.mask_uncertain,
            )
            print(f"Val set size: {len(self.val_dataset)} samples")

        if stage == "test":
            # Create the test dataset
            self.test_dataset = BBBC021Dataset(
                data_file=self.data_file,
                split="test",
                transform=self.test_transform,
                mask_uncertain=self.mask_uncertain,
            )
            print(f"Test set size: {len(self.test_dataset)} samples")

            # Create the test dataset
            self.cal_dataset = BBBC021Dataset(
                data_file=self.data_file,
                split="cal",
                transform=self.test_transform,
                mask_uncertain=self.mask_uncertain,
            )
            print(f"Cal set size: {len(self.cal_dataset)} samples")

        if stage == "val":
            # Create the test dataset
            self.val_dataset = BBBC021Dataset(
                data_file=self.data_file,
                split="val",
                transform=self.test_transform,
                mask_uncertain=self.mask_uncertain,
            )
            print(f"Val set size: {len(self.val_dataset)} samples")

        print("--- Data setup complete ---")

    def train_dataloader(self):
        """Returns the DataLoader for the training set."""
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=0,  # Adjust based on your system
            pin_memory=True,
        )

    def val_dataloader(self):
        """Returns the DataLoader for the calibration/validation set."""
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=0,  # Adjust based on your system
            pin_memory=True,
        )

    def calibration_dataloader(self):
        """Returns the DataLoader for the calibration/validation set."""
        return DataLoader(
            self.cal_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=0,  # Adjust based on your system
            pin_memory=True,
        )

    def test_dataloader(self):
        """Returns the DataLoader for the test set."""
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=0,  # Adjust based on your system
            pin_memory=True,
        )


if __name__ == "__main__":
    data_module = BBBC021DataModule(
        data_dir="./data/BBBC021",
        batch_size=32,
        test_split=0.2,
        cal_split=0.15,
        seed=42,
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

    print("\nSetting up for validation...")
    data_module.setup(stage="val")
    val_loader = data_module.val_dataloader()
    features, targets, mask = next(iter(val_loader))
    print(f"Feature batch shape: {features.shape}")
    print(f"Target batch shape: {targets.shape}")
    print(f"Mask batch shape: {mask.shape}")
    print("-------------------------------")

    print("\nSetting up for calibration...")
    data_module.setup(stage="test")
    cal_loader = data_module.calibration_dataloader()
    features, targets, mask = next(iter(cal_loader))
    print(f"Feature batch shape: {features.shape}")
    print(f"Target batch shape: {targets.shape}")
    print(f"Mask batch shape: {mask.shape}")
    print("-------------------------------")
