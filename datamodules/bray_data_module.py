import torch
from torch.utils.data import DataLoader, random_split

from datamodules.base_data_module import BaseDataModule
from datasets.bray_dataset import BrayDataset


class BrayDataModule(BaseDataModule):
    def __init__(
        self,
        batch_size=64,
        data_dir="_data/gigadb",
        cal_split=0.2,
        mask_uncertain=True,
        normalize_features=False,
        feature_noise_std=0.0,
    ):
        """
        Lightning DataModule for BrayDataset.

        Args:
            batch_size (int): Batch size for dataloaders
            data_dir (str): Base directory for data files
            val_split (float): Validation split ratio from training data
            cal_split (float): Calibration split ratio from training data
            mask_uncertain (bool): Whether to mask uncertain values during training
            normalize_features (bool): Whether to normalize features
            feature_noise_std (float): Standard deviation of noise to add to features during training
        """
        super().__init__()
        self.batch_size = batch_size
        self.data_dir = data_dir
        self.cal_split = cal_split
        self.mask_uncertain = mask_uncertain
        self.feature_noise_std = feature_noise_std
        self.seed = 42

        # Define file paths
        self.data_file = f"{data_dir}/gigadb.csv"
        self.labels_file = f"{data_dir}/gigadb_top_30_moas.csv"

        # Define transforms
        self.train_transform = None
        self.test_transform = None

        self._num_classes = None
        self._embedding_dim = None

        print("Bray DM init")
        print(self.data_file)

    def prepare_data(self):
        """
        Download or prepare the dataset if needed
        This method is called only once and on only one GPU
        """
        # Nothing to do here as we're using existing CSV files
        pass

    def setup(self, stage: str):
        """
        Setup the dataset splits for the given stage

        Args:
            stage (str): One of 'fit', 'validate', 'test'
        """
        print("Bray DM setup")
        if stage == "fit":
            # Create the full training dataset
            train_split = BrayDataset(
                data_file=self.data_file,
                labels_file=self.labels_file,
                split_column="hier_split",
                split_value="train",
                transform=self.train_transform,
                mask_uncertain=self.mask_uncertain,
            )

            torch.manual_seed(self.seed)
            train_size = int((1 - self.cal_split) * len(train_split))
            cal_size = len(train_split) - train_size

            # create training and calibration dataset
            self.train_dataset, self.cal_dataset = random_split(
                train_split, [train_size, cal_size]
            )

            self.val_dataset = BrayDataset(
                data_file=self.data_file,
                labels_file=self.labels_file,
                split_column="hier_split",
                split_value="test",
                transform=self.test_transform,
                mask_uncertain=self.mask_uncertain,
            )

            # Store dimensions
            self._num_classes = len(train_split.moa_columns)
            self._embedding_dim = len(train_split.feature_columns)

        if stage == "test":
            # Create the test dataset
            # self.cal_dataset = BrayDataset(
            #     data_file=self.data_file,
            #     labels_file=self.labels_file,
            #     split_column="hier_split",
            #     split_value="valid",
            #     transform=self.test_transform,
            #     mask_uncertain=False,
            # )
            self.test_dataset = BrayDataset(
                data_file=self.data_file,
                labels_file=self.labels_file,
                split_column="hier_split",
                split_value="test",
                transform=self.test_transform,
                mask_uncertain=False,
            )

            if self._num_classes is None:
                self._num_classes = len(self.test_dataset.moa_columns)
                self._embedding_dim = len(self.test_dataset.feature_columns)

            print(f"Test set: {len(self.test_dataset)} samples")

    @property
    def num_classes(self):
        return self._num_classes

    @property
    def embedding_dim(self):
        return self._embedding_dim

    def train_dataloader(self):
        """Returns the training dataloader"""
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=4,
            persistent_workers=True,
            pin_memory=True,
        )

    def val_dataloader(self):
        """Returns the training dataloader"""
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=True,
        )

    def calibration_dataloader(self):
        """Returns the training dataloader"""
        return DataLoader(
            self.cal_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=True,
        )

    def test_dataloader(self):
        """Returns the test dataloader"""
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=True,
        )
