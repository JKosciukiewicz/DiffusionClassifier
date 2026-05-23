import torch
from torch.utils.data import DataLoader
from datamodules.base_data_module import BaseDataModule
from datasets.image_feature_dataset import ImageFeatureDataset


class ImageFeatureDataModule(BaseDataModule):
    def __init__(
        self,
        data_file: str,
        split_col: str = "split",
        batch_size: int = 64,
        num_workers: int = 4,
        mask_uncertain: bool = True,
        persistent_workers: bool = True,
        pin_memory: bool = True,
    ):
        """
        Generic Lightning DataModule for ImageFeatureDataset.

        Args:
            data_file (str): Path to the CSV file containing features and targets.
            split_col (str): Column name used for splitting ('train', 'val', 'test', 'cal').
            batch_size (int): Batch size for dataloaders.
            num_workers (int): Number of workers for data loading.
            mask_uncertain (bool): Whether to mask uncertain (0) target values.
            persistent_workers (bool): Whether to keep workers alive between epochs.
            pin_memory (bool): Whether to pin memory for GPU transfer.
        """
        super().__init__()
        self.data_file = data_file
        self.split_col = split_col
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.mask_uncertain = mask_uncertain
        self.persistent_workers = persistent_workers
        self.pin_memory = pin_memory

        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None
        self.cal_dataset = None

        self._num_classes = None
        self._embedding_dim = None

    def setup(self, stage: str = None):
        """Set up datasets for different stages."""
        if stage == "fit" or stage is None:
            self.train_dataset = ImageFeatureDataset(
                data_file=self.data_file,
                split="train",
                split_col=self.split_col,
                mask_uncertain=self.mask_uncertain,
            )
            self.val_dataset = ImageFeatureDataset(
                data_file=self.data_file,
                split="val",
                split_col=self.split_col,
                mask_uncertain=self.mask_uncertain,
            )
            self._num_classes = len(self.train_dataset.get_target_names())
            self._embedding_dim = len(self.train_dataset.get_feature_names())

        if stage == "test" or stage is None:
            self.test_dataset = ImageFeatureDataset(
                data_file=self.data_file,
                split="test",
                split_col=self.split_col,
                mask_uncertain=self.mask_uncertain,
            )
            if self._num_classes is None:
                self._num_classes = len(self.test_dataset.get_target_names())
                self._embedding_dim = len(self.test_dataset.get_feature_names())

        if stage == "predict" or stage == "cal" or stage is None:
            try:
                self.cal_dataset = ImageFeatureDataset(
                    data_file=self.data_file,
                    split="cal",
                    split_col=self.split_col,
                    mask_uncertain=self.mask_uncertain,
                )
            except Exception as e:
                print(f"Warning: Could not load calibration split: {e}")

    @property
    def num_classes(self):
        return self._num_classes

    @property
    def embedding_dim(self):
        return self._embedding_dim

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            persistent_workers=self.persistent_workers,
            pin_memory=self.pin_memory,
        )

    def val_dataloader(self):
        if self.val_dataset is None:
            return None
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            persistent_workers=self.persistent_workers,
            pin_memory=self.pin_memory,
        )

    def test_dataloader(self):
        if self.test_dataset is None:
            return None
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            persistent_workers=self.persistent_workers,
            pin_memory=self.pin_memory,
        )

    def calibration_dataloader(self):
        if self.cal_dataset is None:
            return None
        return DataLoader(
            self.cal_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            persistent_workers=self.persistent_workers,
            pin_memory=self.pin_memory,
        )
