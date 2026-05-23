import torch
from torch.utils.data import DataLoader
from datamodules.base_data_module import BaseDataModule
from datasets.image_feature_dataset import ImageFeatureDataset


class BrayDinoDataModule(BaseDataModule):
    def __init__(
        self,
        batch_size=64,
        data_file="_data/bray_dino/bray_dino_complete.csv",
        split_col="hier_split_0",
        mask_uncertain=True,
        num_workers=4,
    ):
        """
        Lightning DataModule for Bray DINO features using ImageFeatureDataset.

        Args:
            batch_size (int): Batch size for dataloaders
            data_file (str): Path to the DINO features CSV
            split_col (str): Column name for data splits (e.g., 'hier_split_0')
            mask_uncertain (bool): Whether to mask uncertain values (0)
            num_workers (int): Number of workers for data loading
        """
        super().__init__()
        self.batch_size = batch_size
        self.data_file = data_file
        self.split_col = split_col
        self.mask_uncertain = mask_uncertain
        self.num_workers = num_workers

        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None
        self.cal_dataset = None

        self._num_classes = None
        self._embedding_dim = None

    def setup(self, stage: str = None):
        if stage == "fit" or stage is None:
            self.train_dataset = ImageFeatureDataset(
                data_file=self.data_file,
                split="train",
                split_col=self.split_col,
                mask_uncertain=self.mask_uncertain,
            )
            
            # Try val split first
            self.val_dataset = ImageFeatureDataset(
                data_file=self.data_file,
                split="val",
                split_col=self.split_col,
                mask_uncertain=self.mask_uncertain,
            )
            
            # Fallback to cal split if val is empty
            if len(self.val_dataset) == 0:
                print("Val split not found or empty, trying cal split for validation...")
                self.val_dataset = ImageFeatureDataset(
                    data_file=self.data_file,
                    split="cal",
                    split_col=self.split_col,
                    mask_uncertain=self.mask_uncertain,
                )
                
            # Final fallback to test split if both val and cal are empty
            if len(self.val_dataset) == 0:
                print("Cal split empty, using test split for validation...")
                self.val_dataset = ImageFeatureDataset(
                    data_file=self.data_file,
                    split="test",
                    split_col=self.split_col,
                    mask_uncertain=self.mask_uncertain,
                )

            self._num_classes = len(self.train_dataset.moa_columns)
            self._embedding_dim = len(self.train_dataset.feature_columns)

        if stage == "test" or stage is None:
            self.test_dataset = ImageFeatureDataset(
                data_file=self.data_file,
                split="test",
                split_col=self.split_col,
                mask_uncertain=False,
            )
            try:
                self.cal_dataset = ImageFeatureDataset(
                    data_file=self.data_file,
                    split="cal",
                    split_col=self.split_col,
                    mask_uncertain=False,
                )
            except Exception:
                self.cal_dataset = None

            if self._num_classes is None:
                self._num_classes = len(self.test_dataset.moa_columns)
                self._embedding_dim = len(self.test_dataset.feature_columns)

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
            persistent_workers=True,
            pin_memory=True,
        )

    def val_dataloader(self):
        if self.val_dataset is None:
            return None
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=True,
        )

    def test_dataloader(self):
        if self.test_dataset is None:
            return None
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=True,
        )

    def calibration_dataloader(self):
        if self.cal_dataset is None:
            return None
        return DataLoader(
            self.cal_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=True,
        )
