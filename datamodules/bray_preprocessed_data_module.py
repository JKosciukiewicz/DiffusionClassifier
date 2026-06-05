import torch
from torch.utils.data import DataLoader, random_split

from datamodules.base_data_module import BaseDataModule
from datasets.bray_preprocessed_dataset import BrayPreprocessedDataset


class BrayPreprocessedDataModule(BaseDataModule):
    def __init__(
        self,
        npz_path: str,
        batch_size: int = 64,
        cal_split: float = 0.2,
        mask_uncertain: bool = True,
        treat_uncertain_as_negative: bool = False,
        feature_noise_std: float = 0.0,
        ternary_labels: bool = False,
    ):
        super().__init__()
        self.npz_path = npz_path
        self.batch_size = batch_size
        self.cal_split = cal_split
        self.mask_uncertain = mask_uncertain
        self.treat_uncertain_as_negative = treat_uncertain_as_negative
        self.feature_noise_std = feature_noise_std
        self.ternary_labels = ternary_labels
        self._num_classes = None
        self._embedding_dim = None

    def setup(self, stage: str):
        kwargs = dict(
            npz_path=self.npz_path,
            ternary_labels=self.ternary_labels,
            treat_uncertain_as_negative=self.treat_uncertain_as_negative,
            mask_uncertain=self.mask_uncertain,
        )
        if stage == "fit":
            train_full = BrayPreprocessedDataset(
                split_value="train", noise_std=self.feature_noise_std, **kwargs
            )
            self._num_classes = len(train_full.moa_columns)
            self._embedding_dim = train_full.features_data.shape[1]

            train_size = int((1 - self.cal_split) * len(train_full))
            cal_size = len(train_full) - train_size
            torch.manual_seed(42)
            self.train_dataset, self.cal_dataset = random_split(
                train_full, [train_size, cal_size]
            )

            self.val_dataset = BrayPreprocessedDataset(
                split_value="test", noise_std=0.0, **kwargs
            )

        if stage == "test":
            self.test_dataset = BrayPreprocessedDataset(
                split_value="test", noise_std=0.0, **kwargs
            )
            if self._num_classes is None:
                self._num_classes = len(self.test_dataset.moa_columns)
                self._embedding_dim = self.test_dataset.features_data.shape[1]

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
            num_workers=4,
            persistent_workers=True,
            pin_memory=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=0,
            pin_memory=True,
        )

    def calibration_dataloader(self):
        return DataLoader(
            self.cal_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=True,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=True,
        )
