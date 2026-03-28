import torch
import torchvision
from torch.utils.data import DataLoader, random_split

from datamodules.base_data_module import BaseDataModule
from datasets.two_digit_mnist_dataset import TwoDigitMNISTDataset


class TwoDigitMNISTDataModule(BaseDataModule):
    def __init__(
        self,
        batch_size=128,
        data_dir="_data/dual_mnist/raw/",
        noise_std=0.4,
    ):
        super().__init__()
        self.batch_size = batch_size
        self.data_dir = data_dir
        self.seed = 42
        self.noise_std = noise_std

        # Training transform: convert image to tensor and add a fuckton of noise
        self.train_transform = torchvision.transforms.Compose(
            [
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Lambda(
                    lambda x: x + torch.randn_like(x) * self.noise_std
                ),
            ]
        )

        # Test/Validation transform: clean image (no noise)
        self.test_val_transform = torchvision.transforms.Compose(
            [
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Lambda(
                    lambda x: x + torch.randn_like(x) * self.noise_std
                ),
            ]
        )

    def setup(self, stage: str) -> None:
        if stage == "fit":
            self.train_dataset = TwoDigitMNISTDataset(
                image_dir=self.data_dir,
                csv_file="_data/dual_mnist/raw/train.csv",
                transform=self.train_transform,  # apply noise for training
            )
            self.val_dataset = TwoDigitMNISTDataset(
                image_dir=self.data_dir,
                csv_file="_data/dual_mnist/raw/val.csv",
                transform=self.test_val_transform,  # apply noise for training
            )

        if stage == "test":
            self.test_dataset = TwoDigitMNISTDataset(
                image_dir=self.data_dir,
                csv_file="_data/dual_mnist/raw/test.csv",
                transform=self.test_val_transform,  # apply noise for training
            )

        if stage == "validate":
            self.val_dataset = TwoDigitMNISTDataset(
                image_dir=self.data_dir,
                csv_file="_data/dual_mnist/raw/val.csv",
                transform=self.test_val_transform,  # apply noise for training
            )

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False)
