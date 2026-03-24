import torchvision
import torch
from torch.utils.data import DataLoader, random_split

from datamodules.base_data_module import BaseDataModule
from datasets.two_digit_mnist_dataset import TwoDigitMNISTDataset


class TwoDigitMNISTDataModule(BaseDataModule):
    def __init__(
        self,
        batch_size=128,
        data_dir="./data/two_digit_mnist",
        cal_split=0.2,
        noise_std=0.7,
    ):
        super().__init__()
        self.batch_size = batch_size
        self.data_dir = data_dir
        self.cal_split = cal_split  # Calibration split ratio
        self.seed = 42
        self.noise_std = noise_std

        # Training transform: convert image to tensor and add a fuckton of noise
        self.train_transform = torchvision.transforms.Compose(
            [
                torchvision.transforms.ToTensor(),
            ]
        )

        # Test/Validation transform: clean image (no noise)
        self.test_transform = torchvision.transforms.Compose(
            [
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Lambda(
                    lambda x: x + torch.randn_like(x) * self.noise_std
                ),
            ]
        )

    def setup(self, stage: str) -> None:
        if stage == "fit":
            full_train_dataset = TwoDigitMNISTDataset(
                csv_file="data/two_digit_mnist/train/train_labels.csv",
                img_dir="data/two_digit_mnist/train/images",
                transform=self.train_transform,  # apply noise for training
            )

            # Ensure the split is always the same
            torch.manual_seed(self.seed)
            train_size = int((1 - self.cal_split) * len(full_train_dataset))
            cal_size = len(full_train_dataset) - train_size
            self.train_dataset, self.cal_dataset = random_split(
                full_train_dataset, [train_size, cal_size]
            )

        if stage == "test":
            self.test_dataset = TwoDigitMNISTDataset(
                csv_file="data/two_digit_mnist/test/test_labels.csv",
                img_dir="data/two_digit_mnist/test/images",
                transform=self.test_transform,  # clean images for testing
            )

        if stage == "validate":
            self.val_dataset = TwoDigitMNISTDataset(
                csv_file="data/two_digit_mnist/val/val_labels.csv",
                img_dir="data/two_digit_mnist/val/images",
                transform=self.test_transform,  # clean images for validation
            )

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)

    def calibration_dataloader(self):
        return DataLoader(self.cal_dataset, batch_size=self.batch_size, shuffle=False)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False)
