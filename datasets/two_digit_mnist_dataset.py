import ast
import os

import polars as pl
import torch
from PIL import Image
from torch.utils.data import Dataset


class TwoDigitMNISTDataset(Dataset):
    def __init__(
        self,
        csv_file,
        image_dir=None,
        image_col="image_path",
        digit_prefix="digit",
        transform=None,
    ):
        """
        Args:
            csv_file (str): Path to the CSV file with filenames and labels.
            img_dir (str): Directory with all the images.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.labels_df = pl.read_csv(csv_file)
        self.image_dir = image_dir
        self.transform = transform
        self.image_col = image_col
        self.digit_prefix = digit_prefix

        self.labels_df = self.labels_df.with_columns(
            pl.concat_list(pl.selectors.contains("digit")).alias("label")
        ).drop(pl.selectors.contains("digit"))

    def __len__(self):
        return len(self.labels_df)

    def __getitem__(self, idx):
        """
        :param idx:
        :return:
            image (torch.tensor)
            label (torch.tensor)
            mask (torch.ones) param required for biological data, dummy param to skip masking and allow for common trainer
        """
        row = self.labels_df.row(idx, named=True)
        if self.image_dir:
            img_name = os.path.join(self.image_dir, row[self.image_col])
        else:
            img_name = row[self.image_col]

        label = torch.tensor(row["label"], dtype=torch.float32)

        image = Image.open(img_name).convert("L")

        if self.transform:
            image = self.transform(image)

        return image, label, torch.ones(label.shape[0])
