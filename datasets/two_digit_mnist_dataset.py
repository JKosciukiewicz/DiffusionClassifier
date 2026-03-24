import os
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset
import ast


def convert_label_to_list(label):
    if isinstance(label, str):
        try:
            # Convert the string representation of a list to an actual list
            return ast.literal_eval(label)
        except (ValueError, SyntaxError):
            # Handle cases where conversion fails
            print(f"Failed to convert label: {label}")
            return label
    return label


class TwoDigitMNISTDataset(Dataset):
    def __init__(self, csv_file, img_dir, transform=None):
        """
        Args:
            csv_file (str): Path to the CSV file with filenames and labels.
            img_dir (str): Directory with all the images.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.labels_df = pd.read_csv(csv_file)
        self.img_dir = img_dir
        self.transform = transform
        self.labels_df["Label"] = self.labels_df["Label"].apply(convert_label_to_list)

    def __len__(self):
        return len(self.labels_df)

    def __getitem__(self, idx):
        # Get the file name and label
        """
        :param idx:
        :return:
            image (image)
            label (label)
            mask (torch.ones) param required for biological data, dummy param to skip masking and allow for common trainer
        """
        img_name = os.path.join(self.img_dir, self.labels_df.iloc[idx].Filename)
        label = torch.tensor(self.labels_df.iloc[idx].Label, dtype=torch.float32)
        # Load the image
        image = Image.open(img_name).convert("L")

        # Apply transformations if any
        if self.transform:
            image = self.transform(image)

        return image, label, torch.ones(label.shape[0])
