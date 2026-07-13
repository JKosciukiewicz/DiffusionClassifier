import numpy as np
import torch
from torch.utils.data import Dataset


class BrayPreprocessedDataset(Dataset):
    def __init__(
        self,
        npz_path: str,
        split_value: str = "train",
        noise_std: float = 0.0,
        ternary_labels: bool = False,
        treat_uncertain_as_negative: bool = False,
        mask_uncertain: bool = True,
        fold: int = 0,
    ):
        data = np.load(npz_path, allow_pickle=True)
        self.moa_columns = list(data["moa_columns"])
        self.noise_std = noise_std

        # Use per-fold split column when available (shape N×5), else fall back to split.
        if "splits" in data:
            split_col = data["splits"][:, fold]
        else:
            split_col = data["split"]
        idx = split_col == split_value
        features = data["features"][idx]
        labels_raw = data["labels"][idx].astype(np.float32)

        if ternary_labels:
            masks = (labels_raw != 0).astype(np.float32)
            targets = labels_raw
        elif treat_uncertain_as_negative:
            masks = np.ones_like(labels_raw)
            targets = np.where(labels_raw < 0, 0, labels_raw)
        elif mask_uncertain:
            masks = (labels_raw != 0).astype(np.float32)
            targets = np.where(labels_raw < 0, 0, labels_raw)
        else:
            masks = np.ones_like(labels_raw)
            targets = np.where(labels_raw < 0, 0, labels_raw)

        self.features_data = features
        self.targets_data = targets
        self.masks_data = masks

        print(f"BrayPreprocessed [{split_value}]: {len(features)} samples, "
              f"{features.shape[1]} features, {len(self.moa_columns)} classes")

    def __len__(self):
        return len(self.features_data)

    def __getitem__(self, idx):
        features = torch.from_numpy(self.features_data[idx])
        targets = torch.from_numpy(self.targets_data[idx])
        mask = torch.from_numpy(self.masks_data[idx])

        if self.noise_std > 0:
            features = features + torch.randn_like(features) * self.noise_std

        return features, targets, mask
