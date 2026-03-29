import os

import polars as pl
import polars.selectors as cs
import torch
from PIL import Image
from torch.utils.data import Dataset


class TwoDigitMNISTDataset(Dataset):
    """
    Dataset for the two-digit MNIST
    csv_file : str
        Path to the split CSV file (train / val / test).
    image_dir : str | None
        Root directory prepended to the ``image_col`` paths stored in the CSV.
        Pass ``None`` if the CSV already contains absolute paths.
    image_col : str
        Name of the image-path column.
    digit_prefix : str
        Shared prefix of the clean label columns (default ``"digit"``).
        The dataset expects columns named ``{digit_prefix}_0`` …
        ``{digit_prefix}_9``.  Masked columns are expected to be named
        ``"masked_{digit_prefix}_0"`` … etc.
    transform : callable | None
        Optional transform applied to the PIL image before returning.
    use_masked_labels : bool
        If ``True``, use the ``masked_digit_*`` columns as the model-facing
        label and derive a proper binary mask from them.  Falls back silently
        to all-ones masking when the masked columns are absent from the CSV
        (e.g. when loading a plain dataset).
    mask_symbol : float
        The sentinel value that was written into masked positions during
        dataset generation (default ``-1.0``).  Used to build the binary mask:
        positions whose masked-label value equals this are set to ``0`` in the
        mask tensor.
    """

    def __init__(
        self,
        csv_file: str,
        image_dir: str | None = None,
        image_col: str = "image_path",
        digit_prefix: str = "digit",
        transform=None,
        use_masked_labels: bool = False,
        mask_symbol: float = -1.0,
    ):
        self.image_dir = image_dir
        self.transform = transform
        self.image_col = image_col
        self.digit_prefix = digit_prefix
        self.use_masked_labels = use_masked_labels
        self.mask_symbol = mask_symbol

        df = pl.read_csv(csv_file)

        clean_prefix = f"{digit_prefix}_"
        masked_prefix = f"masked_{digit_prefix}_"

        df = df.with_columns(
            cs.starts_with(clean_prefix)
            .cast(pl.Float32)
            .name.map(lambda c: c.replace(clean_prefix, "", 1))
        )
        clean_cols = sorted(
            [c for c in df.columns if c.startswith(clean_prefix)],
            key=lambda c: int(c.split("_")[-1]),
        )
        df = df.with_columns(
            pl.concat_list([pl.col(c) for c in clean_cols]).alias("_clean_label")
        ).drop(clean_cols)

        masked_cols = sorted(
            [c for c in df.columns if c.startswith(masked_prefix)],
            key=lambda c: int(c.split("_")[-1]),
        )
        self._has_masked_labels = len(masked_cols) > 0

        if self._has_masked_labels:
            df = df.with_columns(
                pl.concat_list([pl.col(c).cast(pl.Float32) for c in masked_cols]).alias(
                    "_masked_label"
                )
            ).drop(masked_cols)

        self.labels_df = df

    def __len__(self) -> int:
        return len(self.labels_df)

    def __getitem__(self, idx: int):
        """
        Returns
        image : torch.Tensor
            Transformed image tensor.
        label : torch.Tensor  shape (num_classes,)  float32
            * ``use_masked_labels=True`` and masked columns are present:
              the masked label vector (contains ``mask_symbol`` at unknown
              positions).
            * Otherwise: the clean two-hot label vector.
        mask : torch.Tensor  shape (num_classes,)  float32
            * ``1.0`` at positions that are **observed / known**.
            * ``0.0`` at positions replaced by ``mask_symbol`` (uncertain).
            * All-ones when masking is disabled or not available in the CSV.
        """
        row = self.labels_df.row(idx, named=True)

        img_path = row[self.image_col]
        if self.image_dir:
            img_path = os.path.join(self.image_dir, img_path)

        image = Image.open(img_path).convert("L")
        if self.transform:
            image = self.transform(image)

        clean_label = torch.tensor(row["_clean_label"], dtype=torch.float32)

        if self.use_masked_labels and self._has_masked_labels:
            masked_label = torch.tensor(row["_masked_label"], dtype=torch.float32)
            # mask=1 where the value is observed, mask=0 where it was replaced
            mask = (masked_label != self.mask_symbol).float()
            return image, masked_label, mask

        # Default: clean label, all-ones mask (backward-compatible)
        return image, clean_label, torch.ones(clean_label.shape[0])
