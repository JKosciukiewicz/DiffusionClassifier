import random
from pathlib import Path

import numpy as np
import polars as pl
from PIL import Image
from torchvision import datasets, transforms
from tqdm import tqdm


def preprocess_mnist(mnist_dataset: datasets.MNIST) -> np.ndarray:
    """Pre-resize all MNIST images to 64x64 numpy arrays once, avoiding
    repeated per-sample PIL/tensor conversions."""
    resize = transforms.Resize((64, 64))
    images = []
    for img, _ in tqdm(mnist_dataset, desc="Preprocessing MNIST"):
        img_resized = resize(img).squeeze(0).numpy()
        images.append((img_resized * 255).astype(np.uint8))
    return np.stack(images)  # (N, 64, 64)


def scale_digit(digit: np.ndarray, scale: float) -> np.ndarray:
    """Scale a 64x64 digit by `scale`, returning a 64x64 array with the result
    centred — so placement logic stays unchanged regardless of scale."""
    new_size = max(1, round(64 * scale))
    pil_img = Image.fromarray(digit).resize((new_size, new_size), Image.BILINEAR)
    scaled = np.array(pil_img, dtype=np.uint8)

    out = np.zeros((64, 64), dtype=np.uint8)
    dy = (64 - new_size) // 2
    dx = (64 - new_size) // 2
    src_y1 = max(0, -dy)
    src_y2 = src_y1 + min(64, new_size)
    src_x1 = max(0, -dx)
    src_x2 = src_x1 + min(64, new_size)
    dst_y1 = max(0, dy)
    dst_y2 = dst_y1 + (src_y2 - src_y1)
    dst_x1 = max(0, dx)
    dst_x2 = dst_x1 + (src_x2 - src_x1)
    out[dst_y1:dst_y2, dst_x1:dst_x2] = scaled[src_y1:src_y2, src_x1:src_x2]
    return out


def two_hot_encode(label1: int, label2: int, num_classes: int = 10) -> np.ndarray:
    """Encode two digit labels as a single two-hot vector of length num_classes."""
    vec = np.zeros(num_classes, dtype=np.float32)
    vec[label1] = 1.0
    vec[label2] = 1.0
    return vec


def sample_non_overlapping_positions(
    canvas_size: int = 128,
    digit_size: int = 64,
    max_jitter: int = 20,
) -> tuple:
    """
    Place digit 1 on the left half and digit 2 on the right half of the canvas,
    then apply an independent random jitter to each, clamped to canvas bounds.
    """
    half = canvas_size // 2
    centre_y = (canvas_size - digit_size) // 2

    base1 = (0, centre_y)
    base2 = (half, centre_y)

    def jitter(base_x: int, base_y: int) -> tuple[int, int]:
        max_xy = canvas_size - digit_size
        x = base_x + random.randint(-max_jitter, max_jitter)
        y = base_y + random.randint(-max_jitter, max_jitter)
        return (max(0, min(x, max_xy)), max(0, min(y, max_xy)))

    return jitter(*base1), jitter(*base2)


def generate_dataset(
    mnist_dataset: datasets.MNIST,
    num_samples: int,
    canvas_size: int = 128,
    max_jitter: int = 20,
    scale_range: tuple[float, float] = (0.9, 1.1),
) -> tuple[np.ndarray, np.ndarray]:
    """
    Generate a dataset of canvas_size x canvas_size images, each containing
    two non-overlapping MNIST digits.

    Returns
    -------
    images : np.ndarray, shape (num_samples, canvas_size, canvas_size), uint8
    labels : np.ndarray, shape (num_samples, 10), float32  — two-hot encoded
    """
    all_images = preprocess_mnist(mnist_dataset)
    all_labels = np.array([label for _, label in mnist_dataset])

    images = np.zeros((num_samples, canvas_size, canvas_size), dtype=np.uint8)
    labels = np.zeros((num_samples, 10), dtype=np.float32)

    n = len(mnist_dataset)
    scale_lo, scale_hi = scale_range

    for i in tqdm(range(num_samples), desc="Generating samples"):
        idx1, idx2 = random.sample(range(n), 2)

        pos1, pos2 = sample_non_overlapping_positions(
            canvas_size, max_jitter=max_jitter
        )
        x1, y1 = pos1
        x2, y2 = pos2

        canvas = np.zeros((canvas_size, canvas_size), dtype=np.uint8)

        for idx, (px, py) in [(idx2, (x2, y2)), (idx1, (x1, y1))]:
            scale = random.uniform(scale_lo, scale_hi)
            digit = scale_digit(all_images[idx], scale)
            region = canvas[py : py + 64, px : px + 64]
            mask = digit > 0
            region[mask] = digit[mask]

        images[i] = canvas
        labels[i] = two_hot_encode(all_labels[idx1], all_labels[idx2])

    return images, labels


def compute_split_indices(
    num_samples: int,
    train_frac: float,
    val_frac: float,
    seed: int = 42,
) -> dict[str, np.ndarray]:
    """
    Randomly shuffle and partition `num_samples` indices into train / val / test.

    Parameters
    ----------
    num_samples : total number of generated samples
    train_frac  : fraction for training   (e.g. 0.7)
    val_frac    : fraction for validation (e.g. 0.15)
                  test fraction is implicitly 1 - train_frac - val_frac
    seed        : random seed for reproducibility

    Returns
    -------
    dict with keys "train", "val", "test" mapping to index arrays.
    """
    if not (0 < train_frac < 1 and 0 < val_frac < 1):
        raise ValueError("train_frac and val_frac must be in (0, 1).")
    test_frac = 1.0 - train_frac - val_frac
    if test_frac <= 0:
        raise ValueError(
            f"train_frac + val_frac = {train_frac + val_frac:.3f} ≥ 1.0; "
            "no samples left for the test split."
        )

    rng = np.random.default_rng(seed)
    indices = rng.permutation(num_samples)

    n_train = round(num_samples * train_frac)
    n_val = round(num_samples * val_frac)
    # test gets whatever remains to avoid off-by-one rounding errors
    n_test = num_samples - n_train - n_val

    print(
        f"Split sizes  →  train: {n_train}  |  val: {n_val}  |  test: {n_test}  "
        f"({train_frac:.0%} / {val_frac:.0%} / {test_frac:.0%})"
    )

    return {
        "train": indices[:n_train],
        "val": indices[n_train : n_train + n_val],
        "test": indices[n_train + n_val :],
    }


def save_dataset_with_splits(
    images: np.ndarray,
    labels: np.ndarray,
    dataset_dir: str | Path,
    train_frac: float = 0.7,
    val_frac: float = 0.15,
    seed: int = 42,
) -> None:
    """
    Save all images into a single ``images/`` directory and write one CSV per
    split (``train.csv``, ``val.csv``, ``test.csv``) alongside it.

    Layout
    ------
    {dataset_dir}/
        images/
            00000.png
            00001.png
            ...
        train.csv
        val.csv
        test.csv

    Each CSV has columns:
        image_path, digit_0, digit_1, ..., digit_9
    where ``image_path`` is relative to ``dataset_dir`` for portability.

    Parameters
    ----------
    images     : (N, H, W) uint8 array
    labels     : (N, 10) float32 two-hot array
    dataset_dir: root output directory
    train_frac : fraction of samples assigned to training split
    val_frac   : fraction of samples assigned to validation split
    seed       : RNG seed used for shuffling before splitting
    """
    dataset_dir = Path(dataset_dir)
    images_dir = dataset_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    n_samples = len(images)
    n_digits = labels.shape[1]

    image_paths: list[str] = []
    for i in tqdm(range(n_samples), desc="Saving images"):
        img_path = images_dir / f"{i:05d}.png"
        Image.fromarray(images[i]).save(img_path)
        # Store path relative to dataset_dir so CSVs are portable
        image_paths.append(str(img_path.relative_to(dataset_dir)))

    splits = compute_split_indices(n_samples, train_frac, val_frac, seed=seed)

    for split_name, idx in splits.items():
        label_cols = {
            f"digit_{d}": labels[idx, d].astype(np.uint8).tolist()
            for d in range(n_digits)
        }
        df = pl.DataFrame({"image_path": [image_paths[i] for i in idx], **label_cols})
        csv_path = dataset_dir / f"{split_name}.csv"
        df.write_csv(csv_path)
        print(f"Saved {split_name:5s} CSV ({len(idx):>6} rows) → {csv_path}")


if __name__ == "__main__":
    transform = transforms.Compose([transforms.ToTensor()])
    mnist = datasets.MNIST(root="_data", train=True, download=True, transform=transform)

    images, labels = generate_dataset(
        mnist_dataset=mnist,
        num_samples=5000,
        canvas_size=128,
        max_jitter=20,
        scale_range=(0.9, 1.1),
    )

    save_dataset_with_splits(
        images=images,
        labels=labels,
        dataset_dir="_data/dual_mnist/raw",
        train_frac=0.70,  # 70 % → 7 000 samples
        val_frac=0.15,  # 15 % → 1 500 samples  (test = remaining 15 %)
        seed=42,
    )

    print(f"\nImages shape : {images.shape}")
    print(f"Labels shape : {labels.shape}")
    print(f"Example label: {labels[0]}")
