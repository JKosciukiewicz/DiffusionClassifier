"""
generate_occluded_single_digit_mnist.py
"""

import random
from pathlib import Path

import numpy as np
import polars as pl
from PIL import Image
from torchvision import datasets, transforms
from tqdm import tqdm


def generate_dataset(
    mnist_dataset: datasets.MNIST,
    num_samples: int,
    occlusion_frac: float = 0.0,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Returns
    -------
    images  : (N, 28, 28) uint8  — blank when occluded
    labels  : (N,) int64         — true class, always set
    visible : (N,) bool          — False when canvas is blank
    """
    rng = np.random.default_rng(seed)
    all_images = np.stack([img.squeeze(0).numpy() for img, _ in mnist_dataset])
    all_images = (all_images * 255).astype(np.uint8)
    all_labels = np.array([lbl for _, lbl in mnist_dataset])

    n_occluded = round(num_samples * occlusion_frac)
    visible = np.ones(num_samples, dtype=bool)
    visible[rng.choice(num_samples, size=n_occluded, replace=False)] = False

    indices = rng.integers(0, len(mnist_dataset), size=num_samples)
    images = all_images[indices].copy()
    images[~visible] = 0

    print(
        f"Total: {num_samples} | visible: {visible.sum()} | occluded: {(~visible).sum()}"
    )
    return images, all_labels[indices], visible


def _to_one_hot(labels: np.ndarray, num_classes: int = 10) -> np.ndarray:
    """Convert integer labels to a (N, num_classes) uint8 one-hot array."""
    one_hot = np.zeros((len(labels), num_classes), dtype=np.uint8)
    one_hot[np.arange(len(labels)), labels] = 1
    return one_hot


def save_dataset(
    images: np.ndarray,
    labels: np.ndarray,
    visible: np.ndarray,
    dataset_dir: str | Path,
    train_frac: float = 0.70,
    val_frac: float = 0.15,
    num_classes: int = 10,
    seed: int = 42,
) -> None:
    dataset_dir = Path(dataset_dir)
    images_dir = dataset_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    n = len(images)
    image_paths = []
    for i in tqdm(range(n), desc="Saving images"):
        p = images_dir / f"{i:05d}.png"
        Image.fromarray(images[i]).save(p)
        image_paths.append(str(p.relative_to(dataset_dir)))

    one_hot = _to_one_hot(labels, num_classes=num_classes)  # (N, num_classes)

    rng = np.random.default_rng(seed)
    idx = rng.permutation(n)
    n_train = round(n * train_frac)
    n_val = round(n * val_frac)
    splits = {
        "train": idx[:n_train],
        "val": idx[n_train : n_train + n_val],
        "test": idx[n_train + n_val :],
    }

    for name, split_idx in splits.items():
        df = pl.DataFrame(
            {
                "image_path": [image_paths[i] for i in split_idx],
                **{
                    f"label_{c}": one_hot[split_idx, c].tolist()
                    for c in range(num_classes)
                },
                "visible": visible[split_idx].tolist(),
            }
        )
        df.write_csv(dataset_dir / f"{name}.csv")
        print(f"{name}: {len(split_idx)} rows")


if __name__ == "__main__":
    mnist = datasets.MNIST(
        root="_data", train=True, download=True, transform=transforms.ToTensor()
    )
    images, labels, visible = generate_dataset(
        mnist_dataset=mnist,
        num_samples=10_000,
        occlusion_frac=0.0,
        seed=42,
    )
    save_dataset(images, labels, visible, "_data/single_mnist_occluded_0/raw")
