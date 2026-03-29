"""
generate_occluded_masked_2_digit_mnist.py
=========================================

Extends ``generate_occluded_2_digit_mnist`` with **label masking**:
a configurable fraction of samples have some of the *zero* entries in
their two-hot label vector replaced by a special ``mask_symbol`` value,
simulating a classifier that is uncertain about which classes are *absent*.

New parameters (relative to the original generator)
----------------------------------------------------
label_mask_frac   : float in [0, 1]
    Fraction of samples whose label vector is partially masked.
max_zeros_to_mask : int >= 1
    Maximum number of zero-valued label entries that may be replaced per
    sample.  The actual count is drawn uniformly from [1, max_zeros_to_mask].
mask_symbol       : float
    Value written into the masked positions.  Common choices:
        -1.0  – "unknown" sentinel compatible with most loss functions
         0.5  – "maximum-entropy" soft label
        float("nan") – explicit NaN (requires downstream NaN handling)

Additional metadata
-------------------
``num_masked``       int  (N,)  – zeros replaced in each sample (0 if not masked)
``masked_positions`` str  (N,)  – comma-separated indices that were masked
                                   (empty string when num_masked == 0)
"""

import random
from enum import Enum
from pathlib import Path

import numpy as np
import polars as pl
from PIL import Image
from torchvision import datasets, transforms
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Sample-type taxonomy (unchanged from base generator)
# ---------------------------------------------------------------------------


class SampleType(str, Enum):
    """Visibility status of the two digits in a canvas sample."""

    FULL = "full"
    A_ONLY = "a_only"
    B_ONLY = "b_only"
    NEITHER = "neither"


# ---------------------------------------------------------------------------
# Preprocessing helpers (unchanged)
# ---------------------------------------------------------------------------


def preprocess_mnist(mnist_dataset: datasets.MNIST) -> np.ndarray:
    """Pre-resize all MNIST images to 64×64 uint8 arrays once."""
    resize = transforms.Resize((64, 64))
    images = []
    for img, _ in tqdm(mnist_dataset, desc="Preprocessing MNIST"):
        img_resized = resize(img).squeeze(0).numpy()
        images.append((img_resized * 255).astype(np.uint8))
    return np.stack(images)  # (N, 64, 64)


def scale_digit(digit: np.ndarray, scale: float) -> np.ndarray:
    """Scale a 64×64 digit by `scale`, keeping it centred in a 64×64 frame."""
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
    """Two-hot vector of length `num_classes`."""
    vec = np.zeros(num_classes, dtype=np.float32)
    vec[label1] = 1.0
    vec[label2] = 1.0
    return vec


def sample_non_overlapping_positions(
    canvas_size: int = 128,
    digit_size: int = 64,
    max_jitter: int = 20,
) -> tuple:
    """Left-half / right-half placement with independent random jitter."""
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


# ---------------------------------------------------------------------------
# Occlusion schedule (unchanged)
# ---------------------------------------------------------------------------


def _assign_sample_types(
    num_samples: int,
    occlusion_frac: float,
    neither_frac: float,
    rng: np.random.Generator,
) -> list[SampleType]:
    """
    Randomly assign a SampleType to every sample index.

    Parameters
    ----------
    occlusion_frac : fraction of samples where *exactly one* digit is absent.
    neither_frac   : fraction of samples where *both* digits are absent.
    """
    if occlusion_frac < 0 or neither_frac < 0:
        raise ValueError("Fractions must be non-negative.")
    if occlusion_frac + neither_frac > 1.0:
        raise ValueError(
            f"occlusion_frac ({occlusion_frac}) + neither_frac ({neither_frac}) "
            "exceeds 1.0."
        )

    n_occluded = round(num_samples * occlusion_frac)
    n_neither = round(num_samples * neither_frac)
    n_full = num_samples - n_occluded - n_neither

    n_a_only = n_occluded // 2
    n_b_only = n_occluded - n_a_only

    type_list: list[SampleType] = (
        [SampleType.FULL] * n_full
        + [SampleType.A_ONLY] * n_a_only
        + [SampleType.B_ONLY] * n_b_only
        + [SampleType.NEITHER] * n_neither
    )
    rng.shuffle(type_list)  # type: ignore[arg-type]

    print(
        f"Sample-type breakdown  →  "
        f"full: {n_full}  |  a_only: {n_a_only}  |  b_only: {n_b_only}  "
        f"|  neither: {n_neither}  "
        f"({n_full / num_samples:.0%} / {n_a_only / num_samples:.0%} / "
        f"{n_b_only / num_samples:.0%} / {n_neither / num_samples:.0%})"
    )
    return type_list


# ---------------------------------------------------------------------------
# Label masking  (NEW)
# ---------------------------------------------------------------------------


def _apply_label_masking(
    labels: np.ndarray,
    label_mask_frac: float,
    max_zeros_to_mask: int,
    mask_symbol: float,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Randomly replace zero-valued entries in two-hot labels with ``mask_symbol``.

    Only the *zero* positions are eligible for masking — the hot (1.0) positions
    are never touched, preserving knowledge of which classes are present while
    simulating uncertainty about which classes are truly absent.

    Parameters
    ----------
    labels           : (N, C) float32 two-hot array (will be copied, not modified).
    label_mask_frac  : Fraction of samples to mask.
    max_zeros_to_mask: Maximum zeros replaced per selected sample.
                       Actual count ~ Uniform{1, …, max_zeros_to_mask}.
    mask_symbol      : Replacement value for masked positions.
    rng              : NumPy random generator for reproducibility.

    Returns
    -------
    masked_labels     : (N, C) float32 — copy of labels with masking applied.
    num_masked        : (N,)  int32  — count of zeros replaced per sample.
    masked_positions  : (N,)  object — comma-separated strings of masked indices,
                                       empty string when no masking occurred.
    """
    if label_mask_frac < 0.0 or label_mask_frac > 1.0:
        raise ValueError(f"label_mask_frac must be in [0, 1], got {label_mask_frac}.")
    if max_zeros_to_mask < 1:
        raise ValueError(f"max_zeros_to_mask must be >= 1, got {max_zeros_to_mask}.")

    num_samples, num_classes = labels.shape
    masked_labels = labels.copy()

    n_to_mask = round(num_samples * label_mask_frac)
    mask_indices = rng.choice(num_samples, size=n_to_mask, replace=False)
    mask_set = set(mask_indices.tolist())

    num_masked = np.zeros(num_samples, dtype=np.int32)
    masked_positions = np.empty(num_samples, dtype=object)
    masked_positions[:] = ""

    for i in range(num_samples):
        if i not in mask_set:
            continue

        # Eligible positions: those that are currently 0.0
        zero_positions = np.where(masked_labels[i] == 0.0)[0]
        if len(zero_positions) == 0:
            # Edge case: both-same-digit label has no zeros to mask (unlikely but safe)
            continue

        k = int(rng.integers(1, min(max_zeros_to_mask, len(zero_positions)) + 1))
        chosen = rng.choice(zero_positions, size=k, replace=False)
        chosen.sort()

        masked_labels[i, chosen] = mask_symbol
        num_masked[i] = k
        masked_positions[i] = ",".join(str(p) for p in chosen.tolist())

    n_actually_masked = int((num_masked > 0).sum())
    avg_k = float(num_masked[num_masked > 0].mean()) if n_actually_masked > 0 else 0.0
    print(
        f"Label masking  →  samples masked: {n_actually_masked}  "
        f"({n_actually_masked / num_samples:.0%})  |  "
        f"avg zeros replaced per masked sample: {avg_k:.2f}  |  "
        f"mask_symbol: {mask_symbol}"
    )
    return masked_labels, num_masked, masked_positions


# ---------------------------------------------------------------------------
# Core generator (extended)
# ---------------------------------------------------------------------------


def generate_dataset(
    mnist_dataset: datasets.MNIST,
    num_samples: int,
    canvas_size: int = 128,
    max_jitter: int = 20,
    scale_range: tuple[float, float] = (0.9, 1.1),
    # --- image occlusion ---
    occlusion_frac: float = 0.0,
    neither_frac: float = 0.0,
    # --- label masking (NEW) ---
    label_mask_frac: float = 0.0,
    max_zeros_to_mask: int = 3,
    mask_symbol: float = -1.0,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, np.ndarray]]:
    """
    Generate a dataset of canvas images with optional image occlusion **and**
    optional label masking.

    Image occlusion
    ---------------
    occlusion_frac : Fraction of samples where exactly one digit is not rendered.
                     Splits 50/50 between A_ONLY and B_ONLY.
    neither_frac   : Fraction of samples where *both* digits are absent.

    Label masking  (NEW)
    --------------------
    label_mask_frac   : Fraction of samples whose label vector is partially masked.
    max_zeros_to_mask : Upper bound on the number of zero-valued label entries
                        replaced per selected sample.  Actual count is drawn
                        uniformly from [1, max_zeros_to_mask].
    mask_symbol       : Value written into masked positions (e.g. -1.0, 0.5, NaN).

    Returns
    -------
    images        : (N, canvas_size, canvas_size)  uint8
    labels        : (N, 10)  float32  clean two-hot vectors (no masking)
    masked_labels : (N, 10)  float32  labels with mask_symbol in selected zeros
    metadata      : dict with keys
                      "label_a"          – int8   (N,)
                      "label_b"          – int8   (N,)
                      "visible_a"        – bool   (N,)
                      "visible_b"        – bool   (N,)
                      "sample_type"      – object (N,)  SampleType value strings
                      "num_masked"       – int32  (N,)  zeros replaced per sample
                      "masked_positions" – object (N,)  comma-separated index strings
    """
    rng = np.random.default_rng(seed)
    all_images = preprocess_mnist(mnist_dataset)
    all_labels = np.array([lbl for _, lbl in mnist_dataset])
    n_mnist = len(mnist_dataset)
    scale_lo, scale_hi = scale_range

    sample_types = _assign_sample_types(num_samples, occlusion_frac, neither_frac, rng)

    images = np.zeros((num_samples, canvas_size, canvas_size), dtype=np.uint8)
    labels = np.zeros((num_samples, 10), dtype=np.float32)
    m_label_a = np.zeros(num_samples, dtype=np.int8)
    m_label_b = np.zeros(num_samples, dtype=np.int8)
    m_visible_a = np.zeros(num_samples, dtype=bool)
    m_visible_b = np.zeros(num_samples, dtype=bool)
    m_types = np.empty(num_samples, dtype=object)

    for i in tqdm(range(num_samples), desc="Generating samples"):
        idx1, idx2 = random.sample(range(n_mnist), 2)
        stype = sample_types[i]

        render_a = stype in (SampleType.FULL, SampleType.A_ONLY)
        render_b = stype in (SampleType.FULL, SampleType.B_ONLY)

        pos1, pos2 = sample_non_overlapping_positions(
            canvas_size, max_jitter=max_jitter
        )
        x1, y1 = pos1
        x2, y2 = pos2

        canvas = np.zeros((canvas_size, canvas_size), dtype=np.uint8)

        if render_b:
            scale = random.uniform(scale_lo, scale_hi)
            digit = scale_digit(all_images[idx2], scale)
            region = canvas[y2 : y2 + 64, x2 : x2 + 64]
            mask = digit > 0
            region[mask] = digit[mask]

        if render_a:
            scale = random.uniform(scale_lo, scale_hi)
            digit = scale_digit(all_images[idx1], scale)
            region = canvas[y1 : y1 + 64, x1 : x1 + 64]
            mask = digit > 0
            region[mask] = digit[mask]

        images[i] = canvas
        labels[i] = two_hot_encode(all_labels[idx1], all_labels[idx2])
        m_label_a[i] = all_labels[idx1]
        m_label_b[i] = all_labels[idx2]
        m_visible_a[i] = render_a
        m_visible_b[i] = render_b
        m_types[i] = stype.value

    # ---- apply label masking -----------------------------------------------
    masked_labels, num_masked, masked_positions = _apply_label_masking(
        labels=labels,
        label_mask_frac=label_mask_frac,
        max_zeros_to_mask=max_zeros_to_mask,
        mask_symbol=mask_symbol,
        rng=rng,
    )

    metadata = {
        "label_a": m_label_a,
        "label_b": m_label_b,
        "visible_a": m_visible_a,
        "visible_b": m_visible_b,
        "sample_type": m_types,
        # masking metadata
        "num_masked": num_masked,
        "masked_positions": masked_positions,
    }
    return images, labels, masked_labels, metadata


# ---------------------------------------------------------------------------
# Splitting & saving (extended)
# ---------------------------------------------------------------------------


def compute_split_indices(
    num_samples: int,
    train_frac: float,
    val_frac: float,
    seed: int = 42,
) -> dict[str, np.ndarray]:
    """Shuffle and partition into train / val / test index arrays."""
    if not (0 < train_frac < 1 and 0 < val_frac < 1):
        raise ValueError("train_frac and val_frac must be in (0, 1).")
    test_frac = 1.0 - train_frac - val_frac
    if test_frac <= 0:
        raise ValueError(
            f"train_frac + val_frac = {train_frac + val_frac:.3f} >= 1.0; "
            "no samples left for the test split."
        )

    rng = np.random.default_rng(seed)
    indices = rng.permutation(num_samples)

    n_train = round(num_samples * train_frac)
    n_val = round(num_samples * val_frac)
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
    masked_labels: np.ndarray,
    metadata: dict[str, np.ndarray],
    dataset_dir: str | Path,
    train_frac: float = 0.70,
    val_frac: float = 0.15,
    seed: int = 42,
) -> None:
    """
    Save images and write per-split CSVs.

    Layout
    ------
    {dataset_dir}/
        images/
            00000.png
            ...
        train.csv
        val.csv
        test.csv

    CSV columns
    -----------
    image_path,
    digit_0 … digit_9          (clean two-hot, uint8)
    masked_digit_0 … masked_digit_9  (masked labels, float32;
                                      contains mask_symbol where masked)
    label_a, label_b,
    visible_a, visible_b,
    sample_type,
    num_masked,                (int: how many zeros were replaced)
    masked_positions           (str: comma-separated indices, "" if none)
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
        image_paths.append(str(img_path.relative_to(dataset_dir)))

    splits = compute_split_indices(n_samples, train_frac, val_frac, seed=seed)

    for split_name, idx in splits.items():
        # Clean label columns (int-friendly since they are strict 0/1)
        clean_label_cols = {
            f"digit_{d}": labels[idx, d].astype(np.uint8).tolist()
            for d in range(n_digits)
        }
        # Masked label columns (float, may contain mask_symbol)
        masked_label_cols = {
            f"masked_digit_{d}": masked_labels[idx, d].tolist() for d in range(n_digits)
        }

        df = pl.DataFrame(
            {
                "image_path": [image_paths[i] for i in idx],
                **clean_label_cols,
                **masked_label_cols,
                # occlusion metadata
                "label_a": metadata["label_a"][idx].tolist(),
                "label_b": metadata["label_b"][idx].tolist(),
                "visible_a": metadata["visible_a"][idx].tolist(),
                "visible_b": metadata["visible_b"][idx].tolist(),
                "sample_type": metadata["sample_type"][idx].tolist(),
                # masking metadata
                "num_masked": metadata["num_masked"][idx].tolist(),
                "masked_positions": metadata["masked_positions"][idx].tolist(),
            }
        )
        csv_path = dataset_dir / f"{split_name}.csv"
        df.write_csv(csv_path)
        print(f"Saved {split_name:5s} CSV ({len(idx):>6} rows) → {csv_path}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    transform = transforms.Compose([transforms.ToTensor()])
    mnist = datasets.MNIST(root="_data", train=True, download=True, transform=transform)

    images, labels, masked_labels, metadata = generate_dataset(
        mnist_dataset=mnist,
        num_samples=10_000,
        canvas_size=128,
        max_jitter=20,
        scale_range=(0.9, 1.1),
        # --- image occlusion ---
        # 30 % of samples have exactly one digit absent (15 % a_only, 15 % b_only)
        occlusion_frac=0.30,
        # 5 % extreme: both digits absent, canvas is blank
        neither_frac=0.05,
        # --- label masking ---
        # 40 % of samples get some zero entries replaced
        label_mask_frac=0.40,
        # replace between 1 and 3 zeros per masked sample
        max_zeros_to_mask=3,
        # use -1.0 as the "unknown / uncertain" sentinel
        mask_symbol=-1.0,
        seed=42,
    )

    save_dataset_with_splits(
        images=images,
        labels=labels,
        masked_labels=masked_labels,
        metadata=metadata,
        dataset_dir="_data/dual_mnist_occluded_masked/raw",
        train_frac=0.70,
        val_frac=0.15,
        seed=42,
    )

    # quick sanity check
    print(f"\nImages shape        : {images.shape}")
    print(f"Labels shape        : {labels.shape}")
    print(f"Masked labels shape : {masked_labels.shape}")

    # find a sample that was actually masked for a meaningful example
    masked_idx = int(np.argmax(metadata["num_masked"] > 0))
    print(f"\nExample masked sample (index {masked_idx}):")
    print(f"  clean label    : {labels[masked_idx]}")
    print(f"  masked label   : {masked_labels[masked_idx]}")
    print(f"  num_masked     : {metadata['num_masked'][masked_idx]}")
    print(f"  masked_positions: {metadata['masked_positions'][masked_idx]}")
    print(
        f"  label_a={metadata['label_a'][masked_idx]}  "
        f"label_b={metadata['label_b'][masked_idx]}  "
        f"visible_a={metadata['visible_a'][masked_idx]}  "
        f"visible_b={metadata['visible_b'][masked_idx]}  "
        f"type={metadata['sample_type'][masked_idx]}"
    )
