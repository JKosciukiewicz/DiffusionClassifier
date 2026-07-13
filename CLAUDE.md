# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

**Environment:** Uses `uv` with Python 3.12 and a `.venv` at the project root.

```bash
# Install dependencies
uv sync

# Run a training script
uv run python scripts/train_bray.py
uv run python scripts/train_diffusion.py

# Run via LightningCLI with a config file
uv run python -m lightning.pytorch.cli fit --config configs/diffusion_mnist.yaml
```

There are no automated tests in this project.

## Architecture

This project implements a **diffusion-based multi-label classifier** — it treats classification as a denoising diffusion process over label vectors rather than standard direct prediction.

### Core Idea

At training time, labels are scaled to `[-1, 1]`, Gaussian noise is added at a random timestep, and the model learns to predict either the noise (`objective="noise"`) or the denoised labels (`objective="labels"`). At inference, the DDIM scheduler iteratively denoises from random noise to a predicted label vector, with optional Classifier-Free Guidance (CFG).

### Module Layout

- **`models/`** — Pure PyTorch `nn.Module` definitions:
  - `diffusion_autoencoder.py`: Primary denoising network (`DiffusionAutoencoder`). Encodes input features + sinusoidal time embeddings into a conditioning context, then uses stacked `CrossAttentionBlock`s to denoise the noisy label vector. Supports CFG via a 15% feature-dropout during training.
  - `diffusion_classifier.py`: Older/simpler MLP-based denoiser (`DiffusionClassifier`) that concatenates features + noisy labels + timestep.
  - `clip_extractor.py`: Frozen CLIP backbone for image feature extraction.
  - `autoencoder.py`, `cnn.py`: Additional backbone options.

- **`lightning_models/`** — PyTorch Lightning wrappers:
  - `lightning_diffusion_classifier.py`: Main training module (`LightningDiffusionClassifier`). Owns the `DDIMScheduler`, backbone, and `DiffusionAutoencoder`. Implements `predict_labels()` for iterative DDIM generation with dual-forward CFG. Validation uses 20 DDIM steps; test uses 50.
  - `lightning_cnn.py`, `lightning_autoencoder.py`: Backbone lightning modules that can be loaded as frozen feature extractors via `backbone_ckpt_path`.
  - `base_model.py`: Marker base class (`BaseModel(L.LightningModule)`).

- **`datamodules/`** — Lightning `DataModule` subclasses, each paired with a dataset in `datasets/`:
  - `BrayDataModule` / `BrayDataset`: Cell morphology features (CSV) → MoA multi-label prediction. Primary dataset. Data at `_data/gigadb/`.
  - `TwoDigitMNISTDataModule`: Image-based multi-label (two overlapping digits). Data at `_data/dual_mnist_occluded/`.
  - `BBBC021DataModule`: Cell imaging dataset.

- **`datasets/`** — `torch.utils.data.Dataset` implementations. `BrayDataset` uses Polars for efficient CSV loading.

- **`scripts/`** — Standalone training entry points (not importable modules). Each script instantiates a datamodule, model, callbacks, and `WandbLogger`, then calls `trainer.fit()` / `trainer.test()`.

- **`loss/`** — Custom loss functions (`MaskedBCELoss`).

- **`configs/`** — YAML configs for `LightningCLI`.

### Backbone Types

`LightningDiffusionClassifier` accepts `backbone_type` in `{"none", "cnn", "autoencoder", "clip"}`:
- `"none"`: Input features are pre-extracted (e.g., Bray/GigaDB cell morphology features passed directly as vectors).
- `"cnn"` / `"autoencoder"`: Load a frozen checkpoint, call `.extract_features(x)`.
- `"clip"`: Frozen `CLIPExtractor` wrapping OpenAI CLIP.

### Data Format

Each batch is a tuple `(x, y, mask)`:
- `x`: feature vector or image tensor
- `y`: binary label vector `[0, 1]` (float)
- `mask`: boolean mask for valid/certain labels (used in loss and metrics to ignore uncertain entries)

### Key Hyperparameters

- `objective`: `"noise"` (predict added noise, default) or `"labels"` (predict denoised labels directly)
- `cfg_scale`: CFG guidance scale (default `2.0`); set to `1.0` to disable
- `masked_loss`: Whether metrics exclude uncertain label entries
- `num_timesteps`: DDIM training timesteps (default `1000`)
- Checkpoints saved to `./checkpoints/<dataset>/<model_type>/`
- Experiment tracking via Weights & Biases (`wandb`)

### Instructions
Don't add unnecesary bloat like extra classes for thing that could be functions, unit tests etc. This is a research project, code quality is not a priority yet. If unsure ask.