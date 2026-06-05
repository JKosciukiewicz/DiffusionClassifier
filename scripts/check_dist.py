import lightning as L
import torch  # Make sure torch is imported to read stats
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger

from datamodules.bray_data_module import BrayDataModule
from datamodules.bray_preprocessed_data_module import BrayPreprocessedDataModule
from lightning_models.lightning_cfm_classifier import LightningCFMClassifier


def main():
    bray_datamodule = BrayPreprocessedDataModule(
        npz_path="_data/gigadb/bray_top_5_moas.npz",
        batch_size=64,
        mask_uncertain=False,
        treat_uncertain_as_negative=True,
        feature_noise_std=0.0,
        ternary_labels=True,
    )

    bray_datamodule.setup("fit")

    # 1. Get the training dataloader
    train_loader = bray_datamodule.train_dataloader()

    # 2. Grab a single batch safely now
    batch = next(iter(train_loader))

    # 3. Unpack the features (assuming features, labels, mask format)
    features = batch[0]

    # 4. Calculate and print the statistics
    print("=== Pre-extracted Feature Statistics ===")
    print(f"Feature Tensor Shape: {features.shape}")
    print(f"Mean:                 {features.mean().item():.4f}")
    print(f"Standard Deviation:   {features.std().item():.4f}")
    print(f"Minimum Value:        {features.min().item():.4f}")
    print(f"Maximum Value:        {features.max().item():.4f}")
    print("========================================")


if __name__ == "__main__":
    main()
