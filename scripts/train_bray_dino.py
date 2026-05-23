import lightning as L
import torch
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger

from datamodules.bray_dino_data_module import BrayDinoDataModule
from lightning_models.lightning_diffusion_classifier import LightningDiffusionClassifier
from loss.masked_bce_loss import MaskedBCELoss

# Configuration for Bray DINO
data_file = "_data/bray_dino/bray_dino_complete.csv"
split_col = "hier_split_0"  # Can be hier_split_0 to hier_split_4

bray_datamodule = BrayDinoDataModule(
    batch_size=1024,
    data_file=data_file,
    split_col=split_col,
    mask_uncertain=True,
)


diffusion = LightningDiffusionClassifier(
    alpha=0.5,
    num_classes=30,
    embedding_dim=384,
    lr=1e-4,
    loss_fn=MaskedBCELoss,
    masked_loss=True,
    backbone_type="none",
)

checkpoint_callback = ModelCheckpoint(
    save_top_k=1,
    monitor="val/roc_auc",
    mode="max",
    dirpath="./checkpoints/bray_dino/diffusion",
    filename="diffusion-{epoch:02d}-{val/roc_auc:.4f}",
)

logger = WandbLogger(
    project="diffusion_bray_dino",
    name=f"bray_dino_noise_{split_col}",
)

trainer = L.Trainer(
    max_epochs=50,
    callbacks=[checkpoint_callback],
    check_val_every_n_epoch=5,
    logger=logger,
)

if __name__ == "__main__":
    trainer.fit(model=diffusion, datamodule=bray_datamodule)
    trainer.test(model=diffusion, datamodule=bray_datamodule)
