import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint

from lightning.pytorch.loggers import WandbLogger
from datamodules.bray_data_module import BrayDataModule
from lightning_models.lightning_diffusion_classifier import LightningDiffusionClassifier
from loss.masked_bce_loss import MaskedBCELoss

bray_datamodule = BrayDataModule(
    batch_size=128, data_dir="_data/gigadb", mask_uncertain=True
)
# bray_datamodule.setup("fit")

# num_classes = bray_datamodule.num_classes
# embedding_dim = bray_datamodule.embedding_dim
model_channels = 1024

diffusion = LightningDiffusionClassifier(
    alpha=0.05,
    num_classes=30,
    embedding_dim=4643,
    model_channels=model_channels,
    lr=1e-4,
    loss_fn=MaskedBCELoss,
    masked_loss=True,
    backbone_type="none",  # Use "none" for pre-extracted features
)

checkpoint_callback = ModelCheckpoint(
    save_top_k=1,
    monitor="validation_roc_auc",
    mode="max",
    dirpath="./checkpoints/bray/diffusion",
    filename="diffusion-{epoch:02d}-{validation_roc_auc:.4f}",
)

logger = WandbLogger(project="diffusion_classifier", name="diffusion_bray")

trainer = L.Trainer(
    max_epochs=200,
    callbacks=[checkpoint_callback],
    check_val_every_n_epoch=5,
    logger=logger
)

if __name__ == "__main__":
    trainer.fit(model=diffusion, datamodule=bray_datamodule)
    trainer.test(model=diffusion, datamodule=bray_datamodule)
