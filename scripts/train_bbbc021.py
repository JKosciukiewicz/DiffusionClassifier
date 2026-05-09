import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint

from lightning.pytorch.loggers import WandbLogger
from datamodules.bbbc021_data_module import BBBC021DataModule
from lightning_models.lightning_diffusion_classifier import LightningDiffusionClassifier
from loss.masked_bce_loss import MaskedBCELoss

bbbc021_datamodule = BBBC021DataModule(
    batch_size=128, 
    data_dir="_data/BBBC021",
    mask_uncertain=True
)
bbbc021_datamodule.setup("fit")

num_classes = bbbc021_datamodule.num_classes
embedding_dim = bbbc021_datamodule.embedding_dim
model_channels = 768

diffusion = LightningDiffusionClassifier(
    alpha=0.05,
    num_classes=num_classes,
    embedding_dim=embedding_dim,
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
    dirpath="./checkpoints/bbbc021/diffusion",
    filename="diffusion-{epoch:02d}-{validation_roc_auc:.4f}",
)

logger = WandbLogger(project="diffusion_classifier", name="diffusion_bbbc021")

trainer = L.Trainer(
    max_epochs=200,
    callbacks=[checkpoint_callback],
    check_val_every_n_epoch=5,
    logger=logger
)

if __name__ == "__main__":
    trainer.fit(model=diffusion, datamodule=bbbc021_datamodule)
    trainer.test(model=diffusion, datamodule=bbbc021_datamodule)
