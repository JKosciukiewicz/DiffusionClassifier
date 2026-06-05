import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger

from datamodules.bray_preprocessed_data_module import BrayPreprocessedDataModule
from lightning_models.lightning_mlp_classifier import LightningMLPClassifier

bray_datamodule = BrayPreprocessedDataModule(
    npz_path="_data/gigadb/bray_top_30_moas.npz",
    batch_size=64,
    mask_uncertain=True,
    treat_uncertain_as_negative=False,
    feature_noise_std=0.0,
    ternary_labels=False,
)

model = LightningMLPClassifier(
    num_classes=30,
    embedding_dim=4558,
    lr=1e-4,
    masked_loss=True,
)

checkpoint_callback = ModelCheckpoint(
    save_top_k=1,
    monitor="val/roc_auc",
    mode="max",
    dirpath="./checkpoints/bray/mlp",
    filename="mlp-{epoch:02d}-{val/roc_auc:.4f}",
)

logger = WandbLogger(
    project="bray_mlp",
    name="mlp_top_30(all)",
)

trainer = L.Trainer(
    max_epochs=200,
    callbacks=[checkpoint_callback],
    check_val_every_n_epoch=5,
    logger=logger,
)

if __name__ == "__main__":
    trainer.fit(model=model, datamodule=bray_datamodule)
    trainer.test(model=model, datamodule=bray_datamodule)
