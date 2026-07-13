import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger

from datamodules.bray_preprocessed_data_module import BrayPreprocessedDataModule
from lightning_models.lightning_mlp_classifier import LightningMLPClassifier

num_classes = 30

bray_datamodule = BrayPreprocessedDataModule(
    npz_path=f"_data/bray_dino/bray_dino_top_{num_classes}_moas.npz",
    batch_size=64,
    mask_uncertain=True,
    treat_uncertain_as_negative=False,
    feature_noise_std=0.0,
    ternary_labels=False,
)

model = LightningMLPClassifier(
    num_classes=num_classes,
    embedding_dim=384,
    lr=1e-4,
    masked_loss=True,
)

checkpoint_callback = ModelCheckpoint(
    save_top_k=1,
    monitor="val/roc_auc",
    mode="max",
    dirpath="./checkpoints/bray_dino/mlp",
    filename="mlp-{epoch:02d}-{val/roc_auc:.4f}",
)

logger = WandbLogger(
    project="bray_dino_mlp",
    name=f"mlp_top_{num_classes}",
)

trainer = L.Trainer(
    max_epochs=100,
    callbacks=[checkpoint_callback],
    check_val_every_n_epoch=5,
    logger=logger,
)

if __name__ == "__main__":
    trainer.fit(model=model, datamodule=bray_datamodule)
    trainer.test(model=model, datamodule=bray_datamodule)
