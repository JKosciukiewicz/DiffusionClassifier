import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger

from datamodules.bray_dino_data_module import BrayDinoDataModule
from lightning_models.lightning_autoencoder import LightningAutoencoder


def train_autoencoder():
    # Configuration for Bray DINO
    data_file = "_data/bray_dino/bray_dino_complete.csv"
    split_col = "hier_split_0"  # Can be hier_split_0 to hier_split_4

    bray_datamodule = BrayDinoDataModule(
        batch_size=1024,
        data_file=data_file,
        split_col=split_col,
        mask_uncertain=True,
    )
    bray_datamodule.setup("fit")

    input_dim = bray_datamodule.embedding_dim
    hidden_dim = 128
    latent_dim = 32

    model = LightningAutoencoder(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        latent_dim=latent_dim,
        lr=1e-3,
    )

    checkpoint_callback = ModelCheckpoint(
        save_top_k=1,
        monitor="val/loss",
        mode="min",
        dirpath="./checkpoints/autoencoder",
        filename="autoencoder-{epoch:02d}-{val/loss:.4f}",
    )

    logger = WandbLogger(project="Autoencoder", name="autoencoder_bray_dino")

    trainer = L.Trainer(
        max_epochs=10,
        callbacks=[checkpoint_callback],
        logger=logger,
    )

    trainer.fit(model=model, datamodule=bray_datamodule)


if __name__ == "__main__":
    train_autoencoder()
