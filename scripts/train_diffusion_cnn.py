import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint

from datamodules.two_digit_mnist_data_module import TwoDigitMNISTDataModule
from lightning_models import LightningDiffusionClassifier
from loss.masked_bce_loss import MaskedBCELoss

two_digit_mnist_datamodule = TwoDigitMNISTDataModule(
    batch_size=128, data_dir="_data/dual_mnist_occluded/raw"
)

diffusion = LightningDiffusionClassifier(
    alpha=0.05,
    num_classes=10,
    embedding_dim=128,  # Use 128 for CNN, 512 for CLIP (ViT-B/32)
    lr=1e-4,
    loss_fn=MaskedBCELoss,
    masked_loss=True,
    backbone_type="cnn",  # "cnn" or "clip"
    cnn_ckpt_path="checkpoints/mnist_occluded/cnn/cnn-epoch=09-train_loss=0.1351-val_loss=0.3547.ckpt",
    # clip_model_name="ViT-B/32",  # Only needed if backbone_type="clip"
)

diffusion.configure_optimizers()


checkpoint_callback = ModelCheckpoint(
    save_top_k=1,
    monitor="train_loss",
    mode="min",
    dirpath="./checkpoints/mnist_occluded_masked_cnn/diffusion",
    filename="diffusion-{epoch:02d}-{train_loss:.4f}-{validation_loss:.4f}",
    every_n_epochs=40,
)

trainer = L.Trainer(
    max_epochs=200, callbacks=[checkpoint_callback], check_val_every_n_epoch=3
)
trainer.fit(model=diffusion, datamodule=two_digit_mnist_datamodule)
trainer.test(model=diffusion, datamodule=two_digit_mnist_datamodule)
