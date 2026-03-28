import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint

from datamodules.two_digit_mnist_data_module import TwoDigitMNISTDataModule
from lightning_models import LightningDiffusionClassifier

two_digit_mnist_datamodule = TwoDigitMNISTDataModule(batch_size=32)

diffusion = LightningDiffusionClassifier(
    alpha=0.05,
    num_classes=10,
    embedding_dim=128,
    lr=1e-4,
    cnn_ckpt_path="checkpoints/mnist/cnn/cnn-epoch=09-train_loss=0.0276.ckpt",
)

diffusion.configure_optimizers()


checkpoint_callback = ModelCheckpoint(
    save_top_k=1,
    monitor="train_loss",
    mode="min",
    dirpath="./checkpoints/mnist/diffusion",
    filename="cnn-{epoch:02d}-{train_loss:.4f}",
    every_n_epochs=10,
)

trainer = L.Trainer(max_epochs=10, callbacks=[checkpoint_callback])
trainer.fit(model=diffusion, datamodule=two_digit_mnist_datamodule)
