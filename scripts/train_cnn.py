import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint

from datamodules.two_digit_mnist_data_module import TwoDigitMNISTDataModule
from lightning_models import LightningCNN

two_digit_mnist_datamodule = TwoDigitMNISTDataModule(
    batch_size=128, data_dir="_data/dual_mnist_occluded/raw"
)
cnn = LightningCNN(learning_rate=1e-3)
cnn.configure_optimizers()


checkpoint_callback = ModelCheckpoint(
    save_top_k=1,
    monitor="val_loss",
    mode="min",
    dirpath="./checkpoints/mnist_occluded/cnn",
    filename="cnn-{epoch:02d}-{train_loss:.4f}-{val_loss:.4f}",
    every_n_epochs=2,
)


trainer = L.Trainer(
    max_epochs=20, callbacks=[checkpoint_callback], check_val_every_n_epoch=2
)
trainer.fit(model=cnn, datamodule=two_digit_mnist_datamodule)
