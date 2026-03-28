import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint

from datamodules.two_digit_mnist_data_module import TwoDigitMNISTDataModule
from lightning_models import LightningCNN

two_digit_mnist_datamodule = TwoDigitMNISTDataModule(batch_size=32)
cnn = LightningCNN()
cnn.configure_optimizers()


checkpoint_callback = ModelCheckpoint(
    save_top_k=1,
    monitor="train_loss",
    mode="min",
    dirpath="./checkpoints/mnist/cnn",
    filename="cnn-{epoch:02d}-{train_loss:.4f}",
    every_n_epochs=10,
)


trainer = L.Trainer(max_epochs=10, callbacks=[checkpoint_callback])
trainer.fit(model=cnn, datamodule=two_digit_mnist_datamodule)
