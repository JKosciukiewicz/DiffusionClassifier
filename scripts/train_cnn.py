import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger
from torchvision import transforms

from datamodules.two_digit_mnist_data_module import TwoDigitMNISTDataModule
from lightning_models import LightningCNN

two_digit_mnist_datamodule = TwoDigitMNISTDataModule(
    batch_size=128,
    transform=transforms.Compose(
        [
            transforms.Resize((128, 128)),
        ]
    ),
    data_dir="/Users/jkosciukiewicz/Developer/Research/DiffusionClassifier/_data/single_mnist_occluded_0/raw/",
)
cnn = LightningCNN(learning_rate=1e-3)
cnn.configure_optimizers()


checkpoint_callback = ModelCheckpoint(
    save_top_k=1,
    monitor="val_loss",
    mode="min",
    dirpath="./checkpoints/mnist_occluded/cnn_1_digit/",
    filename="cnn-{epoch:02d}-{train_loss:.4f}-{val_loss:.4f}",
    every_n_epochs=2,
)

logger = WandbLogger(project="diffusion_classifier", name="mnist_occluded_cnn_1_digit")

trainer = L.Trainer(
    max_epochs=20, 
    callbacks=[checkpoint_callback], 
    check_val_every_n_epoch=2,
    logger=logger
)
trainer.fit(model=cnn, datamodule=two_digit_mnist_datamodule)
