import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint
from torch.nn import CrossEntropyLoss
from torchvision import transforms

from datamodules.two_digit_mnist_data_module import TwoDigitMNISTDataModule
from lightning_models import LightningDiffusionClassifier
from loss.masked_bce_loss import MaskedBCELoss

two_digit_mnist_datamodule = TwoDigitMNISTDataModule(
    batch_size=128,
    transform=transforms.Compose(
        [
            transforms.Resize((128, 128)),
        ]
    ),
    data_dir="/Users/jkosciukiewicz/Developer/Research/DiffusionClassifier/_data/single_mnist_occluded_0/raw/",
)

diffusion = LightningDiffusionClassifier(
    alpha=0.05,
    num_classes=10,
    embedding_dim=128, # Use 128 for CNN, 512 for CLIP (ViT-B/32)
    lr=1e-4,
    loss_fn=CrossEntropyLoss,
    masked_loss=False,
    backbone_type="cnn", # "cnn" or "clip"
    cnn_ckpt_path="checkpoints/mnist_occluded/cnn_1_digit/cnn-epoch=19-train_loss=0.0015-val_loss=0.0179.ckpt",
    # clip_model_name="ViT-B/32", # Only needed if backbone_type="clip"
)

diffusion.configure_optimizers()


checkpoint_callback = ModelCheckpoint(
    save_top_k=1,
    monitor="train_loss",
    mode="min",
    dirpath="./checkpoints/mnist_occluded_masked/diffusion_1_digit",
    filename="diffusion-{epoch:02d}-{train_loss:.4f}-{validation_loss:.4f}",
    every_n_epochs=10,
)

trainer = L.Trainer(
    max_epochs=10, callbacks=[checkpoint_callback], check_val_every_n_epoch=3
)
trainer.fit(model=diffusion, datamodule=two_digit_mnist_datamodule)
trainer.test(model=diffusion, datamodule=two_digit_mnist_datamodule)
