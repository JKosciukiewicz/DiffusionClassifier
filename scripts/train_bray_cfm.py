import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger

from datamodules.bray_data_module import BrayDataModule
from datamodules.bray_preprocessed_data_module import BrayPreprocessedDataModule
from lightning_models.lightning_cfm_classifier import (
    LightningCFMClassifier,
)

# bray_datamodule = BrayDataModule(
#     batch_size=64,
#     data_dir="_data/gigadb",
#     label_file="/Users/jkosciukiewicz/Developer/Research/DiffusionClassifier/_data/gigadb/gigadb_top_5_moas.csv",
#     mask_uncertain=True,
#     treat_uncertain_as_negative=False,
#     feature_noise_std=0.0,
#     ternary_labels=False,
# )

num_classes = 30
lr = 1e-4
num_blocks = 4
ternary_labels = True
mask_uncertain = False
unknown_as_negative = False

bray_datamodule = BrayPreprocessedDataModule(
    npz_path=f"_data/gigadb/bray_top_{num_classes}_moas.npz",
    batch_size=64,
    mask_uncertain=mask_uncertain,
    treat_uncertain_as_negative=unknown_as_negative,
    feature_noise_std=0.0,
    ternary_labels=ternary_labels,
)

flow = LightningCFMClassifier(
    num_classes=num_classes,
    embedding_dim=4558,
    lr=lr,
    cfm_method="vanilla",  # vanilla or ot
    num_blocks=num_blocks,
    masked_loss=False if unknown_as_negative or ternary_labels else True,
    backbone_type="none",  # Use "none" for pre-extracted features
    weight_decay=1e-8,
    ternary_labels=ternary_labels,
    t_power=1.0,  # low-t (high-noise) emphasis; >1 pushes the model to use labels, 1.0 = uniform
)

checkpoint_callback = ModelCheckpoint(
    save_top_k=1,
    monitor="val/roc_auc",
    mode="max",
    dirpath="./checkpoints/bray/flow_matching",
    filename="flow-{epoch:02d}-{val/roc_auc:.4f}",
)

logger = WandbLogger(
    project="flow_matching",
    name=f"bray_top:{num_classes}__lr:{lr}__blocks:{num_blocks}__ternary:{ternary_labels}__masked:{mask_uncertain}__unknown_as_netagive:{unknown_as_negative}",
)

trainer = L.Trainer(
    max_epochs=100,
    callbacks=[checkpoint_callback],
    check_val_every_n_epoch=5,
    logger=logger,
)

if __name__ == "__main__":
    trainer.fit(model=flow, datamodule=bray_datamodule)
    trainer.test(model=flow, datamodule=bray_datamodule)
