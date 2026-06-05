import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger

from datamodules.bray_data_module import BrayDataModule
from datamodules.bray_preprocessed_data_module import BrayPreprocessedDataModule
from lightning_models.lightning_flow_matching_classifier import (
    LightningFlowMatchingClassifier,
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
bray_datamodule = BrayPreprocessedDataModule(
    npz_path="_data/gigadb/bray_top_5_moas.npz",
    batch_size=64,
    mask_uncertain=True,
    treat_uncertain_as_negative=False,
    feature_noise_std=0.0,
    ternary_labels=False,
)

flow = LightningFlowMatchingClassifier(
    num_classes=5,
    embedding_dim=4558,
    lr=5e-4,
    masked_loss=False,  # false with uncertain as negative, true otherwise (impacts how metrics are calculated)
    backbone_type="none",  # Use "none" for pre-extracted features
    cfg_scale=2.0,
    num_sampling_steps=20,
    num_blocks=3,
    weight_decay=1e-8,
    ternary_labels=True,
)

checkpoint_callback = ModelCheckpoint(
    save_top_k=1,
    monitor="val/roc_auc",
    mode="max",
    dirpath="./checkpoints/bray/flow_matching",
    filename="flow-{epoch:02d}-{val/roc_auc:.4f}",
)

logger = WandbLogger(
    project="diffusion_bray_flow",
    name="flow_matching_bray_top_5_test_blocks=3_lr5e4",
)

trainer = L.Trainer(
    max_epochs=200,
    callbacks=[checkpoint_callback],
    check_val_every_n_epoch=5,
    logger=logger,
)

if __name__ == "__main__":
    trainer.fit(model=flow, datamodule=bray_datamodule)
    trainer.test(model=flow, datamodule=bray_datamodule)
