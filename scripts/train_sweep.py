import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger

import wandb
from datamodules.bray_preprocessed_data_module import BrayPreprocessedDataModule
from lightning_models.lightning_cfm_classifier import LightningCFMClassifier


def main():
    # 1. Initialize wandb to fetch this run's specific hyperparameters
    wandb.init()
    config = wandb.config

    # Static settings
    num_classes = 30
    ternary_labels = True
    mask_uncertain = False
    unknown_as_negative = False

    # 2. Extract swept parameters from config
    lr = config.lr
    num_blocks = config.num_blocks
    t_power = config.t_power

    # DataModule
    bray_datamodule = BrayPreprocessedDataModule(
        npz_path=f"_data/gigadb/bray_top_{num_classes}_moas.npz",
        batch_size=64,
        mask_uncertain=mask_uncertain,
        treat_uncertain_as_negative=unknown_as_negative,
        feature_noise_std=0.0,
        ternary_labels=ternary_labels,
    )

    # Model (Injecting the swept hyperparams here)
    flow = LightningCFMClassifier(
        num_classes=num_classes,
        embedding_dim=4558,
        lr=lr,
        cfm_method="vanilla",
        num_blocks=num_blocks,
        masked_loss=False if unknown_as_negative or ternary_labels else True,
        backbone_type="none",
        weight_decay=1e-8,
        ternary_labels=ternary_labels,
        t_power=t_power,
    )

    checkpoint_callback = ModelCheckpoint(
        save_top_k=1,
        monitor="val/roc_auc",
        mode="max",
        dirpath="./checkpoints/bray/flow_matching",
        filename="flow-{epoch:02d}-{val/roc_auc:.4f}",
    )

    # 3. Use the existing wandb run instead of creating a new one
    logger = WandbLogger(
        project="flow_matching",
        name=f"bray_top:{num_classes}__lr:{lr}__blocks:{num_blocks}__t_power:{t_power}",
    )

    trainer = L.Trainer(
        max_epochs=100,
        callbacks=[checkpoint_callback],
        check_val_every_n_epoch=5,
        logger=logger,
    )

    trainer.fit(model=flow, datamodule=bray_datamodule)
    trainer.test(model=flow, datamodule=bray_datamodule)


if __name__ == "__main__":
    main()
