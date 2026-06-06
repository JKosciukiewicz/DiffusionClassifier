import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger

import wandb
from datamodules.bray_preprocessed_data_module import BrayPreprocessedDataModule
from lightning_models.lightning_cfm_classifier import LightningCFMClassifier

# 1. Define the sweep configuration dictionary
sweep_config = {
    "method": "bayes",
    "metric": {"name": "val/roc_auc", "goal": "maximize"},
    "parameters": {
        "lr": {"values": [1e-4, 3e-4, 5e-4]},
        "num_blocks": {"values": [8, 12]},
        "t_power": {"values": [2.0, 3.0, 5.0]},
        "label_dropout": {"values": [0.1, 0.2, 0.3, 0.5]},
        "cond_dim": {"values": [256, 384, 512]},
    },
}


def sweep_train_step():
    # 2. Initialize the specific run determined by the sweep controller
    wandb.init()
    config = wandb.config

    # Static settings
    num_classes = 3
    ternary_labels = True
    mask_uncertain = True
    unknown_as_negative = False

    # Extract swept parameters
    lr = config.lr
    num_blocks = config.num_blocks
    t_power = config.t_power
    label_dropout = config.label_dropout
    cond_dim = config.cond_dim

    # DataModule
    bray_datamodule = BrayPreprocessedDataModule(
        npz_path=f"_data/gigadb/bray_top_{num_classes}_moas.npz",
        batch_size=256,
        mask_uncertain=mask_uncertain,
        treat_uncertain_as_negative=unknown_as_negative,
        feature_noise_std=0.0,
        ternary_labels=ternary_labels,
    )

    # Model
    flow = LightningCFMClassifier(
        num_classes=num_classes,
        embedding_dim=4558,
        lr=lr,
        cfm_method="vanilla",
        num_blocks=num_blocks,
        cond_dim=cond_dim,
        masked_loss=False if unknown_as_negative or ternary_labels else True,
        backbone_type="none",
        normalization_layer=True,
        weight_decay=1e-8,
        ternary_labels=ternary_labels,
        t_power=t_power,
        label_dropout=label_dropout,
    )

    checkpoint_callback = ModelCheckpoint(
        save_top_k=1,
        monitor="val/roc_auc",
        mode="max",
        dirpath="./checkpoints/bray/flow_matching",
        filename="flow-{epoch:02d}-{val/roc_auc:.4f}",
    )

    # Setup logger to hook into the current sweep run
    logger = WandbLogger(
        project=f"flow_matching_top_{num_classes}_sweeps",
        name=f"bray_top:{num_classes}__lr:{lr}__blocks:{num_blocks}__t_power:{t_power}__ldrop:{label_dropout}__cond:{cond_dim}",
        experiment=wandb.run,
    )

    trainer = L.Trainer(
        max_epochs=200,
        callbacks=[checkpoint_callback],
        check_val_every_n_epoch=5,
        logger=logger,
    )

    trainer.fit(model=flow, datamodule=bray_datamodule)
    trainer.test(model=flow, datamodule=bray_datamodule)


if __name__ == "__main__":
    # 3. Register the sweep with W&B central server
    sweep_id = wandb.sweep(sweep=sweep_config, project="flow_matching_sweep2_top_3")

    wandb.agent(sweep_id, function=sweep_train_step, count=30)
