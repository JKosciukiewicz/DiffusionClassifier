import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger

import wandb
from datamodules.bray_preprocessed_data_module import BrayPreprocessedDataModule
from lightning_models.lightning_cfm_classifier import LightningCFMClassifier

# 1. Define the sweep configuration dictionary
sweep_config = {
    "method": "grid",  # Tries every combination
    "metric": {"name": "val/roc_auc", "goal": "maximize"},
    "parameters": {
        "num_blocks": {"values": [4, 8, 12]},
        "lr": {"values": [1e-3, 1e-4, 1e-5]},
        "t_power": {"values": [1.0, 2.0]},
    },
}


def sweep_train_step():
    # 2. Initialize the specific run determined by the sweep controller
    wandb.init()
    config = wandb.config

    # Static settings
    num_classes = 30
    ternary_labels = True
    mask_uncertain = False
    unknown_as_negative = False

    # Extract swept parameters
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

    # Model
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

    # Setup logger to hook into the current sweep run
    logger = WandbLogger(
        project="flow_matching_top_30_sweeps",
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
    # 3. Register the sweep with W&B central server
    sweep_id = wandb.sweep(sweep=sweep_config, project="flow_matching")

    # 4. Start the agent locally to run through the grid
    # count=18 means it will execute exactly the 18 combinations (3 * 3 * 2) and then exit
    wandb.agent(sweep_id, function=sweep_train_step, count=18)
