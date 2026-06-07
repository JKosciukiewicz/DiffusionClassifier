import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger

import wandb
from datamodules.bray_preprocessed_data_module import BrayPreprocessedDataModule
from lightning_models.lightning_cfm_classifier import LightningCFMClassifier

sweep_config = {
    "method": "bayes",
    "metric": {"name": "val/roc_auc", "goal": "maximize"},
    "parameters": {
        "num_blocks": {"values": [1, 2, 4, 8, 12]},
        "cfm_method": {"values": ["vanilla", "ot"]},
        "lr": {"values": [1e-3, 1e-4, 5e-4]},
        "normalization_layer": {"values": [True, False]},
        "t_power": {"values": [1.0, 2.0, 3.0, 5.0]},
        "label_dropout": {"values": [0.1, 0.2, 0.3, 0.5]},
        "cond_dim": {"values": [128, 256, 384]},
    },
}


def sweep_train_step():
    wandb.init()
    config = wandb.config

    num_classes = 30
    ternary_labels = True
    mask_uncertain = False
    unknown_as_negative = False

    lr = config.lr
    num_blocks = config.num_blocks
    t_power = config.t_power
    normalization_layer = config.normalization_layer
    cfm_method = config.cfm_method
    label_dropout = config.label_dropout
    cond_dim = config.cond_dim

    bray_datamodule = BrayPreprocessedDataModule(
        npz_path=f"_data/gigadb/bray_dino_top_{num_classes}_moas.npz",
        batch_size=256,
        mask_uncertain=mask_uncertain,
        treat_uncertain_as_negative=unknown_as_negative,
        feature_noise_std=0.0,
        ternary_labels=ternary_labels,
    )

    flow = LightningCFMClassifier(
        num_classes=num_classes,
        embedding_dim=384,
        lr=lr,
        cfm_method=cfm_method,
        num_blocks=num_blocks,
        masked_loss=False if unknown_as_negative or ternary_labels else True,
        backbone_type="none",
        normalization_layer=normalization_layer,
        weight_decay=1e-8,
        ternary_labels=ternary_labels,
        t_power=t_power,
        cond_dim=cond_dim,
        label_dropout=label_dropout,
    )

    checkpoint_callback = ModelCheckpoint(
        save_top_k=1,
        monitor="val/roc_auc",
        mode="max",
        dirpath="/net/pr2/projects/plgrid/plggwtln/jk/checkpoints/bray_dino/flow_matching",
        filename="flow-{epoch:02d}-{val/roc_auc:.4f}",
    )

    logger = WandbLogger(
        project=f"flow_matching_dino_top_{num_classes}_sweeps",
        name=f"dino_top:{num_classes}__lr:{lr}__blocks:{num_blocks}__t_power:{t_power}",
        experiment=wandb.run,
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
    sweep_id = wandb.sweep(
        sweep=sweep_config, project="flow_matching_dino_sweeps_top_30"
    )
    wandb.agent(sweep_id, function=sweep_train_step, count=100)
