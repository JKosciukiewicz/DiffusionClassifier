import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger

import wandb
from datamodules.bray_preprocessed_data_module import BrayPreprocessedDataModule
from lightning_models.lightning_mlp_classifier import LightningMLPClassifier

# Discriminative baseline for the CFM sweep in train_sweep_plgrid_top3.py.
# Same data, same splits, same metric, same budget — only the model differs.
sweep_config = {
    "method": "bayes",
    "metric": {"name": "val/roc_auc", "goal": "maximize"},
    "parameters": {
        "num_blocks": {"values": [1, 2, 4, 8, 12]},
        "hidden_dim": {"values": [128, 256, 512]},
        "lr": {"values": [1e-3, 1e-4, 5e-4]},
        "normalization_layer": {"values": [True, False]},
        "dropout": {"values": [0.0, 0.1, 0.3]},
        "weight_decay": {"values": [1e-8, 1e-4, 1e-2]},
    },
}


def sweep_train_step():
    wandb.init()
    config = wandb.config

    num_classes = 3
    # The MLP is trained with BCE, so labels must be {0, 1} + a validity mask
    # rather than the CFM's ternary {-1, 0, +1}. mask_uncertain=True gives the
    # *same* supervision: identical known/unknown entries, identical positives.
    ternary_labels = False
    mask_uncertain = True
    unknown_as_negative = False

    lr = config.lr
    num_blocks = config.num_blocks
    hidden_dim = config.hidden_dim
    normalization_layer = config.normalization_layer
    dropout = config.dropout
    weight_decay = config.weight_decay

    bray_datamodule = BrayPreprocessedDataModule(
        npz_path=f"_data/gigadb/bray_dino_top_{num_classes}_moas.npz",
        batch_size=256,
        mask_uncertain=mask_uncertain,
        treat_uncertain_as_negative=unknown_as_negative,
        feature_noise_std=0.0,
        ternary_labels=ternary_labels,
    )

    mlp = LightningMLPClassifier(
        num_classes=num_classes,
        embedding_dim=384,
        lr=lr,
        masked_loss=not unknown_as_negative,
        hidden_dims=[hidden_dim] * num_blocks,
        dropout=dropout,
        weight_decay=weight_decay,
        normalization_layer=normalization_layer,
    )

    checkpoint_callback = ModelCheckpoint(
        save_top_k=1,
        monitor="val/roc_auc",
        mode="max",
        dirpath="/net/pr2/projects/plgrid/plggwtln/jk/checkpoints/bray_dino/mlp",
        filename="mlp-{epoch:02d}-{val/roc_auc:.4f}",
    )

    logger = WandbLogger(
        project=f"mlp_dino_top_{num_classes}_sweeps",
        name=f"dino_top:{num_classes}__lr:{lr}__blocks:{num_blocks}__hidden:{hidden_dim}",
        experiment=wandb.run,
    )

    trainer = L.Trainer(
        max_epochs=100,
        callbacks=[checkpoint_callback],
        check_val_every_n_epoch=5,
        logger=logger,
    )

    trainer.fit(model=mlp, datamodule=bray_datamodule)
    trainer.test(model=mlp, datamodule=bray_datamodule)


if __name__ == "__main__":
    sweep_id = wandb.sweep(sweep=sweep_config, project="mlp_dino_sweeps_top_3")
    wandb.agent(sweep_id, function=sweep_train_step, count=100)
