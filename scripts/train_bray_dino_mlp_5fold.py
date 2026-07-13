import lightning as L
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger

import wandb
from datamodules.bray_preprocessed_data_module import BrayPreprocessedDataModule
from lightning_models.lightning_mlp_classifier import LightningMLPClassifier

matplotlib.use("Agg")

num_classes = 3
npz_path = f"_data/bray_dino/bray_dino_top_{num_classes}_moas.npz"
auc_thresholds = [0.5, 0.6, 0.7]

_data = np.load(npz_path, allow_pickle=True)
moa_columns = list(_data["moa_columns"])
short_names = [c.replace("moa_", "") for c in moa_columns]

wandb_group = f"kfold_mlp_dino_top{num_classes}"
wandb_project = "bray_dino_mlp_5fold_fixed_w_ece"

fold_metrics = []  # result dicts from trainer.test(), one per fold
fold_per_class = []  # per-class AUC lists for cross-fold AUC boxplot
fold_y_pred = []  # raw predictions collected for pooled summary metrics
fold_y_true = []
fold_y_mask = []

if __name__ == "__main__":
    for fold in range(5):
        dm = BrayPreprocessedDataModule(
            npz_path=npz_path,
            batch_size=64,
            mask_uncertain=True,
            treat_uncertain_as_negative=False,
            feature_noise_std=0.0,
            ternary_labels=False,
            fold=fold,
        )

        model = LightningMLPClassifier(
            num_classes=num_classes,
            embedding_dim=384,
            lr=1e-4,
            masked_loss=True,
            moa_columns=moa_columns,
            auc_thresholds=auc_thresholds,
            log_per_fold_details=False,  # threshold counts + boxplot only in summary
        )

        checkpoint_callback = ModelCheckpoint(
            save_top_k=1,
            monitor="val/roc_auc",
            mode="max",
            dirpath=f"./checkpoints/bray_dino/mlp_5fold/fold_{fold}",
            filename="mlp-{epoch:02d}-{val/roc_auc:.4f}",
        )

        logger = WandbLogger(
            project=wandb_project,
            name=f"fold_{fold}",
            group=wandb_group,
        )

        trainer = L.Trainer(
            max_epochs=100,
            callbacks=[checkpoint_callback],
            check_val_every_n_epoch=5,
            logger=logger,
        )

        trainer.fit(model=model, datamodule=dm)
        results = trainer.test(model=model, datamodule=dm)
        fold_metrics.append(results[0])
        fold_per_class.append(model.per_class_auc)

        # Collect raw predictions before wandb.finish() clears the run.
        fold_y_pred.append(torch.cat(model.y_pred).numpy())
        fold_y_true.append(torch.cat(model.y_true).numpy())
        fold_y_mask.append(torch.cat(model.y_mask).numpy().astype(bool))

        wandb.finish()

    # ── Cross-fold summary ──────────────────────────────────────────────────
    # Pool all fold predictions into one array.
    all_y_pred = np.concatenate(fold_y_pred, axis=0)  # (N_total, K)
    all_y_true = np.concatenate(fold_y_true, axis=0)
    all_y_mask = np.concatenate(fold_y_mask, axis=0)

    auc_matrix = np.array(fold_per_class)  # (n_folds, K)

    print("\n=== 5-Fold Average Results ===")
    summary_metrics = {}
    for k in fold_metrics[0].keys():
        vals = [m[k] for m in fold_metrics if k in m]
        mean, std = np.mean(vals), np.std(vals)
        summary_metrics[f"summary/{k.replace('test/', '')}"] = mean
        print(f"  {k}: {mean:.4f} ± {std:.4f}")

    # Threshold counts from pooled per-class AUC (mean across folds).
    print("\nClasses above AUC threshold (pooled):")
    from sklearn.metrics import roc_auc_score

    pooled_per_class_auc = []
    for k in range(num_classes):
        m = all_y_mask[:, k]
        if m.sum() > 0 and len(np.unique(all_y_true[m, k])) > 1:
            pooled_per_class_auc.append(
                roc_auc_score(all_y_true[m, k], all_y_pred[m, k])
            )
        else:
            pooled_per_class_auc.append(np.nan)

    for t in auc_thresholds:
        count = sum(1 for a in pooled_per_class_auc if not np.isnan(a) and a >= t)
        summary_metrics[f"summary/n_classes_auc_above_{t}"] = count
        print(f"  >= {t}: {count} / {num_classes}")

    # Score distribution boxplot from pooled predictions.
    pos_scores = []
    for k in range(num_classes):
        m = all_y_mask[:, k] & (all_y_true[:, k] == 1)
        pos_scores.append(all_y_pred[m, k] if m.sum() > 0 else np.array([]))

    fig_dist, ax_dist = plt.subplots(figsize=(max(6, num_classes * 1.4), 5))
    ax_dist.boxplot(
        pos_scores,
        labels=short_names,
        patch_artist=True,
        showfliers=True,
        flierprops=dict(marker="o", markersize=3, alpha=0.4),
        medianprops=dict(color="black", linewidth=1.5),
        boxprops=dict(facecolor="#4c9be8", alpha=0.7),
    )
    ax_dist.set_ylim(0, 1)
    ax_dist.set_ylabel("Predicted score")
    ax_dist.set_title("Score distribution on true-positive samples — pooled 5 folds")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()

    # Per-class AUC boxplot across folds.
    fig_auc, ax_auc = plt.subplots(figsize=(max(6, num_classes * 1.4), 5))
    sns.boxplot(
        data=[auc_matrix[:, k] for k in range(num_classes)], ax=ax_auc, palette="Set2"
    )
    ax_auc.set_xticks(range(num_classes))
    ax_auc.set_xticklabels(short_names, rotation=45, ha="right")
    for t in auc_thresholds:
        ax_auc.axhline(t, color="gray", linestyle="--", linewidth=0.7, alpha=0.6)
    ax_auc.set_ylim(0, 1)
    ax_auc.set_ylabel("ROC-AUC")
    ax_auc.set_title(f"Per-class ROC-AUC across 5 folds — top {num_classes} MoAs")
    plt.tight_layout()

    summary_run = wandb.init(project=wandb_project, name="summary", group=wandb_group)
    summary_run.log(summary_metrics)
    summary_run.log(
        {
            "summary/per_class_auc_boxplot": wandb.Image(fig_auc),
            "summary/score_distribution": wandb.Image(fig_dist),
        }
    )
    plt.close(fig_auc)
    plt.close(fig_dist)
    summary_run.finish()
