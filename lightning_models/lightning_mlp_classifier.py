from typing import List, Optional

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import wandb
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from torchmetrics.functional.classification import binary_calibration_error

from lightning_models.base_model import BaseModel
from loss import MaskedBCELoss
from models.mlp import MLPClassifier

matplotlib.use("Agg")


def calibration_error(probs, targets, norm: str = "l1", n_bins: int = 15) -> float:
    """Expected (l1) / maximum (max) calibration error over a flat prob/target vector.

    Multi-label with masking, so this is the binary (per-entry) variant rather than
    the multiclass one: every valid (sample, class) pair is one binary prediction.
    """
    probs = torch.as_tensor(probs, dtype=torch.float32)
    targets = torch.as_tensor(targets).long()
    return binary_calibration_error(probs, targets, n_bins=n_bins, norm=norm).item()


class LightningMLPClassifier(BaseModel):
    def __init__(
        self,
        num_classes: int,
        embedding_dim: int,
        lr: float = 1e-4,
        masked_loss: bool = False,
        hidden_dims: Optional[List[int]] = None,
        dropout: float = 0.0,
        weight_decay: float = 0.0,
        normalization_layer: bool = False,
        moa_columns: Optional[List[str]] = None,
        auc_thresholds: List[float] = [0.6, 0.7, 0.8, 0.9],
        log_per_fold_details: bool = True,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.lr = lr
        self.weight_decay = weight_decay
        self.masked_loss = masked_loss
        # Same LayerNorm on the input features as the CFM model, so the two are
        # comparable when sweeping.
        self.feature_norm = (
            nn.LayerNorm(embedding_dim, elementwise_affine=False)
            if normalization_layer
            else nn.Identity()
        )
        self.model = MLPClassifier(
            num_classes=num_classes,
            embedding_dim=embedding_dim,
            hidden_dims=hidden_dims,
            dropout=dropout,
        )
        self.loss_fn = nn.BCELoss()
        self.masked_loss_fn = MaskedBCELoss()
        self.moa_columns = moa_columns or [str(i) for i in range(num_classes)]
        self.auc_thresholds = auc_thresholds
        self.log_per_fold_details = log_per_fold_details
        self.y_pred = []
        self.y_true = []
        self.y_mask = []
        self.per_class_auc: List[float] = []
        self.per_class_ece: List[float] = []

    def forward(self, x):
        return self.model(self.feature_norm(x.float()))

    def _loss(self, pred, y, mask):
        if self.masked_loss:
            return self.masked_loss_fn(pred, y, mask)
        return self.loss_fn(pred, y)

    def training_step(self, batch, batch_idx):
        x, y, mask = batch
        pred = self.forward(x)
        loss = self._loss(pred, y, mask)
        self.log("train/loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y, mask = batch
        pred = self.forward(x)
        loss = self._loss(pred, y, mask)
        self.log("val/loss", loss, prog_bar=True)

        mask_bool = mask.bool()
        y_pred_flat = pred[mask_bool].cpu()
        y_true_flat = y[mask_bool].cpu()

        if len(y_true_flat) > 0 and len(torch.unique(y_true_flat)) > 1:
            try:
                roc_auc = roc_auc_score(y_true_flat.numpy(), y_pred_flat.detach().numpy())
                self.log("val/roc_auc", roc_auc, prog_bar=True)
            except ValueError:
                pass
        if len(y_true_flat) > 0:
            self.log("val/ece", calibration_error(y_pred_flat.detach(), y_true_flat))
        return loss

    def test_step(self, batch, batch_idx):
        x, y, mask = batch
        pred = self.forward(x)
        # Keep full (N, K) tensors — per-class AUC needs them unflattened.
        self.y_pred.append(pred.cpu())
        self.y_true.append(y.cpu())
        self.y_mask.append(mask.bool().cpu())

    def on_test_start(self):
        self.y_pred = []
        self.y_true = []
        self.y_mask = []

    def on_test_epoch_end(self):
        y_true = torch.cat(self.y_true).numpy()          # (N, K)
        y_pred = torch.cat(self.y_pred).numpy()          # (N, K)
        y_mask = torch.cat(self.y_mask).numpy().astype(bool)  # (N, K)

        K = y_true.shape[1]

        # Per-class ROC-AUC and calibration error
        per_class_auc = []
        per_class_ece = []
        for k in range(K):
            m = y_mask[:, k]
            if m.sum() > 0 and len(np.unique(y_true[m, k])) > 1:
                per_class_auc.append(roc_auc_score(y_true[m, k], y_pred[m, k]))
                per_class_ece.append(calibration_error(y_pred[m, k], y_true[m, k]))
            else:
                per_class_auc.append(np.nan)
                per_class_ece.append(np.nan)
        self.per_class_auc = per_class_auc
        self.per_class_ece = per_class_ece

        # Pooled metrics over all (sample, class) pairs where mask is True
        y_true_flat = y_true[y_mask]
        y_pred_flat = y_pred[y_mask]

        metrics = {}
        if len(y_true_flat) > 0 and len(np.unique(y_true_flat)) > 1:
            try:
                y_pred_binary = (y_pred_flat > 0.5).astype(np.float32)
                metrics["test/roc_auc"] = roc_auc_score(y_true_flat, y_pred_flat)
                metrics["test/accuracy"] = accuracy_score(y_true_flat, y_pred_binary)
                metrics["test/precision"] = precision_score(
                    y_true_flat, y_pred_binary, zero_division=0
                )
                metrics["test/recall"] = recall_score(
                    y_true_flat, y_pred_binary, zero_division=0
                )
                metrics["test/f1_score"] = f1_score(
                    y_true_flat, y_pred_binary, zero_division=0
                )
                metrics["test/ece"] = calibration_error(y_pred_flat, y_true_flat)
                metrics["test/mce"] = calibration_error(
                    y_pred_flat, y_true_flat, norm="max"
                )
            except Exception as e:
                print(f"Error calculating pooled metrics: {e}")

        # Per-class AUC / ECE scalars
        for k, auc in enumerate(per_class_auc):
            if not np.isnan(auc):
                metrics[f"test/roc_auc_{self.moa_columns[k]}"] = auc
        for k, ece in enumerate(per_class_ece):
            if not np.isnan(ece):
                metrics[f"test/ece_{self.moa_columns[k]}"] = ece

        if self.log_per_fold_details:
            valid_aucs = [a for a in per_class_auc if not np.isnan(a)]
            for t in self.auc_thresholds:
                metrics[f"test/n_classes_auc_above_{t}"] = sum(1 for a in valid_aucs if a >= t)

        print(f"Test Metrics: {metrics}")
        self.log_dict(metrics)

        if self.log_per_fold_details and self.logger is not None and hasattr(self.logger, "experiment"):
            self._log_score_distribution(y_true, y_pred, y_mask)

    def _log_score_distribution(self, y_true, y_pred, y_mask):
        """Boxplot: for each class, distribution of predicted scores on true-positive samples."""
        import seaborn as sns

        K = y_true.shape[1]
        short_names = [c.replace("moa_", "") for c in self.moa_columns]

        # Collect scores assigned to positive examples for each class.
        pos_scores = []
        for k in range(K):
            m = y_mask[:, k] & (y_true[:, k] == 1)
            pos_scores.append(y_pred[m, k] if m.sum() > 0 else np.array([]))

        fig, ax = plt.subplots(figsize=(max(6, K * 1.4), 5))
        ax.boxplot(
            pos_scores,
            labels=short_names,
            patch_artist=True,
            showfliers=True,
            flierprops=dict(marker="o", markersize=3, alpha=0.4),
            medianprops=dict(color="black", linewidth=1.5),
            boxprops=dict(facecolor="#4c9be8", alpha=0.7),
        )
        ax.set_ylim(0, 1)
        ax.set_ylabel("Predicted score")
        ax.set_title("Score distribution on true-positive samples (per class)")
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()
        self.logger.experiment.log({"test/score_distribution": wandb.Image(fig)})
        plt.close(fig)

    def configure_optimizers(self):
        return torch.optim.AdamW(
            self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )
