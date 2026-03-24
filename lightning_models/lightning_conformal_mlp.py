from typing import Optional, Type

import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score

from models.bio_models.bbbc_mlp import MLPClassifierBBBC
from utils.evaluate_conformal_model import calculate_metrics, calculate_val_metrics
from models.bio_models.baseline_mlp import MLPClassifier, MLPClassifierXlg
from models.bio_models.dino_mlp import MLPClassifierDino, MLPClassifierDinoSmall
from lightning_models.base_model import BaseModel
from lightning_models.loss.weighted_masked_bce import weighted_masked_bce_loss
from utils.conformal_prediction import (
    multiclass_non_conformity_score,
    apply_multiclass_thresholds,
)


class LightningConformalClassifier(BaseModel):
    def __init__(
        self,
        size: str,
        num_classes: int,
        embedding_dim: int,
        lr: float,
        alpha: float,
        residual: bool,
        activation_fn: Type[nn.Module],
        results_path: str = None,
        backbone: Optional[BaseModel] = None,
        backbone_ckpt_path: Optional[str] = None,
        dropout_rate: float = 0.3,
        percent_accept: float = 10.0,
        loss_weighted: bool = False,
        loss_pos_weight: float = 1,
    ):
        super().__init__()

        self.alpha = alpha
        self.num_classes = num_classes
        self.lr = lr
        self.loss_weighted = loss_weighted
        self.loss_pos_weight = loss_pos_weight
        self.results_file = results_path

        self.nonconformity_scores = []
        self.thresholds = None
        self.percent_accept = percent_accept
        self.embedding_dim = embedding_dim

        # Init backbone if specified
        self.backbone = (
            type(backbone).load_from_checkpoint(backbone_ckpt_path)
            if backbone
            else None
        )

        # Init model
        if size == "sm":
            self.model = MLPClassifierDino(
                num_classes=num_classes,
                embedding_dim=embedding_dim,
                dropout_rate=dropout_rate,
                residual=residual,
                activation_fn=activation_fn,
            )

        elif size == "xsm":
            self.model = MLPClassifierDinoSmall(
                num_classes=num_classes,
                embedding_dim=embedding_dim,
                dropout_rate=dropout_rate,
                residual=residual,
                activation_fn=activation_fn,
            )

        elif size == "md":
            self.model = MLPClassifierBBBC(
                num_classes=num_classes,
                embedding_dim=embedding_dim,
                dropout_rate=dropout_rate,
                residual=residual,
                activation_fn=activation_fn,
            )
        elif size == "lg":
            self.model = MLPClassifier(
                num_classes=num_classes,
                embedding_dim=embedding_dim,
                dropout_rate=dropout_rate,
                residual=residual,
                activation_fn=activation_fn,
            )
        elif size == "xlg":
            self.model = MLPClassifierXlg(
                num_classes=num_classes,
                embedding_dim=embedding_dim,
                dropout_rate=dropout_rate,
                residual=residual,
                activation_fn=activation_fn,
            )

        # For evaluation/metric accumulation:
        self.y_pred_conf = []
        self.y_true = []
        self.y_pred_conf_bt = []
        self.y_true_bt = []
        self.y_pred_raw = []  # Store raw probabilities for ROC AUC
        self.y_true_raw = []  # Store corresponding true labels
        self.ex_rejected = 0
        self.cls_rejected = 0

        # wandb
        self.peak_roc_auc = 0
        self.roc_aucs = []

    def on_test_start(self):
        """Called when test starts - ensure thresholds are set."""
        if self.thresholds is None:
            print("Thresholds from alpha")
            self._set_thresholds_from_alpha()

    def _set_thresholds_from_alpha(self):
        """Automatically set thresholds based on alpha parameter."""
        self.thresholds = np.ones(self.num_classes) * (1 - self.alpha)
        print(
            f"Automatically set thresholds based on alpha={self.alpha}: {self.thresholds}"
        )

    def set_thresholds(self, thresholds):
        """Manual override for thresholds if needed."""
        self.thresholds = thresholds

    def configure_optimizers(self):
        return torch.optim.Adam(self.model.parameters(), lr=self.lr)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        predicted_eps = self.model(features=features)
        return predicted_eps

    def on_validation_start(self) -> None:
        self.roc_aucs = []
        self._set_thresholds_from_alpha()

    def on_validation_end(self):
        mean_roc_auc = np.mean(np.array(self.roc_aucs))
        if mean_roc_auc > self.peak_roc_auc:
            self.peak_roc_auc = mean_roc_auc
            self.logger.log_metrics(
                {"peak_roc_auc": self.peak_roc_auc}, step=self.current_epoch
            )
            self.logger.log_metrics(
                {"peak_roc_auc_epoch": self.current_epoch}, step=self.current_epoch
            )

    def validation_step(self, batch, batch_idx):
        x, y, mask = batch

        if self.backbone:
            x = self.backbone.extract_features(x)

        y_pred = self.forward(x)
        loss = weighted_masked_bce_loss(
            y_pred,
            y,
            mask,
            pos_weight=self.loss_pos_weight,
            weighted=self.loss_weighted,
        )
        self.log("validation_loss", loss, prog_bar=True)

        mask = mask.cpu()
        y_true = y.cpu() * mask
        y_pred = y_pred.cpu() * mask
        y_true_flat = y_true[mask.bool()]
        y_pred_flat = y_pred[mask.bool()]

        if len(y_true_flat) > 0 and len(torch.unique(y_true_flat)) > 1:
            try:
                roc_auc = roc_auc_score(y_true_flat.numpy(), y_pred_flat.numpy())
                self.log("validation_roc_auc", roc_auc, prog_bar=True)
                self.roc_aucs.append(roc_auc)

            except ValueError:
                pass  # Skip if still only one class after masking

        return loss

    def training_step(self, batch, batch_idx):
        x, y, mask = batch

        if self.backbone:
            x = self.backbone.extract_features(x)

        predicted_y = self.forward(x)

        loss = weighted_masked_bce_loss(
            predicted_y,
            y,
            mask,
            pos_weight=self.loss_pos_weight,
            weighted=self.loss_weighted,
        )
        self.log("train_loss", loss, prog_bar=True)
        y_true_flat = y[mask.bool()]
        y_pred_flat = predicted_y[mask.bool()]

        # Compute ROC AUC for training monitoring
        y_true_flat = y[mask.bool()].cpu()  # Fixed: use y instead of predicted_y
        y_pred_flat = predicted_y[mask.bool()].cpu()

        if len(y_true_flat) > 0 and len(torch.unique(y_true_flat)) > 1:
            try:
                roc_auc = roc_auc_score(y_true_flat.detach().numpy(), y_pred_flat.detach().numpy())
                self.log("train_roc_auc", roc_auc, prog_bar=True)
            except ValueError:
                pass


        return loss

    # New methods for percentage-based threshold calculation
    def calibration_step(self, batch, batch_idx):
        x, y, mask = batch
        with torch.no_grad():
            y_pred = self.forward(x)

        for ypred, ytrue, masked in zip(y_pred, y, mask):
            bool_mask = masked.bool()
            score = multiclass_non_conformity_score(ypred, ytrue, mask=bool_mask)
            self.nonconformity_scores.append(score)

    def compute_thresholds(self):
        if not self.nonconformity_scores:
            raise ValueError("No nonconformity scores recorded. Run calibration first.")

        calib_scores = np.array(self.nonconformity_scores)
        print(calib_scores.shape)
        self.thresholds = np.nanquantile(calib_scores, 1 - self.alpha, axis=0)
        print(self.thresholds)
        return self.thresholds

    def test_step(self, batch, batch_idx):
        x, y, mask = batch
        y_pred = self.forward(x)
        bool_mask = mask.bool()
        y_pred = y_pred[bool_mask]
        y_pred = y_pred.cpu().numpy().reshape(-1, 1)
        y = y[bool_mask]
        y = y.cpu().numpy().reshape(-1, 1)

        # Store raw predictions for ROC AUC calculation
        self.y_pred_raw.extend(y_pred.flatten())
        self.y_true_raw.extend(y.flatten())

        for idx, sample_pred in enumerate(y_pred):
            y_true_sample = y[idx]
            (
                y_pred_conf,
                y_true_conf,
                ex_rejected,
                cls_rejected,
                y_pred_bt,
                y_true_bt,
            ) = apply_multiclass_thresholds(sample_pred, y_true_sample, self.thresholds)

            self.y_true.append(y_true_conf)
            self.y_pred_conf.append(y_pred_conf)
            self.ex_rejected += ex_rejected
            self.cls_rejected += cls_rejected
            self.y_true_bt.append(y_true_bt)
            self.y_pred_conf_bt.append(y_pred_bt)

    def on_test_epoch_end(self):
        # Calculate metrics at the end of testing
        metrics = calculate_metrics(
            self.y_true,
            self.y_pred_conf,
            self.y_true_bt,
            self.y_pred_conf_bt,
            self.ex_rejected,
            self.cls_rejected,
            self.alpha,
        )
        print(metrics)

        # # Log all metrics
        # for name, value in metrics.items():
        #     wandb.log({name: value, "percent_accept": self.percent_accept})

        # Save metrics to CSV file
        if self.results_file:
            # Convert metrics to DataFrame
            metrics_df = pd.DataFrame([metrics])
            metrics_df["alpha"] = self.alpha

            # Add metadata
            metrics_df["embedding_dim"] = self.embedding_dim
            metrics_df["num_classes"] = self.num_classes
            metrics_df["learning_rate"] = self.lr

            # Save to CSV
            metrics_df.to_csv(self.results_file, index=False)
            print(f"Results saved to {self.results_file}")

    def reset_params(self):
        # Reset evaluation parameters
        self.y_pred_conf = []
        self.y_true = []
        self.y_pred_conf_bt = []
        self.y_true_bt = []
        self.y_pred_raw = []
        self.y_true_raw = []
        self.ex_rejected = 0
        self.cls_rejected = 0
