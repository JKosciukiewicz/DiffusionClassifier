from typing import List, Optional

import matplotlib
matplotlib.use("Agg")
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
from torchcfm.conditional_flow_matching import ConditionalFlowMatcher

from lightning_models.base_model import BaseModel
from models.card_velocity_net import CARDVelocityNet


class LightningCARDClassifier(BaseModel):
    """CARD-style multi-label classifier via Flow Matching in label space.

    Instead of the generative Bayes route p(x|y) -> p(y|x), this models p(y|x)
    directly: a velocity field v(y_t, t, x) flows Gaussian noise -> label vector
    conditioned on the feature vector x.

    At inference, running the ODE from z~N(0,I^K) to z_1 and applying sigmoid
    gives calibrated per-class probabilities directly, with no Bayes inversion.
    Multiple ODE samples are averaged to reduce stochasticity.
    """

    def __init__(
        self,
        num_classes: int,
        embedding_dim: int,
        lr: float = 1e-4,
        num_blocks: int = 6,
        hidden_dim: int = 256,
        cond_dim: int = 256,
        dropout: float = 0.0,
        num_integration_steps: int = 20,
        test_integration_steps: int = 50,
        num_pred_samples: int = 5,
        normalization_layer: bool = True,
        weight_decay: float = 1e-4,
        moa_columns: Optional[List[str]] = None,
        auc_thresholds: List[float] = [0.6, 0.7, 0.8, 0.9],
        log_per_fold_details: bool = True,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.lr = lr
        self.weight_decay = weight_decay
        self.num_classes = num_classes
        self.num_integration_steps = num_integration_steps
        self.test_integration_steps = test_integration_steps
        self.num_pred_samples = num_pred_samples
        self.moa_columns = moa_columns or [str(i) for i in range(num_classes)]
        self.auc_thresholds = auc_thresholds
        self.log_per_fold_details = log_per_fold_details

        if normalization_layer:
            self.feature_norm = nn.LayerNorm(embedding_dim, elementwise_affine=False)
        else:
            self.feature_norm = nn.Identity()

        self.net = CARDVelocityNet(
            feature_dim=embedding_dim,
            num_classes=num_classes,
            hidden_dim=hidden_dim,
            cond_dim=cond_dim,
            num_blocks=num_blocks,
            dropout=dropout,
        )
        self.fm = ConditionalFlowMatcher(sigma=0.0)

        self.y_pred: List[torch.Tensor] = []
        self.y_true: List[torch.Tensor] = []
        self.y_mask: List[torch.Tensor] = []
        self.per_class_auc: List[float] = []

    # ------------------------------------------------------------------ training
    def _card_loss(
        self, x: torch.Tensor, y: torch.Tensor, mask: torch.Tensor
    ) -> torch.Tensor:
        y0 = torch.randn_like(y)
        t = torch.rand(x.shape[0], device=x.device)
        t, yt, ut = self.fm.sample_location_and_conditional_flow(y0, y, t=t)
        v = self.net(yt, t, x)
        # Only penalise on known label entries.
        per_entry = (v - ut) ** 2
        return (per_entry * mask).sum() / (mask.sum() + 1e-8)

    def training_step(self, batch, batch_idx):
        x, y, mask = batch
        x = self.feature_norm(x.float())
        loss = self._card_loss(x, y.float(), mask.float())
        self.log("train/loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y, mask = batch
        x = self.feature_norm(x.float())
        loss = self._card_loss(x, y.float(), mask.float())
        self.log("val/loss", loss, prog_bar=True)

        y_pred = self._predict(x, num_steps=self.num_integration_steps)
        mask_bool = mask.bool()
        y_pred_flat = y_pred[mask_bool].cpu()
        y_true_flat = y[mask_bool].cpu()

        if len(y_true_flat) > 0 and len(torch.unique(y_true_flat)) > 1:
            try:
                roc_auc = roc_auc_score(y_true_flat.numpy(), y_pred_flat.numpy())
                self.log("val/roc_auc", roc_auc, prog_bar=True)
            except ValueError:
                pass
        return loss

    # ------------------------------------------------------------------ inference
    @torch.no_grad()
    def _predict(self, x: torch.Tensor, num_steps: int) -> torch.Tensor:
        """Euler integration: noise -> labels, averaged over num_pred_samples runs."""
        B, K = x.shape[0], self.num_classes
        dt = 1.0 / num_steps
        preds = []
        for _ in range(self.num_pred_samples):
            z = torch.randn(B, K, device=x.device)
            for i in range(num_steps):
                t = torch.full((B,), i * dt, device=x.device)
                v = self.net(z, t, x)
                z = z + v * dt
            preds.append(torch.sigmoid(z))
        return torch.stack(preds).mean(0)  # (B, K)

    # ------------------------------------------------------------------ test
    def test_step(self, batch, batch_idx):
        x, y, mask = batch
        x = self.feature_norm(x.float())
        y_pred = self._predict(x, num_steps=self.test_integration_steps)
        self.y_pred.append(y_pred.cpu())
        self.y_true.append(y.cpu())
        self.y_mask.append(mask.bool().cpu())

    def on_test_start(self):
        self.y_pred = []
        self.y_true = []
        self.y_mask = []

    def on_test_epoch_end(self):
        y_pred = torch.cat(self.y_pred).numpy()          # (N, K)
        y_true = torch.cat(self.y_true).numpy()          # (N, K)
        y_mask = torch.cat(self.y_mask).numpy().astype(bool)

        K = y_true.shape[1]

        per_class_auc = []
        for k in range(K):
            m = y_mask[:, k]
            if m.sum() > 0 and len(np.unique(y_true[m, k])) > 1:
                per_class_auc.append(roc_auc_score(y_true[m, k], y_pred[m, k]))
            else:
                per_class_auc.append(np.nan)
        self.per_class_auc = per_class_auc

        y_true_flat = y_true[y_mask]
        y_pred_flat = y_pred[y_mask]

        metrics = {}
        if len(y_true_flat) > 0 and len(np.unique(y_true_flat)) > 1:
            try:
                y_pred_binary = (y_pred_flat > 0.5).astype(np.float32)
                metrics["test/roc_auc"] = roc_auc_score(y_true_flat, y_pred_flat)
                metrics["test/accuracy"] = accuracy_score(y_true_flat, y_pred_binary)
                metrics["test/precision"] = precision_score(y_true_flat, y_pred_binary, zero_division=0)
                metrics["test/recall"] = recall_score(y_true_flat, y_pred_binary, zero_division=0)
                metrics["test/f1_score"] = f1_score(y_true_flat, y_pred_binary, zero_division=0)
            except Exception as e:
                print(f"Error calculating metrics: {e}")

        for k, auc in enumerate(per_class_auc):
            if not np.isnan(auc):
                metrics[f"test/roc_auc_{self.moa_columns[k]}"] = auc

        if self.log_per_fold_details:
            valid_aucs = [a for a in per_class_auc if not np.isnan(a)]
            for t in self.auc_thresholds:
                metrics[f"test/n_classes_auc_above_{t}"] = sum(1 for a in valid_aucs if a >= t)

        print(f"Test Metrics: {metrics}")
        self.log_dict(metrics)

        if self.log_per_fold_details and self.logger is not None and hasattr(self.logger, "experiment"):
            self._log_score_distribution(y_true, y_pred, y_mask)

    def _log_score_distribution(self, y_true, y_pred, y_mask):
        K = y_true.shape[1]
        short_names = [c.replace("moa_", "") for c in self.moa_columns]
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
        return torch.optim.AdamW(self.net.parameters(), lr=self.lr, weight_decay=self.weight_decay)
