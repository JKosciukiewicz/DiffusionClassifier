import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from lightning_models.base_model import BaseModel
from models.mlp import MLPClassifier


class LightningMLPClassifier(BaseModel):
    def __init__(self, num_classes: int, embedding_dim: int, lr: float = 1e-4, masked_loss: bool = False):
        super().__init__()
        self.lr = lr
        self.masked_loss = masked_loss
        self.model = MLPClassifier(num_classes=num_classes, embedding_dim=embedding_dim)
        self.loss_fn = nn.BCELoss()
        self.y_pred = []
        self.y_true = []

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y, mask = batch
        pred = self.forward(x)
        if self.masked_loss:
            loss = ((pred - y) ** 2 * mask).sum() / (mask.sum() + 1e-8)
        else:
            loss = self.loss_fn(pred, y)
        self.log("train/loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y, mask = batch
        pred = self.forward(x)
        if self.masked_loss:
            loss = ((pred - y) ** 2 * mask).sum() / (mask.sum() + 1e-8)
        else:
            loss = self.loss_fn(pred, y)
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
        return loss

    def test_step(self, batch, batch_idx):
        x, y, mask = batch
        pred = self.forward(x)
        mask_bool = mask.bool()
        self.y_pred.append(pred[mask_bool].cpu())
        self.y_true.append(y[mask_bool].cpu())

    def on_test_start(self):
        self.y_pred = []
        self.y_true = []

    def on_test_epoch_end(self):
        y_true = torch.cat(self.y_true).numpy()
        y_pred = torch.cat(self.y_pred).detach().numpy()

        if len(y_true) > 0 and len(np.unique(y_true)) > 1:
            try:
                y_pred_binary = (y_pred > 0.5).astype(np.float32)
                metrics = {
                    "test/roc_auc": roc_auc_score(y_true, y_pred),
                    "test/accuracy": accuracy_score(y_true, y_pred_binary),
                    "test/precision": precision_score(y_true, y_pred_binary, zero_division=0),
                    "test/recall": recall_score(y_true, y_pred_binary, zero_division=0),
                    "test/f1_score": f1_score(y_true, y_pred_binary, zero_division=0),
                }
                print(f"Test Metrics: {metrics}")
                self.log_dict(metrics)
            except Exception as e:
                print(f"Error calculating metrics: {e}")

    def configure_optimizers(self):
        return torch.optim.Adam(self.model.parameters(), lr=self.lr)
