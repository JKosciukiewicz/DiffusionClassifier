from typing import Optional, Type

import numpy as np
import torch
import torch.nn as nn
from diffusers import DDPMScheduler
from sklearn.metrics import roc_auc_score

from lightning_models.base_model import BaseModel
from lightning_models.lightning_cnn import LightningCNN
from models.diffusion_classifier import DiffusionClassifier
from utils.conformal_prediction import (
    apply_multiclass_thresholds,
)
from utils.evaluate_conformal_model import calculate_metrics

torch.manual_seed(42)


class LightningDiffusionClassifier(BaseModel):
    def __init__(
        self,
        num_classes: int,
        embedding_dim: int,
        lr: float,
        alpha: float,
        residual: bool = True,
        activation_fn: Type[nn.Module] = nn.GELU,
        loss_fn: Type[nn.Module] = nn.BCELoss,
        masked_loss: bool = False,
        cnn_ckpt_path: Optional[str] = None,
        dropout_rate: float = 0.3,
        num_timesteps: int = 1000,
    ):
        super().__init__()

        # Core parameters
        self.alpha = alpha
        self.num_classes = num_classes
        self.lr = lr
        self.embedding_dim = embedding_dim

        self.num_timesteps = num_timesteps

        self.nonconformity_scores = []
        self.thresholds = None
        self.masked_loss = masked_loss

        self.cnn = LightningCNN.load_from_checkpoint(cnn_ckpt_path).eval()

        self.model = DiffusionClassifier(
            num_classes=num_classes,
            embedding_dim=embedding_dim,
            dropout_rate=dropout_rate,
            residual=residual,
            activation_fn=activation_fn,
        )

        self.loss_fn = loss_fn()

        self.noise_scheduler = DDPMScheduler(
            num_train_timesteps=self.num_timesteps, beta_schedule="squaredcos_cap_v2"
        )

        self.y_pred_conf = []
        self.y_true = []
        self.y_pred_conf_bt = []
        self.y_true_bt = []
        self.ex_rejected = 0
        self.cls_rejected = 0

        # ROC AUC tracking
        self.peak_roc_auc = 0
        self.roc_aucs = []

    def forward(
        self,
        features: torch.Tensor,
        labels: torch.Tensor,
        inference=False,
    ) -> torch.Tensor:
        batch_size = labels.shape[0]
        device = labels.device
        noise = torch.randn_like(labels)

        if inference:
            # Inference -> label = gaussian noise, timestep = max possible timestep
            timesteps = torch.full(
                (batch_size,),
                self.num_timesteps,
                device=device,
                dtype=torch.long,
            )
            noisy_labels = torch.rand_like(labels).to(features.device)
        else:
            # Training -> label with random noise added based on the timestep, timestep selected randomly
            timesteps = torch.randint(
                0,
                self.num_timesteps,
                (batch_size,),
                device=device,
            ).long()
            noisy_labels = self.noise_scheduler.add_noise(labels, noise, timesteps)

        # Predict added noise using the model
        predicted_noise = self.model(
            features=features,
            noisy_labels=noisy_labels,
            timesteps=timesteps.unsqueeze(1),
        )

        # Reconstruct clean labels via denoising
        return predicted_noise

    def training_step(self, batch, batch_idx):
        """Training step with corrected BCE loss computation."""
        x, y, mask = batch

        x = self.cnn.extract_features(x)
        predicted_y = self.forward(x, y)

        if self.masked_loss:
            loss = self.loss_fn(predicted_y, y, mask)
        else:
            loss = self.loss_fn(predicted_y, y)

        self.log("train_loss", loss, prog_bar=True)

        # Compute ROC AUC for training monitoring
        y_true_flat = y[mask.bool()].cpu()  # Fixed: use y instead of predicted_y
        y_pred_flat = predicted_y[mask.bool()].cpu()

        if len(y_true_flat) > 0 and len(torch.unique(y_true_flat)) > 1:
            try:
                roc_auc = roc_auc_score(
                    y_true_flat.detach().numpy(), y_pred_flat.detach().numpy()
                )
                self.log("train_roc_auc", roc_auc, prog_bar=True)
            except ValueError:
                pass

        return loss

    def validation_step(self, batch, batch_idx):
        """Validation step with multi-timestep sampling for robust evaluation."""
        x, y, mask = batch
        x = self.cnn.extract_features(x)

        predicted_y = self.forward(x, y, inference=True)

        if self.masked_loss:
            loss = self.loss_fn(predicted_y, y, mask)
        else:
            loss = self.loss_fn(predicted_y, y)

        self.log("validation_loss", loss, prog_bar=True)

        # Move tensors to CPU for metric computation
        mask = mask.cpu()
        y_true = y.cpu()
        y_pred = predicted_y.cpu()
        y_true_flat = y_true[mask.bool()]
        y_pred_flat = y_pred[mask.bool()]

        if len(y_true_flat) > 0 and len(torch.unique(y_true_flat)) > 1:
            try:
                roc_auc = roc_auc_score(y_true_flat.numpy(), y_pred_flat.numpy())
                self.log("validation_roc_auc", roc_auc, prog_bar=True)
                self.roc_aucs.append(roc_auc)
            except ValueError:
                pass

        return loss

    def test_step(self, batch, batch_idx):
        """Test step with multi-timestep sampling for final evaluation."""
        x, y, mask = batch
        x = self.cnn.extract_features(x)

        y_pred = self.forward(x, y, inference=True)
        bool_mask = mask.bool()

        y_pred = y_pred[bool_mask].cpu()
        y = y[bool_mask].cpu()

        (
            y_pred_conf,
            y_true_conf,
            ex_rejected,
            cls_rejected,
            y_pred_bt,
            y_true_bt,
        ) = apply_multiclass_thresholds(y_pred, y, self.thresholds)

        self.y_true.append(y_true_conf)
        self.y_pred_conf.append(y_pred_conf)
        self.ex_rejected += ex_rejected
        self.cls_rejected += cls_rejected
        self.y_true_bt.append(y_true_bt)
        self.y_pred_conf_bt.append(y_pred_bt)

    def on_fit_end(self):
        """Called when fit ends - automatically set thresholds based on alpha."""
        self._set_thresholds_from_alpha()

    def on_test_start(self):
        """Called when test starts - ensure thresholds are set."""
        if self.thresholds is None:
            self._set_thresholds_from_alpha()

    def on_validation_start(self) -> None:
        """Initialize validation tracking."""
        self.roc_aucs = []
        self._set_thresholds_from_alpha()

    def on_validation_end(self):
        """Track peak validation performance."""
        mean_roc_auc = np.mean(np.array(self.roc_aucs)) if self.roc_aucs else 0
        if mean_roc_auc > self.peak_roc_auc:
            self.peak_roc_auc = mean_roc_auc
            self.logger.log_metrics(
                {"peak_roc_auc": self.peak_roc_auc}, step=self.current_epoch
            )
            self.logger.log_metrics(
                {"peak_roc_auc_epoch": self.current_epoch}, step=self.current_epoch
            )

    def on_test_epoch_end(self):
        """Compute and save final test metrics."""
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

    def _set_thresholds_from_alpha(self):
        """Automatically set thresholds based on alpha parameter."""
        self.thresholds = np.ones(self.num_classes) * (1 - self.alpha)

    def set_thresholds(self, thresholds):
        """Manual override for thresholds if needed."""
        self.thresholds = thresholds

    def configure_optimizers(self):
        """Configure Adam optimizer for training."""
        return torch.optim.Adam(self.model.parameters(), lr=self.lr)

    def reset_params(self):
        """Reset evaluation parameters for new test run."""
        self.y_pred_conf = []
        self.y_true = []
        self.y_pred_conf_bt = []
        self.y_true_bt = []
        self.ex_rejected = 0
        self.cls_rejected = 0
