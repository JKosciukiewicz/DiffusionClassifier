from typing import Optional, Type

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
from lightning_models.lightning_autoencoder import LightningAutoencoder
from lightning_models.lightning_cnn import LightningCNN
from models.clip_extractor import CLIPExtractor
from models.flow_matching_autoencoder import FlowMatchingAutoencoder

torch.manual_seed(42)


class IdentityBackbone(nn.Module):
    def extract_features(self, x):
        return x

    def forward(self, x):
        return x


class LightningFlowMatchingClassifier(BaseModel):
    """Multi-label classifier trained with Conditional (Rectified) Flow Matching.

    Instead of learning to denoise (as the diffusion variant does), the model
    learns a velocity field that transports Gaussian noise to the label vector
    along straight-line paths. At inference the ODE dx/dt = v(x, t, features) is
    integrated with an Euler solver from t=0 (noise) to t=1 (labels), optionally
    with Classifier-Free Guidance.
    """

    def __init__(
        self,
        num_classes: int,
        embedding_dim: int,
        lr: float,
        loss_fn: Optional[Type[nn.Module]] = None,
        masked_loss: bool = False,
        backbone_type: str = "none",
        backbone_ckpt_path: Optional[str] = None,
        clip_model_name: str = "ViT-B/32",
        dropout_rate: float = 0.3,
        cfg_scale: float = 2.0,
        num_sampling_steps: int = 20,
        num_blocks: int = 6,
        weight_decay: float = 1e-4,
        ternary_labels: bool = False,
        logit_mean: float = 0.0,
        logit_std: float = 1.0,
    ):
        super().__init__()

        self.num_classes = num_classes
        self.lr = lr
        self.weight_decay = weight_decay
        self.embedding_dim = embedding_dim
        self.cfg_scale = cfg_scale
        self.num_sampling_steps = num_sampling_steps
        self.masked_loss = masked_loss
        self.ternary_labels = ternary_labels
        self.logit_mean = logit_mean
        self.logit_std = logit_std

        if backbone_type == "cnn":
            if backbone_ckpt_path is None:
                raise ValueError(
                    "backbone_ckpt_path must be provided when backbone_type is 'cnn'"
                )
            self.backbone = LightningCNN.load_from_checkpoint(backbone_ckpt_path).eval()
        elif backbone_type == "autoencoder":
            if backbone_ckpt_path is None:
                raise ValueError(
                    "backbone_ckpt_path must be provided when backbone_type is 'autoencoder'"
                )
            self.backbone = LightningAutoencoder.load_from_checkpoint(
                backbone_ckpt_path
            ).eval()
        elif backbone_type == "clip":
            self.backbone = CLIPExtractor(model_name=clip_model_name).eval()
        elif backbone_type == "none":
            self.backbone = IdentityBackbone()
        else:
            raise ValueError(f"Unknown backbone type: {backbone_type}")

        self.model = FlowMatchingAutoencoder(
            feature_dim=self.embedding_dim,
            label_dim=self.num_classes,
            dropout_rate=dropout_rate,
            num_blocks=num_blocks,
        )

        self.loss_fn = loss_fn() if loss_fn is not None else nn.MSELoss()

        self.y_pred = []
        self.y_true = []
        self.roc_aucs = []

    def forward(
        self,
        features: torch.Tensor,
        x_t: torch.Tensor,
        t: torch.Tensor,
        force_uncond: bool = False,
    ) -> torch.Tensor:
        return self.model(features=features, x_t=x_t, t=t, force_uncond=force_uncond)

    def _sample_t(self, batch_size: int, device: torch.device) -> torch.Tensor:
        """SD3-style logit-normal timestep sampling: t = sigmoid(N(mean, std)).

        Concentrates training time on the middle of the path (away from the
        easy t->0 / t->1 endpoints), which typically improves flow matching
        over uniform sampling. Set logit_std large / use uniform to revert.
        """
        return torch.sigmoid(
            self.logit_mean + self.logit_std * torch.randn(batch_size, device=device)
        )

    @torch.no_grad()
    def predict_labels(
        self, features: torch.Tensor, num_steps: Optional[int] = None
    ) -> torch.Tensor:
        """Euler integration of the learned velocity ODE with CFG.

        Starts from pure noise at t=0 and integrates to t=1, where the state is
        the predicted label vector (in [-1, 1]), then rescaled to [0, 1].
        """
        if num_steps is None:
            num_steps = self.num_sampling_steps

        device = features.device
        batch_size = features.shape[0]

        # x0 ~ N(0, I)
        y = torch.randn((batch_size, self.num_classes), device=device)
        dt = 1.0 / num_steps

        for i in range(num_steps):
            t = torch.full((batch_size,), i * dt, device=device)

            cond_v = self.forward(features, y, t)
            if self.cfg_scale != 1.0:
                uncond_v = self.forward(features, y, t, force_uncond=True)
                v = uncond_v + self.cfg_scale * (cond_v - uncond_v)
            else:
                v = cond_v

            y = y + dt * v

        y_unscaled = (y + 1.0) / 2.0
        return torch.clamp(y_unscaled, 0.0, 1.0)

    def training_step(self, batch, batch_idx):
        x, y, mask = batch
        x = self.backbone.extract_features(x)
        x = torch.nn.functional.dropout(x, p=0.1, training=self.training)

        # Endpoints of the flow path. Ternary labels are already in [-1, 1];
        # binary [0, 1] labels need scaling.
        x1 = y if self.ternary_labels else y * 2.0 - 1.0
        x0 = torch.randn_like(x1)

        # Sample a time per element on the path, interpolate, and form the
        # straight-line (constant) target velocity.
        t = self._sample_t(y.shape[0], y.device)
        t_exp = t.view(-1, 1)
        x_t = (1.0 - t_exp) * x0 + t_exp * x1
        target_v = x1 - x0

        prediction = self.forward(x, x_t, t)

        per_elem_loss = (prediction - target_v) ** 2
        if self.masked_loss:
            per_sample_loss = (per_elem_loss * mask).sum(dim=-1) / (
                mask.sum(dim=-1) + 1e-8
            )
        else:
            per_sample_loss = per_elem_loss.mean(dim=-1)
        loss = per_sample_loss.mean()
        self.log("train/loss", loss, prog_bar=True)

        with torch.no_grad():
            # Recover the clean label endpoint: x1 = x_t + (1 - t) * v.
            # Measured on the high-noise half (t < 0.5), where the model must
            # rely on the conditioning rather than the partially-clean state.
            high_noise_mask = t < 0.5
            if high_noise_mask.sum() > 0:
                x1_hat = x_t + (1.0 - t_exp) * prediction
                x1_hat_unscaled = torch.clamp((x1_hat + 1.0) / 2.0, 0.0, 1.0)

                y_true_flat = y[high_noise_mask].cpu()
                if self.ternary_labels:
                    y_true_flat = (y_true_flat > 0).float()
                y_pred_flat = x1_hat_unscaled[high_noise_mask].cpu()

                if len(torch.unique(y_true_flat)) > 1:
                    try:
                        roc_auc = roc_auc_score(
                            y_true_flat.numpy(), y_pred_flat.numpy()
                        )
                        self.log("train/roc_auc", roc_auc, prog_bar=True)
                    except ValueError:
                        pass

        return loss

    def validation_step(self, batch, batch_idx):
        x, y, mask = batch
        x = self.backbone.extract_features(x)

        x1 = y if self.ternary_labels else y * 2.0 - 1.0
        x0 = torch.randn_like(x1)
        t = self._sample_t(y.shape[0], y.device)
        t_exp = t.view(-1, 1)
        x_t = (1.0 - t_exp) * x0 + t_exp * x1
        target_v = x1 - x0
        prediction = self.forward(x, x_t, t)

        if self.masked_loss:
            loss = ((prediction - target_v) ** 2 * mask).sum() / (mask.sum() + 1e-8)
        else:
            loss = self.loss_fn(prediction, target_v)
        self.log("val/loss", loss, prog_bar=True)

        predicted_y = self.predict_labels(x, num_steps=self.num_sampling_steps)

        mask_bool = mask.cpu().bool()
        y_true_flat = y.cpu()[mask_bool]
        y_pred_flat = predicted_y.cpu()[mask_bool]

        if self.ternary_labels:
            y_true_flat = (y_true_flat > 0).float()

        if len(y_true_flat) > 0 and len(torch.unique(y_true_flat)) > 1:
            try:
                roc_auc = roc_auc_score(y_true_flat.numpy(), y_pred_flat.numpy())
                self.log("val/roc_auc", roc_auc, prog_bar=True)
                self.roc_aucs.append(roc_auc)
            except ValueError:
                pass

        return loss

    def test_step(self, batch, batch_idx):
        x, y, mask = batch
        x = self.backbone.extract_features(x)

        # Higher-fidelity sampling for the final test metrics.
        predicted_y = self.predict_labels(x, num_steps=max(self.num_sampling_steps, 50))
        mask_bool = mask.bool()

        y_pred_flat = predicted_y[mask_bool].cpu()
        y_true_flat = y[mask_bool].cpu()

        if self.ternary_labels:
            y_true_flat = (y_true_flat > 0).float()

        self.y_true.append(y_true_flat)
        self.y_pred.append(y_pred_flat)

    def on_test_start(self):
        self.reset_params()

    def on_validation_epoch_start(self) -> None:
        self.roc_aucs = []

    def on_test_epoch_end(self):
        y_true = torch.cat(self.y_true).numpy()
        y_pred = torch.cat(self.y_pred).numpy()

        metrics = {}
        if len(y_true) > 0 and len(np.unique(y_true)) > 1:
            try:
                y_pred_binary = (y_pred > 0.5).astype(np.float32)

                metrics["test/roc_auc"] = roc_auc_score(y_true, y_pred)
                metrics["test/accuracy"] = accuracy_score(y_true, y_pred_binary)
                metrics["test/precision"] = precision_score(
                    y_true, y_pred_binary, zero_division=0
                )
                metrics["test/recall"] = recall_score(
                    y_true, y_pred_binary, zero_division=0
                )
                metrics["test/f1_score"] = f1_score(
                    y_true, y_pred_binary, zero_division=0
                )

            except Exception as e:
                print(f"Error calculating metrics: {e}")

        print(f"Test Metrics: {metrics}")
        self.log_dict(metrics)

    def configure_optimizers(self):
        return torch.optim.AdamW(
            self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )

    def reset_params(self):
        self.y_pred = []
        self.y_true = []
