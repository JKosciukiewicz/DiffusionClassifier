from typing import Optional, Type

import numpy as np
import torch
import torch.nn as nn
from diffusers import DDIMScheduler  # Swapped to DDIM
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
from models.diffusion_autoencoder import DiffusionAutoencoder

torch.manual_seed(42)


class IdentityBackbone(nn.Module):
    def extract_features(self, x):
        return x

    def forward(self, x):
        return x


class LightningDiffusionClassifier(BaseModel):
    def __init__(
        self,
        num_classes: int,
        embedding_dim: int,
        lr: float,
        alpha: float,
        residual: bool = True,
        activation_fn: Type[nn.Module] = nn.GELU,
        loss_fn: Optional[Type[nn.Module]] = None,
        objective: str = "noise",
        masked_loss: bool = False,
        backbone_type: str = "none",
        backbone_ckpt_path: Optional[str] = None,
        clip_model_name: str = "ViT-B/32",
        dropout_rate: float = 0.3,
        num_timesteps: int = 1000,
        model_channels: int = 768,
        cfg_scale: float = 2.0,
        ternary_labels: bool = False,
    ):
        super().__init__()

        # Core parameters
        self.alpha = alpha
        self.num_classes = num_classes
        self.lr = lr
        self.embedding_dim = embedding_dim
        self.model_channels = model_channels
        self.objective = objective
        self.num_timesteps = num_timesteps
        self.cfg_scale = cfg_scale
        self.masked_loss = masked_loss
        self.ternary_labels = ternary_labels

        if backbone_type == "cnn":
            if backbone_ckpt_path is None:
                raise ValueError(
                    "cnn_ckpt_path must be provided when backbone_type is 'cnn'"
                )
            self.backbone = LightningCNN.load_from_checkpoint(backbone_ckpt_path).eval()
        elif backbone_type == "autoencoder":
            if backbone_ckpt_path is None:
                raise ValueError(
                    "autoencoder_ckpt_path must be provided when backbone_type is 'autoencoder'"
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

        self.model = DiffusionAutoencoder(
            feature_dim=self.embedding_dim,
            label_dim=self.num_classes,
            dropout_rate=dropout_rate,
            use_sigmoid=(objective == "labels"),
        )

        self.loss_fn = loss_fn() if loss_fn is not None else nn.MSELoss()

        # FIX: Switched to DDIM to allow safe step-skipping during inference
        self.noise_scheduler = DDIMScheduler(
            num_train_timesteps=self.num_timesteps, beta_schedule="squaredcos_cap_v2"
        )

        self.y_pred = []
        self.y_true = []
        self.roc_aucs = []

    def forward(
        self,
        features: torch.Tensor,
        noisy_labels: torch.Tensor,
        timesteps: torch.Tensor,
    ) -> torch.Tensor:
        return self.model(
            features=features,
            noisy_labels=noisy_labels,
            timesteps=timesteps.unsqueeze(1) if timesteps.ndim == 1 else timesteps,
        )

    def predict_labels(
        self, features: torch.Tensor, num_steps: int = 50
    ) -> torch.Tensor:
        """Iterative DDIM generation loop with Classifier-Free Guidance.

        Safely accepts lower step counts (e.g., 20 or 50) for massive speedups.
        """
        device = features.device
        batch_size = features.shape[0]

        if self.objective == "labels":
            timesteps = torch.full(
                (batch_size,), self.num_timesteps - 1, device=device, dtype=torch.long
            )
            noisy_labels = torch.randn((batch_size, self.num_classes), device=device)
            pred = self.forward(features, noisy_labels, timesteps)
            return torch.clamp((pred + 1.0) / 2.0, 0.0, 1.0)

        # Initialize from pure random noise
        y_t = torch.randn((batch_size, self.num_classes), device=device)

        # DDIM modifies the internal timesteps array to safely skip intervals
        self.noise_scheduler.set_timesteps(num_steps, device=device)

        # Create a null feature vector for the unconditional CFG baseline path
        uncond_features = torch.zeros_like(features)

        for t in self.noise_scheduler.timesteps:
            t_batch = t.expand(batch_size)

            # --- Classifier-Free Guidance Dual-Forward Pass ---
            cond_pred = self.forward(features, y_t, t_batch)
            uncond_pred = self.forward(uncond_features, y_t, t_batch)

            # Extrapolate away from the unconditioned state
            predicted_noise = uncond_pred + self.cfg_scale * (cond_pred - uncond_pred)

            # DDIM handles the adjusted variance math for step skipping here
            y_t = self.noise_scheduler.step(predicted_noise, t, y_t).prev_sample

        y_unscaled = (y_t + 1.0) / 2.0
        return torch.clamp(y_unscaled, 0.0, 1.0)

    def training_step(self, batch, batch_idx):
        x, y, mask = batch
        x = self.backbone.extract_features(x)
        x = torch.nn.functional.dropout(x, p=0.1, training=self.training)

        # Ternary labels are already in [-1, 1]; binary [0,1] labels need scaling
        y_scaled = y if self.ternary_labels else y * 2.0 - 1.0
        noise = torch.randn_like(y_scaled)

        # Curriculum schedule: first 20% of training is fully uniform (warmup),
        # then a cosine ramp drives both timestep skew and label masking toward
        # their final values. Cosine gives a slow start and slow finish with most
        # change happening in the middle, avoiding abrupt difficulty jumps.
        progress = min(self.current_epoch / max(self.trainer.max_epochs - 1, 1), 1.0)
        # Delay onset: clamp the first 20% to 0, then re-scale [0.2, 1.0] → [0.0, 1.0]
        effective_progress = max(0.0, (progress - 0.1) / 0.9)
        # Cosine ramp: 0.0 → 1.0 with slow start and slow finish
        cosine_ramp = float((1 - np.cos(np.pi * effective_progress)) / 2)

        # [2] Timestep exponent: 1.0 (uniform) → 0.5 (skewed toward high-t).
        # u^exponent with exponent < 1 concentrates mass toward 1 (high timesteps).
        exponent = 1.0 - 0.2 * cosine_ramp
        u = torch.rand(y.shape[0], device=y.device)
        timesteps = (
            (u.pow(exponent) * self.num_timesteps)
            .long()
            .clamp(0, self.num_timesteps - 1)
        )

        noisy_y = self.noise_scheduler.add_noise(y_scaled, noise, timesteps)

        # [3] Noise floor: add small fixed noise so labels are never fully clean.
        noisy_y = noisy_y + 0.1 * torch.randn_like(noisy_y)

        # [1] Label masking probability: 0.0 → 0.3, same cosine ramp.
        # Early training sees real noisy labels; late training simulates inference
        # by replacing 30% of noisy_y with pure noise.
        label_mask_p = 0.3 * cosine_ramp
        label_drop = (
            (torch.rand(y.shape[0], device=y.device) < label_mask_p)
            .float()
            .unsqueeze(1)
        )
        noisy_y = (1.0 - label_drop) * noisy_y + label_drop * torch.randn_like(noisy_y)

        self.log("train/curriculum_progress", progress, prog_bar=False)
        self.log("train/label_mask_p", label_mask_p, prog_bar=False)
        self.log("train/timestep_exponent", exponent, prog_bar=False)

        prediction = self.forward(x, noisy_y, timesteps)

        target = y_scaled if self.objective == "labels" else noise

        # Per-sample MSE (B,)
        per_elem_loss = (prediction - target) ** 2
        if self.masked_loss:
            per_sample_loss = (per_elem_loss * mask).sum(dim=-1) / (
                mask.sum(dim=-1) + 1e-8
            )
        else:
            per_sample_loss = per_elem_loss.mean(dim=-1)

        # [4] No Min-SNR weighting: Min-SNR downweights high-t gradients, but
        # high-t is exactly where the model must learn to use conditioning rather
        # than the noisy label. Plain mean keeps all timesteps equally weighted.
        loss = per_sample_loss.mean()
        self.log("train/loss", loss, prog_bar=True)

        with torch.no_grad():
            high_noise_mask = timesteps > (self.num_timesteps // 4)
            if high_noise_mask.sum() > 0 and self.objective == "noise":
                alpha_prod_t = (
                    self.noise_scheduler.alphas_cumprod[timesteps]
                    .to(y.device)
                    .view(-1, 1)
                )
                pred_y0 = (
                    noisy_y - torch.sqrt(1 - alpha_prod_t) * prediction
                ) / torch.sqrt(alpha_prod_t)
                pred_y0_unscaled = torch.clamp((pred_y0 + 1.0) / 2.0, 0.0, 1.0)

                y_true_flat = y[high_noise_mask].cpu()
                if self.ternary_labels:
                    y_true_flat = (y_true_flat > 0).float()
                y_pred_flat = pred_y0_unscaled[high_noise_mask].cpu()

                if len(torch.unique(y_true_flat)) > 1:
                    roc_auc = roc_auc_score(y_true_flat.numpy(), y_pred_flat.numpy())
                    self.log("train/roc_auc", roc_auc, prog_bar=True)

        return loss

    def validation_step(self, batch, batch_idx):
        x, y, mask = batch
        x = self.backbone.extract_features(x)

        y_scaled = y if self.ternary_labels else y * 2.0 - 1.0
        noise = torch.randn_like(y_scaled)
        timesteps = torch.randint(
            0, self.num_timesteps, (y.shape[0],), device=y.device
        ).long()
        noisy_y = self.noise_scheduler.add_noise(y_scaled, noise, timesteps)
        prediction = self.forward(x, noisy_y, timesteps)

        target = y_scaled if self.objective == "labels" else noise
        if self.masked_loss:
            loss = ((prediction - target) ** 2 * mask).sum() / (mask.sum() + 1e-8)
        else:
            loss = self.loss_fn(prediction, target)
        self.log("val/loss", loss, prog_bar=True)

        # FIX: Blazing fast evaluation loop using only 20 steps (Safe under DDIM)
        predicted_y = self.predict_labels(x, num_steps=20)

        mask_cpu = mask.cpu()
        y_true = y.cpu()
        y_pred = predicted_y.cpu()
        mask_bool = mask_cpu.bool()
        y_true_flat = y_true[mask_bool]
        y_pred_flat = y_pred[mask_bool]

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

        # FIX: Test loop balanced at 50 steps for higher fidelity metrics
        predicted_y = self.predict_labels(x, num_steps=50)
        mask_bool = mask.bool()

        y_pred_flat = predicted_y[mask_bool].cpu()
        y_true_flat = y[mask_bool].cpu()

        if self.ternary_labels:
            y_true_flat = (y_true_flat > 0).float()

        self.y_true.append(y_true_flat)
        self.y_pred.append(y_pred_flat)

    def on_fit_end(self):
        pass

    def on_test_start(self):
        self.reset_params()

    def on_validation_epoch_start(self) -> None:
        self.roc_aucs = []

    def on_validation_epoch_end(self):
        pass

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
        return torch.optim.Adam(self.model.parameters(), lr=self.lr)

    def reset_params(self):
        self.y_pred = []
        self.y_true = []
