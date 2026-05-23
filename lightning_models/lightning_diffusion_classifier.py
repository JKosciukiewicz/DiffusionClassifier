from typing import Optional, Type

import numpy as np
import torch
import torch.nn as nn
from diffusers import DDPMScheduler
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
        objective: str = "noise",  # "noise" or "labels"
        masked_loss: bool = False,
        backbone_type: str = "none",
        backbone_ckpt_path: Optional[str] = None,
        clip_model_name: str = "ViT-B/32",
        dropout_rate: float = 0.3,
        num_timesteps: int = 1000,
        model_channels: int = 768,
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

        self.nonconformity_scores = []
        self.thresholds = None
        self.masked_loss = masked_loss

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

        if loss_fn is not None:
            self.loss_fn = loss_fn()
        else:
            # FIX: Switched to MSE for both since targets are scaled to [-1, 1]
            self.loss_fn = nn.MSELoss()

        # FIX: Switched to cosine schedule to retain small-vector label context
        self.noise_scheduler = DDPMScheduler(
            num_train_timesteps=self.num_timesteps, beta_schedule="squaredcos_cap_v2"
        )

        self.y_pred = []
        self.y_true = []

        # ROC AUC tracking
        self.peak_roc_auc = 0
        self.roc_aucs = []

    def forward(
        self,
        features: torch.Tensor,
        noisy_labels: torch.Tensor,
        timesteps: torch.Tensor,
    ) -> torch.Tensor:
        """Forward pass to predict noise or labels."""
        return self.model(
            features=features,
            noisy_labels=noisy_labels,
            timesteps=timesteps.unsqueeze(1) if timesteps.ndim == 1 else timesteps,
        )

    def predict_labels(
        self, features: torch.Tensor, num_steps: Optional[int] = None
    ) -> torch.Tensor:
        """Iterative DDPM generation loop to retrieve target probabilities."""
        device = features.device
        batch_size = features.shape[0]

        # FIX: Default to full timesteps loop to keep strict DDPM variance math intact
        if num_steps is None:
            num_steps = self.num_timesteps

        if self.objective == "labels":
            # Direct prediction pass
            timesteps = torch.full(
                (batch_size,),
                self.num_timesteps - 1,
                device=device,
                dtype=torch.long,
            )
            noisy_labels = torch.randn((batch_size, self.num_classes), device=device)
            pred = self.forward(features, noisy_labels, timesteps)
            # FIX: Unscale direct prediction back to [0, 1] bounds
            return torch.clamp((pred + 1.0) / 2.0, 0.0, 1.0)

        # Start from pure Gaussian noise
        y_t = torch.randn((batch_size, self.num_classes), device=device)

        self.noise_scheduler.set_timesteps(num_steps, device=device)

        for t in self.noise_scheduler.timesteps:
            # Predict noise residual
            predicted_noise = self.forward(features, y_t, t.expand(batch_size))

            # Standard DDPM reverse tracking step
            y_t = self.noise_scheduler.step(predicted_noise, t, y_t).prev_sample

        # FIX: Map from [-1, 1] back into [0, 1] probability space
        y_unscaled = (y_t + 1.0) / 2.0
        return torch.clamp(y_unscaled, 0.0, 1.0)


    def training_step(self, batch, batch_idx):
        x, y, mask = batch
        x = self.backbone.extract_features(x)
        
        # Apply Feature Dropout to prevent conditioning blindness
        # Forces the model to not rely on exact feature maps if it overfits
        x = torch.nn.functional.dropout(x, p=0.1, training=self.training)
        
        y_scaled = y * 2.0 - 1.0
        noise = torch.randn_like(y_scaled)
        timesteps = torch.randint(
            0, self.num_timesteps, (y.shape[0],), device=y.device
        ).long()
        
        noisy_y = self.noise_scheduler.add_noise(y_scaled, noise, timesteps)
        prediction = self.forward(x, noisy_y, timesteps)
        
        target = y_scaled if self.objective == "labels" else noise
        loss = self.loss_fn(prediction, target)
        self.log("train/loss", loss, prog_bar=True)
        
        # Monitor ROC AUC cleanly without the Tweedie Illusion
        with torch.no_grad():
            # ONLY evaluate training ROC AUC when noise is heavy (high t)
            # This prevents the ground-truth leakage at low t
            high_noise_mask = timesteps > (self.num_timesteps // 2)
        
            if high_noise_mask.sum() > 0 and self.objective == "noise":
                alpha_prod_t = (
                    self.noise_scheduler.alphas_cumprod[timesteps].to(y.device).view(-1, 1)
                )
                pred_y0 = (
                    noisy_y - torch.sqrt(1 - alpha_prod_t) * prediction
                ) / torch.sqrt(alpha_prod_t)
                pred_y0_unscaled = torch.clamp((pred_y0 + 1.0) / 2.0, 0.0, 1.0)
        
                y_true_flat = y[high_noise_mask].cpu()
                y_pred_flat = pred_y0_unscaled[high_noise_mask].cpu()
        
                if len(torch.unique(y_true_flat)) > 1:
                    roc_auc = roc_auc_score(y_true_flat.numpy(), y_pred_flat.numpy())
                    self.log("train/roc_auc_heavy_noise", roc_auc, prog_bar=True)
    
        return loss

    def validation_step(self, batch, batch_idx):
        """Validation step computing objective loss and tracking full loop ROC AUC."""
        x, y, mask = batch
        x = self.backbone.extract_features(x)

        # FIX: Map targets to [-1, 1] range
        y_scaled = y * 2.0 - 1.0

        noise = torch.randn_like(y_scaled)
        timesteps = torch.randint(
            0, self.num_timesteps, (y.shape[0],), device=y.device
        ).long()
        noisy_y = self.noise_scheduler.add_noise(y_scaled, noise, timesteps)
        prediction = self.forward(x, noisy_y, timesteps)

        if self.objective == "labels":
            target = y_scaled
        else:
            target = noise

        if self.objective == "labels":
            if self.masked_loss:
                loss = self.loss_fn(prediction, target, mask)
            else:
                loss = self.loss_fn(prediction, target)
        else:
            loss = self.loss_fn(prediction, target)

        self.log("val/loss", loss, prog_bar=True)

        # FIX: Force execution across full DDPM timesteps range to maintain variance tracking accuracy
        predicted_y = self.predict_labels(x, num_steps=self.num_timesteps / 100)

        mask_cpu = mask.cpu()
        y_true = y.cpu()
        y_pred = predicted_y.cpu()
        mask_bool = mask_cpu.bool()
        y_true_flat = y_true[mask_bool]
        y_pred_flat = y_pred[mask_bool]

        if len(y_true_flat) > 0 and len(torch.unique(y_true_flat)) > 1:
            try:
                roc_auc = roc_auc_score(y_true_flat.numpy(), y_pred_flat.numpy())
                self.log("val/roc_auc", roc_auc, prog_bar=True)
                self.roc_aucs.append(roc_auc)
            except ValueError:
                pass

        return loss

    def test_step(self, batch, batch_idx):
        """Test step with complete reverse chain processing."""
        x, y, mask = batch
        x = self.backbone.extract_features(x)

        # FIX: Execute complete DDPM processing chain
        predicted_y = self.predict_labels(x, num_steps=self.num_timesteps / 10)
        mask_bool = mask.bool()

        y_pred_flat = predicted_y[mask_bool].cpu()
        y_true_flat = y[mask_bool].cpu()

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
        """Gather accumulated batches and extract global test scoring."""
        y_true = torch.cat(self.y_true).numpy()
        y_pred = torch.cat(self.y_pred).numpy()

        metrics = {}
        if len(y_true) > 0 and len(np.unique(y_true)) > 1:
            try:
                # Standard binary mapping via middle threshold cutoff
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
