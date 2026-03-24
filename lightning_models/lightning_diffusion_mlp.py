from typing import Optional, Type

import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score

from utils.evaluate_conformal_model import calculate_metrics
from models.bio_models.diffusion_classifier_lg import (
    DiffusionClassifierLg,
    DiffusionClassifierXlg,
)
from models.bio_models.diffusion_classifier_md import DiffusionClassifierMd
from models.bio_models.diffusion_classifier_sm import (
    DiffusionClassifierSm,
    DiffusionClassifierXsm,
)
from diffusers import DDPMScheduler
from lightning_models.base_model import BaseModel
from lightning_models.loss.weighted_masked_bce import weighted_masked_bce_loss
from utils.conformal_prediction import (
    multiclass_non_conformity_score,
    apply_multiclass_thresholds,
)
torch.manual_seed(42)

class LightningDiffusionClassifier(BaseModel):
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
            num_inference_timesteps: int = 1000,
    ):
        super().__init__()

        # Core parameters
        self.alpha = alpha
        self.num_classes = num_classes
        self.lr = lr
        self.loss_weighted = loss_weighted
        self.loss_pos_weight = loss_pos_weight
        self.results_file = results_path
        self.embedding_dim = embedding_dim

        # Multi-timestep sampling parameters
        self.num_inference_timesteps = num_inference_timesteps

        # Conformal prediction variables
        self.nonconformity_scores = []
        self.thresholds = None
        self.percent_accept = percent_accept

        # Initialize backbone if specified
        self.backbone = (
            type(backbone).load_from_checkpoint(backbone_ckpt_path)
            if backbone
            else None
        )

        # Initialize model based on size
        # NOTE: Assuming these classes have been converted to LabelGenerator architecture
        # with forward(features, noisy_labels, timesteps) -> predicted_noise
        if size == "sm":
            self.model = DiffusionClassifierSm(
                num_classes=num_classes,
                embedding_dim=embedding_dim,
                dropout_rate=dropout_rate,
                residual=residual,
                activation_fn=activation_fn,
            )
        elif size == "xsm":
            self.model = DiffusionClassifierXsm(
                num_classes=num_classes,
                embedding_dim=embedding_dim,
                dropout_rate=dropout_rate,
                residual=residual,
                activation_fn=activation_fn,
            )
        elif size == "md":
            self.model = DiffusionClassifierMd(
                num_classes=num_classes,
                embedding_dim=embedding_dim,
                dropout_rate=dropout_rate,
                residual=residual,
                activation_fn=activation_fn,
            )
        elif size == "lg":
            self.model = DiffusionClassifierLg(
                num_classes=num_classes,
                embedding_dim=embedding_dim,
                dropout_rate=dropout_rate,
                residual=residual,
                activation_fn=activation_fn,
            )
        elif size == "xlg":
            self.model = DiffusionClassifierXlg(
                num_classes=num_classes,
                embedding_dim=embedding_dim,
                dropout_rate=dropout_rate,
                residual=residual,
                activation_fn=activation_fn,
            )

        self.loss_fn = nn.MSELoss()
        # Noise scheduler initialization
        self.noise_scheduler = DDPMScheduler(
            num_train_timesteps=1000, beta_schedule="squaredcos_cap_v2"
        )

        # Evaluation metric accumulation
        self.y_pred_conf = []
        self.y_true = []
        self.y_pred_conf_bt = []
        self.y_true_bt = []
        self.ex_rejected = 0
        self.cls_rejected = 0

        # ROC AUC tracking
        self.peak_roc_auc = 0
        self.roc_aucs = []

    def forward(self, features: torch.Tensor, labels: torch.Tensor, num_timesteps: int = 1, inference=False) -> torch.Tensor:
        """
        Forward pass with optional multi-timestep sampling following diffusion classifier methodology.

        Args:
            features: Input features (batch_size, embedding_dim)
            labels: Input labels (batch_size, num_classes)
            num_timesteps: Number of timesteps to sample for averaging (default=1 for training)

        Returns:
            predicted_labels: Averaged predictions across timesteps (batch_size, num_classes)

        Mathematical formulation:
        $$\\hat{y} = \\frac{1}{N}\\sum_{i=1}^{N} \\sigma\\left(x_{t_i} - \\epsilon_\\theta(x_{t_i}, f, t_i)\\right)$$

        Where:
        - $N$ = num_timesteps
        - $t_i$ = randomly sampled timesteps
        - $x_{t_i}$ = noisy labels at timestep $t_i$
        - $f$ = input features
        - $\\epsilon_\\theta$ = noise prediction network
        - $\\sigma$ = sigmoid activation
        """
        # if num_timesteps == 1:
        #     return self._single_timestep_forward(features, labels, inference)
        # else:
        #     return self._multi_timestep_forward(features, labels, num_timesteps)
        #return  self._single_timestep_forward(features, labels, inference)
        return  self._single_timestep_forward(features, labels)

    # def _single_timestep_forward(
    #     self, features: torch.Tensor, labels: torch.Tensor, inference:bool = False
    # ) -> torch.Tensor:
    #     """
    #     Single timestep forward pass optimized for training efficiency.
    #
    #     Implements the core diffusion process:
    #     1. Sample random timestep $t \\sim U[0, T]$
    #     2. Sample noise $\\epsilon \\sim \\mathcal{N}(0, I)$
    #     3. Create noisy labels: $x_t = \\sqrt{\\bar{\\alpha}_t} y + \\sqrt{1-\\bar{\\alpha}_t} \\epsilon$
    #     4. Predict noise: $\\hat{\\epsilon} = \\epsilon_\\theta(x_t, f, t)$
    #     5. Reconstruct: $\\hat{y} = \\text{scheduler.step}(\\hat{\\epsilon}, t, x_t)$
    #     """
    #     batch_size = labels.shape[0]
    #     device = labels.device
    #     noise = torch.randn_like(labels)
    #     timesteps = torch.randint(
    #         0,
    #         self.noise_scheduler.config.num_train_timesteps,
    #         (batch_size,),
    #         device=device,
    #     ).long()
    #
    #     noisy_labels = self.noise_scheduler.add_noise(labels, noise, timesteps)
    #
    #     # Predict added noise using the model
    #     predicted_noise = self.model(
    #         features=features,
    #         noisy_labels=noisy_labels,
    #         timesteps=timesteps.unsqueeze(1),
    #     )
    #
    #     loss = self.loss_fn(predicted_noise, noise)
    #
    #     # Reconstruct clean labels via denoising step (process each sample individually)
    #     predicted_ys = []
    #     for ts, outputs, label in zip(timesteps, predicted_noise, noisy_labels):
    #         # Add batch dimension back since scheduler expects batched tensors
    #         outputs_batched = outputs.unsqueeze(
    #             0
    #         )  # Add batch dim: [features] -> [1, features]
    #         label_batched = label.unsqueeze(
    #             0
    #         )  # Add batch dim: [features] -> [1, features]
    #         if inference:
    #             ts = torch.tensor(self.noise_scheduler.config.num_train_timesteps)
    #         ts_batched = ts.unsqueeze(0)  # Add batch dim: [] -> [1]
    #
    #         predicted_y = self.noise_scheduler.step(
    #             outputs_batched, ts_batched, label_batched
    #         ).pred_original_sample
    #         predicted_ys.append(
    #             predicted_y.squeeze(0)
    #         )  # Remove batch dim for consistency
    #
    #     # Stack all predictions back into a batch
    #     pred_ys = torch.stack(predicted_ys, dim=0)
    #     return loss, pred_ys

    def _single_timestep_forward(self, features: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        Single timestep forward pass optimized for training efficiency.

        Implements the core diffusion process:
        1. Sample random timestep $t \\sim U[0, T]$
        2. Sample noise $\\epsilon \\sim \\mathcal{N}(0, I)$
        3. Create noisy labels: $x_t = \\sqrt{\\bar{\\alpha}_t} y + \\sqrt{1-\\bar{\\alpha}_t} \\epsilon$
        4. Predict noise: $\\hat{\\epsilon} = \\epsilon_\\theta(x_t, f, t)$
        5. Reconstruct: $\\hat{y} = \\sigma(x_t - \\hat{\\epsilon})$
        """
        batch_size = labels.shape[0]
        device = labels.device
        #noise = torch.randn_like(labels)
        noise = torch.normal(mean = 0.040524, std=0.197185, size=labels.shape).to(device)
        timesteps = torch.randint(
            0, self.noise_scheduler.config.num_train_timesteps,
            (batch_size,), device=device
        ).long()

        # Apply forward diffusion process
        noisy_labels = self.noise_scheduler.add_noise(labels, noise, timesteps)

        # Predict added noise using the model
        predicted_noise = self.model(
            features=features,
            noisy_labels=noisy_labels,
            timesteps=timesteps.unsqueeze(1)
        )

        # Reconstruct clean labels via denoising
        return predicted_noise

    def _multi_timestep_forward(self, features: torch.Tensor, labels: torch.Tensor, noise, num_timesteps: int) -> torch.Tensor:
        """
        Multi-timestep sampling for robust inference as described in Li et al.

        Implements Monte Carlo estimation over multiple timesteps to reduce variance:
        $$\\mathbb{E}_{t,\\epsilon}[\\|\\epsilon - \\epsilon_\\theta(x_t, f, t)\\|^2]$$

        By averaging predictions across $N$ independent samples of $(t_i, \\epsilon_i)$ pairs.
        """
        batch_size = labels.shape[0]
        device = labels.device
        predictions = []

        # Perform multiple independent forward passes
        for _ in range(num_timesteps):
            timesteps = torch.randint(
                0,
                self.noise_scheduler.config.num_train_timesteps,
                (batch_size,),
                device=device,
            ).long()

            noisy_labels = self.noise_scheduler.add_noise(labels, noise, timesteps)

            predicted_noise = self.model(
                features=features,
                noisy_labels=noisy_labels,
                timesteps=timesteps.unsqueeze(1)
            )

            predictions.append(predicted_noise)

        # Compute ensemble average for improved stability
        averaged_predictions = torch.stack(predictions).mean(dim=0)
        return averaged_predictions

    def training_step(self, batch, batch_idx):
        """Training step with corrected BCE loss computation."""
        x, y, mask = batch

        # Extract backbone features if specified
        if self.backbone:
            x = self.backbone.extract_features(x)

        # Single timestep forward pass for efficiency
        #loss, predicted_y = self.forward(x, y, num_timesteps=1)
        predicted_y = self.forward(x, y, num_timesteps=1)

        #Compute weighted masked BCE loss
        loss = weighted_masked_bce_loss(
            predicted_y,
            y,
            mask,
            pos_weight=self.loss_pos_weight,
            weighted=self.loss_weighted,
        )

        self.log("train_loss", loss, prog_bar=True)
        # self.log("bce_loss", bce_loss, prog_bar=True)

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

    def validation_step(self, batch, batch_idx):
        """Validation step with multi-timestep sampling for robust evaluation."""
        x, y, mask = batch
        faux_y = torch.randn(y.shape).to(x.device)
        # Extract backbone features if specified
        if self.backbone:
            x = self.backbone.extract_features(x)

        # Multi-timestep sampling for more robust validation predictions
        #loss, predicted_y = self.forward(x, faux_y, num_timesteps= 1, inference = True)
        predicted_y = self.forward(x, faux_y, num_timesteps=1, inference=True)
        # Compute validation loss
        loss = weighted_masked_bce_loss(
            predicted_y,
            y,
            mask,
            pos_weight=self.loss_pos_weight,
            weighted=self.loss_weighted,
        )

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
        faux_y = torch.randn(y.shape).to(x.device)
        noise = torch.randn_like(y)


        #loss, y_pred = self.forward(x, faux_y, num_timesteps=1, inference = True)
        y_pred = self.forward(x, faux_y, num_timesteps=1, inference=True)
        bool_mask = mask.bool()

        # Apply masking and prepare for conformal prediction
        y_pred = y_pred[bool_mask].cpu()
        y = y[bool_mask].cpu()
        # Create a mask for where y == 1

        # Apply conformal prediction thresholds - iterate over each sample
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

    def compute_thresholds(self):
        """Compute conformal prediction thresholds based on calibration scores."""
        if not self.nonconformity_scores:
            raise ValueError("No nonconformity scores recorded. Run calibration first.")

        # Convert to numpy array for percentile calculation
        calib_scores = np.array(self.nonconformity_scores)

        # Calculate threshold at the given percentile
        quantile = self.percent_accept / 100.0
        self.thresholds = np.quantile(calib_scores, quantile, axis=0)

        print(f"Computed Diffusion Thresholds for percent_accept={self.percent_accept}%")
        print(self.thresholds)
        return self.thresholds

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

        # # Save metrics to CSV file
        # if self.results_file:
        #     metrics_df = pd.DataFrame([metrics])
        #     metrics_df["alpha"] = self.alpha
        #     metrics_df["embedding_dim"] = self.embedding_dim
        #     metrics_df["num_classes"] = self.num_classes
        #     metrics_df["learning_rate"] = self.lr
        #     metrics_df["num_inference_timesteps"] = self.num_inference_timesteps
        #
        #     metrics_df.to_csv(self.results_file, index=False)
        #     print(f"Results saved to {self.results_file}")

    def _set_thresholds_from_alpha(self):
        """Automatically set thresholds based on alpha parameter."""
        self.thresholds = np.ones(self.num_classes) * (1 - self.alpha)
        print(f"Automatically set thresholds based on alpha={self.alpha}: {self.thresholds}")

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