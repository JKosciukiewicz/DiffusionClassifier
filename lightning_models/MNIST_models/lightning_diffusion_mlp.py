import lightning as L
import torch
import torch.nn.functional as F
import numpy as np
from diffusers import DDPMScheduler

from models.diffusion_classifier import DiffusionMLP
from lightning_models.MNIST_models.lightning_cnn import LightningCNN
from utils.conformal_prediction import (
    multiclass_conformal_thresholds,
    multiclass_non_conformity_score,
    apply_multiclass_thresholds,
)

from utils.evaluate_conformal_model import calculate_metrics


class LightningDiffusionMLP(L.LightningModule):
    def __init__(self, num_classes=10, embedding_dim=128, alpha=0.05):
        super().__init__()
        self.model = DiffusionMLP(num_classes=num_classes, embedding_dim=embedding_dim)
        self.feature_extractor = LightningCNN.load_from_checkpoint(
            "lightning_logs/cnn/cnn-epoch=09-train_loss=0.02.ckpt"
        )
        self.feature_extractor.freeze()
        self.feature_extractor.eval()
        self.noise_scheduler = DDPMScheduler(
            num_train_timesteps=1000, beta_schedule="linear"
        )

        # Conformal prediction variables:
        self.nonconformity_scores = []  # Calibration scores (list of arrays, shape: [n_calib, num_classes])
        self.thresholds = None  # Per-class thresholds computed during calibration
        self.alpha = alpha

        # For evaluation/metric accumulation:
        self.y_pred_conf = []
        self.y_true = []
        self.y_pred_conf_bt = []
        self.y_true_bt = []
        self.ex_rejected = 0
        self.cls_rejected = 0

    def forward(self, x, y):
        """
        Forward pass: extract features, add noise via DDPMScheduler, and predict eps.
        """
        features = self.feature_extractor.extract_features(x)
        timesteps = torch.randint(
            0, self.noise_scheduler.config.num_train_timesteps, (y.shape[0],)
        ).long()
        pred_noise = self.model(features=features, labels=y)
        noisy_labels = self.noise_scheduler.add_noise(y, pred_noise, timesteps)
        predicted_eps = self.model(features=features, labels=noisy_labels)
        return predicted_eps

    def training_step(self, batch, batch_idx):
        x, y = batch
        predicted_eps = self.forward(x, y)
        loss = F.binary_cross_entropy(predicted_eps, y)
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def configure_optimizers(self, lr=1e-3):
        return torch.optim.Adam(self.model.parameters(), lr=lr)

    # Conformal prediction: Calibration step
    def calibration_step(self, batch, batch_idx):
        x, y = batch
        y_pred = self.forward(x, y)
        # Compute nonconformity score for each sample in the batch:
        for i in range(y.shape[0]):
            score = multiclass_non_conformity_score(y_pred[i], y[i])
            self.nonconformity_scores.append(score)

    def compute_thresholds(self):
        """Compute per-class conformal thresholds after calibration."""
        if not self.nonconformity_scores:
            raise ValueError("No nonconformity scores recorded. Run calibration first.")
        self.thresholds = multiclass_conformal_thresholds(
            alpha=1 - self.alpha, calibration_scores=self.nonconformity_scores
        )
        print("Computed Conformal Thresholds:")
        print(self.thresholds)
        return self.thresholds

    def set_thresholds(self, thresholds):
        self.thresholds = thresholds

    def test_step(self, batch, batch_idx):
        x, y = batch
        y_pred = self.forward(x, y).cpu().numpy()
        y = y.cpu().numpy()
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

        metrics = calculate_metrics(
            self.y_true,
            self.y_pred_conf,
            self.y_true_bt,
            self.y_pred_conf_bt,
            self.ex_rejected,
            self.cls_rejected,
            self.alpha,
        )
        self.log_dict(metrics)

    def on_test_epoch_end(self) -> None:
        # Concatenate tie-breaker predictions from all test batches.
        # Expected shape: (total_samples, num_classes) e.g. (12000, 10)
        y_pred_all = np.vstack(self.y_pred_conf_bt)
        # plot_ambiguous_class_distribution(y_pred_all)

    def predict(self, x, y_dummy):
        """
        Predict conformal labels on new data.
        Since true labels are unavailable, pass a dummy label vector (e.g. filled with -1)
        to simulate candidate scores.
        """
        self.eval()
        with torch.no_grad():
            y_pred = self.forward(x, y_dummy).cpu().numpy()
        predictions = []
        for sample_pred in y_pred:
            # Create a dummy true label vector of same shape:
            dummy_vec = np.full(sample_pred.shape, -1)
            y_pred_conf, _, _, _, y_pred_bt, _ = apply_multiclass_thresholds(
                sample_pred, dummy_vec, self.thresholds
            )
            # You can choose to return the conformal prediction (which might be -1 for ambiguous)
            # or the tie-breaker result.
            predictions.append(y_pred_conf)
        return np.array(predictions)
