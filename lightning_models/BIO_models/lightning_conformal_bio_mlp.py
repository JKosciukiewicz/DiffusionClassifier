# import lightning as L
# import torch
# import pandas as pd
# import os
# from utils.evaluate_conformal_model import calculate_metrics
# from models.bio_models.baseline_mlp import MLPClassifier
# from utils.conformal_prediction import (
#     multiclass_conformal_thresholds,
#     multiclass_non_conformity_score,
#     apply_multiclass_thresholds
# )
# from lightning_models.loss.weighted_masked_bce import weighted_masked_bce_loss
#
#
# class LightningConformalMLP(L.LightningModule):
#     def __init__(self, num_classes: int, embedding_dim: int, dropout_rate: float = 0.3, percent_accept: float = 10.0,
#                  lr=1e-3):
#         super().__init__()
#         self.model = MLPClassifier(num_classes=num_classes, embedding_dim=embedding_dim, dropout_rate=dropout_rate)
#         # Conformal prediction variables:
#         self.nonconformity_scores = []  # Calibration scores (list of arrays, shape: [n_calib, num_classes])
#         self.thresholds = None  # Per-class thresholds computed during calibration
#         self.percent_accept = percent_accept  # Percentage of scores to accept
#         self.lr = lr
#
#         # For evaluation/metric accumulation:
#         self.y_pred_conf = []
#         self.y_true = []
#         self.y_pred_conf_bt = []
#         self.y_true_bt = []
#         self.ex_rejected = 0
#         self.cls_rejected = 0
#         # File path for saving results
#         self.results_file = None
#
#     def configure_optimizers(self):
#         return torch.optim.Adam(self.model.parameters(), lr=self.lr)
#
#     def forward(self, features: torch.Tensor) -> torch.Tensor:
#         pred_y = self.model(features=features)
#         return pred_y
#
#     def training_step(self, batch, batch_idx):
#         x, y, mask = batch
#         predicted_y = self.forward(x)
#         loss = weighted_masked_bce_loss(predicted_y, y, mask, weighted=True)
#         self.log("train_loss", loss, prog_bar=True)
#         return loss
#
#     # Conformal prediction: Calibration step
#     def calibration_step(self, batch, batch_idx):
#         x, y, mask = batch
#         y_pred = self.forward(x)
#         bool_mask = mask.bool()
#         # Apply the boolean mask for indexing
#         y_pred = y_pred[bool_mask]
#         y_pred = y_pred.reshape(-1, 1)
#         y = y[bool_mask]
#         y = y.reshape(-1, 1)
#         # Compute nonconformity scores for each sample in the batch.
#         for i in range(y.shape[0]):
#             # multiclass_non_conformity_score expects tensors
#             score = multiclass_non_conformity_score(y_pred[i], y[i])
#             self.nonconformity_scores.append(score)
#
#     def compute_thresholds(self):
#         if not self.nonconformity_scores:
#             raise ValueError("No nonconformity scores recorded. Run calibration first.")
#
#         # Use percent_accept to calculate thresholds
#         self.thresholds = multiclass_conformal_thresholds(
#             percent_accept=self.a, calibration_scores=self.nonconformity_scores
#         )
#         print(f"Computed Conformal Thresholds for percent_accept={self.percent_accept}%")
#         print(self.thresholds)
#         print(len(self.thresholds))
#         return self.thresholds
#
#     def set_thresholds(self, thresholds):
#         self.thresholds = thresholds
#
#     def test_step(self, batch, batch_idx):
#         x, y, mask = batch
#         y_pred = self.forward(x)
#         bool_mask = mask.bool()
#         # Apply the boolean mask for indexing
#         y_pred = y_pred[bool_mask]
#         y_pred = y_pred.cpu().numpy().reshape(-1, 1)
#         y = y[bool_mask]
#         y = y.cpu().numpy().reshape(-1, 1)
#
#         for idx, sample_pred in enumerate(y_pred):
#             y_true_sample = y[idx]
#             (y_pred_conf, y_true_conf, ex_rejected, cls_rejected,
#              y_pred_bt, y_true_bt) = apply_multiclass_thresholds(
#                 sample_pred, y_true_sample, self.thresholds
#             )
#             self.y_true.append(y_true_conf)
#             self.y_pred_conf.append(y_pred_conf)
#             self.ex_rejected += ex_rejected
#             self.cls_rejected += cls_rejected
#             self.y_true_bt.append(y_true_bt)
#             self.y_pred_conf_bt.append(y_pred_bt)
#
#     def on_test_epoch_end(self):
#         # Calculate metrics at the end of the test epoch
#         # Convert percent_accept to equivalent alpha for metrics calculation
#         equivalent_alpha = 1 - (self.percent_accept / 100.0)
#
#         metrics = calculate_metrics(
#             self.y_true, self.y_pred_conf,
#             self.y_true_bt, self.y_pred_conf_bt,
#             self.ex_rejected, self.cls_rejected, equivalent_alpha
#         )
#
#         # Log metrics
#         self.log_dict(metrics)
#
#         # Also log the percent_accept value alongside metrics for easier tracking
#         metrics["percent_accept"] = self.percent_accept
#
#         # Save metrics to CSV file if a results file path is specified
#         if self.results_file:
#             # Convert metrics to DataFrame
#             metrics_df = pd.DataFrame([metrics])
#
#             # Add model hyperparameters to the metrics dataframe
#             metrics_df['embedding_dim'] = 4643
#             metrics_df['num_classes'] = 30
#             metrics_df['learning_rate'] = self.lr
#             metrics_df['dropout_rate'] = 0.3
#
#             # Create directory if it doesn't exist
#             os.makedirs(os.path.dirname(self.results_file), exist_ok=True)
#
#             # Save to CSV
#             metrics_df.to_csv(self.results_file, index=False)
#             print(f"Results saved to {self.results_file}")
#
#         print(f"Test completed for percent_accept={self.percent_accept}%")
#         print(f"Metrics: {metrics}")
#
#     def reset_params(self):
#         # Reset all evaluation parameters
#         self.nonconformity_scores = []
#         self.thresholds = None
#         self.y_pred_conf = []
#         self.y_true = []
#         self.y_pred_conf_bt = []
#         self.y_true_bt = []
#         self.ex_rejected = 0
#         self.cls_rejected = 0

import lightning as L
import torch
import pandas as pd
import os
from utils.evaluate_conformal_model import calculate_metrics
from models.bio_models.baseline_mlp import MLPClassifier
from utils.conformal_prediction import (
    multiclass_conformal_thresholds,
    multiclass_non_conformity_score,
    apply_multiclass_thresholds,
)
from lightning_models.base_model import BaseModel
from lightning_models.loss.weighted_masked_bce import weighted_masked_bce_loss


class LightningConformalMLP(BaseModel):
    def __init__(
        self,
        num_classes: int,
        embedding_dim: int,
        dropout_rate: float = 0.3,
        alpha=0.05,
        lr=1e-3,
    ):
        super().__init__()
        self.model = MLPClassifier(
            num_classes=num_classes,
            embedding_dim=embedding_dim,
            dropout_rate=dropout_rate,
        )
        # Conformal prediction variables:
        self.nonconformity_scores = []  # Calibration scores (list of arrays, shape: [n_calib, num_classes])
        self.thresholds = None  # Per-class thresholds computed during calibration
        self.alpha = alpha
        self.lr = lr

        # For evaluation/metric accumulation:
        self.y_pred_conf = []
        self.y_true = []
        self.y_pred_conf_bt = []
        self.y_true_bt = []
        self.ex_rejected = 0
        self.cls_rejected = 0
        # File path for saving results
        self.results_file = None

    def configure_optimizers(self):
        return torch.optim.Adam(self.model.parameters(), lr=self.lr)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        pred_y = self.model(features=features)
        return pred_y

    def training_step(self, batch, batch_idx):
        x, y, mask = batch
        predicted_y = self.forward(x)
        loss = weighted_masked_bce_loss(predicted_y, y, mask, weighted=True)
        self.log("train_loss", loss, prog_bar=True)
        return loss

    # Conformal prediction: Calibration step
    def calibration_step(self, batch, batch_idx):
        x, y = batch
        y_pred = self.forward(x)
        # Compute nonconformity scores for each sample in the batch.
        for i in range(y.shape[0]):
            # multiclass_non_conformity_score expects tensors
            score = multiclass_non_conformity_score(y_pred[i], y[i])
            self.nonconformity_scores.append(score)

    def compute_thresholds(self):
        if not self.nonconformity_scores:
            raise ValueError("No nonconformity scores recorded. Run calibration first.")
        # Here alpha for thresholds is self.alpha
        self.thresholds = multiclass_conformal_thresholds(
            alpha=self.alpha, calibration_scores=self.nonconformity_scores
        )
        print(f"Computed Conformal Thresholds for alpha={self.alpha}")
        print(self.thresholds)
        print(len(self.thresholds))
        return self.thresholds

    def set_thresholds(self, thresholds):
        self.thresholds = thresholds

    def test_step(self, batch, batch_idx):
        x, y, mask = batch
        y_pred = self.forward(x)
        y_pred = y_pred * mask
        y_pred = y_pred.cpu().numpy()
        y = y * mask
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

    def on_test_epoch_end(self):
        # Calculate metrics at the end of the test epoch
        metrics = calculate_metrics(
            self.y_true,
            self.y_pred_conf,
            self.y_true_bt,
            self.y_pred_conf_bt,
            self.ex_rejected,
            self.cls_rejected,
            self.alpha,
        )

        # Log metrics - they will be associated with the current step and alpha value
        self.log_dict(metrics)

        # Also log the alpha value alongside metrics for easier tracking
        metrics["alpha"] = self.alpha

        # Save metrics to CSV file if a results file path is specified
        if self.results_file:
            # Convert metrics to DataFrame
            metrics_df = pd.DataFrame([metrics])

            # Add model hyperparameters to the metrics dataframe
            metrics_df["embedding_dim"] = 4643
            metrics_df["num_classes"] = 30
            metrics_df["learning_rate"] = self.lr
            metrics_df["dropout_rate"] = 0.3

            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(self.results_file), exist_ok=True)

            # Save to CSV
            metrics_df.to_csv(self.results_file, index=False)
            print(f"Results saved to {self.results_file}")

        print(f"Test completed for alpha={self.alpha}")
        print(f"Metrics: {metrics}")

    def reset_params(self):
        # Conformal prediction variables:
        self.nonconformity_scores = []  # Reset will be overridden in main script
        self.thresholds = None  # Per-class thresholds computed during calibration
        # For evaluation/metric accumulation:
        self.y_pred_conf = []
        self.y_true = []
        self.y_pred_conf_bt = []
        self.y_true_bt = []
        self.ex_rejected = 0
        self.cls_rejected = 0
