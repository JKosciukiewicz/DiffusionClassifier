import lightning as L
import torch
import torch.nn.functional as F
import numpy as np
from models.mlp_classifier import MLPClassifier
from lightning_models.MNIST_models.lightning_cnn import LightningCNN
from utils.conformal_prediction import (
    multiclass_conformal_thresholds,
    multiclass_non_conformity_score,
    apply_multiclass_thresholds,
)
from utils.evaluate_conformal_model import calculate_metrics


class LightningConformalMLP(L.LightningModule):
    def __init__(self, alpha, num_classes=10, embedding_dim=128):
        super().__init__()
        self.model = MLPClassifier(num_classes=num_classes, embedding_dim=embedding_dim)
        self.feature_extractor = LightningCNN.load_from_checkpoint(
            "lightning_logs/cnn/cnn-epoch=09-train_loss=0.02.ckpt"
        )
        self.feature_extractor.freeze()
        self.feature_extractor.eval()
        self.nonconformity_scores = []  # list to hold calibration scores
        self.thresholds = None  # per-class thresholds (to be computed in calibration)
        self.alpha = alpha
        # For evaluation of rejection (below threshold)
        self.y_pred_conf = []
        self.y_true = []
        self.y_pred_conf_bt = []
        self.y_true_bt = []
        self.ex_rejected = 0
        self.cls_rejected = 0

    def forward(self, x):
        features = self.feature_extractor.extract_features(x)
        pred_y = self.model(features=features)
        return pred_y

    def training_step(self, batch, batch_idx):
        x, y = batch
        predicted_y = self.forward(x)
        loss = F.binary_cross_entropy(predicted_y, y)
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.model.parameters(), lr=1e-3)

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
        """Compute thresholds after calibration"""
        if not self.nonconformity_scores:
            raise ValueError("No nonconformity scores recorded. Run calibration first.")
        # Here alpha for thresholds is 1 - self.alpha (so that coverage is ~1 - alpha)
        self.thresholds = multiclass_conformal_thresholds(
            alpha=1 - self.alpha, calibration_scores=self.nonconformity_scores
        )
        print("Computed Conformal Thresholds")
        print(self.thresholds)
        print(len(self.thresholds))
        return self.thresholds

    def test_step(self, batch, batch_idx):
        x, y = batch
        y_pred = self.forward(x)
        y_pred = y_pred.cpu().numpy()
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

    # def on_test_epoch_end(self) -> None:
    # plot_ambiguous_class_distribution(self.y_pred_conf_bt)

    def predict(self, x):
        """
        Predict conformal labels on new data.
        Since true labels are not available, we pass a dummy vector (e.g. -1 for every class)
        into apply_multiclass_thresholds and return only the conformal prediction.
        """
        self.eval()  # ensure evaluation mode
        with torch.no_grad():
            y_pred = self.forward(x)
        y_pred = y_pred.cpu().numpy()
        predictions = []
        # For each test sample, create a dummy true label vector.
        dummy_true = -1  # use -1 as dummy value
        for sample_pred in y_pred:
            # Create a dummy vector with same shape as sample_pred.
            dummy_vec = np.full(sample_pred.shape, dummy_true)
            # apply_multiclass_thresholds expects both y_pred and y_true.
            # Here we ignore the y_true output and use only the conformal prediction.
            y_pred_conf, _, _, _, y_pred_bt, _ = apply_multiclass_thresholds(
                sample_pred, dummy_vec, self.thresholds
            )
            predictions.append(y_pred_conf)
        return np.array(predictions)
