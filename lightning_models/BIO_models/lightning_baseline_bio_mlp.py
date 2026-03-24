import lightning as L
import torch

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)

from models.bio_models.baseline_mlp import MLPClassifier
from models.bio_models.mlp_variants import (
    MLPClassifierLarger,
    MLPClassifierSmaller,
    MLPClassifierSmallest,
)
from lightning_models.loss.weighted_masked_bce import weighted_masked_bce_loss


class LightningMLPClassifier(L.LightningModule):
    def __init__(
        self,
        num_classes: int,
        embedding_dim: int,
        dropout_rate: float = 0.3,
        version: str = "default",
    ):
        super().__init__()
        if version == "larger":
            self.model = MLPClassifier(
                num_classes=num_classes,
                embedding_dim=embedding_dim,
                dropout_rate=dropout_rate,
            )
        elif version == "small":
            self.model = MLPClassifierLarger(
                num_classes=num_classes,
                embedding_dim=embedding_dim,
                dropout_rate=dropout_rate,
            )
        elif version == "smallest":
            self.model = MLPClassifierSmaller(
                num_classes=num_classes,
                embedding_dim=embedding_dim,
                dropout_rate=dropout_rate,
            )
        else:
            self.model = MLPClassifierSmallest(
                num_classes=num_classes,
                embedding_dim=embedding_dim,
                dropout_rate=dropout_rate,
            )

        self.save_hyperparameters()

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.model(features)

    def training_step(self, batch, batch_idx):
        x, y, mask = batch
        predicted_y = self.forward(x)
        loss = weighted_masked_bce_loss(predicted_y, y, mask, weighted=True)
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def test_step(self, batch, batch_idx):
        x, y, mask = batch
        predicted_y = self.forward(x)
        loss = weighted_masked_bce_loss(predicted_y, y, mask)
        self.log("test_loss", loss)

        predicted_y = predicted_y * mask

        # Get binary predictions using threshold
        y_pred_bin = (predicted_y > 0.5).float()

        # Use only masked values for evaluation
        valid_mask = mask.bool()

        # Use only the masked targets and predictions for evaluation
        # Convert to binary for proper metric calculation
        y_true_masked = (y[valid_mask] > 0.5).float().cpu().numpy()
        y_pred_masked = y_pred_bin[valid_mask].cpu().numpy()
        y_pred_proba_masked = predicted_y[valid_mask].cpu().numpy()

        # If we have valid predictions, compute metrics
        if len(y_true_masked) > 0:
            accuracy = accuracy_score(y_true_masked, y_pred_masked)
            precision = precision_score(
                y_true_masked, y_pred_masked, average="macro", zero_division=0
            )
            recall = recall_score(
                y_true_masked, y_pred_masked, average="macro", zero_division=0
            )
            f1 = f1_score(
                y_true_masked, y_pred_masked, average="macro", zero_division=0
            )

            # Calculate ROC AUC
            try:
                roc_auc = roc_auc_score(
                    y_true_masked, y_pred_proba_masked, average="macro"
                )
            except ValueError:
                # This can happen if all samples belong to one class
                roc_auc = float("nan")

            # Log metrics
            metrics = {
                "test_accuracy": accuracy,
                "test_precision": precision,
                "test_recall": recall,
                "test_f1_score": f1,
                "test_roc_auc": roc_auc,
            }
            self.log_dict(metrics)

    # def test_step(self, batch, batch_idx):
    #     x, y, mask = batch
    #     # print(x[0])
    #     # print(y[0])
    #     # print(mask[0])
    #     y_pred = self.forward(x, y)
    #     y_pred=y_pred * mask
    #     y_pred = y_pred.cpu().numpy()
    #     y = y * mask
    #     y = y.cpu().numpy()
    #     for idx, sample_pred in enumerate(y_pred):
    #         y_true_sample = y[idx]
    #         (y_pred_conf, y_true_conf, ex_rejected, cls_rejected,
    #          y_pred_bt, y_true_bt) = apply_multiclass_thresholds(sample_pred, y_true_sample, self.thresholds)
    #         self.y_true.append(y_true_conf)
    #         self.y_pred_conf.append(y_pred_conf)
    #         self.ex_rejected += ex_rejected
    #         self.cls_rejected += cls_rejected
    #         self.y_true_bt.append(y_true_bt)
    #         self.y_pred_conf_bt.append(y_pred_bt)
    #
    #     metrics = calculate_metrics(
    #         self.y_true, self.y_pred_conf,
    #         self.y_true_bt, self.y_pred_conf_bt,
    #         self.ex_rejected, self.cls_rejected, self.alpha
    #     )
    #
    #     self.log_dict(metrics)

    def configure_optimizers(self):
        return torch.optim.Adam(self.model.parameters(), lr=1e-3)
