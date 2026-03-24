import lightning as L
import torch
import torch.nn.functional as F
from models.mlp_classifier import MLPClassifier
from lightning_models.MNIST_models.lightning_cnn import LightningCNN
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


class LightningMLP(L.LightningModule):
    def __init__(self, num_classes=10, embedding_dim=128):
        super().__init__()
        self.model = MLPClassifier(num_classes=num_classes, embedding_dim=embedding_dim)
        self.feature_extractor = LightningCNN.load_from_checkpoint(
            "lightning_logs/cnn/cnn-epoch=09-train_loss=0.02.ckpt"
        )
        self.feature_extractor.freeze()
        self.feature_extractor.eval()

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

    def test_step(self, batch, batch_idx):
        x, y = batch
        y_pred = self.forward(x)
        # Apply a fixed threshold to obtain binary predictions
        y_pred_bin = (y_pred > 0.5).float()

        # Convert tensors to numpy arrays for scikit-learn metrics
        y_true_np = y.cpu().numpy()
        y_pred_np = y_pred_bin.cpu().numpy()

        # Compute metrics using scikit-learn with average set for multilabel evaluation
        accuracy = accuracy_score(y_true_np, y_pred_np)
        precision = precision_score(
            y_true_np, y_pred_np, average="macro", zero_division=0
        )
        recall = recall_score(y_true_np, y_pred_np, average="macro", zero_division=0)
        f1 = f1_score(y_true_np, y_pred_np, average="macro", zero_division=0)

        # Build metrics dictionary matching the previous format
        metrics = {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
        }
        self.log_dict(metrics)
