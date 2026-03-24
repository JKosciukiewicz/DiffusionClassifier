from models.diffusion_classifier import DiffusionMLP
from lightning_models.MNIST_models.lightning_cnn import LightningCNN
import lightning as L
from diffusers import DDPMScheduler
import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


class LightningDiffusionMLP(L.LightningModule):
    def __init__(self, num_classes=10, embedding_dim=128):
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

    def forward(self, x, y):
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
