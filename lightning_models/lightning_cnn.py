from typing import Any

import torch
import torch.nn.functional as F

from lightning_models.base_model import BaseModel
from models.cnn import CNNMultiLabel


class LightningCNN(BaseModel):
    def __init__(self, num_classes=10, embedding_dim=128, learning_rate=1e-3):
        super().__init__()
        self.learning_rate = learning_rate
        self.model = CNNMultiLabel(num_classes=num_classes, embedding_dim=embedding_dim)

    def forward(self, x) -> Any:
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y, _ = batch
        y_pred = self.model(x)
        loss = F.binary_cross_entropy(y_pred, y)

        # Log loss for monitoring
        self.log("train_loss", loss, prog_bar=True)

        return loss

    def validation_step(self, batch, batch_idx):
        x, y, _ = batch
        y_pred = self.model(x)
        loss = F.binary_cross_entropy(y_pred, y)

        self.log("val_loss", loss, prog_bar=True)

        return loss

    def extract_features(self, x):
        return self.model.extract_features(x)

    def configure_optimizers(self):
        return torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
