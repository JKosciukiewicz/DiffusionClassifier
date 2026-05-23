import torch
import torch.nn.functional as F
from lightning_models.base_model import BaseModel
from models.autoencoder import Autoencoder

class LightningAutoencoder(BaseModel):
    def __init__(self, input_dim: int = 384, hidden_dim: int = 128, latent_dim: int = 32, dropout_rate: float = 0.3, lr: float = 1e-3):
        super().__init__()
        self.save_hyperparameters()
        self.lr = lr
        self.model = Autoencoder(input_dim=input_dim, hidden_dim=hidden_dim, latent_dim=latent_dim, dropout_rate=dropout_rate)

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        # Assuming the batch is (x, y, mask) as in other models, but for AE we only need x
        x, _, _ = batch
        x = x.float()
        x_hat = self.model(x)
        loss = F.mse_loss(x_hat, x)
        self.log("train/loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, _, _ = batch
        x = x.float()
        x_hat = self.model(x)
        loss = F.mse_loss(x_hat, x)
        self.log("val/loss", loss, prog_bar=True)
        return loss

    def extract_features(self, x):
        return self.model.extract_features(x)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)
