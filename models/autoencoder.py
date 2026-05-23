import torch
import torch.nn as nn
from typing import Any


class Autoencoder(nn.Module):
    def __init__(self, input_dim: int = 384, hidden_dim: int = 128, latent_dim: int = 32, dropout_rate: float = 0.3) -> None:
        super().__init__()

        # Encoder: input_dim -> hidden_dim -> latent_dim
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Dropout(dropout_rate),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim),
            nn.Dropout(dropout_rate),
            nn.ReLU(),
        )

        # Decoder: latent_dim -> hidden_dim -> input_dim
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.Dropout(dropout_rate),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim),
        )

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.encoder(x)
        x_hat = self.decoder(z)
        return x_hat
