import torch
import torch.nn as nn
from typing import Type


class DiffusionClassifierSm(nn.Module):
    """
    Small diffusion classifier with class conditioning.

    Args:
        num_classes: Number of output classes
        embedding_dim: DINO feature dimensionality
        dropout_rate: Base dropout probability
        residual: Whether to use residual connections
        activation_fn: Activation function class
    """

    def __init__(
        self,
        num_classes: int,
        embedding_dim: int,
        dropout_rate: float = 0.2,
        residual: bool = False,
        activation_fn: Type[nn.Module] = nn.GELU,
    ) -> None:
        super().__init__()

        self.residual = residual
        input_dim = embedding_dim + num_classes + 1

        self.input_norm = nn.LayerNorm(input_dim)

        self.fc1 = nn.Linear(input_dim, 512)
        self.ln1 = nn.LayerNorm(512)

        self.fc2 = nn.Linear(512, 256)
        self.ln2 = nn.LayerNorm(256)

        self.fc3 = nn.Linear(256, 128)
        self.ln3 = nn.LayerNorm(128)

        self.fc4 = nn.Linear(128, num_classes)

        if self.residual:
            self.res_proj1 = nn.Linear(input_dim, 512)
            self.res_proj2 = nn.Linear(512, 256)
            self.res_proj3 = nn.Linear(256, 128)

        self.activation = activation_fn()
        self.sigmoid = nn.Sigmoid()
        self.dropout1 = nn.Dropout(dropout_rate)
        self.dropout2 = nn.Dropout(dropout_rate * 1.5)

    def forward(
        self,
        features: torch.Tensor,
        noisy_labels: torch.Tensor,
        timesteps: torch.Tensor,
    ) -> torch.Tensor:
        """
        Predict noise given features, noisy labels, and timesteps.

        Args:
            features: (batch_size, embedding_dim)
            noisy_labels: (batch_size, num_classes)
            timesteps: (batch_size, 1)

        Returns:
            predicted_noise: (batch_size, num_classes)
        """
        # Normalize timesteps to [0, 1]
        timesteps_norm = timesteps.float() / 1000.0

        # Concatenate all inputs
        x = torch.cat([features, noisy_labels, timesteps_norm], dim=1)
        x = self.input_norm(x)

        if self.residual:
            x = self._forward_with_residual(x)
        else:
            x = self._forward_standard(x)

        # Predict noise (no activation - can be positive/negative)
        x = self.fc4(x)
        predicted_noise = self.sigmoid(x)
        return predicted_noise

    def _forward_with_residual(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.res_proj1(x)
        x = self.fc1(x)
        x = self.ln1(x)
        x = self.activation(x + residual)
        x = self.dropout1(x)

        residual = self.res_proj2(x)
        x = self.fc2(x)
        x = self.ln2(x)
        x = self.activation(x + residual)
        x = self.dropout2(x)

        residual = self.res_proj3(x)
        x = self.fc3(x)
        x = self.ln3(x)
        x = self.activation(x + residual)
        x = self.dropout2(x)

        return x

    def _forward_standard(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.ln1(x)
        x = self.activation(x)
        x = self.dropout1(x)

        x = self.fc2(x)
        x = self.ln2(x)
        x = self.activation(x)
        x = self.dropout2(x)

        x = self.fc3(x)
        x = self.ln3(x)
        x = self.activation(x)
        x = self.dropout2(x)

        return x


class DiffusionClassifierXsm(nn.Module):
    """
    Extra small diffusion classifier.

    Args:
        num_classes: Number of output classes
        embedding_dim: DINO feature dimensionality
        dropout_rate: Base dropout probability
        residual: Whether to use residual connections
        activation_fn: Activation function class
    """

    def __init__(
        self,
        num_classes: int,
        embedding_dim: int,
        dropout_rate: float = 0.2,
        residual: bool = False,
        activation_fn: Type[nn.Module] = nn.GELU,
    ) -> None:
        super().__init__()

        self.residual = residual
        input_dim = embedding_dim + num_classes + 1

        self.input_norm = nn.LayerNorm(input_dim)

        self.fc1 = nn.Linear(input_dim, 128)
        self.ln1 = nn.LayerNorm(128)

        self.fc2 = nn.Linear(128, 64)
        self.ln2 = nn.LayerNorm(64)

        self.fc3 = nn.Linear(64, num_classes)


        if self.residual:
            self.res_proj1 = nn.Linear(input_dim, 128)
            self.res_proj2 = nn.Linear(128, 64)

        self.activation = activation_fn()
        self.sigmoid = nn.Sigmoid()
        self.dropout1 = nn.Dropout(dropout_rate)
        self.dropout2 = nn.Dropout(dropout_rate * 1.5)

    def forward(
        self,
        features: torch.Tensor,
        noisy_labels: torch.Tensor,
        timesteps: torch.Tensor,
    ) -> torch.Tensor:
        """
        Predict noise given features, noisy labels, and timesteps.

        Args:
            features: (batch_size, embedding_dim)
            noisy_labels: (batch_size, num_classes)
            timesteps: (batch_size, 1)

        Returns:
            predicted_noise: (batch_size, num_classes)
        """
        # Normalize timesteps to [0, 1]
        timesteps_norm = timesteps.float() / 1000.0

        # Concatenate all inputs
        x = torch.cat([features, noisy_labels, timesteps_norm], dim=1)

        if self.residual:
            x = self._forward_with_residual(x)
        else:
            x = self._forward_standard(x)

        # Predict noise (no activation - can be positive/negative)
        x = self.fc3(x)
        predicted_noise = self.sigmoid(x)
        return predicted_noise

    def _forward_with_residual(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.res_proj1(x)
        x = self.fc1(x)
        x = self.ln1(x)
        x = self.activation(x + residual)
        x = self.dropout1(x)

        residual = self.res_proj2(x)
        x = self.fc2(x)
        x = self.ln2(x)
        x = self.activation(x + residual)
        x = self.dropout2(x)

        return x

    def _forward_standard(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.ln1(x)
        x = self.activation(x)
        x = self.dropout1(x)

        x = self.fc2(x)
        x = self.ln2(x)
        x = self.activation(x)
        x = self.dropout2(x)

        return x
