import torch
import torch.nn as nn
from typing import Type


class DiffusionClassifierLg(nn.Module):
    """
    Large diffusion-based classifier for Mechanism of Action (MoA) prediction with class conditioning.


    Args:
        num_classes: Number of output classes for MoA prediction
        embedding_dim: Input feature dimensionality from cell image embeddings
        dropout_rate: Base dropout probability
        residual: Whether to use residual connections between layers
        activation_fn: Activation function class (e.g., nn.ReLU, nn.GELU)
    """

    def __init__(
        self,
        num_classes: int,
        embedding_dim: int,
        dropout_rate: float,
        residual: bool,
        activation_fn: Type[nn.Module],
    ) -> None:
        """Initialize diffusion classifier with class conditioning architecture."""
        super().__init__()

        self.num_classes = num_classes
        self.embedding_dim = embedding_dim
        self.residual = residual

        # Input combines features + class labels + timesteps
        input_dim = embedding_dim + num_classes + 1

        self.fc1 = nn.Linear(input_dim, 2048)
        self.ln1 = nn.LayerNorm(2048)

        self.fc2 = nn.Linear(2048, 1024)
        self.ln2 = nn.LayerNorm(1024)

        self.fc3 = nn.Linear(1024, 512)
        self.ln3 = nn.LayerNorm(512)

        self.fc4 = nn.Linear(512, 256)
        self.ln4 = nn.LayerNorm(256)

        self.fc5 = nn.Linear(256, 128)
        self.ln5 = nn.LayerNorm(128)

        self.fc_out = nn.Linear(128, num_classes)

        if self.residual:
            self.res_proj1 = nn.Linear(input_dim, 2048)
            self.res_proj2 = nn.Linear(2048, 1024)
            self.res_proj3 = nn.Linear(1024, 512)
            self.res_proj4 = nn.Linear(512, 256)
            self.res_proj5 = nn.Linear(256, 128)

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
        x = self.fc_out(x)
        predicted_noise = self.sigmoid(x)
        return predicted_noise

    def _forward_with_residual(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with residual connections."""
        # Block 1: (embedding_dim + num_classes) -> 2048
        residual = self.res_proj1(x)
        x = self.fc1(x)
        x = self.ln1(x)
        x = self.activation(x + residual)
        x = self.dropout1(x)

        # Block 2: 2048 -> 1024
        residual = self.res_proj2(x)
        x = self.fc2(x)
        x = self.ln2(x)
        x = self.activation(x + residual)
        x = self.dropout2(x)

        # Block 3: 1024 -> 512
        residual = self.res_proj3(x)
        x = self.fc3(x)
        x = self.ln3(x)
        x = self.activation(x + residual)
        x = self.dropout2(x)

        # Block 4: 512 -> 256
        residual = self.res_proj4(x)
        x = self.fc4(x)
        x = self.ln4(x)
        x = self.activation(x + residual)
        x = self.dropout2(x)

        # Block 5: 256 -> 128
        residual = self.res_proj5(x)
        x = self.fc5(x)
        x = self.ln5(x)
        x = self.activation(x + residual)
        x = self.dropout2(x)

        return x

    def _forward_standard(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass without residual connections."""
        # Block 1: (embedding_dim + num_classes) -> 2048
        x = self.fc1(x)
        x = self.ln1(x)
        x = self.activation(x)
        x = self.dropout1(x)

        # Block 2: 2048 -> 1024
        x = self.fc2(x)
        x = self.ln2(x)
        x = self.activation(x)
        x = self.dropout2(x)

        # Block 3: 1024 -> 512
        x = self.fc3(x)
        x = self.ln3(x)
        x = self.activation(x)
        x = self.dropout2(x)

        # Block 4: 512 -> 256
        x = self.fc4(x)
        x = self.ln4(x)
        x = self.activation(x)
        x = self.dropout2(x)

        # Block 5: 256 -> 128
        x = self.fc5(x)
        x = self.ln5(x)
        x = self.activation(x)
        x = self.dropout2(x)

        return x


class DiffusionClassifierXlg(nn.Module):
    """
    Extra large diffusion-based classifier for Mechanism of Action (MoA) prediction with class conditioning.


    Args:
        num_classes: Number of output classes for MoA prediction
        embedding_dim: Input feature dimensionality from cell image embeddings
        dropout_rate: Base dropout probability
        residual: Whether to use residual connections between layers
        activation_fn: Activation function class (e.g., nn.ReLU, nn.GELU)
    """

    def __init__(
        self,
        num_classes: int,
        embedding_dim: int,
        dropout_rate: float,
        residual: bool,
        activation_fn: Type[nn.Module],
    ) -> None:
        """Initialize diffusion classifier with class conditioning architecture."""
        super().__init__()

        self.num_classes = num_classes
        self.embedding_dim = embedding_dim
        self.residual = residual
        self.sigmoid = nn.Sigmoid()

        # Input combines features + class labels + timesteps
        input_dim = embedding_dim + num_classes + 1

        self.fc1 = nn.Linear(input_dim, 4096)
        self.ln1 = nn.LayerNorm(4096)

        self.fc2 = nn.Linear(4096, 2048)
        self.ln2 = nn.LayerNorm(2048)

        self.fc3 = nn.Linear(2048, 1024)
        self.ln3 = nn.LayerNorm(1024)

        self.fc4 = nn.Linear(1024, 512)
        self.ln4 = nn.LayerNorm(512)

        self.fc5 = nn.Linear(512, 256)
        self.ln5 = nn.LayerNorm(256)

        self.fc6 = nn.Linear(256, 128)
        self.ln6 = nn.LayerNorm(128)

        self.fc_out = nn.Linear(128, num_classes)

        if self.residual:
            self.res_proj1 = nn.Linear(input_dim, 4096)
            self.res_proj2 = nn.Linear(4096, 2048)
            self.res_proj3 = nn.Linear(2048, 1024)
            self.res_proj4 = nn.Linear(1024, 512)
            self.res_proj5 = nn.Linear(512, 256)
            self.res_proj6 = nn.Linear(256, 128)

        self.activation = activation_fn()
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
        x = self.fc_out(x)
        predicted_noise = self.sigmoid(x)
        return predicted_noise

    def _forward_with_residual(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with residual connections."""
        # Block 1: (embedding_dim + num_classes) -> 2048
        residual = self.res_proj1(x)
        x = self.fc1(x)
        x = self.ln1(x)
        x = self.activation(x + residual)
        x = self.dropout1(x)

        # Block 2: 2048 -> 1024
        residual = self.res_proj2(x)
        x = self.fc2(x)
        x = self.ln2(x)
        x = self.activation(x + residual)
        x = self.dropout2(x)

        # Block 3: 1024 -> 512
        residual = self.res_proj3(x)
        x = self.fc3(x)
        x = self.ln3(x)
        x = self.activation(x + residual)
        x = self.dropout2(x)

        # Block 4: 512 -> 256
        residual = self.res_proj4(x)
        x = self.fc4(x)
        x = self.ln4(x)
        x = self.activation(x + residual)
        x = self.dropout2(x)

        # Block 5: 256 -> 128
        residual = self.res_proj5(x)
        x = self.fc5(x)
        x = self.ln5(x)
        x = self.activation(x + residual)
        x = self.dropout2(x)

        residual = self.res_proj6(x)
        x = self.fc6(x)
        x = self.ln6(x)
        x = self.activation(x + residual)
        x = self.dropout2(x)

        return x

    def _forward_standard(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass without residual connections."""
        # Block 1: (embedding_dim + num_classes) -> 2048
        x = self.fc1(x)
        x = self.ln1(x)
        x = self.activation(x)
        x = self.dropout1(x)

        # Block 2: 2048 -> 1024
        x = self.fc2(x)
        x = self.ln2(x)
        x = self.activation(x)
        x = self.dropout2(x)

        # Block 3: 1024 -> 512
        x = self.fc3(x)
        x = self.ln3(x)
        x = self.activation(x)
        x = self.dropout2(x)

        # Block 4: 512 -> 256
        x = self.fc4(x)
        x = self.ln4(x)
        x = self.activation(x)
        x = self.dropout2(x)

        # Block 5: 256 -> 128
        x = self.fc5(x)
        x = self.ln5(x)
        x = self.activation(x)
        x = self.dropout2(x)

        # Block 5: 256 -> 128
        x = self.fc6(x)
        x = self.ln6(x)
        x = self.activation(x)
        x = self.dropout2(x)

        return x
