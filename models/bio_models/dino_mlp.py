import torch
import torch.nn as nn
from typing import Type


class MLPClassifierDino(nn.Module):
    """DINO-based MLP classifier with configurable architecture."""

    def __init__(
        self,
        num_classes: int,
        embedding_dim: int,
        dropout_rate: float = 0.2,
        residual: bool = False,
        activation_fn: Type[nn.Module] = nn.GELU,
        use_layer_norm: bool = False,
    ) -> None:
        super().__init__()

        self.residual = residual
        self.use_layer_norm = use_layer_norm

        self.input_norm = nn.LayerNorm(embedding_dim)

        self.fc1 = nn.Linear(embedding_dim, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 128)
        self.fc4 = nn.Linear(128, num_classes)

        if self.use_layer_norm:
            self.ln1 = nn.LayerNorm(512)
            self.ln2 = nn.LayerNorm(256)
            self.ln3 = nn.LayerNorm(128)

        if self.residual:
            self.res_proj1 = nn.Linear(embedding_dim, 512)
            self.res_proj2 = nn.Linear(512, 256)
            self.res_proj3 = nn.Linear(256, 128)

        self.activation = activation_fn()
        self.dropout1 = nn.Dropout(dropout_rate)
        self.dropout2 = nn.Dropout(dropout_rate * 1.5)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        x = self.input_norm(features)

        if self.residual:
            x = self._forward_with_residual(x)
        else:
            x = self._forward_standard(x)

        x = self.fc4(x)
        return torch.sigmoid(x)

    def _forward_with_residual(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.res_proj1(x)
        x = self.fc1(x)
        if self.use_layer_norm:
            x = self.ln1(x)
        x = self.activation(x + residual)
        x = self.dropout1(x)

        residual = self.res_proj2(x)
        x = self.fc2(x)
        if self.use_layer_norm:
            x = self.ln2(x)
        x = self.activation(x + residual)
        x = self.dropout2(x)

        residual = self.res_proj3(x)
        x = self.fc3(x)
        if self.use_layer_norm:
            x = self.ln3(x)
        x = self.activation(x + residual)
        x = self.dropout2(x)

        return x

    def _forward_standard(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        if self.use_layer_norm:
            x = self.ln1(x)
        x = self.activation(x)
        x = self.dropout1(x)

        x = self.fc2(x)
        if self.use_layer_norm:
            x = self.ln2(x)
        x = self.activation(x)
        x = self.dropout2(x)

        x = self.fc3(x)
        if self.use_layer_norm:
            x = self.ln3(x)
        x = self.activation(x)
        x = self.dropout2(x)

        return x


class MLPClassifierDinoSmall(nn.Module):
    """Compact DINO MLP classifier with residual connections."""

    def __init__(
        self,
        num_classes: int,
        embedding_dim: int,
        dropout_rate: float = 0.2,
        residual: bool = True,
        activation_fn: Type[nn.Module] = nn.GELU,
        use_layer_norm: bool = True,
    ) -> None:
        super().__init__()

        self.residual = residual
        self.use_layer_norm = use_layer_norm

        self.input_norm = nn.LayerNorm(embedding_dim)

        self.fc1 = nn.Linear(embedding_dim, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, num_classes)

        if self.use_layer_norm:
            self.ln1 = nn.LayerNorm(128)
            self.ln2 = nn.LayerNorm(64)
            self.ln3 = nn.LayerNorm(num_classes)

        if self.residual:
            self.proj1 = nn.Linear(embedding_dim, 128)
            self.proj2 = nn.Linear(128, 64)
            self.proj3 = nn.Linear(64, num_classes)

        self.activation = activation_fn()
        self.dropout1 = nn.Dropout(dropout_rate)
        self.dropout2 = nn.Dropout(dropout_rate * 1.5)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        x = self.input_norm(features)

        if self.residual:
            x = self._forward_with_residual(x)
        else:
            x = self._forward_standard(x)

        return torch.sigmoid(x)

    def _forward_with_residual(self, x: torch.Tensor) -> torch.Tensor:
        identity = self.proj1(x)
        x = self.fc1(x)
        if self.use_layer_norm:
            x = self.ln1(x)
        x = self.activation(x)
        x = self.dropout1(x)
        x = x + identity  # This one is necessary

        identity = self.proj2(x)
        x = self.fc2(x)
        if self.use_layer_norm:
            x = self.ln2(x)
        x = self.activation(x)
        x = self.dropout2(x)
        # x = x + identity #Removing this one leads to even better results smh

        identity = self.proj3(x)
        x = self.fc3(x)
        if self.use_layer_norm:
            x = self.ln3(x)
        x = self.activation(x)
        # x = x + identity Removing this layer is crucial for BBBC DINO representation to work

        return x

    def _forward_standard(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        if self.use_layer_norm:
            x = self.ln1(x)
        x = self.activation(x)
        x = self.dropout1(x)

        x = self.fc2(x)
        if self.use_layer_norm:
            x = self.ln2(x)
        x = self.activation(x)
        x = self.dropout2(x)

        x = self.fc3(x)
        if self.use_layer_norm:
            x = self.ln3(x)
        x = self.activation(x)

        return x
