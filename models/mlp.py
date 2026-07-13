from typing import List, Optional, Sequence

import torch.nn as nn

# Original fixed architecture — kept as the default so existing scripts are unchanged.
DEFAULT_HIDDEN_DIMS = (256, 128, 64, 32, 16)


class MLPClassifier(nn.Module):
    """Residual MLP. Each hidden layer is Linear + a projected residual, then ReLU."""

    def __init__(
        self,
        num_classes: int = 10,
        embedding_dim: int = 128,
        hidden_dims: Optional[Sequence[int]] = None,
        dropout: float = 0.0,
    ):
        super().__init__()
        hidden_dims: List[int] = list(hidden_dims or DEFAULT_HIDDEN_DIMS)
        dims = [embedding_dim, *hidden_dims]

        self.layers = nn.ModuleList(
            nn.Linear(dims[i], dims[i + 1]) for i in range(len(hidden_dims))
        )
        # Projections make the residual add work across changing widths.
        self.res_projs = nn.ModuleList(
            nn.Linear(dims[i], dims[i + 1]) for i in range(len(hidden_dims))
        )
        self.out = nn.Linear(dims[-1], num_classes)

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.sigmoid = nn.Sigmoid()

    def forward(self, features):
        x = features
        for fc, res_proj in zip(self.layers, self.res_projs):
            x = self.relu(fc(x) + res_proj(x))
            x = self.dropout(x)
        return self.sigmoid(self.out(x))
