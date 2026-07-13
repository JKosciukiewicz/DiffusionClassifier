from models.cfm_velocity_net import AdaLNMLPBlock, GaussianFourierFeatures, modulate

import torch
import torch.nn as nn


class CARDVelocityNet(nn.Module):
    """Flow velocity field for CARD: v(y_t, t, x) -> velocity in label space.

    Flows Gaussian noise -> label vector conditioned on the feature vector x.
    This is the inverse of CFMVelocityNet: label space is the "data" dimension,
    feature space is the conditioning dimension.

    Because K << D the network is lightweight — hidden_dim operates on K-dim vectors.
    The feature vector x is encoded into cond_dim and drives every AdaLN block.
    """

    def __init__(
        self,
        feature_dim: int,
        num_classes: int,
        hidden_dim: int = 256,
        cond_dim: int = 256,
        num_blocks: int = 6,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.num_classes = num_classes

        self.feature_encoder = nn.Sequential(
            nn.Linear(feature_dim, cond_dim),
            nn.SiLU(),
            nn.Linear(cond_dim, cond_dim),
        )
        self.time_embed = nn.Sequential(
            GaussianFourierFeatures(cond_dim),
            nn.Linear(cond_dim, cond_dim),
            nn.SiLU(),
            nn.Linear(cond_dim, cond_dim),
        )

        self.label_proj = nn.Linear(num_classes, hidden_dim)
        self.blocks = nn.ModuleList([
            AdaLNMLPBlock(hidden_dim, cond_dim, dropout=dropout)
            for _ in range(num_blocks)
        ])
        self.final_norm = nn.LayerNorm(hidden_dim, elementwise_affine=False)
        self.final_modulation = nn.Sequential(nn.SiLU(), nn.Linear(cond_dim, 2 * hidden_dim))
        self.final_linear = nn.Linear(hidden_dim, num_classes)

        self._init_weights()

    def _init_weights(self):
        for block in self.blocks:
            nn.init.zeros_(block.modulation[-1].weight)
            nn.init.zeros_(block.modulation[-1].bias)
        nn.init.zeros_(self.final_modulation[-1].weight)
        nn.init.zeros_(self.final_modulation[-1].bias)
        nn.init.zeros_(self.final_linear.weight)
        nn.init.zeros_(self.final_linear.bias)

    def forward(self, y_t: torch.Tensor, t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        if t.ndim == 2:
            t = t.squeeze(1)
        c = self.time_embed(t.float()) + self.feature_encoder(x.float())
        h = self.label_proj(y_t.float())
        for block in self.blocks:
            h = block(h, c)
        shift, scale = self.final_modulation(c).chunk(2, dim=-1)
        h = modulate(self.final_norm(h), shift, scale)
        return self.final_linear(h)
