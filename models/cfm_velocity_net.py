import math

import torch
import torch.nn as nn


class GaussianFourierFeatures(nn.Module):
    """Sinusoidal features for continuous time t in [0, 1].

    A fixed bank of Gaussian frequencies maps the scalar t to sin/cos features.
    This is the continuous-time analogue of the usual sinusoidal timestep
    embedding (no `t * 1000` discretization hack).
    """

    def __init__(self, dim, scale=4.0):
        super().__init__()
        assert dim % 2 == 0
        self.register_buffer("freqs", torch.randn(dim // 2) * scale)

    def forward(self, t):
        proj = t[:, None] * self.freqs[None, :] * 2.0 * math.pi
        return torch.cat((proj.sin(), proj.cos()), dim=-1)


def modulate(x, shift, scale):
    """adaLN modulation for a flat (B, C) vector."""
    return x * (1.0 + scale) + shift


class AdaLNMLPBlock(nn.Module):
    """Residual MLP block with AdaLN-Zero conditioning.

    The (time + label) conditioning vector regresses a shift/scale/gate for the
    block. With the modulation linear zero-initialized the gate starts at 0, so
    the block starts as an identity map and a deep stack trains stably from the
    start. This is the DiT adaLN-Zero idea lifted onto a plain MLP — strong,
    stable t+class conditioning without attention or patches.
    """

    def __init__(self, hidden_dim, cond_dim, mlp_ratio=4, dropout=0.0):
        super().__init__()
        self.norm = nn.LayerNorm(hidden_dim, elementwise_affine=False)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * mlp_ratio),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * mlp_ratio, hidden_dim),
        )
        # shift / scale / gate for the MLP sublayer.
        self.modulation = nn.Sequential(nn.SiLU(), nn.Linear(cond_dim, 3 * hidden_dim))

    def forward(self, x, c):
        shift, scale, gate = self.modulation(c).chunk(3, dim=-1)
        h = modulate(self.norm(x), shift, scale)
        return x + gate * self.mlp(h)


class CFMVelocityNet(nn.Module):
    """AdaLN-Zero MLP velocity field for Conditional Flow Matching over feature profiles.

    Models p(features | labels): predicts the velocity v(x_t, t, y) that
    transports Gaussian noise x0 ~ N(0, I) at t=0 to a feature profile x1 at
    t=1 along the straight-line path, conditioned on a (ternary) multi-label
    vector y in {-1, 0, +1}^K (0 = "unknown").

    "DiT-flavored MLP": the time and label embeddings are summed into a single
    conditioning vector that AdaLN-modulates every residual MLP block.
    """

    def __init__(
        self,
        feature_dim,
        num_classes,
        hidden_dim=512,
        cond_dim=256,
        num_blocks=6,
        mlp_ratio=4,
        dropout=0.0,
    ):
        super().__init__()
        self.feature_dim = feature_dim
        self.num_classes = num_classes

        # Continuous-time embedding (sinusoidal Fourier features -> MLP).
        self.time_embed = nn.Sequential(
            GaussianFourierFeatures(cond_dim),
            nn.Linear(cond_dim, cond_dim),
            nn.SiLU(),
            nn.Linear(cond_dim, cond_dim),
        )
        # Label conditioning: ternary {-1, 0, +1}^K -> conditioning vector.
        self.label_embed = nn.Sequential(
            nn.Linear(num_classes, cond_dim),
            nn.SiLU(),
            nn.Linear(cond_dim, cond_dim),
        )

        self.input_proj = nn.Linear(feature_dim, hidden_dim)
        self.blocks = nn.ModuleList(
            [
                AdaLNMLPBlock(hidden_dim, cond_dim, mlp_ratio, dropout)
                for _ in range(num_blocks)
            ]
        )

        self.final_norm = nn.LayerNorm(hidden_dim, elementwise_affine=False)
        self.final_modulation = nn.Sequential(
            nn.SiLU(), nn.Linear(cond_dim, 2 * hidden_dim)
        )
        self.final_linear = nn.Linear(hidden_dim, feature_dim)

        self._init_weights()

    def _init_weights(self):
        # AdaLN-Zero: zero the modulation outputs so every block (and the head)
        # starts as an identity map and the velocity field starts at zero.
        for block in self.blocks:
            nn.init.zeros_(block.modulation[-1].weight)
            nn.init.zeros_(block.modulation[-1].bias)
        nn.init.zeros_(self.final_modulation[-1].weight)
        nn.init.zeros_(self.final_modulation[-1].bias)
        nn.init.zeros_(self.final_linear.weight)
        nn.init.zeros_(self.final_linear.bias)

    def forward(self, x_t, t, y):
        if t.ndim == 2:
            t = t.squeeze(1)
        # Combined time + label conditioning drives every block's adaLN.
        c = self.time_embed(t.float()) + self.label_embed(y.float())

        h = self.input_proj(x_t)
        for block in self.blocks:
            h = block(h, c)

        shift, scale = self.final_modulation(c).chunk(2, dim=-1)
        h = modulate(self.final_norm(h), shift, scale)
        return self.final_linear(h)
