import math

import torch
import torch.nn as nn


class GaussianFourierFeatures(nn.Module):
    """Random Fourier features for continuous time t in [0, 1].

    Avoids the discrete-step `t * 1000` hack: a fixed bank of Gaussian
    frequencies maps the scalar t to a smooth high-dimensional embedding sized
    for the [0, 1] range.
    """

    def __init__(self, dim, scale=4.0):
        super().__init__()
        assert dim % 2 == 0
        self.register_buffer("freqs", torch.randn(dim // 2) * scale)

    def forward(self, t):
        proj = t[:, None] * self.freqs[None, :] * 2.0 * math.pi
        return torch.cat((proj.sin(), proj.cos()), dim=-1)


def modulate(x, shift, scale):
    """adaLN modulation. x: (B, T, C); shift/scale: (B, C)."""
    return x * (1.0 + scale.unsqueeze(1)) + shift.unsqueeze(1)


class DiTBlock(nn.Module):
    """DiT-style block with adaLN-Zero time conditioning.

    Label tokens attend to each other (self-attention, capturing label
    co-occurrence) and to the feature tokens (cross-attention). Time conditions
    the block by modulating the scale/shift/gate of each sublayer. With the
    modulation linear zero-initialized, every block starts as an identity map.
    """

    def __init__(self, hidden_dim, num_heads=8, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_dim, elementwise_affine=False)
        self.self_attn = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            batch_first=True,
            dropout=dropout,
        )

        self.norm2 = nn.LayerNorm(hidden_dim, elementwise_affine=False)
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            batch_first=True,
            dropout=dropout,
        )

        self.norm3 = nn.LayerNorm(hidden_dim, elementwise_affine=False)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
        )

        # adaLN: shift/scale/gate for self-attn, cross-attn and MLP (9 vectors).
        self.modulation = nn.Sequential(
            nn.SiLU(), nn.Linear(hidden_dim, 9 * hidden_dim)
        )

    def forward(self, x, feat_tokens, t_emb):
        (
            shift_sa,
            scale_sa,
            gate_sa,
            shift_ca,
            scale_ca,
            gate_ca,
            shift_mlp,
            scale_mlp,
            gate_mlp,
        ) = self.modulation(t_emb).chunk(9, dim=1)

        h = modulate(self.norm1(x), shift_sa, scale_sa)
        attn_out, _ = self.self_attn(h, h, h)
        x = x + gate_sa.unsqueeze(1) * attn_out

        h = modulate(self.norm2(x), shift_ca, scale_ca)
        cross_out, _ = self.cross_attn(query=h, key=feat_tokens, value=feat_tokens)
        x = x + gate_ca.unsqueeze(1) * cross_out

        h = modulate(self.norm3(x), shift_mlp, scale_mlp)
        x = x + gate_mlp.unsqueeze(1) * self.mlp(h)

        return x


class FlowMatchingAutoencoder(nn.Module):
    """Velocity field for Conditional (Rectified) Flow Matching over label vectors.

    Given a point x_t on the linear path between noise x0 ~ N(0, I) and the data
    endpoint x1 (labels in [-1, 1]) at continuous time t in [0, 1], predicts the
    velocity v(x_t, t, features) that transports x0 toward x1. The target velocity
    of the straight-line path is the constant (x1 - x0).

    Architecture (DiT-style):
      - Each label scalar becomes its own token (+ learned positional embedding),
        so the model represents labels individually and learns their correlations.
      - Features are projected into several tokens used as cross-attention context.
      - Time conditions every block via adaLN-Zero (scale/shift/gate modulation).
      - CFG uses a learned null feature-token set for the unconditional path.
    """

    def __init__(
        self,
        feature_dim=4643,
        label_dim=5,
        dropout_rate=0.1,
        num_blocks=6,
        num_feature_tokens=8,
        num_heads=8,
    ):
        super().__init__()
        self.feature_dim = feature_dim
        self.label_dim = label_dim
        self.hidden_dim = 256
        self.num_feature_tokens = num_feature_tokens

        # 1. Feature tokens (cross-attention context).
        self.feature_encoder = nn.Sequential(
            nn.Linear(self.feature_dim, 512),
            nn.LayerNorm(512),
            nn.SiLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(512, num_feature_tokens * self.hidden_dim),
        )
        self.feature_norm = nn.LayerNorm(self.hidden_dim)
        # Learned null context for the unconditional (CFG) path.
        self.null_tokens = nn.Parameter(
            torch.randn(num_feature_tokens, self.hidden_dim) * 0.02
        )

        # 2. Continuous-time embedding (Fourier features -> MLP).
        self.time_dim = 256
        self.time_mlp = nn.Sequential(
            GaussianFourierFeatures(self.time_dim),
            nn.Linear(self.time_dim, self.hidden_dim),
            nn.SiLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
        )

        # 3. Label tokens: one token per label scalar + learned positions.
        self.label_embed = nn.Linear(1, self.hidden_dim)
        self.label_pos = nn.Parameter(torch.randn(label_dim, self.hidden_dim) * 0.02)

        self.blocks = nn.ModuleList(
            [
                DiTBlock(self.hidden_dim, num_heads=num_heads, dropout=dropout_rate)
                for _ in range(num_blocks)
            ]
        )

        # 4. Per-token velocity head with a final adaLN modulation.
        self.final_norm = nn.LayerNorm(self.hidden_dim, elementwise_affine=False)
        self.final_modulation = nn.Sequential(
            nn.SiLU(), nn.Linear(self.hidden_dim, 2 * self.hidden_dim)
        )
        self.final_linear = nn.Linear(self.hidden_dim, 1)

        self._init_weights()

    def _init_weights(self):
        # adaLN-Zero: zero the modulation outputs so each block (and the head)
        # starts as an identity map; the velocity field starts at zero.
        for block in self.blocks:
            nn.init.zeros_(block.modulation[-1].weight)
            nn.init.zeros_(block.modulation[-1].bias)
        nn.init.zeros_(self.final_modulation[-1].weight)
        nn.init.zeros_(self.final_modulation[-1].bias)
        nn.init.zeros_(self.final_linear.weight)
        nn.init.zeros_(self.final_linear.bias)

    def forward(self, features, x_t, t, force_uncond=False):
        if t.ndim == 2:
            t = t.squeeze(1)
        batch_size = x_t.shape[0]

        # 1. Time conditioning context.
        t_emb = self.time_mlp(t.float())

        # 2. Feature tokens, with learned-null substitution for CFG.
        feat_tokens = self.feature_encoder(features).view(
            batch_size, self.num_feature_tokens, self.hidden_dim
        )
        feat_tokens = self.feature_norm(feat_tokens)

        null = self.null_tokens.unsqueeze(0).expand(batch_size, -1, -1)
        if force_uncond:
            feat_tokens = null
        elif self.training:
            drop = torch.rand(batch_size, device=feat_tokens.device) < 0.15
            feat_tokens = torch.where(drop[:, None, None], null, feat_tokens)

        # 3. Label tokens.
        x = self.label_embed(x_t.unsqueeze(-1)) + self.label_pos.unsqueeze(0)

        for block in self.blocks:
            x = block(x, feat_tokens, t_emb)

        # 4. Per-token velocity (unbounded).
        shift, scale = self.final_modulation(t_emb).chunk(2, dim=1)
        x = modulate(self.final_norm(x), shift, scale)
        return self.final_linear(x).squeeze(-1)
