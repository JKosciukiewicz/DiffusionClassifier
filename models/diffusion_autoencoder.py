import math

import torch
import torch.nn as nn


class SinusoidalPositionEmbeddings(nn.Module):
    """Maps scalar timestep t to high-dimensional frequency space."""

    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings


class CrossAttentionBlock(nn.Module):
    """Fuses generated labels (Query) with features + time (Key/Value)."""

    def __init__(self, hidden_dim, cond_dim, num_heads=4, dropout=0.1):
        super().__init__()
        self.ln1 = nn.LayerNorm(hidden_dim)
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            kdim=cond_dim,
            vdim=cond_dim,
            num_heads=num_heads,
            batch_first=True,
            dropout=dropout,
        )
        self.ln2 = nn.LayerNorm(hidden_dim)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, cond_emb):
        x_seq = x.unsqueeze(1)
        cond_seq = cond_emb.unsqueeze(1)

        # Cross Attention (Residual)
        attn_out, _ = self.cross_attn(
            query=self.ln1(x_seq), key=cond_seq, value=cond_seq
        )
        x_seq = x_seq + self.dropout(attn_out)

        # MLP (Residual)
        mlp_out = self.mlp(self.ln2(x_seq))
        x_seq = x_seq + mlp_out

        return x_seq.squeeze(1)


class DiffusionAutoencoder(nn.Module):
    def __init__(
        self, feature_dim=4643, label_dim=5, dropout_rate=0.1, use_sigmoid=False
    ):
        super().__init__()
        self.feature_dim = feature_dim
        self.label_dim = label_dim
        self.use_sigmoid = use_sigmoid
        self.hidden_dim = 256

        # 1. Feature Bottleneck
        # FIX: Removed the initial 0.5 dropout which was corrupting the pre-extracted embeddings
        self.encoded_feature_dim = 128
        self.feature_encoder = nn.Sequential(
            nn.Linear(self.feature_dim, 512),
            nn.LayerNorm(512),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(512, self.encoded_feature_dim),
            nn.LayerNorm(self.encoded_feature_dim),
        )

        # 2. Time Embeddings
        self.time_dim = 128
        self.time_embed = SinusoidalPositionEmbeddings(self.time_dim)
        self.time_mlp = nn.Sequential(
            self.time_embed,
            nn.Linear(self.time_dim, self.time_dim),
            nn.GELU(),
            nn.Linear(self.time_dim, self.time_dim),
        )

        self.cond_dim = self.encoded_feature_dim + self.time_dim

        # 3. Main Denoising Path (Processes noisy labels exclusively)
        self.input_projection = nn.Linear(self.label_dim, self.hidden_dim)

        self.blocks = nn.ModuleList(
            [
                CrossAttentionBlock(
                    self.hidden_dim, self.cond_dim, dropout=dropout_rate
                ),
                CrossAttentionBlock(
                    self.hidden_dim, self.cond_dim, dropout=dropout_rate
                ),
                CrossAttentionBlock(
                    self.hidden_dim, self.cond_dim, dropout=dropout_rate
                ),
            ]
        )

        self.output_layer = nn.Sequential(
            nn.LayerNorm(self.hidden_dim), nn.Linear(self.hidden_dim, self.label_dim)
        )

    def forward(self, features, noisy_labels, timesteps, force_uncond=False):
        # 1. Process Time
        if timesteps.ndim == 2:
            timesteps = timesteps.squeeze(1)
        t_emb = self.time_mlp(timesteps.float())

        # 2. Process Features
        feat_emb = self.feature_encoder(features)

        # FIX: Explicit CFG alignment toggled by the evaluation loop
        if force_uncond:
            # Replaces the embedding with strict zero tokens, matching training dropout state
            feat_emb = torch.zeros_like(feat_emb)
        elif self.training:
            # 15% chance to drop the entire feature context for this sample
            drop_mask = (
                torch.rand(feat_emb.shape[0], 1, device=feat_emb.device) > 0.15
            ).float()
            feat_emb = feat_emb * drop_mask

        # Combine Time and Features into cross-attention conditioning context
        cond_emb = torch.cat([t_emb, feat_emb], dim=1)

        # 3. Main Denoising Pass
        x = self.input_projection(noisy_labels)

        for block in self.blocks:
            x = block(x, cond_emb)

        out = self.output_layer(x)

        if self.use_sigmoid:
            out = torch.sigmoid(out)
        return out
