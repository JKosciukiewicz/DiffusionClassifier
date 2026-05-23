import torch
import torch.nn as nn


class ResNetBlock(nn.Module):
    """Upgraded ResNet block with LayerNorm, Dropout, and Conditioning Injection."""

    def __init__(self, in_dim, out_dim, cond_dim=None, dropout=0.3):
        super().__init__()
        self.linear1 = nn.Linear(in_dim, out_dim)
        self.ln1 = nn.LayerNorm(out_dim)  # LayerNorm works much better for diffusion

        # Conditioning projection layer to map the conditioning embedding to the block's width
        if cond_dim is not None:
            self.cond_project = nn.Linear(cond_dim, out_dim)
        else:
            self.cond_project = None

        self.linear2 = nn.Linear(out_dim, out_dim)
        self.ln2 = nn.LayerNorm(out_dim)
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

        # Skip connection (projection if dimensions differ)
        self.shortcut = nn.Identity()
        if in_dim != out_dim:
            self.shortcut = nn.Linear(in_dim, out_dim)

    def forward(self, x, cond_emb=None):
        out = self.linear1(x)
        out = self.ln1(out)
        out = self.activation(out)
        out = self.dropout(out)

        # Inject conditioning context directly into the hidden state
        if cond_emb is not None and self.cond_project is not None:
            out = out + self.cond_project(cond_emb)

        out = self.linear2(out)
        out = self.ln2(out)
        out = self.dropout(out)

        out += self.shortcut(x)  # Residual connection
        out = self.activation(out)
        return out


class DiffusionAutoencoder(nn.Module):
    def __init__(
        self, feature_dim=4643, label_dim=5, dropout_rate=0.4, use_sigmoid=False
    ):
        super().__init__()
        self.feature_dim = feature_dim
        self.label_dim = label_dim
        self.use_sigmoid = use_sigmoid

        # 1. Feature Bottleneck: Compress the massive 4643D vector
        self.encoded_feature_dim = 128
        self.feature_encoder = nn.Sequential(
            nn.Dropout(0.5),  # Extreme input dropout to prevent fingerprinting
            nn.Linear(self.feature_dim, 512),
            nn.LayerNorm(512),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(512, self.encoded_feature_dim),
            nn.LayerNorm(self.encoded_feature_dim),
            nn.ReLU(),
        )

        # The new input to the diffusion U-Net is just encoded_features (128) + labels (5)
        self.input_dim = self.encoded_feature_dim + self.label_dim

        self.embedding_layer = nn.Linear(self.input_dim, 512)

        # Dedicated Time Embedding Network
        self.time_dim = 128
        self.time_mlp = nn.Sequential(
            nn.Linear(1, self.time_dim),
            nn.ReLU(),
            nn.Linear(self.time_dim, self.time_dim),
            nn.ReLU(),
        )

        # Combined Conditioning Dimension (Time + Features)
        self.cond_dim = self.time_dim + self.encoded_feature_dim

        # --- Encoder (Downsample) ---
        self.down1 = ResNetBlock(512, 512, cond_dim=self.cond_dim, dropout=dropout_rate)
        self.down2 = ResNetBlock(512, 256, cond_dim=self.cond_dim, dropout=dropout_rate)
        self.down3 = ResNetBlock(256, 128, cond_dim=self.cond_dim, dropout=dropout_rate)

        # --- Decoder (Upsample) ---
        self.up1 = ResNetBlock(
            128 + 256, 256, cond_dim=self.cond_dim, dropout=dropout_rate
        )
        self.up2 = ResNetBlock(
            256 + 512, 512, cond_dim=self.cond_dim, dropout=dropout_rate
        )
        self.up3 = ResNetBlock(
            512 + 512, 512, cond_dim=self.cond_dim, dropout=dropout_rate
        )

        self.output_layer = nn.Linear(512, self.label_dim)

    def forward(self, features, noisy_labels, timesteps):
        # Process timesteps
        timesteps_norm = timesteps.float() / 1000.0
        if timesteps_norm.ndim == 1:
            timesteps_norm = timesteps_norm.unsqueeze(1)
        t_emb = self.time_mlp(timesteps_norm)

        # Encode the massive feature vector into a compact bottleneck
        feat_emb = self.feature_encoder(features)

        # 2. Classifier-Free Guidance (Condition Dropout)
        # Randomly zero out the features 15% of the time during training ONLY
        if self.training:
            # Create a mask of 0s and 1s. 15% chance of being 0.
            drop_mask = (
                torch.rand(feat_emb.shape[0], 1, device=feat_emb.device) > 0.2
            ).float()
            feat_emb = feat_emb * drop_mask

        # Create the combined conditioning embedding
        cond_emb = torch.cat([t_emb, feat_emb], dim=1)

        # Combine the balanced representations
        x = torch.cat([feat_emb, noisy_labels], dim=1)
        x = self.embedding_layer(x)

        # --- U-Net Pass ---
        d1 = self.down1(x, cond_emb)
        d2 = self.down2(d1, cond_emb)
        d3 = self.down3(d2, cond_emb)

        u1 = self.up1(torch.cat([d3, d2], dim=1), cond_emb)
        u2 = self.up2(torch.cat([u1, d1], dim=1), cond_emb)
        u3 = self.up3(torch.cat([u2, x], dim=1), cond_emb)

        out = self.output_layer(u3)
        if self.use_sigmoid:
            out = torch.sigmoid(out)
        return out
