from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from torchcfm.conditional_flow_matching import (
    ConditionalFlowMatcher,
    ExactOptimalTransportConditionalFlowMatcher,
)

from lightning_models.base_model import BaseModel
from lightning_models.lightning_autoencoder import LightningAutoencoder
from lightning_models.lightning_cnn import LightningCNN
from models.cfm_velocity_net import CFMVelocityNet
from models.clip_extractor import CLIPExtractor

torch.manual_seed(42)


class IdentityBackbone(nn.Module):
    def extract_features(self, x):
        return x

    def forward(self, x):
        return x


class LightningCFMClassifier(BaseModel):
    """Generative (Bayes) multi-label classifier via Conditional Flow Matching.

    Instead of mapping features -> labels directly, this models the *features*
    generatively, conditioned on the labels: p(x | y). A velocity field
    v(x_t, t, y) is trained with CFM to transport noise -> feature profile along
    straight-line paths, conditioned on a ternary label vector y in {-1, 0, +1}^K
    (0 = unknown).

    Classification is then the Bayes route: for each label we measure how well
    the model explains the observed feature x under the hypothesis y_i = +1 vs
    y_i = -1 (all other labels set to 0 = unknown). The "fit" is the conditional
    CFM loss estimated by Monte Carlo; a lower loss means higher likelihood, so

        score_i = E_loss[ y_i = -1 ] - E_loss[ y_i = +1 ]

    is positive when label i is present. This is the cheap FM-loss scoring that
    works for any path/coupling choice (exact-likelihood ODE integration is the
    more expensive, more principled alternative left for later).

    The coupling between noise and data is selected with `cfm_method`:
      - "vanilla": independent coupling (baseline).
      - "ot":      minibatch optimal-transport coupling (straighter paths).
    """

    def __init__(
        self,
        num_classes: int,
        embedding_dim: int,
        lr: float,
        cfm_method: str = "vanilla",
        sigma: float = 0.0,
        masked_loss: bool = False,
        backbone_type: str = "none",
        backbone_ckpt_path: Optional[str] = None,
        clip_model_name: str = "ViT-B/32",
        hidden_dim: int = 512,
        cond_dim: int = 256,
        num_blocks: int = 6,
        dropout: float = 0.0,
        feature_dropout: float = 0.1,
        label_dropout: float = 0.3,
        num_score_samples: int = 10,
        test_score_samples: int = 25,
        weight_decay: float = 1e-4,
        ternary_labels: bool = False,
        t_power: float = 1.0,
    ):
        super().__init__()

        self.num_classes = num_classes
        self.embedding_dim = embedding_dim
        self.lr = lr
        self.weight_decay = weight_decay
        self.masked_loss = masked_loss
        self.feature_dropout = feature_dropout
        self.label_dropout = label_dropout
        self.num_score_samples = num_score_samples
        self.test_score_samples = test_score_samples
        self.ternary_labels = ternary_labels
        self.t_power = t_power

        if backbone_type == "cnn":
            if backbone_ckpt_path is None:
                raise ValueError(
                    "backbone_ckpt_path must be provided when backbone_type is 'cnn'"
                )
            self.backbone = LightningCNN.load_from_checkpoint(backbone_ckpt_path).eval()
        elif backbone_type == "autoencoder":
            if backbone_ckpt_path is None:
                raise ValueError(
                    "backbone_ckpt_path must be provided when backbone_type is 'autoencoder'"
                )
            self.backbone = LightningAutoencoder.load_from_checkpoint(
                backbone_ckpt_path
            ).eval()
        elif backbone_type == "clip":
            self.backbone = CLIPExtractor(model_name=clip_model_name).eval()
        elif backbone_type == "none":
            self.backbone = IdentityBackbone()
        else:
            raise ValueError(f"Unknown backbone type: {backbone_type}")

        # normalize features
        self.feature_norm = nn.LayerNorm(embedding_dim, elementwise_affine=False)

        self.model = CFMVelocityNet(
            feature_dim=embedding_dim,
            num_classes=num_classes,
            hidden_dim=hidden_dim,
            cond_dim=cond_dim,
            num_blocks=num_blocks,
            dropout=dropout,
        )

        # CFM path/coupling. With sigma=0 the conditional path is the exact
        # straight line x_t = (1 - t) x0 + t x1, u_t = x1 - x0.
        if cfm_method == "vanilla":
            self.fm = ConditionalFlowMatcher(sigma=sigma)
        elif cfm_method == "ot":
            self.fm = ExactOptimalTransportConditionalFlowMatcher(sigma=sigma)
        else:
            raise ValueError(
                f"Unknown cfm_method: {cfm_method!r} (use 'vanilla' or 'ot')"
            )
        self.cfm_method = cfm_method

        self.y_pred = []
        self.y_true = []

    # ------------------------------------------------------------------ utils
    def _sample_t(self, n: int, device: torch.device) -> torch.Tensor:
        """Path time sampler. t = U^t_power: t_power > 1 concentrates on low t
        (high noise), where the label conditioning is most informative — this
        is what pushes the velocity field to actually *use* y. t_power = 1 is
        the usual uniform sampling."""
        u = torch.rand(n, device=device)
        return u if self.t_power == 1.0 else u.pow(self.t_power)

    def _to_ternary(self, y: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Conditioning labels in {-1, 0, +1}: known-neg / unknown / known-pos."""
        if self.ternary_labels:
            return y  # dataset already provides {-1, 0, +1}
        return torch.where(mask.bool(), y * 2.0 - 1.0, torch.zeros_like(y))

    def _drop_labels(self, y_cond: torch.Tensor) -> torch.Tensor:
        """Per-entry label dropout to 0 (unknown).

        Teaches p(x | partial labels) so that the single-known-label
        conditionings used at scoring time are in-distribution.
        """
        if not self.training or self.label_dropout <= 0:
            return y_cond
        keep = (torch.rand_like(y_cond) >= self.label_dropout).float()
        return y_cond * keep

    # ------------------------------------------------------------------ scoring
    @torch.no_grad()
    def score_labels(
        self, features: torch.Tensor, num_samples: Optional[int] = None
    ) -> torch.Tensor:
        """Per-label FM-loss score (logit): positive => label present.

        For each label we compare the Monte-Carlo CFM loss under y_i = +1 vs
        y_i = -1 (other labels = 0). Returns a (B, K) tensor of score logits.
        """
        if num_samples is None:
            num_samples = self.num_score_samples

        device = features.device
        B, D = features.shape
        K = self.num_classes

        # (2K, K) hypotheses: rows [+e_0, -e_0, +e_1, -e_1, ...].
        hyp = torch.zeros(2 * K, K, device=device)
        idx = torch.arange(K, device=device)
        hyp[2 * idx, idx] = 1.0
        hyp[2 * idx + 1, idx] = -1.0

        loss_acc = torch.zeros(B, 2 * K, device=device)
        for _ in range(num_samples):
            t = self._sample_t(B, device)
            x0 = torch.randn(B, D, device=device)
            xt = (1.0 - t)[:, None] * x0 + t[:, None] * features
            ut = features - x0

            # Evaluate the velocity under all 2K hypotheses with the SAME (t, x0).
            xt_e = xt[:, None, :].expand(B, 2 * K, D).reshape(-1, D)
            ut_e = ut[:, None, :].expand(B, 2 * K, D).reshape(-1, D)
            t_e = t[:, None].expand(B, 2 * K).reshape(-1)
            y_e = hyp[None].expand(B, 2 * K, K).reshape(-1, K)

            v = self.model(xt_e, t_e, y_e)
            loss_acc += ((v - ut_e) ** 2).mean(dim=1).view(B, 2 * K)

        loss_acc /= num_samples
        loss_pos = loss_acc[:, 0::2]  # (B, K) loss when y_i = +1
        loss_neg = loss_acc[:, 1::2]  # (B, K) loss when y_i = -1
        return loss_neg - loss_pos  # logit: positive => label present

    # ------------------------------------------------------------------ steps
    def _cfm_loss(self, x: torch.Tensor, y_cond: torch.Tensor) -> torch.Tensor:
        x0 = torch.randn_like(x)
        t = self._sample_t(x.shape[0], x.device)
        if self.cfm_method == "ot":
            # OT reorders the batch into the optimal noise<->data coupling. The
            # conditioning labels MUST be reordered with the SAME plan, otherwise
            # the model trains on mismatched (features, label) pairs and learns to
            # ignore the conditioning. `guided_*` returns the reordered y1.
            t, xt, ut, _, y_cond = self.fm.guided_sample_location_and_conditional_flow(
                x0, x, y1=y_cond, t=t
            )
        else:
            t, xt, ut = self.fm.sample_location_and_conditional_flow(x0, x, t=t)
        v = self.model(xt, t, y_cond)
        return ((v - ut) ** 2).mean()

    def training_step(self, batch, batch_idx):
        x, y, mask = batch
        x = self.backbone.extract_features(x)
        x = self.feature_norm(x.float())
        x = F.dropout(x, p=self.feature_dropout, training=self.training)

        y_cond = self._drop_labels(self._to_ternary(y, mask))
        loss = self._cfm_loss(x, y_cond)
        self.log("train/loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y, mask = batch
        x = self.backbone.extract_features(x)
        x = self.feature_norm(x.float())
        loss = self._cfm_loss(x, self._to_ternary(y, mask))
        self.log("val/loss", loss, prog_bar=True)

        logits = self.score_labels(x, num_samples=self.num_score_samples)
        y_pred = torch.sigmoid(logits).cpu()
        y_true = y.cpu()
        if self.ternary_labels:
            y_true = (y_true > 0).float()

        mask_bool = mask.bool().cpu()
        y_true_flat = y_true[mask_bool]
        y_pred_flat = y_pred[mask_bool]

        if len(y_true_flat) > 0 and len(torch.unique(y_true_flat)) > 1:
            try:
                roc_auc = roc_auc_score(y_true_flat.numpy(), y_pred_flat.numpy())
                self.log("val/roc_auc", roc_auc, prog_bar=True)
            except ValueError:
                pass
        return loss

    def test_step(self, batch, batch_idx):
        x, y, mask = batch
        x = self.backbone.extract_features(x)
        x = self.feature_norm(x.float())
        logits = self.score_labels(x, num_samples=self.test_score_samples)
        y_pred = torch.sigmoid(logits).cpu()
        y_true = y.cpu()
        if self.ternary_labels:
            y_true = (y_true > 0).float()

        mask_bool = mask.bool().cpu()
        self.y_pred.append(y_pred[mask_bool])
        self.y_true.append(y_true[mask_bool])

    def on_test_start(self):
        self.y_pred = []
        self.y_true = []

    def on_test_epoch_end(self):
        y_true = torch.cat(self.y_true).numpy()
        y_pred = torch.cat(self.y_pred).numpy()

        metrics = {}
        if len(y_true) > 0 and len(np.unique(y_true)) > 1:
            try:
                y_pred_binary = (y_pred > 0.5).astype(np.float32)
                metrics["test/roc_auc"] = roc_auc_score(y_true, y_pred)
                metrics["test/accuracy"] = accuracy_score(y_true, y_pred_binary)
                metrics["test/precision"] = precision_score(
                    y_true, y_pred_binary, zero_division=0
                )
                metrics["test/recall"] = recall_score(
                    y_true, y_pred_binary, zero_division=0
                )
                metrics["test/f1_score"] = f1_score(
                    y_true, y_pred_binary, zero_division=0
                )
            except Exception as e:
                print(f"Error calculating metrics: {e}")

        print(f"Test Metrics: {metrics}")
        self.log_dict(metrics)

    def configure_optimizers(self):
        return torch.optim.AdamW(
            self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )

    def reset_params(self):
        self.y_pred = []
        self.y_true = []
