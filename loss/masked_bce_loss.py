import torch
import torch.nn as nn


class MaskedBCELoss(nn.Module):
    """
    Computes weighted binary cross entropy loss for imbalanced datasets,
    only for targets with known values (mask=1).

    Args:
        predictions: Model output probabilities (shape: [batch_size, num_classes])
        targets: Ground truth labels (shape: [batch_size, num_classes])
        mask: Binary mask indicating which targets are valid (shape: [batch_size, num_classes])

    Returns:
        Masked BCE loss averaged over valid targets

    Example:
        >>> criterion = MaskedBCELoss()
        >>> loss = criterion(predictions, targets, mask)
    """

    def __init__(self) -> None:
        super().__init__()

    def forward(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        device = predictions.device
        targets = targets.to(device)
        mask = mask.to(device)

        bce = torch.nn.functional.binary_cross_entropy(
            predictions, targets, reduction="none"
        )
        weighted_masked_bce = bce * mask

        return weighted_masked_bce.sum() / mask.sum()
