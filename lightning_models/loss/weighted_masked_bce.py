import torch


def weighted_masked_bce_loss(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    mask: torch.Tensor,
    pos_weight: float = 1,
    weighted: bool = False,
) -> torch.Tensor:
    """
    Computes weighted binary cross entropy loss for imbalanced datasets,
    only for targets with known values (mask=1).

    Args:
        predictions: Model output probabilities (shape: [batch_size, num_classes])
        targets: Ground truth labels (shape: [batch_size, num_classes])
        mask: Binary mask indicating which targets are valid (shape: [batch_size, num_classes])
        weighted: Weight for positive class.

    Returns:
        Masked weighted BCE loss averaged over valid targets
    """
    # Ensure all tensors are on the same device
    device = predictions.device
    targets = targets.to(device)
    mask = mask.to(device)

    # Default weight based on dataset distribution (from your statistics)
    if not weighted:
        pos_weight = 1

    # Create weight tensor where positive samples have higher weight
    weights = torch.ones_like(targets, device=device)
    weights[targets > 0] = pos_weight

    bce = torch.nn.functional.binary_cross_entropy(
        predictions, targets, reduction="none"
    )

    weighted_masked_bce = bce * weights * mask

    # Add small epsilon to avoid division by zero
    mask_sum = mask.sum() + 1e-12
    # if mask_sum == 0:
    #     return torch.tensor(0.0, device=device, requires_grad=True)

    return weighted_masked_bce.sum() / mask_sum
