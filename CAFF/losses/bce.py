import torch
import torch.nn as nn


def bce_loss(logits: torch.Tensor, labels: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    criterion = nn.BCEWithLogitsLoss(reduction='none')
    loss = criterion(logits, labels)
    masked_loss = loss * mask
    return masked_loss.sum() / mask.sum().clamp(min=1.0)
