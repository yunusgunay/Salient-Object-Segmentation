# src/losses.py

# [FINAL REPORT CHANGE] Added BCEDiceLoss combining BCE and soft Dice.
# Original training used plain BCEWithLogitsLoss, which:
#   - For CNN: caused 7/10 epochs of all-background predictions due to class imbalance
#   - For U-Net: optimized pixel-wise error but not spatial overlap, causing over-segmentation
# BCEDiceLoss directly optimizes the Dice coefficient (same metric used for model selection),
# while BCE with pos_weight counteracts foreground/background imbalance.

import torch
import torch.nn as nn


class BCEDiceLoss(nn.Module):
    """
    Weighted combination of BCEWithLogitsLoss and soft Dice loss.

    bce_weight: contribution of BCE term (1 - bce_weight goes to Dice)
    pos_weight: passed to BCEWithLogitsLoss to upweight the foreground class.
                Set to ~2.0 for ECSSD where background is roughly twice as frequent.
    """

    def __init__(self, bce_weight: float = 0.5, pos_weight: torch.Tensor = None):
        super().__init__()
        self.bce_weight = bce_weight
        self.bce = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        bce_loss = self.bce(logits, targets)

        probs = torch.sigmoid(logits)
        smooth = 1e-6
        # Sum over spatial dims, keep batch dim
        intersection = (probs * targets).sum(dim=(1, 2, 3))
        union = probs.sum(dim=(1, 2, 3)) + targets.sum(dim=(1, 2, 3))
        dice_loss = 1.0 - (2.0 * intersection + smooth) / (union + smooth)

        return self.bce_weight * bce_loss + (1.0 - self.bce_weight) * dice_loss.mean()
