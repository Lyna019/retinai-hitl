"""
Loss functions for multi-label retinal classification.

- BCEWithLogitsLoss (weighted): baseline
- Focal Loss: emphasizes hard samples
- Asymmetric Loss (ASL): designed for multi-label imbalance (Ridnik et al. 2021)
"""

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class WeightedBCEWithLogitsLoss(nn.Module):
    """BCEWithLogitsLoss with per-class positive weights."""

    def __init__(self, pos_weight: Optional[torch.Tensor] = None):
        super().__init__()
        self.pos_weight = pos_weight

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return F.binary_cross_entropy_with_logits(
            logits, targets, pos_weight=self.pos_weight
        )


class FocalLoss(nn.Module):
    """
    Multi-label focal loss.
    Down-weights well-classified examples, focusing on hard negatives/positives.

    Args:
        gamma: Focusing parameter (γ=2 is standard).
        alpha: Per-class weights (balances positive/negative). If None, no weighting.
    """

    def __init__(
        self,
        gamma: float = 2.0,
        alpha: Optional[torch.Tensor] = None,
    ):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probs = torch.sigmoid(logits)
        ce_loss = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")

        # Focal modulation
        p_t = probs * targets + (1 - probs) * (1 - targets)
        focal_weight = (1 - p_t) ** self.gamma
        loss = focal_weight * ce_loss

        # Per-class weighting
        if self.alpha is not None:
            alpha = self.alpha.to(logits.device)
            alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
            loss = alpha_t * loss

        return loss.mean()


class AsymmetricLoss(nn.Module):
    """
    Asymmetric Loss for Multi-Label Classification (Ridnik et al., 2021).

    Uses different gamma values for positive and negative samples.
    γ- > γ+ → harder on negatives, which is what you want for imbalanced multi-label.

    Args:
        gamma_pos: Focusing param for positive samples (default 0 = no down-weighting)
        gamma_neg: Focusing param for negative samples (default 4 = strong down-weighting)
        clip: Probability clipping for negatives to prevent training on very easy negatives
    """

    def __init__(
        self,
        gamma_pos: float = 0.0,
        gamma_neg: float = 4.0,
        clip: float = 0.05,
    ):
        super().__init__()
        self.gamma_pos = gamma_pos
        self.gamma_neg = gamma_neg
        self.clip = clip

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probs = torch.sigmoid(logits)

        # Asymmetric clipping for negatives
        probs_neg = probs.clamp(min=self.clip)

        # Separate positive and negative losses
        loss_pos = targets * torch.log(probs.clamp(min=1e-8))
        loss_neg = (1 - targets) * torch.log((1 - probs_neg).clamp(min=1e-8))

        # Focal modulation with asymmetric gamma
        if self.gamma_pos > 0:
            loss_pos = loss_pos * ((1 - probs) ** self.gamma_pos)
        if self.gamma_neg > 0:
            loss_neg = loss_neg * (probs_neg ** self.gamma_neg)

        loss = -(loss_pos + loss_neg)
        return loss.mean()


def build_loss(
    config: dict,
    class_weights: Optional[torch.Tensor] = None,
) -> nn.Module:
    """
    Factory for loss functions.

    Args:
        config: Loss config from base.yaml
        class_weights: Inverse-frequency weights (from compute_class_weights)
    """
    name = config.get("name", "bce_weighted")

    if name == "bce_weighted":
        return WeightedBCEWithLogitsLoss(pos_weight=class_weights)

    elif name == "focal":
        gamma = config.get("focal_gamma", 2.0)
        return FocalLoss(gamma=gamma, alpha=class_weights)

    elif name == "asymmetric":
        return AsymmetricLoss(
            gamma_pos=config.get("asl_gamma_pos", 0.0),
            gamma_neg=config.get("asl_gamma_neg", 4.0),
        )

    else:
        raise ValueError(f"Unknown loss function: {name}")
