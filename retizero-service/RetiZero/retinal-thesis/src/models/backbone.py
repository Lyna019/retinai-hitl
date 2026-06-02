"""
Single-backbone model factory.
Builds ResNet, DenseNet, EfficientNet, ConvNeXt, ViT, Swin from timm.
"""

from typing import Any, Dict, Optional

import timm
import torch
import torch.nn as nn


class SingleBackboneModel(nn.Module):
    """
    Wrapper around a timm backbone for multi-label classification.
    Replaces the classification head with a custom sigmoid-based head.
    """

    def __init__(
        self,
        backbone_name: str,
        num_classes: int = 8,
        pretrained: bool = True,
        drop_path_rate: float = 0.0,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.backbone_name = backbone_name
        self.num_classes = num_classes

        # Create backbone from timm
        self.backbone = timm.create_model(
            backbone_name,
            pretrained=pretrained,
            num_classes=0,  # Remove default head → returns features
            drop_path_rate=drop_path_rate,
        )

        # Get feature dimension from backbone
        self.feature_dim = self.backbone.num_features

        # Custom classification head
        self.head = nn.Sequential(
            nn.LayerNorm(self.feature_dim),
            nn.Dropout(dropout),
            nn.Linear(self.feature_dim, num_classes),
        )

        # For Grad-CAM: store reference to the last feature layer
        self._gradcam_target = self._find_gradcam_target()

    def _find_gradcam_target(self) -> Optional[nn.Module]:
        """Auto-detect the best layer for Grad-CAM based on architecture family."""
        backbone = self.backbone

        # CNN-based: last conv layer before pooling
        for attr in ["layer4", "features", "stages"]:
            if hasattr(backbone, attr):
                module = getattr(backbone, attr)
                if isinstance(module, (nn.Sequential, nn.ModuleList)):
                    return module[-1]

        # Transformer-based: last block/layer
        for attr in ["blocks", "layers"]:
            if hasattr(backbone, attr):
                module = getattr(backbone, attr)
                if isinstance(module, (nn.Sequential, nn.ModuleList)):
                    return module[-1]

        return None

    @property
    def gradcam_target_layer(self) -> Optional[nn.Module]:
        """Return the target layer for Grad-CAM visualization."""
        return self._gradcam_target

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass. Returns raw logits (no sigmoid).
        Sigmoid is applied in the loss function (BCEWithLogitsLoss).
        """
        features = self.backbone(x)  # (B, feature_dim)
        logits = self.head(features)  # (B, num_classes)
        return logits

    def get_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extract features before the classification head (for ensemble stacking)."""
        return self.backbone(x)

    def freeze_backbone(self) -> None:
        """Freeze all backbone parameters (for warmup training)."""
        for param in self.backbone.parameters():
            param.requires_grad = False

    def unfreeze_backbone(self) -> None:
        """Unfreeze all backbone parameters."""
        for param in self.backbone.parameters():
            param.requires_grad = True


# ============================================================
# timm backbone name mapping
# ============================================================

BACKBONE_REGISTRY = {
    "resnet50": "resnet50",
    "densenet121": "densenet121",
    "efficientnet_b4": "efficientnet_b4",
    "efficientnet_b5": "efficientnet_b5",
    "convnext_base": "convnext_base.fb_in22k_ft_in1k",
    "vit_base_patch16_224": "vit_base_patch16_224.augreg_in21k_ft_in1k",
    "swin_base_patch4_window7_224": "swin_base_patch4_window7_224.ms_in22k_ft_in1k",
}


def build_single_model(config: Dict[str, Any]) -> SingleBackboneModel:
    """
    Factory function for single-backbone models.

    Args:
        config: Model config dict with 'backbone', 'num_classes', etc.
    """
    backbone_key = config["backbone"]
    backbone_name = BACKBONE_REGISTRY.get(backbone_key, backbone_key)

    # ViT-specific overrides
    vit_cfg = config.get("vit", {})
    drop_path = vit_cfg.get("drop_path_rate", 0.0)

    model = SingleBackboneModel(
        backbone_name=backbone_name,
        num_classes=config.get("num_classes", 8),
        pretrained=config.get("pretrained", "imagenet") is not None,
        drop_path_rate=drop_path,
    )

    return model
