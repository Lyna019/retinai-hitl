"""
Hybrid CNN+Transformer architectures.

Hybrid A — Dual-Branch Fusion:
  CNN branch (e.g. EfficientNet-B4) + Transformer branch (e.g. Swin-Base)
  → Adaptive pooling → Concatenation → Fusion MLP → Sigmoid outputs

Hybrid B — CNN Backbone + Transformer Head:
  CNN backbone → Feature map → Reshape to sequence → Transformer encoder
  → [CLS] token → Linear head → Sigmoid outputs
"""

import math
from typing import Any, Dict, Optional

import timm
import torch
import torch.nn as nn

from src.models.backbone import BACKBONE_REGISTRY


# ============================================================
# Hybrid A: Dual-Branch Fusion
# ============================================================

class DualBranchHybrid(nn.Module):
    """
    Two pretrained branches (CNN + Transformer) with learned fusion.
    Each branch extracts features independently; features are concatenated
    and passed through a fusion MLP.
    """

    def __init__(
        self,
        cnn_name: str = "efficientnet_b4",
        transformer_name: str = "swin_base_patch4_window7_224.ms_in22k_ft_in1k",
        num_classes: int = 8,
        fusion_dim: int = 512,
        fusion_dropout: float = 0.3,
        pretrained: bool = True,
    ):
        super().__init__()

        # Resolve backbone names
        cnn_name = BACKBONE_REGISTRY.get(cnn_name, cnn_name)
        transformer_name = BACKBONE_REGISTRY.get(transformer_name, transformer_name)

        # CNN branch
        self.cnn_branch = timm.create_model(
            cnn_name, pretrained=pretrained, num_classes=0
        )
        self.cnn_dim = self.cnn_branch.num_features

        # Transformer branch
        self.transformer_branch = timm.create_model(
            transformer_name, pretrained=pretrained, num_classes=0
        )
        self.transformer_dim = self.transformer_branch.num_features

        # Fusion MLP
        concat_dim = self.cnn_dim + self.transformer_dim
        self.fusion = nn.Sequential(
            nn.LayerNorm(concat_dim),
            nn.Linear(concat_dim, fusion_dim),
            nn.GELU(),
            nn.Dropout(fusion_dropout),
            nn.Linear(fusion_dim, fusion_dim),
            nn.GELU(),
            nn.Dropout(fusion_dropout),
            nn.Linear(fusion_dim, num_classes),
        )

        self.num_classes = num_classes

        # For Grad-CAM: target the last layer of CNN branch (more interpretable)
        self._gradcam_target = self._find_last_conv(self.cnn_branch)

    def _find_last_conv(self, model: nn.Module) -> Optional[nn.Module]:
        """Find last convolutional/feature block for Grad-CAM."""
        for attr in ["layer4", "features", "stages"]:
            if hasattr(model, attr):
                module = getattr(model, attr)
                if isinstance(module, (nn.Sequential, nn.ModuleList)):
                    return module[-1]
        return None

    @property
    def gradcam_target_layer(self) -> Optional[nn.Module]:
        return self._gradcam_target

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        cnn_features = self.cnn_branch(x)          # (B, cnn_dim)
        trans_features = self.transformer_branch(x)  # (B, transformer_dim)
        combined = torch.cat([cnn_features, trans_features], dim=1)
        logits = self.fusion(combined)
        return logits

    def get_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extract concatenated features before fusion head."""
        cnn_features = self.cnn_branch(x)
        trans_features = self.transformer_branch(x)
        return torch.cat([cnn_features, trans_features], dim=1)

    def freeze_branches(self) -> None:
        """Freeze both branches (for warmup: only train fusion MLP)."""
        for param in self.cnn_branch.parameters():
            param.requires_grad = False
        for param in self.transformer_branch.parameters():
            param.requires_grad = False

    def unfreeze_branches(self) -> None:
        """Unfreeze both branches."""
        for param in self.cnn_branch.parameters():
            param.requires_grad = True
        for param in self.transformer_branch.parameters():
            param.requires_grad = True


# ============================================================
# Hybrid B: CNN Backbone + Transformer Encoder Head
# ============================================================

class CNNTransformerHead(nn.Module):
    """
    CNN backbone extracts spatial feature maps, which are reshaped into a
    sequence and processed by a lightweight Transformer encoder.
    A learnable [CLS] token aggregates global information.
    """

    def __init__(
        self,
        cnn_name: str = "efficientnet_b4",
        num_classes: int = 8,
        transformer_layers: int = 2,
        transformer_heads: int = 8,
        transformer_dim_ff: int = 2048,
        transformer_dropout: float = 0.1,
        pretrained: bool = True,
    ):
        super().__init__()

        cnn_name = BACKBONE_REGISTRY.get(cnn_name, cnn_name)

        # CNN backbone — extract feature maps (not pooled)
        self.cnn = timm.create_model(
            cnn_name,
            pretrained=pretrained,
            num_classes=0,
            global_pool="",  # No global pooling → keep spatial dims
        )

        # Determine feature map dimensions by running a dummy forward pass
        with torch.no_grad():
            dummy = torch.zeros(1, 3, 224, 224)
            feat = self.cnn(dummy)
            if feat.dim() == 4:
                # CNN output: (B, C, H, W)
                self.feat_channels = feat.shape[1]
                self.feat_h = feat.shape[2]
                self.feat_w = feat.shape[3]
                self.seq_len = self.feat_h * self.feat_w
                self.d_model = self.feat_channels
            elif feat.dim() == 3:
                # Some backbones return (B, seq_len, d_model)
                self.seq_len = feat.shape[1]
                self.d_model = feat.shape[2]
                self.feat_channels = self.d_model
                self.feat_h = self.feat_w = int(math.sqrt(self.seq_len))
            else:
                raise ValueError(f"Unexpected feature shape: {feat.shape}")

        # Learnable [CLS] token
        self.cls_token = nn.Parameter(torch.randn(1, 1, self.d_model) * 0.02)

        # Positional encoding
        self.pos_embed = nn.Parameter(
            torch.randn(1, self.seq_len + 1, self.d_model) * 0.02
        )

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.d_model,
            nhead=transformer_heads,
            dim_feedforward=transformer_dim_ff,
            dropout=transformer_dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=transformer_layers
        )

        # Classification head from [CLS] token
        self.head = nn.Sequential(
            nn.LayerNorm(self.d_model),
            nn.Dropout(0.3),
            nn.Linear(self.d_model, num_classes),
        )

        self.num_classes = num_classes

        # Grad-CAM target: last CNN feature layer
        self._gradcam_target = self._find_last_conv(self.cnn)

    def _find_last_conv(self, model: nn.Module) -> Optional[nn.Module]:
        for attr in ["layer4", "features", "stages"]:
            if hasattr(model, attr):
                module = getattr(model, attr)
                if isinstance(module, (nn.Sequential, nn.ModuleList)):
                    return module[-1]
        return None

    @property
    def gradcam_target_layer(self) -> Optional[nn.Module]:
        return self._gradcam_target

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B = x.shape[0]

        # CNN feature extraction
        feat = self.cnn(x)

        # Reshape to sequence
        if feat.dim() == 4:
            feat = feat.flatten(2).transpose(1, 2)  # (B, HW, C)
        # feat is now (B, seq_len, d_model)

        # Prepend [CLS] token
        cls_tokens = self.cls_token.expand(B, -1, -1)
        feat = torch.cat([cls_tokens, feat], dim=1)  # (B, seq_len+1, d_model)

        # Add positional encoding
        feat = feat + self.pos_embed[:, : feat.shape[1], :]

        # Transformer encoder
        feat = self.transformer_encoder(feat)

        # Extract [CLS] token → classify
        cls_out = feat[:, 0]
        logits = self.head(cls_out)

        return logits

    def freeze_cnn(self) -> None:
        for param in self.cnn.parameters():
            param.requires_grad = False

    def unfreeze_cnn(self) -> None:
        for param in self.cnn.parameters():
            param.requires_grad = True


# ============================================================
# Factory
# ============================================================

def build_hybrid_model(config: Dict[str, Any]) -> nn.Module:
    """
    Factory for hybrid models.

    Config must have 'hybrid' key with 'type' = 'dual_branch' or 'cnn_transformer_head'.
    """
    hybrid_cfg = config["hybrid"]
    hybrid_type = hybrid_cfg["type"]
    num_classes = config.get("num_classes", 8)

    if hybrid_type == "dual_branch":
        return DualBranchHybrid(
            cnn_name=hybrid_cfg["cnn_branch"],
            transformer_name=hybrid_cfg["transformer_branch"],
            num_classes=num_classes,
            fusion_dim=hybrid_cfg.get("fusion_dim", 512),
            fusion_dropout=hybrid_cfg.get("fusion_dropout", 0.3),
        )

    elif hybrid_type == "cnn_transformer_head":
        return CNNTransformerHead(
            cnn_name=hybrid_cfg["cnn_backbone"],
            num_classes=num_classes,
            transformer_layers=hybrid_cfg.get("transformer_layers", 2),
            transformer_heads=hybrid_cfg.get("transformer_heads", 8),
            transformer_dim_ff=hybrid_cfg.get("transformer_dim_ff", 2048),
            transformer_dropout=hybrid_cfg.get("transformer_dropout", 0.1),
        )

    else:
        raise ValueError(f"Unknown hybrid type: {hybrid_type}")
