"""
Unified model factory.
Single entry point to build any architecture from config.
"""

from typing import Any, Dict

import torch.nn as nn

from src.models.backbone import build_single_model
from src.models.hybrid import build_hybrid_model


def build_model(config: Dict[str, Any]) -> nn.Module:
    """
    Build model from config dict.

    Dispatches to single-backbone or hybrid builder based on config['model']['family'].
    """
    model_cfg = config.get("model", config)
    family = model_cfg.get("family", "cnn")

    if family == "hybrid":
        model = build_hybrid_model(model_cfg)
    else:
        model = build_single_model(model_cfg)

    return model
