"""
Reproducibility, logging, and general utilities.
"""

import os
import random
from typing import Dict, List, Optional

import numpy as np
import torch


def set_seed(seed: int = 42) -> None:
    """Set all random seeds for full reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # Deterministic operations (slight perf hit, worth it for reproducibility)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)


def get_device() -> torch.device:
    """Get best available device."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def count_parameters(model: torch.nn.Module) -> Dict[str, int]:
    """Count trainable and total parameters."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {"total": total, "trainable": trainable, "frozen": total - trainable}


def compute_class_weights(
    labels: np.ndarray,
    cap: float = 10.0,
) -> np.ndarray:
    """
    Compute inverse-frequency class weights for multi-label, capped at `cap`.

    Args:
        labels: Binary label matrix (N, C)
        cap: Maximum weight to avoid extreme values for very rare classes
    """
    pos_counts = labels.sum(axis=0).astype(float)
    neg_counts = labels.shape[0] - pos_counts
    # Avoid division by zero
    pos_counts = np.maximum(pos_counts, 1.0)
    weights = neg_counts / pos_counts
    weights = np.minimum(weights, cap)
    return weights.astype(np.float32)
