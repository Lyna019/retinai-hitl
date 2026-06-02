"""
Evaluation metrics for multi-label retinal classification.

Includes: AUC (macro/micro/per-label), F1, precision, recall,
specificity, Brier score, ECE, and per-label threshold optimization.
"""

from typing import Dict, List, Optional, Tuple

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    brier_score_loss,
)


def compute_all_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    label_names: List[str],
    threshold: float = 0.5,
    per_label_thresholds: Optional[Dict[str, float]] = None,
) -> Dict[str, float]:
    """
    Compute all evaluation metrics for multi-label classification.

    Args:
        y_true: Binary ground truth (N, C)
        y_prob: Predicted probabilities (N, C)
        label_names: List of class names
        threshold: Default threshold for binarization
        per_label_thresholds: Optional per-label thresholds (overrides default)

    Returns:
        Dict of metric_name → value
    """
    n_classes = y_true.shape[1]
    metrics = {}

    # --- AUC ---
    try:
        # Per-label AUC
        per_label_auc = {}
        for i, name in enumerate(label_names):
            if y_true[:, i].sum() > 0 and y_true[:, i].sum() < len(y_true):
                per_label_auc[name] = roc_auc_score(y_true[:, i], y_prob[:, i])
            else:
                per_label_auc[name] = float("nan")
        metrics["per_label_auc"] = per_label_auc

        # Macro AUC (excluding NaN labels)
        valid_aucs = [v for v in per_label_auc.values() if not np.isnan(v)]
        metrics["macro_auc"] = np.mean(valid_aucs) if valid_aucs else 0.0

        # Micro AUC
        metrics["micro_auc"] = roc_auc_score(y_true, y_prob, average="micro")
    except ValueError as e:
        metrics["macro_auc"] = 0.0
        metrics["micro_auc"] = 0.0
        metrics["per_label_auc"] = {name: 0.0 for name in label_names}

    # --- Binarize predictions ---
    if per_label_thresholds:
        y_pred = np.zeros_like(y_prob)
        for i, name in enumerate(label_names):
            t = per_label_thresholds.get(name, threshold)
            y_pred[:, i] = (y_prob[:, i] >= t).astype(int)
    else:
        y_pred = (y_prob >= threshold).astype(int)

    # --- F1 ---
    metrics["f1_macro"] = f1_score(y_true, y_pred, average="macro", zero_division=0)
    metrics["f1_micro"] = f1_score(y_true, y_pred, average="micro", zero_division=0)

    # Per-label F1
    per_label_f1 = f1_score(y_true, y_pred, average=None, zero_division=0)
    metrics["per_label_f1"] = {name: float(per_label_f1[i]) for i, name in enumerate(label_names)}

    # --- Precision / Recall ---
    metrics["precision_macro"] = precision_score(y_true, y_pred, average="macro", zero_division=0)
    metrics["recall_macro"] = recall_score(y_true, y_pred, average="macro", zero_division=0)

    # --- Specificity (per label, then macro) ---
    specificities = []
    for i in range(n_classes):
        tn = ((y_true[:, i] == 0) & (y_pred[:, i] == 0)).sum()
        fp = ((y_true[:, i] == 0) & (y_pred[:, i] == 1)).sum()
        spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        specificities.append(spec)
    metrics["specificity_macro"] = np.mean(specificities)

    # --- Brier Score (per label + macro) ---
    brier_scores = []
    for i, name in enumerate(label_names):
        bs = brier_score_loss(y_true[:, i], y_prob[:, i])
        brier_scores.append(bs)
    metrics["brier_score_macro"] = np.mean(brier_scores)
    metrics["per_label_brier"] = {name: brier_scores[i] for i, name in enumerate(label_names)}

    # --- ECE (Expected Calibration Error) ---
    metrics["ece"] = compute_ece(y_true, y_prob, n_bins=15)

    return metrics


def compute_ece(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bins: int = 15,
) -> float:
    """
    Expected Calibration Error for multi-label classification.
    Computed across all labels jointly.
    """
    # Flatten to treat each label independently
    y_true_flat = y_true.flatten()
    y_prob_flat = y_prob.flatten()

    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    total = len(y_true_flat)

    for i in range(n_bins):
        mask = (y_prob_flat >= bin_boundaries[i]) & (y_prob_flat < bin_boundaries[i + 1])
        n_in_bin = mask.sum()
        if n_in_bin == 0:
            continue
        avg_confidence = y_prob_flat[mask].mean()
        avg_accuracy = y_true_flat[mask].mean()
        ece += (n_in_bin / total) * abs(avg_accuracy - avg_confidence)

    return float(ece)


def optimize_thresholds(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    label_names: List[str],
    search_range: Tuple[float, float] = (0.1, 0.9),
    search_step: float = 0.01,
) -> Dict[str, float]:
    """
    Find optimal per-label thresholds that maximize F1 on validation set.

    Returns dict of label_name → optimal_threshold.
    """
    thresholds = {}
    for i, name in enumerate(label_names):
        best_t, best_f1 = 0.5, 0.0
        for t in np.arange(search_range[0], search_range[1], search_step):
            preds = (y_prob[:, i] >= t).astype(int)
            f1 = f1_score(y_true[:, i], preds, zero_division=0)
            if f1 > best_f1:
                best_t, best_f1 = t, f1
        thresholds[name] = float(best_t)
    return thresholds


def compute_reliability_diagram_data(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bins: int = 15,
) -> Dict[str, np.ndarray]:
    """
    Compute data for reliability diagram plotting.

    Returns dict with 'bin_centers', 'bin_accuracy', 'bin_confidence', 'bin_count'.
    """
    y_true_flat = y_true.flatten()
    y_prob_flat = y_prob.flatten()

    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    bin_centers = (bin_boundaries[:-1] + bin_boundaries[1:]) / 2
    bin_accuracy = np.zeros(n_bins)
    bin_confidence = np.zeros(n_bins)
    bin_count = np.zeros(n_bins)

    for i in range(n_bins):
        mask = (y_prob_flat >= bin_boundaries[i]) & (y_prob_flat < bin_boundaries[i + 1])
        n_in_bin = mask.sum()
        if n_in_bin > 0:
            bin_accuracy[i] = y_true_flat[mask].mean()
            bin_confidence[i] = y_prob_flat[mask].mean()
            bin_count[i] = n_in_bin

    return {
        "bin_centers": bin_centers,
        "bin_accuracy": bin_accuracy,
        "bin_confidence": bin_confidence,
        "bin_count": bin_count,
    }
