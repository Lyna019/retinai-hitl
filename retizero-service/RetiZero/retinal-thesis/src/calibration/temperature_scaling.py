"""
Temperature scaling for model calibration.
Learns a single temperature parameter T on the validation set
to improve predicted probability calibration.
"""

from typing import Dict, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import LBFGS

import wandb

from src.training.metrics import compute_ece, compute_reliability_diagram_data


class TemperatureScaling(nn.Module):
    """
    Post-hoc temperature scaling.

    Learns a single scalar T that divides logits before sigmoid.
    T > 1 → softer predictions (less overconfident)
    T < 1 → sharper predictions
    """

    def __init__(self):
        super().__init__()
        self.temperature = nn.Parameter(torch.ones(1) * 1.5)

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        return logits / self.temperature

    def calibrate(
        self,
        model: nn.Module,
        val_loader: torch.utils.data.DataLoader,
        device: torch.device,
        max_iter: int = 50,
    ) -> float:
        """
        Optimize temperature on validation set.

        Returns: learned temperature value.
        """
        model.eval()
        self.to(device)

        # Collect all logits and labels
        all_logits = []
        all_labels = []
        with torch.no_grad():
            for batch in val_loader:
                images = batch["image"].to(device)
                labels = batch["label"].to(device)
                logits = model(images)
                all_logits.append(logits)
                all_labels.append(labels)

        all_logits = torch.cat(all_logits, dim=0)
        all_labels = torch.cat(all_labels, dim=0)

        # Optimize temperature using LBFGS
        optimizer = LBFGS([self.temperature], lr=0.01, max_iter=max_iter)

        def closure():
            optimizer.zero_grad()
            scaled_logits = self(all_logits)
            loss = F.binary_cross_entropy_with_logits(scaled_logits, all_labels)
            loss.backward()
            return loss

        optimizer.step(closure)

        return self.temperature.item()


def calibrate_and_report(
    model: nn.Module,
    val_loader: torch.utils.data.DataLoader,
    test_loader: torch.utils.data.DataLoader,
    device: torch.device,
    label_names: list,
) -> Dict[str, float]:
    """
    Full calibration pipeline:
    1. Learn temperature on validation set
    2. Apply to test set
    3. Compute ECE, Brier score before/after calibration
    4. Generate reliability diagrams
    5. Log everything to W&B

    Returns: Dict with calibration metrics.
    """
    temp_scaler = TemperatureScaling()
    learned_temp = temp_scaler.calibrate(model, val_loader, device)
    print(f"[Calibration] Learned temperature: {learned_temp:.4f}")

    # Collect test predictions (before and after calibration)
    model.eval()
    temp_scaler.eval()
    all_logits = []
    all_labels = []

    with torch.no_grad():
        for batch in test_loader:
            images = batch["image"].to(device)
            labels = batch["label"]
            logits = model(images).cpu()
            all_logits.append(logits)
            all_labels.append(labels)

    logits = torch.cat(all_logits, dim=0)
    labels = torch.cat(all_labels, dim=0).numpy()

    # Before calibration
    probs_before = torch.sigmoid(logits).numpy()
    ece_before = compute_ece(labels, probs_before)

    # After calibration
    with torch.no_grad():
        scaled_logits = temp_scaler(logits.to(device)).cpu()
    probs_after = torch.sigmoid(scaled_logits).numpy()
    ece_after = compute_ece(labels, probs_after)

    # Reliability diagrams
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    for ax, probs, title, ece in [
        (ax1, probs_before, "Before Calibration", ece_before),
        (ax2, probs_after, "After Calibration", ece_after),
    ]:
        data = compute_reliability_diagram_data(labels, probs)
        ax.bar(
            data["bin_centers"],
            data["bin_accuracy"],
            width=1.0 / len(data["bin_centers"]),
            alpha=0.6,
            label="Accuracy",
            edgecolor="black",
        )
        ax.plot([0, 1], [0, 1], "r--", label="Perfect calibration")
        ax.set_xlabel("Predicted Probability")
        ax.set_ylabel("Observed Frequency")
        ax.set_title(f"{title}\nECE = {ece:.4f}")
        ax.legend()
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)

    plt.tight_layout()
    wandb.log({"calibration/reliability_diagram": wandb.Image(fig)})
    plt.close(fig)

    results = {
        "temperature": learned_temp,
        "ece_before": ece_before,
        "ece_after": ece_after,
        "ece_improvement": ece_before - ece_after,
    }

    wandb.log({f"calibration/{k}": v for k, v in results.items()})

    return results
