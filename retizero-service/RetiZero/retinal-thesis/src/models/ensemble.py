"""
Ensemble strategies: simple averaging and learned stacking.
"""

from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
import torch.nn as nn
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold

import joblib


class AveragingEnsemble:
    """
    Simple averaging of sigmoid outputs from multiple models.
    No additional training needed.
    """

    def __init__(self, models: List[nn.Module]):
        self.models = models

    @torch.no_grad()
    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """Average sigmoid outputs across all models."""
        outputs = []
        for model in self.models:
            model.eval()
            logits = model(x)
            probs = torch.sigmoid(logits)
            outputs.append(probs)
        return torch.stack(outputs).mean(dim=0)


class StackingEnsemble:
    """
    Learned stacking: logistic regression on concatenated logits.
    Trained on validation set using 5-fold CV.
    """

    def __init__(
        self,
        models: List[nn.Module],
        num_classes: int = 8,
    ):
        self.models = models
        self.num_classes = num_classes
        # One logistic regression per label (multi-label = independent binary)
        self.meta_learners: List[LogisticRegression] = []

    @torch.no_grad()
    def extract_logits(
        self,
        dataloader: torch.utils.data.DataLoader,
        device: torch.device,
    ) -> tuple:
        """Extract concatenated logits from all models on a dataset."""
        all_logits = []
        all_labels = []

        for batch in dataloader:
            images = batch["image"].to(device)
            labels = batch["label"].numpy()

            batch_logits = []
            for model in self.models:
                model.eval()
                logits = model(images).cpu().numpy()
                batch_logits.append(logits)

            # Concatenate logits from all models: (B, num_models * num_classes)
            concat = np.concatenate(batch_logits, axis=1)
            all_logits.append(concat)
            all_labels.append(labels)

        return np.vstack(all_logits), np.vstack(all_labels)

    def fit(
        self,
        val_loader: torch.utils.data.DataLoader,
        device: torch.device,
    ) -> None:
        """Train meta-learners on validation set logits."""
        logits, labels = self.extract_logits(val_loader, device)

        self.meta_learners = []
        for c in range(self.num_classes):
            lr = LogisticRegression(
                C=1.0, max_iter=1000, solver="lbfgs", random_state=42
            )
            lr.fit(logits, labels[:, c].astype(int))
            self.meta_learners.append(lr)

    @torch.no_grad()
    def predict(
        self,
        x: torch.Tensor,
        device: torch.device,
    ) -> torch.Tensor:
        """Predict using meta-learners on concatenated logits."""
        batch_logits = []
        for model in self.models:
            model.eval()
            logits = model(x).cpu().numpy()
            batch_logits.append(logits)

        concat = np.concatenate(batch_logits, axis=1)

        probs = np.zeros((concat.shape[0], self.num_classes))
        for c, lr in enumerate(self.meta_learners):
            probs[:, c] = lr.predict_proba(concat)[:, 1]

        return torch.from_numpy(probs).float().to(device)

    def save(self, path: str) -> None:
        joblib.dump(self.meta_learners, path)

    def load(self, path: str) -> None:
        self.meta_learners = joblib.load(path)
