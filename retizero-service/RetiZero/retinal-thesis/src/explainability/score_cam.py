"""
Score-CAM: gradient-free class activation mapping.

Referenced in thesis §1.6.2 — used by Huang et al. (Swin-MCSFNet) for RIQA.
Unlike Grad-CAM, weights are computed by forward-passing each activation map
as a mask and measuring its effect on the target score. This avoids gradient
saturation issues and provides more faithful attribution.

Reference: Wang et al., "Score-CAM: Score-Weighted Visual Explanations
for Convolutional Neural Networks", CVPR 2020.
"""

from typing import Optional

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class ScoreCAM:
    """
    Score-CAM implementation for multi-label classification.

    For each activation channel k:
      1. Upsample activation map A_k to input resolution
      2. Normalize A_k to [0, 1]
      3. Mask input: X_masked = X ⊙ A_k
      4. Forward pass → get class score S_k for target class
      5. Weight = S_k (the increase in confidence caused by that channel)
    Final CAM = ReLU(Σ weight_k * A_k)
    """

    def __init__(self, model: nn.Module, target_layer: nn.Module):
        self.model = model
        self.target_layer = target_layer
        self.activations = None
        self._register_hook()

    def _register_hook(self) -> None:
        def hook_fn(module, input, output):
            if isinstance(output, torch.Tensor):
                self.activations = output.detach()
            elif isinstance(output, tuple):
                self.activations = output[0].detach()
        self.target_layer.register_forward_hook(hook_fn)

    @torch.no_grad()
    def generate(
        self,
        image: torch.Tensor,
        target_class: int,
        batch_size: int = 16,
    ) -> np.ndarray:
        """
        Generate Score-CAM heatmap.

        Args:
            image: (1, C, H, W) input tensor
            target_class: class index
            batch_size: how many channels to process in parallel

        Returns:
            Heatmap (H, W), values in [0, 1]
        """
        self.model.eval()
        device = image.device
        H, W = image.shape[2], image.shape[3]

        # Forward pass to capture activations
        logits = self.model(image)
        baseline_score = torch.sigmoid(logits[0, target_class]).item()

        if self.activations is None:
            return np.zeros((H, W))

        acts = self.activations.squeeze(0)  # (C_feat, h, w)

        # Handle transformer 3D output → reshape to spatial
        if acts.dim() == 2:
            N, C = acts.shape
            h = w = int(np.sqrt(N))
            if h * w != N:
                acts = acts[1:, :]  # remove [CLS]
                N -= 1
                h = w = int(np.sqrt(N))
            acts = acts.reshape(h, w, C).permute(2, 0, 1)  # (C, h, w)

        n_channels = acts.shape[0]

        # Upsample all activation maps to input resolution
        acts_upsampled = F.interpolate(
            acts.unsqueeze(0), size=(H, W), mode="bilinear", align_corners=False
        ).squeeze(0)  # (C_feat, H, W)

        # Normalize each channel to [0, 1]
        for k in range(n_channels):
            a_min = acts_upsampled[k].min()
            a_max = acts_upsampled[k].max()
            if a_max - a_min > 1e-8:
                acts_upsampled[k] = (acts_upsampled[k] - a_min) / (a_max - a_min)
            else:
                acts_upsampled[k] = 0.0

        # Compute scores by masking input with each activation map
        scores = torch.zeros(n_channels, device=device)
        for start in range(0, n_channels, batch_size):
            end = min(start + batch_size, n_channels)
            masks = acts_upsampled[start:end].unsqueeze(1)  # (bs, 1, H, W)
            masked_inputs = image * masks  # broadcast (bs, C, H, W)
            batch_logits = self.model(masked_inputs)
            batch_scores = torch.sigmoid(batch_logits[:, target_class])
            scores[start:end] = batch_scores

        # Scores become channel weights
        weights = F.relu(scores - baseline_score)  # Only keep channels that increase confidence

        # Weighted combination
        cam = (weights.view(-1, 1, 1) * acts_upsampled).sum(dim=0)
        cam = F.relu(cam).cpu().numpy()

        if cam.max() > 0:
            cam = cam / cam.max()

        cam = cv2.resize(cam, (W, H))
        return cam
