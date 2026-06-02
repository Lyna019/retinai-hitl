"""
Grad-CAM heatmap generation.

This module handles GENERATION only. For quantitative validation,
see faithfulness.py and validation_pipeline.py.
"""

from typing import Dict, List, Optional

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import wandb


class GradCAM:
    """
    Grad-CAM for multi-label classification.
    Works with CNN (4D activations) and Transformer (3D activations).
    """

    def __init__(self, model: nn.Module, target_layer: nn.Module):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        self._register_hooks()

    def _register_hooks(self) -> None:
        def forward_hook(module, input, output):
            if isinstance(output, torch.Tensor):
                self.activations = output.detach()
            elif isinstance(output, tuple):
                self.activations = output[0].detach()

        def backward_hook(module, grad_input, grad_output):
            if isinstance(grad_output, tuple):
                self.gradients = grad_output[0].detach()
            else:
                self.gradients = grad_output.detach()

        self.target_layer.register_forward_hook(forward_hook)
        self.target_layer.register_full_backward_hook(backward_hook)

    def generate(
        self,
        image: torch.Tensor,
        target_class: int,
    ) -> np.ndarray:
        """
        Generate Grad-CAM heatmap for a specific class.

        Args:
            image: (1, C, H, W) input tensor
            target_class: class index

        Returns:
            Heatmap (H, W), values in [0, 1]
        """
        self.model.eval()
        image.requires_grad_(True)

        logits = self.model(image)
        self.model.zero_grad()
        logits[0, target_class].backward(retain_graph=True)

        if self.gradients is None or self.activations is None:
            return np.zeros((image.shape[2], image.shape[3]))

        gradients = self.gradients
        activations = self.activations

        # Handle Transformer 3D → 4D
        if gradients.dim() == 3:
            B, N, C = gradients.shape
            h = w = int(np.sqrt(N))
            if h * w != N:
                gradients = gradients[:, 1:, :]
                activations = activations[:, 1:, :]
                N -= 1
                h = w = int(np.sqrt(N))
            gradients = gradients.reshape(B, h, w, C).permute(0, 3, 1, 2)
            activations = activations.reshape(B, h, w, C).permute(0, 3, 1, 2)

        weights = gradients.mean(dim=(2, 3), keepdim=True)
        cam = (weights * activations).sum(dim=1, keepdim=True)
        cam = F.relu(cam).squeeze().cpu().numpy()

        if cam.max() > 0:
            cam = cam / cam.max()

        cam = cv2.resize(cam, (image.shape[3], image.shape[2]))
        return cam

    def generate_all_classes(
        self,
        image: torch.Tensor,
        label_names: List[str],
        threshold: float = 0.5,
    ) -> Dict[str, np.ndarray]:
        """Generate heatmaps for all predicted-positive classes."""
        self.model.eval()
        with torch.no_grad():
            logits = self.model(image)
            probs = torch.sigmoid(logits).squeeze().cpu().numpy()

        heatmaps = {}
        for i, name in enumerate(label_names):
            if probs[i] >= threshold:
                heatmaps[name] = self.generate(image, target_class=i)
        return heatmaps


def create_gradcam_grid(
    model: nn.Module,
    samples: Dict[str, torch.Tensor],
    device: torch.device,
    label_names: Optional[List[str]] = None,
) -> wandb.Image:
    """
    Generate Grad-CAM grid for periodic W&B logging during training.
    Used by Trainer every N epochs.
    """
    if label_names is None:
        label_names = [f"Class_{i}" for i in range(samples["labels"].shape[1])]

    target_layer = getattr(model, "gradcam_target_layer", None)
    if target_layer is None:
        return None

    gradcam = GradCAM(model, target_layer)
    images = samples["images"].to(device)

    n_samples = min(len(images), 10)
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])

    fig, axes = plt.subplots(n_samples, 3, figsize=(12, 4 * n_samples))
    if n_samples == 1:
        axes = axes[np.newaxis, :]

    for i in range(n_samples):
        img_tensor = images[i:i + 1]
        img_np = images[i].cpu().numpy().transpose(1, 2, 0)
        img_np = (img_np * std + mean).clip(0, 1)

        model.eval()
        with torch.no_grad():
            probs = torch.sigmoid(model(img_tensor)).squeeze().cpu().numpy()

        target_class = probs.argmax()
        cam = gradcam.generate(img_tensor, target_class)

        heatmap_color = cv2.applyColorMap((cam * 255).astype(np.uint8), cv2.COLORMAP_JET)
        heatmap_color = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB) / 255.0
        overlay = 0.6 * img_np + 0.4 * heatmap_color

        axes[i, 0].imshow(img_np)
        axes[i, 0].set_title("Original", fontsize=9)
        axes[i, 0].axis("off")

        axes[i, 1].imshow(cam, cmap="jet")
        axes[i, 1].set_title(f"Grad-CAM: {label_names[target_class]}", fontsize=9)
        axes[i, 1].axis("off")

        axes[i, 2].imshow(overlay)
        axes[i, 2].set_title(f"Overlay (p={probs[target_class]:.2f})", fontsize=9)
        axes[i, 2].axis("off")

    plt.tight_layout()
    wandb_image = wandb.Image(fig)
    plt.close(fig)
    return wandb_image
