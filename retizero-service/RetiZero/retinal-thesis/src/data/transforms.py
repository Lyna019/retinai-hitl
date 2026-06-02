"""
Data augmentation transforms.
Uses Albumentations for all spatial and photometric transforms.
"""

from typing import Any, Dict, Optional

import albumentations as A
from albumentations.pytorch import ToTensorV2

# ImageNet normalization constants
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def build_train_transform(
    resolution: int = 224,
    config: Optional[Dict[str, Any]] = None,
) -> A.Compose:
    """
    Build training augmentation pipeline.

    Args:
        resolution: Input resolution (images should already be this size)
        config: Augmentation config from base.yaml. If None, use defaults.
                Pass config=False or empty dict to disable augmentation.
    """
    transforms = []

    if config is not None and config is not False:
        # Geometric
        if config.get("horizontal_flip", 0) > 0:
            transforms.append(A.HorizontalFlip(p=config["horizontal_flip"]))

        if config.get("rotation_limit", 0) > 0:
            transforms.append(
                A.Rotate(limit=config["rotation_limit"], p=0.5, border_mode=0)
            )

        # Shift-Scale
        ss = config.get("shift_scale", {})
        if ss:
            transforms.append(
                A.ShiftScaleRotate(
                    shift_limit=ss.get("shift_limit", 0.05),
                    scale_limit=ss.get("scale_limit", 0.1),
                    rotate_limit=0,
                    p=ss.get("p", 0.3),
                    border_mode=0,
                )
            )

        # Photometric
        if config.get("brightness_limit", 0) > 0 or config.get("contrast_limit", 0) > 0:
            transforms.append(
                A.RandomBrightnessContrast(
                    brightness_limit=config.get("brightness_limit", 0.2),
                    contrast_limit=config.get("contrast_limit", 0.2),
                    p=0.5,
                )
            )

        cj = config.get("color_jitter", {})
        if cj:
            transforms.append(
                A.ColorJitter(
                    brightness=cj.get("brightness", 0.1),
                    contrast=cj.get("contrast", 0.1),
                    saturation=cj.get("saturation", 0.1),
                    hue=cj.get("hue", 0.05),
                    p=cj.get("p", 0.3),
                )
            )

        # Cutout regularization
        co = config.get("coarse_dropout", {})
        if co:
            transforms.append(
                A.CoarseDropout(
                    max_holes=co.get("max_holes", 4),
                    max_height=co.get("max_height", 32),
                    max_width=co.get("max_width", 32),
                    fill_value=0,
                    p=co.get("p", 0.3),
                )
            )

    # Always normalize and convert to tensor
    transforms.append(A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD))
    transforms.append(ToTensorV2())

    return A.Compose(transforms)


def build_val_transform(resolution: int = 224) -> A.Compose:
    """Validation/test transform: normalize only."""
    return A.Compose([
        A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ToTensorV2(),
    ])


def build_tta_transforms(resolution: int = 224) -> list:
    """
    Build TTA (Test-Time Augmentation) transforms.
    Returns list of transform pipelines to apply at inference.
    Results should be averaged.
    """
    base_norm = [A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD), ToTensorV2()]

    transforms = [
        # Original
        A.Compose(base_norm),
        # Horizontal flip
        A.Compose([A.HorizontalFlip(p=1.0)] + base_norm),
    ]

    # 5-crop: center + 4 corners (implemented as shift augmentation)
    crop_size = int(resolution * 0.9)
    for dx, dy in [(-0.05, -0.05), (-0.05, 0.05), (0.05, -0.05), (0.05, 0.05)]:
        transforms.append(
            A.Compose([
                A.ShiftScaleRotate(
                    shift_limit=(dx, dx),
                    scale_limit=(-0.1, -0.1),
                    rotate_limit=0,
                    p=1.0,
                    border_mode=0,
                ),
            ] + base_norm)
        )

    return transforms
