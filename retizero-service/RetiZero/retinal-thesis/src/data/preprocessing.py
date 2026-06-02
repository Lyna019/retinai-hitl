"""
Fundus image preprocessing pipeline.
Steps: border removal → resize → CLAHE → normalize.
"""

from typing import Dict, Optional, Tuple

import cv2
import numpy as np


def remove_black_border(image: np.ndarray, threshold: int = 10) -> np.ndarray:
    """
    Remove black borders from fundus images using threshold-based circular crop.
    Finds the retinal region and crops tightly around it.

    Args:
        image: BGR or RGB image (H, W, 3)
        threshold: Pixel intensity threshold for border detection
    """
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY) if image.ndim == 3 else image
    _, binary = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)

    # Find contours of the bright (retinal) region
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return image

    # Take the largest contour (the retina)
    largest = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(largest)

    # Add small padding to avoid cutting retinal edges
    pad = 5
    x = max(0, x - pad)
    y = max(0, y - pad)
    w = min(image.shape[1] - x, w + 2 * pad)
    h = min(image.shape[0] - y, h + 2 * pad)

    return image[y : y + h, x : x + w]


def apply_clahe(
    image: np.ndarray,
    clip_limit: float = 2.0,
    tile_grid_size: int = 8,
) -> np.ndarray:
    """
    Apply CLAHE to improve contrast of fundus images.
    Operates on the L channel of LAB color space to preserve color info.

    Args:
        image: RGB image (H, W, 3), uint8
        clip_limit: CLAHE clip limit
        tile_grid_size: CLAHE tile size
    """
    lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
    clahe = cv2.createCLAHE(
        clipLimit=clip_limit,
        tileGridSize=(tile_grid_size, tile_grid_size),
    )
    lab[:, :, 0] = clahe.apply(lab[:, :, 0])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)


def preprocess_fundus(
    image: np.ndarray,
    resolution: int = 224,
    config: Optional[Dict] = None,
) -> np.ndarray:
    """
    Full preprocessing pipeline for a single fundus image.

    Args:
        image: RGB image (H, W, 3), uint8
        resolution: Target resolution (square)
        config: Preprocessing config dict from base.yaml

    Returns:
        Preprocessed RGB image (resolution, resolution, 3), uint8
    """
    if config is None:
        config = {
            "border_removal": True,
            "clahe": {"enabled": True, "clip_limit": 2.0, "tile_grid_size": 8},
        }

    # Step 1: Border removal
    if config.get("border_removal", True):
        image = remove_black_border(image)

    # Step 2: Resize to target resolution
    image = cv2.resize(image, (resolution, resolution), interpolation=cv2.INTER_AREA)

    # Step 3: CLAHE
    clahe_cfg = config.get("clahe", {})
    if clahe_cfg.get("enabled", True):
        image = apply_clahe(
            image,
            clip_limit=clahe_cfg.get("clip_limit", 2.0),
            tile_grid_size=clahe_cfg.get("tile_grid_size", 8),
        )

    return image
