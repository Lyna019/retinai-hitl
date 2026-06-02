"""
PyTorch Dataset for multi-label retinal fundus classification.
Handles loading, preprocessing, and augmentation.
"""

from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from src.data.preprocessing import preprocess_fundus


class RetinalFundusDataset(Dataset):
    """
    Multi-label retinal fundus dataset.

    Args:
        image_paths: List of paths to fundus images.
        labels: Binary label matrix (N, num_classes).
        resolution: Target image resolution.
        transform: Albumentations transform pipeline.
        preprocessing_config: Config for fundus preprocessing (border removal, CLAHE).
        cache_preprocessed: If True, cache preprocessed images in memory.
    """

    def __init__(
        self,
        image_paths: List[str],
        labels: np.ndarray,
        resolution: int = 224,
        transform: Optional[Callable] = None,
        preprocessing_config: Optional[Dict] = None,
        cache_preprocessed: bool = False,
    ):
        assert len(image_paths) == labels.shape[0], (
            f"Mismatch: {len(image_paths)} images vs {labels.shape[0]} labels"
        )
        self.image_paths = image_paths
        self.labels = labels.astype(np.float32)
        self.resolution = resolution
        self.transform = transform
        self.preprocessing_config = preprocessing_config
        self.cache_preprocessed = cache_preprocessed
        self._cache: Dict[int, np.ndarray] = {}

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        # Load and preprocess image
        if self.cache_preprocessed and idx in self._cache:
            image = self._cache[idx]
        else:
            image = self._load_and_preprocess(idx)
            if self.cache_preprocessed:
                self._cache[idx] = image

        # Apply augmentation / normalization
        if self.transform is not None:
            augmented = self.transform(image=image)
            image_tensor = augmented["image"]
        else:
            image_tensor = torch.from_numpy(image.transpose(2, 0, 1)).float() / 255.0

        label_tensor = torch.from_numpy(self.labels[idx])

        return {
            "image": image_tensor,
            "label": label_tensor,
            "idx": idx,
            "path": str(self.image_paths[idx]),
        }

    def _load_and_preprocess(self, idx: int) -> np.ndarray:
        """Load image from disk and apply preprocessing pipeline."""
        path = self.image_paths[idx]
        image = cv2.imread(str(path))
        if image is None:
            raise FileNotFoundError(f"Cannot load image: {path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        image = preprocess_fundus(
            image,
            resolution=self.resolution,
            config=self.preprocessing_config,
        )
        return image


def build_dataloaders(
    image_paths: List[str],
    labels: np.ndarray,
    split_indices: Dict[str, np.ndarray],
    resolution: int = 224,
    batch_size: int = 32,
    num_workers: int = 4,
    pin_memory: bool = True,
    train_transform: Optional[Callable] = None,
    val_transform: Optional[Callable] = None,
    preprocessing_config: Optional[Dict] = None,
) -> Dict[str, torch.utils.data.DataLoader]:
    """
    Build train/val/test DataLoaders from split indices.

    Returns dict with 'train', 'val', 'test' DataLoader objects.
    """
    dataloaders = {}

    for split_name in ["train", "val", "test"]:
        indices = split_indices[split_name]
        split_paths = [image_paths[i] for i in indices]
        split_labels = labels[indices]

        transform = train_transform if split_name == "train" else val_transform
        shuffle = split_name == "train"
        drop_last = split_name == "train"

        dataset = RetinalFundusDataset(
            image_paths=split_paths,
            labels=split_labels,
            resolution=resolution,
            transform=transform,
            preprocessing_config=preprocessing_config,
        )

        dataloaders[split_name] = torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=pin_memory,
            drop_last=drop_last,
            persistent_workers=num_workers > 0,
        )

    return dataloaders
