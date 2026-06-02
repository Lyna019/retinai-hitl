"""
Label harmonization across RFMiD, ODIR-5K, EyeDisease datasets,
and iterative stratified splitting for multi-label.

Run once to create artifacts/splits.npz and artifacts/label_mapping.json.
"""

import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from iterstrat.ml_stratifiers import MultilabelStratifiedShuffleSplit

# ============================================================
# Label mapping rules — edit this to match your actual CSVs
# ============================================================

LABEL_MAPPING: Dict[str, str] = {
    # --- Diabetic Retinopathy ---
    "diabetic retinopathy": "DR",
    "diabetic_retinopathy": "DR",
    "DR": "DR",
    "dr": "DR",
    "mild npdr": "DR",
    "moderate npdr": "DR",
    "severe npdr": "DR",
    "proliferative dr": "DR",
    "pdr": "DR",
    "npdr": "DR",
    # --- AMD ---
    "age-related macular degeneration": "AMD",
    "amd": "AMD",
    "AMD": "AMD",
    "macular degeneration": "AMD",
    "ARMD": "AMD",
    "armd": "AMD",
    # --- Glaucoma ---
    "glaucoma": "Glaucoma",
    "Glaucoma": "Glaucoma",
    "glaucoma suspect": "Glaucoma",
    # --- Cataract ---
    "cataract": "Cataract",
    "Cataract": "Cataract",
    # --- Hypertensive retinopathy ---
    "hypertensive retinopathy": "Hypertension",
    "hypertension": "Hypertension",
    "HTN": "Hypertension",
    # --- Myopia ---
    "myopia": "Myopia",
    "pathologic myopia": "Myopia",
    "pathological myopia": "Myopia",
    "degenerative myopia": "Myopia",
    "high myopia": "Myopia",
    # --- Normal ---
    "normal": "Normal",
    "Normal": "Normal",
    "N": "Normal",
    "healthy": "Normal",
}

UNIFIED_LABELS = ["DR", "AMD", "Glaucoma", "Cataract", "Hypertension", "Myopia", "Other", "Normal"]


def harmonize_labels(
    raw_labels: List[str],
    mapping: Dict[str, str] = LABEL_MAPPING,
    min_samples: int = 50,
) -> str:
    """Map a single raw label string to unified taxonomy."""
    label = raw_labels.strip().lower() if isinstance(raw_labels, str) else str(raw_labels).strip().lower()
    # Check mapping (case-insensitive lookup)
    for key, value in mapping.items():
        if label == key.lower():
            return value
    return "Other"


def build_multilabel_matrix(
    df: pd.DataFrame,
    label_columns: List[str],
    unified_labels: List[str] = UNIFIED_LABELS,
) -> np.ndarray:
    """
    Convert a DataFrame with disease columns into a binary multi-label matrix.

    Args:
        df: DataFrame where each label_column is binary (0/1) or has disease names.
        label_columns: Columns containing disease indicators.
        unified_labels: Target label names.

    Returns:
        Binary matrix of shape (N, len(unified_labels))
    """
    n = len(df)
    matrix = np.zeros((n, len(unified_labels)), dtype=np.float32)
    label_to_idx = {lbl: i for i, lbl in enumerate(unified_labels)}

    for col in label_columns:
        if col in label_to_idx:
            # Column name matches a unified label directly
            idx = label_to_idx[col]
            matrix[:, idx] = df[col].values.astype(float)
        else:
            # Column contains string label names — harmonize each
            for row_i, val in enumerate(df[col]):
                if pd.notna(val) and val not in [0, "0", "", "nan"]:
                    harmonized = harmonize_labels(str(val))
                    if harmonized in label_to_idx:
                        matrix[row_i, label_to_idx[harmonized]] = 1.0

    # Enforce: if no pathology is active, set Normal=1
    no_pathology = matrix[:, :-1].sum(axis=1) == 0  # All except Normal
    matrix[no_pathology, label_to_idx["Normal"]] = 1.0

    return matrix


def create_splits(
    labels: np.ndarray,
    seed: int = 42,
    train_ratio: float = 0.70,
) -> Dict[str, np.ndarray]:
    """
    Create iterative stratified train/val/test splits for multi-label data.

    Returns dict with 'train', 'val', 'test' index arrays.
    """
    n = labels.shape[0]
    all_idx = np.arange(n)

    # First split: train vs (val+test)
    test_val_ratio = 1.0 - train_ratio
    msss1 = MultilabelStratifiedShuffleSplit(
        n_splits=1, test_size=test_val_ratio, random_state=seed
    )
    train_idx, temp_idx = next(msss1.split(all_idx.reshape(-1, 1), labels))

    # Second split: val vs test (50/50 of remaining)
    msss2 = MultilabelStratifiedShuffleSplit(
        n_splits=1, test_size=0.5, random_state=seed
    )
    val_relative, test_relative = next(
        msss2.split(temp_idx.reshape(-1, 1), labels[temp_idx])
    )
    val_idx = temp_idx[val_relative]
    test_idx = temp_idx[test_relative]

    # Sanity checks
    assert len(set(train_idx) & set(val_idx)) == 0, "Train/val overlap!"
    assert len(set(train_idx) & set(test_idx)) == 0, "Train/test overlap!"
    assert len(set(val_idx) & set(test_idx)) == 0, "Val/test overlap!"
    assert len(train_idx) + len(val_idx) + len(test_idx) == n

    return {"train": train_idx, "val": val_idx, "test": test_idx}


def save_artifacts(
    splits: Dict[str, np.ndarray],
    label_mapping_used: Dict[str, str],
    label_distribution: Dict[str, Dict[str, int]],
    output_dir: str = "artifacts",
) -> None:
    """Save split indices and label mapping as artifacts."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save splits
    np.savez(
        output_dir / "splits.npz",
        train=splits["train"],
        val=splits["val"],
        test=splits["test"],
    )

    # Save label mapping
    artifact_data = {
        "unified_labels": UNIFIED_LABELS,
        "raw_to_unified": label_mapping_used,
        "distribution": label_distribution,
    }
    with open(output_dir / "label_mapping.json", "w") as f:
        json.dump(artifact_data, f, indent=2)

    print(f"[✓] Splits saved to {output_dir / 'splits.npz'}")
    print(f"[✓] Label mapping saved to {output_dir / 'label_mapping.json'}")
    for split_name, indices in splits.items():
        print(f"    {split_name}: {len(indices)} samples")
