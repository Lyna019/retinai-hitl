"""
Data Preparation Script.
Run ONCE before any experiments.

1. Loads RFMiD, ODIR-5K, EyeDisease CSVs
2. Harmonizes labels into unified taxonomy
3. Creates iterative stratified train/val/test splits
4. Saves artifacts (splits.npz, label_mapping.json)
5. Logs artifacts to W&B

Usage: python scripts/prepare_data.py --data_root ./data --config_dir configs
"""

import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

from src.utils.config import load_base_config
from src.utils.helpers import set_seed
from src.data.label_harmonization import (
    LABEL_MAPPING,
    UNIFIED_LABELS,
    harmonize_labels,
    build_multilabel_matrix,
    create_splits,
    save_artifacts,
)


def load_rfmid(data_root: str) -> pd.DataFrame:
    """
    Load RFMiD dataset.
    Adjust paths/columns to match your actual RFMiD CSV structure.
    """
    rfmid_dir = Path(data_root) / "rfmid"
    csv_path = rfmid_dir / "RFMiD_Training_Labels.csv"

    if not csv_path.exists():
        print(f"  [!] RFMiD CSV not found at {csv_path}. Skipping.")
        return pd.DataFrame()

    df = pd.read_csv(csv_path)

    # RFMiD has binary columns for each disease — adapt column names
    # Map RFMiD disease columns to unified labels
    disease_cols = {
        "Disease_Risk": None,  # Binary overall risk — skip
        "DR": "DR",
        "ARMD": "AMD",
        "MH": "Other",
        "DN": "Other",
        "MYA": "Myopia",
        "BRVO": "Other",
        "TSLN": "Other",
        "ERM": "Other",
        "LS": "Other",
        "MS": "Other",
        "CSR": "Other",
        "ODC": "Glaucoma",
        "CRVO": "Other",
        "TV": "Other",
        "AH": "Other",
        "ODP": "Other",
        "ODE": "Other",
        "ST": "Other",
        "AION": "Other",
        "PT": "Other",
        "RT": "Other",
        "RS": "Other",
        "CRS": "Other",
        "EDN": "Other",
        "RPEC": "Other",
        "MHL": "Other",
        "RP": "Other",
        "CWS": "Other",
        "CB": "Other",
        "ODPM": "Other",
        "PRH": "Other",
        "MNF": "Other",
        "HR": "Other",
        "CPED": "Other",
        "CL": "Other",
        "CF": "Other",
        "VH": "Other",
        "MCA": "Other",
        "VS": "Other",
        "BRAO": "Other",
        "PLQ": "Other",
        "HPED": "Other",
        "CSD": "Other",
    }

    # Build unified label columns
    for unified in UNIFIED_LABELS:
        df[unified] = 0

    for col, unified in disease_cols.items():
        if unified and col in df.columns:
            df[unified] = df[unified] | df[col].astype(int)

    # Set Normal
    pathology_cols = [u for u in UNIFIED_LABELS if u != "Normal"]
    df["Normal"] = (df[pathology_cols].sum(axis=1) == 0).astype(int)

    # Add image path
    df["image_path"] = df["ID"].apply(
        lambda x: str(rfmid_dir / "Training" / f"{x}.png")
    )
    df["source"] = "rfmid"

    return df[["image_path", "source"] + UNIFIED_LABELS]


def load_odir(data_root: str) -> pd.DataFrame:
    """
    Load ODIR-5K dataset.
    Adjust to match your ODIR CSV structure.
    """
    odir_dir = Path(data_root) / "odir"
    csv_path = odir_dir / "full_df.csv"

    if not csv_path.exists():
        print(f"  [!] ODIR CSV not found at {csv_path}. Skipping.")
        return pd.DataFrame()

    df = pd.read_csv(csv_path)

    # ODIR typically has columns: N, D, G, C, A, H, M, O
    # N=Normal, D=DR, G=Glaucoma, C=Cataract, A=AMD, H=Hypertension, M=Myopia, O=Other
    col_map = {
        "N": "Normal",
        "D": "DR",
        "G": "Glaucoma",
        "C": "Cataract",
        "A": "AMD",
        "H": "Hypertension",
        "M": "Myopia",
        "O": "Other",
    }

    for odir_col, unified in col_map.items():
        if odir_col in df.columns:
            df[unified] = df[odir_col].astype(int)
        else:
            df[unified] = 0

    # Ensure all unified labels exist
    for label in UNIFIED_LABELS:
        if label not in df.columns:
            df[label] = 0

    # Image path
    if "filename" in df.columns:
        df["image_path"] = df["filename"].apply(
            lambda x: str(odir_dir / "images" / x)
        )
    elif "Left-Fundus" in df.columns:
        # Handle bilateral data — use left eye
        df["image_path"] = df["Left-Fundus"].apply(
            lambda x: str(odir_dir / "images" / x)
        )

    df["source"] = "odir"

    return df[["image_path", "source"] + UNIFIED_LABELS]


def load_eyedisease(data_root: str) -> pd.DataFrame:
    """
    Load Eye Disease (Mendeley/Bangladesh) dataset.
    This dataset is typically multi-class (folder-based).
    """
    eye_dir = Path(data_root) / "eyedisease"

    if not eye_dir.exists():
        print(f"  [!] EyeDisease directory not found at {eye_dir}. Skipping.")
        return pd.DataFrame()

    records = []
    class_to_label = {
        "diabetic_retinopathy": "DR",
        "glaucoma": "Glaucoma",
        "cataract": "Cataract",
        "normal": "Normal",
        "age_related_macular_degeneration": "AMD",
        "hypertension": "Hypertension",
        "myopia": "Myopia",
        "pathological_myopia": "Myopia",
    }

    for class_folder in eye_dir.iterdir():
        if not class_folder.is_dir():
            continue

        folder_name = class_folder.name.lower().replace(" ", "_").replace("-", "_")
        unified = class_to_label.get(folder_name, "Other")

        for img_path in class_folder.glob("*"):
            if img_path.suffix.lower() in [".png", ".jpg", ".jpeg", ".bmp", ".tif"]:
                row = {col: 0 for col in UNIFIED_LABELS}
                row[unified] = 1
                row["image_path"] = str(img_path)
                row["source"] = "eyedisease"
                records.append(row)

    return pd.DataFrame(records)[["image_path", "source"] + UNIFIED_LABELS]


def main():
    parser = argparse.ArgumentParser(description="Prepare merged dataset and splits")
    parser.add_argument("--data_root", type=str, default="./data")
    parser.add_argument("--config_dir", type=str, default="configs")
    parser.add_argument("--output_dir", type=str, default="artifacts")
    args = parser.parse_args()

    config = load_base_config(args.config_dir)
    set_seed(config["seed"])

    print("=" * 60)
    print("DATA PREPARATION PIPELINE")
    print("=" * 60)

    # ---- Load datasets ----
    print("\n[1/4] Loading datasets...")
    dfs = []

    df_rfmid = load_rfmid(args.data_root)
    if len(df_rfmid) > 0:
        print(f"  RFMiD: {len(df_rfmid)} images")
        dfs.append(df_rfmid)

    df_odir = load_odir(args.data_root)
    if len(df_odir) > 0:
        print(f"  ODIR:  {len(df_odir)} images")
        dfs.append(df_odir)

    df_eye = load_eyedisease(args.data_root)
    if len(df_eye) > 0:
        print(f"  EyeDisease: {len(df_eye)} images")
        dfs.append(df_eye)

    if not dfs:
        print("ERROR: No datasets loaded. Check your data_root path.")
        return

    # ---- Merge ----
    print("\n[2/4] Merging datasets...")
    merged_df = pd.concat(dfs, ignore_index=True)
    print(f"  Total merged: {len(merged_df)} images")

    # Filter out images that don't exist
    valid_mask = merged_df["image_path"].apply(lambda p: Path(p).exists())
    n_missing = (~valid_mask).sum()
    if n_missing > 0:
        print(f"  [!] {n_missing} images not found on disk — removing them.")
        merged_df = merged_df[valid_mask].reset_index(drop=True)

    # ---- Label distribution ----
    print("\n  Label distribution:")
    label_dist = {}
    for label in UNIFIED_LABELS:
        count = int(merged_df[label].sum())
        pct = count / len(merged_df) * 100
        print(f"    {label:15s}: {count:6d} ({pct:5.1f}%)")
        label_dist[label] = count

    # ---- Merge rare labels ----
    min_threshold = config.get("labels", {}).get("min_samples_threshold", config.get("min_samples_threshold", 50))
    # (Already handled by putting rare labels into "Other" during loading)

    # ---- Save merged CSV ----
    merged_csv_path = Path(args.data_root) / "merged_labels.csv"
    merged_df.to_csv(merged_csv_path, index=False)
    print(f"\n  Merged CSV saved to: {merged_csv_path}")

    # ---- Create splits ----
    print("\n[3/4] Creating stratified splits...")
    labels = merged_df[UNIFIED_LABELS].values.astype(np.float32)
    splits = create_splits(
        labels=labels,
        seed=config["seed"],
        train_ratio=config.get("split", {}).get("train_ratio", 0.70),
    )

    # Distribution per split
    split_dist = {}
    for split_name, indices in splits.items():
        split_labels = labels[indices]
        dist = {label: int(split_labels[:, i].sum()) for i, label in enumerate(UNIFIED_LABELS)}
        split_dist[split_name] = dist
        print(f"\n  {split_name} ({len(indices)} samples):")
        for label, count in dist.items():
            print(f"    {label:15s}: {count:6d}")

    # ---- Save artifacts ----
    print(f"\n[4/4] Saving artifacts to {args.output_dir}...")
    save_artifacts(
        splits=splits,
        label_mapping_used=LABEL_MAPPING,
        label_distribution={"total": label_dist, **split_dist},
        output_dir=args.output_dir,
    )

    print("\n" + "=" * 60)
    print("DATA PREPARATION COMPLETE")
    print(f"  Merged CSV:    {merged_csv_path}")
    print(f"  Splits:        {args.output_dir}/splits.npz")
    print(f"  Label mapping: {args.output_dir}/label_mapping.json")
    print("=" * 60)


if __name__ == "__main__":
    main()
