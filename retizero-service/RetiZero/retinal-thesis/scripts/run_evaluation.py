"""
Phase 4 & 5: Ensemble, Calibration, External Validation, TTA.

Usage:
  python scripts/run_evaluation.py \
      --checkpoint checkpoints/B-03_best.pt \
      --checkpoint checkpoints/B-04_best.pt \
      --checkpoint checkpoints/B-06_best.pt \
      --brset_csv data/brset_labels.csv \
      --brset_images data/brset_images \
      --run_ensemble \
      --run_calibration \
      --run_external \
      --run_tta \
      --run_gradcam_validation
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import wandb

from src.utils.config import load_base_config
from src.utils.helpers import set_seed, get_device
from src.data.dataset import RetinalFundusDataset, build_dataloaders
from src.data.transforms import build_val_transform, build_tta_transforms
from src.data.label_harmonization import UNIFIED_LABELS
from src.models.factory import build_model
from src.models.ensemble import AveragingEnsemble, StackingEnsemble
from src.training.metrics import compute_all_metrics, optimize_thresholds
from src.calibration.temperature_scaling import calibrate_and_report
from src.explainability.gradcam import GradCAM
from src.explainability.validation_pipeline import (
    ExplainabilityValidator,
    compare_architectures_explainability,
)


def load_model_from_checkpoint(ckpt_path: str, device: torch.device):
    """Load a model from checkpoint, reconstructing from saved config."""
    checkpoint = torch.load(ckpt_path, map_location=device)
    config = checkpoint["config"]
    model = build_model(config)
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    model.eval()
    return model, config


@torch.no_grad()
def predict_dataset(model, dataloader, device):
    """Run inference on a dataset, return probs and labels."""
    all_probs = []
    all_labels = []
    for batch in dataloader:
        images = batch["image"].to(device)
        logits = model(images)
        probs = torch.sigmoid(logits).cpu().numpy()
        all_probs.append(probs)
        all_labels.append(batch["label"].numpy())
    return np.vstack(all_probs), np.vstack(all_labels)


@torch.no_grad()
def predict_with_tta(model, dataset, device, resolution):
    """Run TTA inference: apply multiple transforms, average predictions."""
    tta_transforms = build_tta_transforms(resolution)
    all_probs = []

    for i in range(len(dataset)):
        item = dataset[i]
        # Get raw preprocessed image (before normalization)
        # We need to re-transform with each TTA pipeline
        image_np = dataset._load_and_preprocess(i)

        transform_probs = []
        for transform in tta_transforms:
            augmented = transform(image=image_np)
            img_tensor = augmented["image"].unsqueeze(0).to(device)
            logits = model(img_tensor)
            probs = torch.sigmoid(logits).cpu().numpy()
            transform_probs.append(probs)

        # Average across TTA transforms
        avg_probs = np.mean(transform_probs, axis=0)
        all_probs.append(avg_probs)

    return np.vstack(all_probs)


def run_ensemble_evaluation(models, val_loader, test_loader, device, label_names):
    """Phase 4A: Evaluate averaging and stacking ensembles."""
    print("\n" + "=" * 50)
    print("ENSEMBLE EVALUATION")
    print("=" * 50)

    # --- Averaging Ensemble ---
    avg_ensemble = AveragingEnsemble(models)
    all_probs = []
    all_labels = []
    for batch in test_loader:
        images = batch["image"].to(device)
        probs = avg_ensemble.predict(images).cpu().numpy()
        all_probs.append(probs)
        all_labels.append(batch["label"].numpy())

    avg_probs = np.vstack(all_probs)
    test_labels = np.vstack(all_labels)
    avg_metrics = compute_all_metrics(test_labels, avg_probs, label_names)

    print(f"\nAveraging Ensemble:")
    print(f"  Macro AUC: {avg_metrics['macro_auc']:.4f}")
    print(f"  F1 Macro:  {avg_metrics['f1_macro']:.4f}")

    wandb.log({
        "ensemble/averaging/macro_auc": avg_metrics["macro_auc"],
        "ensemble/averaging/f1_macro": avg_metrics["f1_macro"],
    })

    # --- Stacking Ensemble ---
    stack_ensemble = StackingEnsemble(models, num_classes=len(label_names))
    stack_ensemble.fit(val_loader, device)

    all_probs = []
    for batch in test_loader:
        images = batch["image"].to(device)
        probs = stack_ensemble.predict(images, device).cpu().numpy()
        all_probs.append(probs)

    stack_probs = np.vstack(all_probs)
    stack_metrics = compute_all_metrics(test_labels, stack_probs, label_names)

    print(f"\nStacking Ensemble:")
    print(f"  Macro AUC: {stack_metrics['macro_auc']:.4f}")
    print(f"  F1 Macro:  {stack_metrics['f1_macro']:.4f}")

    wandb.log({
        "ensemble/stacking/macro_auc": stack_metrics["macro_auc"],
        "ensemble/stacking/f1_macro": stack_metrics["f1_macro"],
    })

    return {"averaging": avg_metrics, "stacking": stack_metrics}


def run_gradcam_validation(models_dict, test_loader, device, label_names, annotations_path=None):
    """
    Phase 4C: Full explainability validation.

    Uses the validation pipeline from thesis §1.6.3:
      - Grad-CAM + Score-CAM generation
      - Progressive erasure & restoration (Engelmann et al.)
      - Insertion/deletion AUC (Petsiuk et al.)
      - Faithfulness correlation (Selvaraju et al.)
      - Energy-based pointing game
      - Random baseline comparison
      - Cross-architecture faithfulness comparison
    """
    print("\n" + "=" * 50)
    print("EXPLAINABILITY VALIDATION (FULL PIPELINE)")
    print("=" * 50)

    # Load annotations if available (for pointing game)
    annotations = None
    if annotations_path and Path(annotations_path).exists():
        with open(annotations_path) as f:
            annotations = json.load(f)
        print(f"  Loaded {len(annotations)} annotations for pointing game")

    # If multiple models, run cross-architecture comparison
    if len(models_dict) > 1:
        model_layers = {}
        for name, model in models_dict.items():
            target_layer = getattr(model, "gradcam_target_layer", None)
            if target_layer is not None:
                model_layers[name] = (model, target_layer)
            else:
                print(f"  [!] {name}: no Grad-CAM target layer, skipping")

        if model_layers:
            compare_architectures_explainability(
                models=model_layers,
                dataloader=test_loader,
                label_names=label_names,
                device=device,
                n_samples=50,
            )
    else:
        # Single model: full validation
        model_name, model = next(iter(models_dict.items()))
        target_layer = getattr(model, "gradcam_target_layer", None)
        if target_layer is None:
            print("  [!] No Grad-CAM target layer found. Skipping.")
            return

        validator = ExplainabilityValidator(
            model=model,
            target_layer=target_layer,
            label_names=label_names,
            device=device,
            model_name=model_name,
        )
        validator.run_full_validation(
            dataloader=test_loader,
            n_samples=100,
            n_curve_samples=5,
            n_visual_samples=8,
            annotations=annotations,
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", nargs="+", required=True, help="Model checkpoint paths")
    parser.add_argument("--config_dir", default="configs")
    parser.add_argument("--brset_csv", default=None)
    parser.add_argument("--brset_images", default=None)
    parser.add_argument("--annotations", default=None, help="Grad-CAM annotations JSON")
    parser.add_argument("--run_ensemble", action="store_true")
    parser.add_argument("--run_calibration", action="store_true")
    parser.add_argument("--run_external", action="store_true")
    parser.add_argument("--run_tta", action="store_true")
    parser.add_argument("--run_gradcam_validation", action="store_true")
    args = parser.parse_args()

    base_config = load_base_config(args.config_dir)
    set_seed(base_config["seed"])
    device = get_device()

    wandb.init(
        project=base_config["project"]["name"],
        group="phase4-5-evaluation",
        name="final-evaluation",
        tags=["evaluation", "ensemble", "calibration", "external"],
    )

    # Load models
    models = []
    configs = []
    for ckpt_path in args.checkpoint:
        model, config = load_model_from_checkpoint(ckpt_path, device)
        models.append(model)
        configs.append(config)
        print(f"Loaded: {ckpt_path}")

    # Use first model's config for data loading
    config = configs[0]
    label_names = UNIFIED_LABELS

    # Load internal test data
    paths = base_config["paths"]
    splits_data = np.load(paths["split_artifact"])
    df = pd.read_csv(paths["merged_csv"])
    image_paths = df["image_path"].tolist()
    labels = df[[c for c in label_names if c in df.columns]].values.astype(np.float32)

    resolution = config.get("data", config).get("resolution", 224)
    val_transform = build_val_transform(resolution)

    split_indices = {k: splits_data[k] for k in ["train", "val", "test"]}
    dataloaders = build_dataloaders(
        image_paths, labels, split_indices,
        resolution=resolution, batch_size=16,
        train_transform=val_transform, val_transform=val_transform,
        preprocessing_config=base_config.get("preprocessing"),
    )

    best_model = models[0]  # First checkpoint = best model

    # --- Ensemble ---
    if args.run_ensemble and len(models) > 1:
        run_ensemble_evaluation(models, dataloaders["val"], dataloaders["test"], device, label_names)

    # --- Calibration ---
    if args.run_calibration:
        print("\n" + "=" * 50)
        print("CALIBRATION")
        print("=" * 50)
        cal_results = calibrate_and_report(
            best_model, dataloaders["val"], dataloaders["test"], device, label_names
        )
        print(f"  ECE before: {cal_results['ece_before']:.4f}")
        print(f"  ECE after:  {cal_results['ece_after']:.4f}")

    # --- Grad-CAM Validation ---
    if args.run_gradcam_validation:
        # Build models dict for cross-architecture comparison
        models_dict = {}
        for i, (model, config) in enumerate(zip(models, configs)):
            name = config.get("experiment_id", config.get("model", {}).get("backbone", f"model_{i}"))
            models_dict[name] = model
        run_gradcam_validation(models_dict, dataloaders["test"], device, label_names, args.annotations)

    # --- TTA ---
    if args.run_tta:
        print("\n" + "=" * 50)
        print("TEST-TIME AUGMENTATION")
        print("=" * 50)

        # Without TTA
        probs_no_tta, test_labels = predict_dataset(best_model, dataloaders["test"], device)
        metrics_no_tta = compute_all_metrics(test_labels, probs_no_tta, label_names)

        # With TTA
        probs_tta = predict_with_tta(best_model, dataloaders["test"].dataset, device, resolution)
        metrics_tta = compute_all_metrics(test_labels, probs_tta, label_names)

        print(f"  Without TTA - Macro AUC: {metrics_no_tta['macro_auc']:.4f}")
        print(f"  With TTA    - Macro AUC: {metrics_tta['macro_auc']:.4f}")
        print(f"  Improvement: {metrics_tta['macro_auc'] - metrics_no_tta['macro_auc']:+.4f}")

        wandb.log({
            "tta/macro_auc_no_tta": metrics_no_tta["macro_auc"],
            "tta/macro_auc_with_tta": metrics_tta["macro_auc"],
            "tta/improvement": metrics_tta["macro_auc"] - metrics_no_tta["macro_auc"],
        })

    # --- External Validation (BRSET) ---
    if args.run_external and args.brset_csv:
        print("\n" + "=" * 50)
        print("EXTERNAL VALIDATION (BRSET)")
        print("=" * 50)

        brset_df = pd.read_csv(args.brset_csv)
        brset_paths = brset_df["image_path"].tolist()
        brset_label_cols = [c for c in label_names if c in brset_df.columns]
        brset_labels = brset_df[brset_label_cols].values.astype(np.float32)

        brset_dataset = RetinalFundusDataset(
            image_paths=brset_paths,
            labels=brset_labels,
            resolution=resolution,
            transform=val_transform,
            preprocessing_config=base_config.get("preprocessing"),
        )
        brset_loader = torch.utils.data.DataLoader(
            brset_dataset, batch_size=16, shuffle=False, num_workers=4,
        )

        brset_probs, brset_labels_np = predict_dataset(best_model, brset_loader, device)
        brset_metrics = compute_all_metrics(brset_labels_np, brset_probs, brset_label_cols)

        print(f"  BRSET Macro AUC: {brset_metrics['macro_auc']:.4f}")
        print(f"  BRSET F1 Macro:  {brset_metrics['f1_macro']:.4f}")
        print(f"  BRSET ECE:       {brset_metrics['ece']:.4f}")

        # Performance drop analysis
        internal_probs, internal_labels = predict_dataset(best_model, dataloaders["test"], device)
        internal_metrics = compute_all_metrics(internal_labels, internal_probs, label_names)

        drop = internal_metrics["macro_auc"] - brset_metrics["macro_auc"]
        print(f"\n  Internal→External AUC drop: {drop:+.4f}")

        for k, v in brset_metrics.items():
            if isinstance(v, dict):
                for sub_k, sub_v in v.items():
                    if not (isinstance(sub_v, float) and np.isnan(sub_v)):
                        wandb.log({f"external/{k}/{sub_k}": sub_v})
            elif isinstance(v, (int, float)):
                wandb.log({f"external/{k}": v})

        wandb.log({"external/auc_drop_from_internal": drop})

    wandb.finish()
    print("\n[✓] Evaluation complete.")


if __name__ == "__main__":
    main()
