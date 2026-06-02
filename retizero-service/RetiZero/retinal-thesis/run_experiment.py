"""
Main experiment runner.
Usage: python run_experiment.py --experiment_id B-01 --config_dir configs
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import wandb

from src.utils.config import load_experiment_config, flatten_config, get_model_config
from src.utils.helpers import set_seed, get_device, count_parameters, compute_class_weights
from src.data.dataset import RetinalFundusDataset, build_dataloaders
from src.data.transforms import build_train_transform, build_val_transform
from src.data.label_harmonization import UNIFIED_LABELS
from src.models.factory import build_model
from src.losses.losses import build_loss
from src.training.trainer import Trainer
from src.explainability.gradcam import create_gradcam_grid


def main():
    parser = argparse.ArgumentParser(description="Run a retinal pathology experiment")
    parser.add_argument("--experiment_id", type=str, required=True, help="Experiment ID (e.g., B-01)")
    parser.add_argument("--config_dir", type=str, default="configs", help="Config directory")
    parser.add_argument("--data_root", type=str, default=None, help="Override data root path")
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume from")
    args = parser.parse_args()

    # ---- Load config ----
    config = load_experiment_config(args.experiment_id, args.config_dir)
    if args.data_root:
        config["paths"]["data_root"] = args.data_root

    # ---- Seed ----
    set_seed(config["seed"])
    device = get_device()
    print(f"Device: {device}")
    print(f"Experiment: {config['experiment_id']} | Phase: {config.get('phase', 'unknown')}")

    # ---- W&B Init ----
    wandb.init(
        project=config["project"]["name"],
        entity=config["project"].get("entity"),
        group=config.get("phase", "default"),
        name=config["experiment_id"],
        config=flatten_config(config),
        tags=config.get("tags", []),
        reinit=True,
    )

    # ---- Load data ----
    # Load precomputed splits and labels
    paths = config["paths"]
    splits_path = paths.get("split_artifact", "artifacts/splits.npz")
    mapping_path = paths.get("label_mapping", "artifacts/label_mapping.json")

    if not Path(splits_path).exists():
        print(f"ERROR: Split artifact not found at {splits_path}")
        print("Run `python scripts/prepare_data.py` first to create splits.")
        sys.exit(1)

    splits_data = np.load(splits_path)
    split_indices = {
        "train": splits_data["train"],
        "val": splits_data["val"],
        "test": splits_data["test"],
    }

    with open(mapping_path, "r") as f:
        label_info = json.load(f)
    label_names = label_info["unified_labels"]

    # Load image paths and labels from merged CSV
    import pandas as pd
    merged_csv = paths.get("merged_csv", "data/merged_labels.csv")
    df = pd.read_csv(merged_csv)

    image_paths = df["image_path"].tolist()
    label_columns = [col for col in label_names if col in df.columns]
    labels = df[label_columns].values.astype(np.float32)

    # ---- Build transforms ----
    data_cfg = config.get("data", config)
    resolution = data_cfg.get("resolution", 224)
    batch_size = data_cfg.get("batch_size", 32)

    aug_config = config.get("augmentation", None)
    # If augmentation is explicitly None/False (ablation), disable it
    if aug_config is None or aug_config is False:
        train_transform = build_train_transform(resolution, config=None)
    else:
        train_transform = build_train_transform(resolution, config=aug_config)

    val_transform = build_val_transform(resolution)

    # ---- Build dataloaders ----
    preprocessing_config = config.get("preprocessing", None)
    dataloaders = build_dataloaders(
        image_paths=image_paths,
        labels=labels,
        split_indices=split_indices,
        resolution=resolution,
        batch_size=batch_size,
        num_workers=config.get("training", {}).get("num_workers", 4),
        pin_memory=config.get("training", {}).get("pin_memory", True),
        train_transform=train_transform,
        val_transform=val_transform,
        preprocessing_config=preprocessing_config,
    )

    # ---- Build model ----
    model = build_model(config)
    model = model.to(device)
    params = count_parameters(model)
    print(f"Model: {config.get('model', {}).get('backbone', 'hybrid')}")
    print(f"Parameters: {params['trainable']:,} trainable / {params['total']:,} total")
    wandb.run.summary["params_trainable"] = params["trainable"]
    wandb.run.summary["params_total"] = params["total"]

    # ---- Build loss ----
    train_labels = labels[split_indices["train"]]
    class_weights = compute_class_weights(train_labels, cap=config.get("loss", {}).get("class_weight_cap", 10.0))
    class_weights_tensor = torch.from_numpy(class_weights).to(device)

    loss_cfg = config.get("loss", {})
    loss_fn = build_loss(loss_cfg, class_weights=class_weights_tensor)

    # ---- Prepare Grad-CAM samples ----
    gradcam_cfg = config.get("gradcam", {})
    gradcam_samples = None
    gradcam_fn = None

    if gradcam_cfg.get("enabled", False):
        n_samples = gradcam_cfg.get("num_fixed_samples", 50)
        # Take first N samples from validation set
        val_dataset = dataloaders["val"].dataset
        sample_images = []
        sample_labels = []
        for i in range(min(n_samples, len(val_dataset))):
            item = val_dataset[i]
            sample_images.append(item["image"])
            sample_labels.append(item["label"])

        gradcam_samples = {
            "images": torch.stack(sample_images),
            "labels": torch.stack(sample_labels),
        }
        gradcam_fn = lambda model, samples, device: create_gradcam_grid(
            model, samples, device, label_names
        )

    # ---- Log dataset split as W&B artifact ----
    split_artifact = wandb.Artifact(
        name="dataset-splits",
        type="dataset",
        metadata={
            "train_size": len(split_indices["train"]),
            "val_size": len(split_indices["val"]),
            "test_size": len(split_indices["test"]),
        },
    )
    split_artifact.add_file(splits_path)
    split_artifact.add_file(mapping_path)
    wandb.log_artifact(split_artifact)

    # ---- Train ----
    start_time = time.time()

    trainer = Trainer(
        model=model,
        loss_fn=loss_fn,
        config=config,
        label_names=label_names,
        device=device,
        gradcam_fn=gradcam_fn,
        gradcam_samples=gradcam_samples,
    )

    results = trainer.train(
        train_loader=dataloaders["train"],
        val_loader=dataloaders["val"],
    )

    total_time = time.time() - start_time
    wandb.run.summary["total_training_time_min"] = total_time / 60
    wandb.run.summary["gpu_memory_peak_mb"] = (
        torch.cuda.max_memory_allocated() / 1e6 if torch.cuda.is_available() else 0
    )

    print(f"\nTraining complete in {total_time/60:.1f} min")
    print(f"Best val macro-AUC: {results['best_metric']:.4f} at epoch {results['best_epoch']}")
    print(f"Checkpoint: {results['checkpoint_path']}")

    # ---- Evaluate on internal test set ----
    print("\n--- Internal Test Set Evaluation ---")
    from src.training.metrics import compute_all_metrics, optimize_thresholds

    # Load best checkpoint
    checkpoint = torch.load(results["checkpoint_path"], map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    # Predict on test set
    all_probs = []
    all_labels = []
    with torch.no_grad():
        for batch in dataloaders["test"]:
            images = batch["image"].to(device)
            logits = model(images)
            probs = torch.sigmoid(logits).cpu().numpy()
            all_probs.append(probs)
            all_labels.append(batch["label"].numpy())

    test_probs = np.vstack(all_probs)
    test_labels = np.vstack(all_labels)

    # Optimize thresholds on validation set
    val_probs_list = []
    val_labels_list = []
    with torch.no_grad():
        for batch in dataloaders["val"]:
            images = batch["image"].to(device)
            logits = model(images)
            probs = torch.sigmoid(logits).cpu().numpy()
            val_probs_list.append(probs)
            val_labels_list.append(batch["label"].numpy())

    val_probs = np.vstack(val_probs_list)
    val_labels = np.vstack(val_labels_list)

    optimal_thresholds = optimize_thresholds(val_labels, val_probs, label_names)

    # Test metrics with default and optimized thresholds
    test_metrics_default = compute_all_metrics(test_labels, test_probs, label_names, threshold=0.5)
    test_metrics_optimized = compute_all_metrics(
        test_labels, test_probs, label_names, per_label_thresholds=optimal_thresholds
    )

    # Log test metrics
    for k, v in test_metrics_default.items():
        if isinstance(v, dict):
            for sub_k, sub_v in v.items():
                if not (isinstance(sub_v, float) and np.isnan(sub_v)):
                    wandb.run.summary[f"test/default/{k}/{sub_k}"] = sub_v
        else:
            wandb.run.summary[f"test/default/{k}"] = v

    for k, v in test_metrics_optimized.items():
        if isinstance(v, dict):
            for sub_k, sub_v in v.items():
                if not (isinstance(sub_v, float) and np.isnan(sub_v)):
                    wandb.run.summary[f"test/optimized/{k}/{sub_k}"] = sub_v
        else:
            wandb.run.summary[f"test/optimized/{k}"] = v

    wandb.run.summary["test/optimal_thresholds"] = json.dumps(optimal_thresholds)

    print(f"Test macro-AUC: {test_metrics_default['macro_auc']:.4f}")
    print(f"Test F1 (default t=0.5): {test_metrics_default['f1_macro']:.4f}")
    print(f"Test F1 (optimized t):   {test_metrics_optimized['f1_macro']:.4f}")
    print(f"Optimal thresholds: {optimal_thresholds}")

    wandb.finish()
    print("\n[✓] Experiment complete. Results logged to W&B.")


if __name__ == "__main__":
    main()
