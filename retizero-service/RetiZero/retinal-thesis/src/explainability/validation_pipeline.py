"""
Explainability Validation Pipeline.

This is the core publishable contribution of the thesis (§1.6.3):
  "This gap between a heatmap that looks plausible and one that is
   demonstrably faithful to the model's reasoning seems to be one of
   the more important open problems in the field."

Pipeline:
  1. Generate heatmaps with MULTIPLE methods (Grad-CAM, Score-CAM)
  2. Run ALL quantitative faithfulness metrics on each
  3. Compare methods head-to-head
  4. Compare faithfulness ACROSS architectures
  5. Log everything to W&B with publication-ready figures
"""

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

import wandb

from src.explainability.gradcam import GradCAM
from src.explainability.score_cam import ScoreCAM
from src.explainability.faithfulness import (
    compute_single_image_faithfulness,
    progressive_erasure,
    progressive_restoration,
    progressive_random_baseline,
)


# ============================================================
# Multi-Method Heatmap Generator
# ============================================================

class MultiMethodExplainer:
    """
    Generates attribution maps using multiple methods for comparison.
    Currently supports: Grad-CAM, Score-CAM.
    Extensible to LRP, SHAP, Integrated Gradients if needed later.
    """

    def __init__(self, model: nn.Module, target_layer: nn.Module):
        self.model = model
        self.target_layer = target_layer
        self.methods = {}

        # Initialize available methods
        self.methods["GradCAM"] = GradCAM(model, target_layer)
        self.methods["ScoreCAM"] = ScoreCAM(model, target_layer)

    def generate_all(
        self,
        image: torch.Tensor,
        target_class: int,
    ) -> Dict[str, np.ndarray]:
        """Generate heatmaps from all methods for the same image + class."""
        heatmaps = {}
        for method_name, method in self.methods.items():
            try:
                cam = method.generate(image, target_class)
                heatmaps[method_name] = cam
            except Exception as e:
                print(f"  [!] {method_name} failed: {e}")
                heatmaps[method_name] = np.zeros((image.shape[2], image.shape[3]))
        return heatmaps


# ============================================================
# Full Validation Pipeline
# ============================================================

class ExplainabilityValidator:
    """
    Runs the complete explainability validation protocol.

    Designed to produce:
    - Table: per-method faithfulness scores (for thesis Table 4.X)
    - Figure: side-by-side heatmap comparison (for thesis Figure 4.X)
    - Figure: erasure/restoration curves (for thesis Figure 4.X)
    - Figure: cross-architecture faithfulness comparison (for thesis Figure 4.X)
    """

    def __init__(
        self,
        model: nn.Module,
        target_layer: nn.Module,
        label_names: List[str],
        device: torch.device,
        model_name: str = "model",
    ):
        self.model = model
        self.device = device
        self.label_names = label_names
        self.model_name = model_name
        self.explainer = MultiMethodExplainer(model, target_layer)

    def validate_dataset(
        self,
        dataloader: torch.utils.data.DataLoader,
        n_samples: int = 100,
        patch_size: int = 16,
        annotations: Optional[List[Dict]] = None,
    ) -> Dict[str, Dict[str, float]]:
        """
        Run full validation on a dataset subset.

        Args:
            dataloader: test set DataLoader
            n_samples: how many images to evaluate
            patch_size: patch size for erasure/restoration
            annotations: optional list of {"image_idx", "bbox", "class_idx"}

        Returns:
            Nested dict: method_name → metric_name → mean_value
        """
        self.model.eval()
        dataset = dataloader.dataset

        # Build annotation lookup if available
        anno_lookup = {}
        if annotations:
            for a in annotations:
                anno_lookup[a["image_idx"]] = a

        # Collect results per method
        all_results = {method: [] for method in self.explainer.methods}

        print(f"\n[Explainability] Validating {self.model_name} on {n_samples} images...")
        sample_count = 0

        for i in tqdm(range(min(n_samples, len(dataset))), desc="Faithfulness"):
            item = dataset[i]
            image = item["image"].unsqueeze(0).to(self.device)
            label = item["label"].numpy()

            # Find positive classes
            positive_classes = np.where(label > 0.5)[0]
            if len(positive_classes) == 0:
                continue

            # Use the first positive class (or most confident)
            self.model.eval()
            with torch.no_grad():
                logits = self.model(image)
                probs = torch.sigmoid(logits).squeeze().cpu().numpy()
            target_class = positive_classes[probs[positive_classes].argmax()]

            # Generate heatmaps from all methods
            heatmaps = self.explainer.generate_all(image, target_class)

            # Get annotation bbox if available
            bbox = None
            if i in anno_lookup:
                bbox = tuple(anno_lookup[i]["bbox"])

            # Run faithfulness metrics for each method
            for method_name, cam in heatmaps.items():
                metrics = compute_single_image_faithfulness(
                    self.model, image, cam, target_class, self.device,
                    bbox=bbox, patch_size=patch_size,
                )
                all_results[method_name].append(metrics)

            sample_count += 1

        # Aggregate: mean ± std per metric per method
        aggregated = {}
        for method_name, results_list in all_results.items():
            if not results_list:
                continue
            metric_names = results_list[0].keys()
            agg = {}
            for metric in metric_names:
                values = [r[metric] for r in results_list if metric in r and not np.isnan(r[metric])]
                if values:
                    agg[f"{metric}_mean"] = float(np.mean(values))
                    agg[f"{metric}_std"] = float(np.std(values))
            aggregated[method_name] = agg

        return aggregated

    def validate_with_random_baseline(
        self,
        dataloader: torch.utils.data.DataLoader,
        n_samples: int = 30,
        patch_size: int = 16,
    ) -> Dict[str, float]:
        """
        Compute random-order erasure/restoration baselines.
        Used to show that attribution-guided ordering is better than random.
        """
        self.model.eval()
        dataset = dataloader.dataset

        erasure_aucs = []
        restoration_aucs = []

        for i in tqdm(range(min(n_samples, len(dataset))), desc="Random baseline"):
            item = dataset[i]
            image = item["image"].unsqueeze(0).to(self.device)
            label = item["label"].numpy()

            positive_classes = np.where(label > 0.5)[0]
            if len(positive_classes) == 0:
                continue

            with torch.no_grad():
                logits = self.model(image)
                probs = torch.sigmoid(logits).squeeze().cpu().numpy()
            target_class = positive_classes[probs[positive_classes].argmax()]

            baseline = progressive_random_baseline(
                self.model, image, target_class, self.device,
                patch_size=patch_size, n_trials=3,
            )
            erasure_aucs.append(baseline["erasure_auc_mean"])
            restoration_aucs.append(baseline["restoration_auc_mean"])

        return {
            "random_erasure_auc_mean": float(np.mean(erasure_aucs)),
            "random_erasure_auc_std": float(np.std(erasure_aucs)),
            "random_restoration_auc_mean": float(np.mean(restoration_aucs)),
            "random_restoration_auc_std": float(np.std(restoration_aucs)),
        }

    # ============================================================
    # W&B Logging & Figure Generation
    # ============================================================

    def log_method_comparison_table(
        self,
        results: Dict[str, Dict[str, float]],
        random_baseline: Optional[Dict[str, float]] = None,
    ) -> None:
        """Log comparison table of all methods to W&B."""
        rows = []
        for method_name, metrics in results.items():
            row = {"Method": method_name}
            for k, v in metrics.items():
                row[k] = round(v, 4)
            rows.append(row)

        if random_baseline:
            row = {"Method": "Random (baseline)"}
            row["erasure_auc_mean"] = round(random_baseline["random_erasure_auc_mean"], 4)
            row["erasure_auc_std"] = round(random_baseline["random_erasure_auc_std"], 4)
            row["restoration_auc_mean"] = round(random_baseline["random_restoration_auc_mean"], 4)
            row["restoration_auc_std"] = round(random_baseline["random_restoration_auc_std"], 4)
            rows.append(row)

        table = wandb.Table(columns=list(rows[0].keys()), data=[list(r.values()) for r in rows])
        wandb.log({f"explainability/{self.model_name}/method_comparison": table})

    def log_heatmap_comparison_figure(
        self,
        dataloader: torch.utils.data.DataLoader,
        n_images: int = 8,
    ) -> None:
        """
        Generate side-by-side heatmap figure: Original | Grad-CAM | Score-CAM
        for the same images. Publication-ready layout.
        """
        self.model.eval()
        dataset = dataloader.dataset
        methods = list(self.explainer.methods.keys())
        n_cols = 1 + len(methods)  # original + each method

        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])

        fig, axes = plt.subplots(n_images, n_cols, figsize=(4 * n_cols, 3.5 * n_images))
        if n_images == 1:
            axes = axes[np.newaxis, :]

        image_count = 0
        for i in range(len(dataset)):
            if image_count >= n_images:
                break

            item = dataset[i]
            label = item["label"].numpy()
            positive_classes = np.where(label > 0.5)[0]
            if len(positive_classes) == 0:
                continue

            image = item["image"].unsqueeze(0).to(self.device)

            with torch.no_grad():
                logits = self.model(image)
                probs = torch.sigmoid(logits).squeeze().cpu().numpy()
            target_class = positive_classes[probs[positive_classes].argmax()]
            class_name = self.label_names[target_class]

            # Denormalize for display
            img_np = item["image"].numpy().transpose(1, 2, 0)
            img_np = (img_np * std + mean).clip(0, 1)

            # Original image
            axes[image_count, 0].imshow(img_np)
            axes[image_count, 0].set_title(f"{class_name} (p={probs[target_class]:.2f})", fontsize=9)
            axes[image_count, 0].axis("off")
            if image_count == 0:
                axes[0, 0].set_ylabel("Original", fontsize=10, fontweight="bold")

            # Each method
            heatmaps = self.explainer.generate_all(image, target_class)
            for j, method_name in enumerate(methods):
                cam = heatmaps[method_name]
                heatmap_color = cv2.applyColorMap(
                    (cam * 255).astype(np.uint8), cv2.COLORMAP_JET
                )
                heatmap_color = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB) / 255.0
                overlay = 0.55 * img_np + 0.45 * heatmap_color

                axes[image_count, j + 1].imshow(overlay)
                axes[image_count, j + 1].axis("off")
                if image_count == 0:
                    axes[0, j + 1].set_title(method_name, fontsize=10, fontweight="bold")

            image_count += 1

        plt.tight_layout()
        wandb.log({f"explainability/{self.model_name}/heatmap_comparison": wandb.Image(fig)})
        plt.close(fig)

    def log_erasure_restoration_curves(
        self,
        dataloader: torch.utils.data.DataLoader,
        n_images: int = 5,
        patch_size: int = 16,
    ) -> None:
        """
        Plot erasure and restoration curves for Grad-CAM vs Score-CAM vs Random.
        Averaged across n_images. This is the Engelmann-style figure.
        """
        self.model.eval()
        dataset = dataloader.dataset
        methods = list(self.explainer.methods.keys())

        # Collect curves per method
        erasure_curves = {m: [] for m in methods}
        restoration_curves = {m: [] for m in methods}
        random_erasure_curves = []
        random_restoration_curves = []

        image_count = 0
        for i in range(len(dataset)):
            if image_count >= n_images:
                break

            item = dataset[i]
            label = item["label"].numpy()
            positive_classes = np.where(label > 0.5)[0]
            if len(positive_classes) == 0:
                continue

            image = item["image"].unsqueeze(0).to(self.device)

            with torch.no_grad():
                logits = self.model(image)
                probs = torch.sigmoid(logits).squeeze().cpu().numpy()
            target_class = positive_classes[probs[positive_classes].argmax()]

            heatmaps = self.explainer.generate_all(image, target_class)

            for method_name, cam in heatmaps.items():
                er = progressive_erasure(
                    self.model, image, cam, target_class, self.device, patch_size=patch_size
                )
                re = progressive_restoration(
                    self.model, image, cam, target_class, self.device, patch_size=patch_size
                )
                erasure_curves[method_name].append((er["fractions"], er["scores"]))
                restoration_curves[method_name].append((re["fractions"], re["scores"]))

            # Random baseline for this image
            rb = progressive_random_baseline(
                self.model, image, target_class, self.device,
                patch_size=patch_size, n_trials=3,
            )
            # We need actual curves for random, so run one more time with fixed seed
            H, W = image.shape[2], image.shape[3]
            rng = np.random.RandomState(42)
            patches = []
            for y_p in range(0, H, patch_size):
                for x_p in range(0, W, patch_size):
                    patches.append((y_p, x_p, min(y_p + patch_size, H), min(x_p + patch_size, W)))
            order = rng.permutation(len(patches))
            n_patches = len(patches)
            pps = max(n_patches // 20, 1)

            # Random erasure curve
            erased = image.clone().to(self.device)
            scores_e = [torch.sigmoid(self.model(erased))[0, target_class].item()]
            fracs_e = [0.0]
            ec = 0
            for si in range(0, n_patches, pps):
                ei = min(si + pps, n_patches)
                for idx in order[si:ei]:
                    y, x, ye, xe = patches[idx]
                    erased[:, :, y:ye, x:xe] = 0.0
                    ec += 1
                scores_e.append(torch.sigmoid(self.model(erased))[0, target_class].item())
                fracs_e.append(ec / n_patches)
            random_erasure_curves.append((fracs_e, scores_e))

            # Random restoration curve
            restored = torch.zeros_like(image).to(self.device)
            scores_r = [torch.sigmoid(self.model(restored))[0, target_class].item()]
            fracs_r = [0.0]
            rc = 0
            for si in range(0, n_patches, pps):
                ei = min(si + pps, n_patches)
                for idx in order[si:ei]:
                    y, x, ye, xe = patches[idx]
                    restored[:, :, y:ye, x:xe] = image[:, :, y:ye, x:xe].to(self.device)
                    rc += 1
                scores_r.append(torch.sigmoid(self.model(restored))[0, target_class].item())
                fracs_r.append(rc / n_patches)
            random_restoration_curves.append((fracs_r, scores_r))

            image_count += 1

        # Average curves and plot
        colors = {"GradCAM": "#2563eb", "ScoreCAM": "#dc2626", "Random": "#9ca3af"}
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

        # Erasure plot
        for method_name, curves in erasure_curves.items():
            if not curves:
                continue
            # Interpolate to common x-axis
            x_common = np.linspace(0, 1, 50)
            y_interp = []
            for fracs, scores in curves:
                y_interp.append(np.interp(x_common, fracs, scores))
            y_mean = np.mean(y_interp, axis=0)
            y_std = np.std(y_interp, axis=0)
            ax1.plot(x_common, y_mean, label=method_name, color=colors.get(method_name, "#333"), linewidth=2)
            ax1.fill_between(x_common, y_mean - y_std, y_mean + y_std, alpha=0.15, color=colors.get(method_name, "#333"))

        # Random erasure
        if random_erasure_curves:
            x_common = np.linspace(0, 1, 50)
            y_interp = [np.interp(x_common, f, s) for f, s in random_erasure_curves]
            y_mean = np.mean(y_interp, axis=0)
            ax1.plot(x_common, y_mean, "--", label="Random", color=colors["Random"], linewidth=1.5)

        ax1.set_xlabel("Fraction of patches erased (most→least important)")
        ax1.set_ylabel("P(target class)")
        ax1.set_title("Progressive Erasure")
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Restoration plot
        for method_name, curves in restoration_curves.items():
            if not curves:
                continue
            x_common = np.linspace(0, 1, 50)
            y_interp = [np.interp(x_common, f, s) for f, s in curves]
            y_mean = np.mean(y_interp, axis=0)
            y_std = np.std(y_interp, axis=0)
            ax2.plot(x_common, y_mean, label=method_name, color=colors.get(method_name, "#333"), linewidth=2)
            ax2.fill_between(x_common, y_mean - y_std, y_mean + y_std, alpha=0.15, color=colors.get(method_name, "#333"))

        if random_restoration_curves:
            x_common = np.linspace(0, 1, 50)
            y_interp = [np.interp(x_common, f, s) for f, s in random_restoration_curves]
            y_mean = np.mean(y_interp, axis=0)
            ax2.plot(x_common, y_mean, "--", label="Random", color=colors["Random"], linewidth=1.5)

        ax2.set_xlabel("Fraction of patches restored (most→least important)")
        ax2.set_ylabel("P(target class)")
        ax2.set_title("Progressive Restoration")
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        plt.suptitle(f"Erasure & Restoration Faithfulness — {self.model_name}", fontsize=13, fontweight="bold")
        plt.tight_layout()
        wandb.log({f"explainability/{self.model_name}/erasure_restoration": wandb.Image(fig)})
        plt.close(fig)

    # ============================================================
    # Full pipeline: run everything
    # ============================================================

    def run_full_validation(
        self,
        dataloader: torch.utils.data.DataLoader,
        n_samples: int = 100,
        n_curve_samples: int = 5,
        n_visual_samples: int = 8,
        patch_size: int = 16,
        annotations: Optional[List[Dict]] = None,
    ) -> Dict:
        """
        Run the complete explainability validation pipeline:
          1. Quantitative faithfulness metrics (all methods)
          2. Random baseline comparison
          3. Method comparison table → W&B
          4. Heatmap comparison figure → W&B
          5. Erasure/restoration curves → W&B

        Returns: all results as nested dict.
        """
        print(f"\n{'='*60}")
        print(f"EXPLAINABILITY VALIDATION — {self.model_name}")
        print(f"{'='*60}")

        # Step 1: Quantitative metrics
        print("\n[1/4] Computing faithfulness metrics...")
        results = self.validate_dataset(
            dataloader, n_samples=n_samples, patch_size=patch_size, annotations=annotations
        )

        # Print summary
        for method_name, metrics in results.items():
            print(f"\n  {method_name}:")
            for k, v in sorted(metrics.items()):
                print(f"    {k}: {v:.4f}")

        # Step 2: Random baseline
        print("\n[2/4] Computing random baseline...")
        random_baseline = self.validate_with_random_baseline(
            dataloader, n_samples=min(30, n_samples), patch_size=patch_size
        )
        print(f"  Random erasure AUC:      {random_baseline['random_erasure_auc_mean']:.4f}")
        print(f"  Random restoration AUC:  {random_baseline['random_restoration_auc_mean']:.4f}")

        # Step 3: W&B table
        print("\n[3/4] Logging comparison table to W&B...")
        self.log_method_comparison_table(results, random_baseline)

        # Log flat metrics for W&B summary
        for method_name, metrics in results.items():
            for k, v in metrics.items():
                wandb.log({f"explainability/{self.model_name}/{method_name}/{k}": v})

        # Step 4: Visual figures
        print("\n[4/4] Generating figures...")
        self.log_heatmap_comparison_figure(dataloader, n_images=n_visual_samples)
        self.log_erasure_restoration_curves(
            dataloader, n_images=n_curve_samples, patch_size=patch_size
        )

        print(f"\n[✓] Explainability validation complete for {self.model_name}")

        return {
            "method_results": results,
            "random_baseline": random_baseline,
        }


# ============================================================
# Cross-Architecture Comparison
# ============================================================

def compare_architectures_explainability(
    models: Dict[str, Tuple[nn.Module, nn.Module]],
    dataloader: torch.utils.data.DataLoader,
    label_names: List[str],
    device: torch.device,
    n_samples: int = 50,
) -> Dict[str, Dict]:
    """
    Run faithfulness validation across multiple architectures.

    This produces the cross-architecture comparison table for the thesis:
    which backbone produces the most faithful Grad-CAM heatmaps?

    Args:
        models: dict of model_name → (model, target_layer)
        dataloader: test set
        label_names: class names
        device: compute device
        n_samples: images per architecture

    Returns:
        dict of model_name → validation results
    """
    all_results = {}

    for model_name, (model, target_layer) in models.items():
        validator = ExplainabilityValidator(
            model=model,
            target_layer=target_layer,
            label_names=label_names,
            device=device,
            model_name=model_name,
        )
        results = validator.run_full_validation(
            dataloader, n_samples=n_samples, n_curve_samples=3, n_visual_samples=4,
        )
        all_results[model_name] = results

    # Cross-architecture comparison table
    print(f"\n{'='*60}")
    print("CROSS-ARCHITECTURE FAITHFULNESS COMPARISON")
    print(f"{'='*60}")

    rows = []
    for model_name, res in all_results.items():
        for method_name, metrics in res["method_results"].items():
            row = {
                "Architecture": model_name,
                "XAI Method": method_name,
            }
            row.update(metrics)
            rows.append(row)

    if rows:
        table = wandb.Table(
            columns=list(rows[0].keys()),
            data=[list(r.values()) for r in rows],
        )
        wandb.log({"explainability/cross_architecture_comparison": table})

    return all_results
