"""
Quantitative faithfulness metrics for explainability validation.

Maps directly to thesis §1.6.3 literature gaps:
  "Producing a heatmap and validating it are two different things."

Methods implemented (with literature source):

1. Progressive Erasure & Restoration   — Engelmann et al. (§1.6.2)
2. Pointing Game (basic)               — Zhang et al., Selvaraju et al.
3. Energy-Based Pointing Game           — extension of basic PG
4. Insertion / Deletion AUC             — Petsiuk et al. (§1.6.3)
5. Faithfulness Correlation             — Selvaraju et al. (§1.6.3)
6. ROAR (RemOve And Retrain)            — Hooker et al. (simplified version)

Each function takes a heatmap + model + image and returns a scalar metric.
All metrics are model-agnostic: they work with any attribution method
(Grad-CAM, Score-CAM, etc.).
"""

from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm


# ============================================================
# 1. Progressive Erasure & Progressive Restoration
#    Source: Engelmann et al. — thesis §1.6.2
#
#    This is the KEY method your lit review highlights as best practice.
#    Instead of trusting the heatmap visually, you:
#    - Rank image regions by attribution score
#    - Progressive ERASURE: mask regions from MOST to LEAST important
#      → AUC should DROP sharply (important regions truly matter)
#    - Progressive RESTORATION: reveal regions from MOST to LEAST important
#      → AUC should RECOVER quickly (model needs only the highlighted area)
#    - Compare: random region ordering as baseline
# ============================================================

@torch.no_grad()
def progressive_erasure(
    model: nn.Module,
    image: torch.Tensor,
    cam: np.ndarray,
    target_class: int,
    device: torch.device,
    steps: int = 20,
    patch_size: int = 16,
) -> Dict[str, object]:
    """
    Progressive Erasure (Engelmann et al.).

    Divides the image into patches, ranks them by mean attribution,
    then progressively masks patches from MOST important to LEAST.
    Measures how quickly the prediction degrades.

    A faithful heatmap → sharp AUC drop when erasing top-ranked patches.

    Args:
        model: trained model
        image: (1, C, H, W) tensor
        cam: (H, W) attribution heatmap, values in [0, 1]
        target_class: which class to track
        steps: number of erasure steps
        patch_size: size of each erasure patch

    Returns:
        dict with 'scores' (list of probabilities at each step),
        'fractions' (fraction of image erased), 'auc' (area under curve)
    """
    model.eval()
    H, W = image.shape[2], image.shape[3]

    # Divide into patches and compute mean attribution per patch
    patches = []
    for y in range(0, H, patch_size):
        for x in range(0, W, patch_size):
            y_end = min(y + patch_size, H)
            x_end = min(x + patch_size, W)
            mean_attr = cam[y:y_end, x:x_end].mean()
            patches.append((mean_attr, y, x, y_end, x_end))

    # Sort by importance: most important first
    patches.sort(key=lambda p: p[0], reverse=True)

    # Baseline score (no erasure)
    prob = torch.sigmoid(model(image.to(device)))[0, target_class].item()
    scores = [prob]
    fractions = [0.0]

    # Determine erasure schedule
    n_patches = len(patches)
    patches_per_step = max(n_patches // steps, 1)

    # Progressive erasure
    erased_image = image.clone().to(device)
    erased_count = 0

    for step_i in range(0, n_patches, patches_per_step):
        end_i = min(step_i + patches_per_step, n_patches)
        for _, y, x, y_end, x_end in patches[step_i:end_i]:
            erased_image[:, :, y:y_end, x:x_end] = 0.0  # Black-out
            erased_count += 1

        prob = torch.sigmoid(model(erased_image))[0, target_class].item()
        scores.append(prob)
        fractions.append(erased_count / n_patches)

    # AUC of the degradation curve (lower = more faithful heatmap)
    x_vals = np.array(fractions)
    y_vals = np.array(scores)
    auc = float(np.trapz(y_vals, x_vals))

    return {"scores": scores, "fractions": fractions, "auc": auc}


@torch.no_grad()
def progressive_restoration(
    model: nn.Module,
    image: torch.Tensor,
    cam: np.ndarray,
    target_class: int,
    device: torch.device,
    steps: int = 20,
    patch_size: int = 16,
) -> Dict[str, object]:
    """
    Progressive Restoration (Engelmann et al.).

    Start from a blank image, progressively RESTORE patches from
    MOST important to LEAST. Measures how quickly prediction recovers.

    A faithful heatmap → sharp AUC recovery when restoring top-ranked patches.

    Returns:
        dict with 'scores', 'fractions', 'auc' (higher = more faithful)
    """
    model.eval()
    H, W = image.shape[2], image.shape[3]

    # Divide into patches and rank
    patches = []
    for y in range(0, H, patch_size):
        for x in range(0, W, patch_size):
            y_end = min(y + patch_size, H)
            x_end = min(x + patch_size, W)
            mean_attr = cam[y:y_end, x:x_end].mean()
            patches.append((mean_attr, y, x, y_end, x_end))

    patches.sort(key=lambda p: p[0], reverse=True)

    # Start from blank
    restored_image = torch.zeros_like(image).to(device)
    prob = torch.sigmoid(model(restored_image))[0, target_class].item()
    scores = [prob]
    fractions = [0.0]

    n_patches = len(patches)
    patches_per_step = max(n_patches // steps, 1)
    restored_count = 0

    for step_i in range(0, n_patches, patches_per_step):
        end_i = min(step_i + patches_per_step, n_patches)
        for _, y, x, y_end, x_end in patches[step_i:end_i]:
            restored_image[:, :, y:y_end, x:x_end] = image[:, :, y:y_end, x:x_end].to(device)
            restored_count += 1

        prob = torch.sigmoid(model(restored_image))[0, target_class].item()
        scores.append(prob)
        fractions.append(restored_count / n_patches)

    x_vals = np.array(fractions)
    y_vals = np.array(scores)
    auc = float(np.trapz(y_vals, x_vals))

    return {"scores": scores, "fractions": fractions, "auc": auc}


@torch.no_grad()
def progressive_random_baseline(
    model: nn.Module,
    image: torch.Tensor,
    target_class: int,
    device: torch.device,
    steps: int = 20,
    patch_size: int = 16,
    n_trials: int = 5,
    seed: int = 42,
) -> Dict[str, object]:
    """
    Random-order erasure/restoration baseline.

    Average over n_trials with random patch orderings.
    This is the control: a faithful heatmap should degrade FASTER than random
    (erasure) and recover FASTER than random (restoration).
    """
    model.eval()
    H, W = image.shape[2], image.shape[3]
    rng = np.random.RandomState(seed)

    patches = []
    for y in range(0, H, patch_size):
        for x in range(0, W, patch_size):
            y_end = min(y + patch_size, H)
            x_end = min(x + patch_size, W)
            patches.append((y, x, y_end, x_end))

    n_patches = len(patches)
    patches_per_step = max(n_patches // steps, 1)

    erasure_aucs = []
    restoration_aucs = []

    for _ in range(n_trials):
        order = rng.permutation(n_patches)

        # Random erasure
        erased = image.clone().to(device)
        scores_e = [torch.sigmoid(model(erased))[0, target_class].item()]
        fracs_e = [0.0]
        erased_count = 0

        for step_i in range(0, n_patches, patches_per_step):
            end_i = min(step_i + patches_per_step, n_patches)
            for idx in order[step_i:end_i]:
                y, x, ye, xe = patches[idx]
                erased[:, :, y:ye, x:xe] = 0.0
                erased_count += 1
            scores_e.append(torch.sigmoid(model(erased))[0, target_class].item())
            fracs_e.append(erased_count / n_patches)

        erasure_aucs.append(float(np.trapz(scores_e, fracs_e)))

        # Random restoration
        restored = torch.zeros_like(image).to(device)
        scores_r = [torch.sigmoid(model(restored))[0, target_class].item()]
        fracs_r = [0.0]
        restored_count = 0

        for step_i in range(0, n_patches, patches_per_step):
            end_i = min(step_i + patches_per_step, n_patches)
            for idx in order[step_i:end_i]:
                y, x, ye, xe = patches[idx]
                restored[:, :, y:ye, x:xe] = image[:, :, y:ye, x:xe].to(device)
                restored_count += 1
            scores_r.append(torch.sigmoid(model(restored))[0, target_class].item())
            fracs_r.append(restored_count / n_patches)

        restoration_aucs.append(float(np.trapz(scores_r, fracs_r)))

    return {
        "erasure_auc_mean": float(np.mean(erasure_aucs)),
        "erasure_auc_std": float(np.std(erasure_aucs)),
        "restoration_auc_mean": float(np.mean(restoration_aucs)),
        "restoration_auc_std": float(np.std(restoration_aucs)),
    }


# ============================================================
# 2. Pointing Game (basic)
#    Source: Zhang et al., referenced in Selvaraju et al.
# ============================================================

def pointing_game(
    cam: np.ndarray,
    bbox: Tuple[int, int, int, int],
) -> float:
    """
    Does the max activation fall inside the ground-truth bounding box?

    Args:
        cam: (H, W) heatmap
        bbox: (x_min, y_min, x_max, y_max)

    Returns: 1.0 (hit) or 0.0 (miss)
    """
    max_y, max_x = np.unravel_index(cam.argmax(), cam.shape)
    x_min, y_min, x_max, y_max = bbox
    return 1.0 if (x_min <= max_x <= x_max and y_min <= max_y <= y_max) else 0.0


# ============================================================
# 3. Energy-Based Pointing Game
#    Extension: what PROPORTION of the heatmap energy is inside the bbox?
#    More informative than binary hit/miss.
# ============================================================

def energy_pointing_game(
    cam: np.ndarray,
    bbox: Tuple[int, int, int, int],
) -> float:
    """
    Proportion of total heatmap energy concentrated inside the bbox.

    More nuanced than basic pointing game: a heatmap that puts 90% of
    its energy on the lesion is better than one that puts 51%.

    Args:
        cam: (H, W) heatmap, values ≥ 0
        bbox: (x_min, y_min, x_max, y_max)

    Returns: float in [0, 1], higher = more faithful
    """
    x_min, y_min, x_max, y_max = bbox
    total_energy = cam.sum()
    if total_energy < 1e-8:
        return 0.0
    bbox_energy = cam[y_min:y_max + 1, x_min:x_max + 1].sum()
    return float(bbox_energy / total_energy)


# ============================================================
# 4. Insertion / Deletion AUC (Petsiuk et al.)
#    Already existed in old code, but rewritten to use patches
#    (matching the erasure/restoration framework) for consistency.
# ============================================================

@torch.no_grad()
def insertion_deletion_auc(
    model: nn.Module,
    image: torch.Tensor,
    cam: np.ndarray,
    target_class: int,
    device: torch.device,
    steps: int = 50,
) -> Dict[str, float]:
    """
    Pixel-level insertion and deletion (Petsiuk et al.).

    Insertion: reveal pixels most-to-least important → track confidence.
    Deletion: mask pixels most-to-least important → track confidence.

    Higher insertion AUC = better. Lower deletion AUC = better.
    """
    model.eval()
    H, W = cam.shape
    n_pixels = H * W
    step_size = max(n_pixels // steps, 1)

    sorted_indices = np.argsort(cam.flatten())[::-1]

    base_prob = torch.sigmoid(model(image.to(device)))[0, target_class].item()

    blank = torch.zeros_like(image)
    insertion_scores = [0.0]
    deletion_scores = [base_prob]

    mask = np.zeros(n_pixels, dtype=bool)

    for step in range(0, n_pixels, step_size):
        end = min(step + step_size, n_pixels)
        mask[sorted_indices[step:end]] = True
        mask_2d = mask.reshape(H, W)
        mask_t = torch.from_numpy(mask_2d).float().unsqueeze(0).unsqueeze(0)
        mask_t = mask_t.expand_as(image).to(device)

        # Insertion
        inserted = blank.to(device) * (1 - mask_t) + image.to(device) * mask_t
        ins_prob = torch.sigmoid(model(inserted))[0, target_class].item()
        insertion_scores.append(ins_prob)

        # Deletion
        deleted = image.to(device) * (1 - mask_t)
        del_prob = torch.sigmoid(model(deleted))[0, target_class].item()
        deletion_scores.append(del_prob)

    x = np.linspace(0, 1, len(insertion_scores))
    return {
        "insertion_auc": float(np.trapz(insertion_scores, x)),
        "deletion_auc": float(np.trapz(deletion_scores, x)),
    }


# ============================================================
# 5. Faithfulness Correlation
#    Source: Selvaraju et al. — thesis §1.6.3
#
#    For each image region:
#      - Compute attribution value (from heatmap)
#      - Compute prediction change when that region is removed
#    Faithfulness = Pearson correlation between the two.
#    A faithful heatmap → high correlation (regions the heatmap calls
#    important actually ARE important for the prediction).
# ============================================================

@torch.no_grad()
def faithfulness_correlation(
    model: nn.Module,
    image: torch.Tensor,
    cam: np.ndarray,
    target_class: int,
    device: torch.device,
    patch_size: int = 16,
) -> float:
    """
    Pearson correlation between attribution values and actual prediction drops.

    For each patch:
      - attribution_k = mean heatmap value in patch k
      - delta_k = baseline_prob - prob_when_patch_k_is_masked

    Faithfulness = corr(attribution, delta)

    Higher correlation = more faithful attribution.

    Returns: Pearson correlation coefficient in [-1, 1]
    """
    model.eval()
    H, W = image.shape[2], image.shape[3]

    baseline_prob = torch.sigmoid(model(image.to(device)))[0, target_class].item()

    attributions = []
    deltas = []

    for y in range(0, H, patch_size):
        for x in range(0, W, patch_size):
            y_end = min(y + patch_size, H)
            x_end = min(x + patch_size, W)

            # Attribution value for this patch
            attr = cam[y:y_end, x:x_end].mean()
            attributions.append(attr)

            # Mask this patch and measure prediction change
            masked = image.clone().to(device)
            masked[:, :, y:y_end, x:x_end] = 0.0
            prob = torch.sigmoid(model(masked))[0, target_class].item()
            delta = baseline_prob - prob  # positive = removing patch hurts prediction
            deltas.append(delta)

    attributions = np.array(attributions)
    deltas = np.array(deltas)

    # Pearson correlation
    if attributions.std() < 1e-8 or deltas.std() < 1e-8:
        return 0.0

    correlation = np.corrcoef(attributions, deltas)[0, 1]
    return float(correlation)


# ============================================================
# 6. Aggregated faithfulness report for a single image
# ============================================================

def compute_single_image_faithfulness(
    model: nn.Module,
    image: torch.Tensor,
    cam: np.ndarray,
    target_class: int,
    device: torch.device,
    bbox: Optional[Tuple[int, int, int, int]] = None,
    patch_size: int = 16,
) -> Dict[str, float]:
    """
    Run all faithfulness metrics on a single image + heatmap.

    Returns a flat dict of metric_name → value.
    """
    results = {}

    # Insertion / Deletion
    ins_del = insertion_deletion_auc(model, image, cam, target_class, device, steps=50)
    results["insertion_auc"] = ins_del["insertion_auc"]
    results["deletion_auc"] = ins_del["deletion_auc"]

    # Progressive Erasure
    erasure = progressive_erasure(model, image, cam, target_class, device, patch_size=patch_size)
    results["erasure_auc"] = erasure["auc"]

    # Progressive Restoration
    restoration = progressive_restoration(model, image, cam, target_class, device, patch_size=patch_size)
    results["restoration_auc"] = restoration["auc"]

    # Faithfulness Correlation
    results["faithfulness_corr"] = faithfulness_correlation(
        model, image, cam, target_class, device, patch_size=patch_size
    )

    # Pointing game (if bbox available)
    if bbox is not None:
        results["pointing_game"] = pointing_game(cam, bbox)
        results["energy_pointing_game"] = energy_pointing_game(cam, bbox)

    return results
