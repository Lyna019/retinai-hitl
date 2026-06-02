# Retinal Pathology AI — Experimental Pipeline

**Thesis**: AI-Assisted Diagnosis for Retinal Pathologies from Fundus Images  
**Authors**: Kaouther Bensedira, Lyna Ikhelef  
**Institution**: ENSIA, in collaboration with Ibn Al Haythem Center

---

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Login to Weights & Biases
wandb login

# 3. Organize your data (see Data Setup below)

# 4. Prepare merged dataset + splits (run ONCE)
python scripts/prepare_data.py --data_root ./data

# 5. Run Phase 1 (backbone screening)
bash scripts/run_phase1.sh
```

---

## Project Structure

```
retinal-thesis/
├── configs/
│   ├── base.yaml                          # Shared config (loss, optimizer, augmentation, etc.)
│   └── experiments/
│       ├── phase1_backbone.yaml           # B-01 to B-06
│       ├── phase2_resolution_loss.yaml    # R-02 to R-06, L-02 to L-03
│       ├── phase3_hybrid.yaml             # H-01, H-02
│       └── ablation_and_external.yaml     # P-01 to P-03, T-01 to T-02
├── src/
│   ├── data/
│   │   ├── dataset.py                     # PyTorch Dataset + DataLoader builder
│   │   ├── label_harmonization.py         # Label mapping + stratified splits
│   │   ├── preprocessing.py               # Border removal, CLAHE, resize
│   │   └── transforms.py                  # Albumentations augmentation + TTA
│   ├── models/
│   │   ├── backbone.py                    # Single-backbone models (timm)
│   │   ├── hybrid.py                      # Dual-branch + CNN-Transformer head
│   │   ├── ensemble.py                    # Averaging + stacking ensembles
│   │   └── factory.py                     # Unified model builder
│   ├── losses/
│   │   └── losses.py                      # BCE, Focal, Asymmetric Loss
│   ├── training/
│   │   ├── trainer.py                     # Training loop + W&B logging
│   │   └── metrics.py                     # AUC, F1, ECE, Brier, threshold optim
│   ├── explainability/
│   │   └── gradcam.py                     # Grad-CAM + faithfulness metrics
│   ├── calibration/
│   │   └── temperature_scaling.py         # Post-hoc calibration
│   └── utils/
│       ├── config.py                      # YAML config loader + merger
│       └── helpers.py                     # Seed, device, class weights
├── scripts/
│   ├── prepare_data.py                    # Data merging + split creation
│   ├── run_phase1.sh                      # Backbone screening
│   ├── run_phase2.sh                      # Resolution + loss study
│   ├── run_phase3.sh                      # Hybrid experiments
│   ├── run_ablation.sh                    # Preprocessing ablation
│   └── run_evaluation.py                  # Ensemble, calibration, external, TTA
├── run_experiment.py                      # Main entry point (any experiment)
├── requirements.txt
└── README.md
```

---

## Data Setup

Place datasets under `./data/` with this structure:

```
data/
├── rfmid/
│   ├── RFMiD_Training_Labels.csv
│   └── Training/
│       ├── 1.png
│       └── ...
├── odir/
│   ├── full_df.csv
│   └── images/
│       ├── 0_left.jpg
│       └── ...
├── eyedisease/
│   ├── diabetic_retinopathy/
│   ├── glaucoma/
│   ├── cataract/
│   ├── normal/
│   └── ...
└── brset_images/          # External validation (BRSET)
    └── ...
```

The `prepare_data.py` script will:
- Load all three datasets
- Harmonize labels into unified taxonomy
- Create iterative stratified 70/15/15 splits
- Save `artifacts/splits.npz` and `artifacts/label_mapping.json`

---

## Running Experiments

### Single Experiment
```bash
python run_experiment.py --experiment_id B-03
```

### Full Pipeline (sequential phases)
```bash
# Phase 1: Backbone screening (6 models at 224px)
bash scripts/run_phase1.sh

# → Check W&B, identify Top-2 backbones
# → Update __TOP1__, __TOP2__ placeholders in phase2 config

# Phase 2: Resolution (224/384/512) + Loss (BCE/Focal/ASL)
bash scripts/run_phase2.sh

# → Lock in champion: backbone + resolution + loss

# Phase 3: Hybrid CNN+Transformer
bash scripts/run_phase3.sh

# Phase 4/5: Ensemble, Calibration, External, TTA, Grad-CAM
python scripts/run_evaluation.py \
    --checkpoint checkpoints/B-03_best.pt checkpoints/B-06_best.pt checkpoints/H-01_best.pt \
    --run_ensemble --run_calibration --run_tta --run_gradcam_validation \
    --run_external --brset_csv data/brset_labels.csv --brset_images data/brset_images

# Preprocessing Ablation
bash scripts/run_ablation.sh
```

---

## W&B Dashboard

All experiments log to project `retinal-pathology-thesis`.

**Groups**: `phase1-backbone-screening`, `phase2a-resolution`, `phase2b-loss`, `phase3-hybrid`, `phase4-5-evaluation`, `ablation-preprocessing`

**Key metrics**: `val/macro_auc`, `val/f1_macro`, `val/ece`, `gradcam/samples`

**Artifacts**: Dataset splits, model checkpoints, Grad-CAM heatmaps

---

## Controlled Variables

Every experiment shares these (unless explicitly ablated):

| Parameter | Value |
|---|---|
| Seed | 42 |
| Split | 70/15/15 iterative stratified |
| Optimizer | AdamW (backbone 1e-4, head 5e-4) |
| Scheduler | CosineAnnealingWarmRestarts (T₀=10, T_mult=2) |
| Loss | BCEWithLogitsLoss (weighted, cap=10×) |
| Early stopping | patience=10 on val macro-AUC |
| Mixed precision | FP16 |
| Augmentation | HFlip, Rotate±30°, BrightnessContrast, ColorJitter, Cutout |
