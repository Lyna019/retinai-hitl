#!/bin/bash
# =============================================================
# Phase 3: Hybrid CNN+Transformer experiments
#
# IMPORTANT: Update __TOP1__, __BEST_RES__ placeholders in
# configs/experiments/phase3_hybrid.yaml first.
# =============================================================
set -e

echo "=============================================="
echo " PHASE 3 — Hybrid CNN+Transformer"
echo "=============================================="

for exp in "H-01" "H-02"; do
    echo ">>> Starting: $exp"
    python run_experiment.py --experiment_id "$exp" "$@"
    echo ">>> Completed: $exp"
done

echo "=============================================="
echo " PHASE 3 COMPLETE"
echo "=============================================="
