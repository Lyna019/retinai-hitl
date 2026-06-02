#!/bin/bash
# =============================================================
# Phase 1: Backbone Screening
# Runs all 6 backbone experiments sequentially.
# =============================================================
set -e

echo "=============================================="
echo " PHASE 1 — Backbone Screening (6 experiments)"
echo "=============================================="

EXPERIMENTS=("B-01" "B-02" "B-03" "B-04" "B-05" "B-06")

for exp in "${EXPERIMENTS[@]}"; do
    echo ""
    echo ">>> Starting experiment: $exp"
    echo "----------------------------------------------"
    python run_experiment.py --experiment_id "$exp" "$@"
    echo ">>> Completed: $exp"
    echo ""
done

echo "=============================================="
echo " PHASE 1 COMPLETE — Check W&B for results"
echo " Compare: wandb.ai/<entity>/retinal-pathology-thesis"
echo "=============================================="
