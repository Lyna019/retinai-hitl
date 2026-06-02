#!/bin/bash
# =============================================================
# Phase 2A: Resolution Study (4 new runs on Top-2 backbones)
# Phase 2B: Loss Function Study (2 new runs)
#
# IMPORTANT: Before running, update the __TOP1__, __TOP2__, etc.
# placeholders in configs/experiments/phase2_resolution_loss.yaml
# with actual backbone names from Phase 1 results.
# =============================================================
set -e

echo "=============================================="
echo " PHASE 2A — Resolution Study"
echo "=============================================="

for exp in "R-02" "R-03" "R-05" "R-06"; do
    echo ">>> Starting: $exp"
    python run_experiment.py --experiment_id "$exp" "$@"
    echo ">>> Completed: $exp"
done

echo ""
echo "=============================================="
echo " PHASE 2B — Loss Function Study"
echo "=============================================="

for exp in "L-02" "L-03"; do
    echo ">>> Starting: $exp"
    python run_experiment.py --experiment_id "$exp" "$@"
    echo ">>> Completed: $exp"
done

echo "=============================================="
echo " PHASE 2 COMPLETE"
echo "=============================================="
