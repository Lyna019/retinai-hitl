#!/bin/bash
# =============================================================
# Preprocessing Ablation
# Runs P-01, P-02, P-03 with champion backbone.
# P-04 = full pipeline (reuse champion run from Phase 2).
#
# IMPORTANT: Update configs/experiments/ablation_and_external.yaml
# with your champion backbone before running.
# =============================================================
set -e

echo "=============================================="
echo " PREPROCESSING ABLATION"
echo "=============================================="

for exp in "P-01" "P-02" "P-03"; do
    echo ">>> Starting: $exp"
    python run_experiment.py --experiment_id "$exp" "$@"
    echo ">>> Completed: $exp"
done

echo "=============================================="
echo " ABLATION COMPLETE"
echo "=============================================="
