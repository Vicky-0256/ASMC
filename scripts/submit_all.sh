#!/bin/bash
# ============================================================
# Submit all 5 batches for MATH500 (100 problems per batch)
# ============================================================

# Usage:
#   ./submit_all.sh              # Submit full comparison (all methods)
#   ./submit_all.sh asmc_only    # Submit ASMC only (faster)

MODE=${1:-full}

cd /rds-d5/user/co-wang1/hpc-work/LLMsample/reasoning-with-sampling/asmc_standalone

# Create results directory
mkdir -p results/qwen_math

if [ "$MODE" == "asmc_only" ]; then
    echo "Submitting ASMC-only jobs for batches 0-4..."
    sbatch --array=0-4 scripts/run_asmc_only.sh
else
    echo "Submitting full comparison jobs for batches 0-4..."
    sbatch --array=0-4 scripts/run_full_comparison.sh
fi

echo "Jobs submitted. Check status with: squeue -u \$USER"
