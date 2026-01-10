#!/bin/bash
#SBATCH -J asmc_only
#SBATCH -A COMPUTERLAB-SL2-GPU
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00
#SBATCH -p ampere
#SBATCH --output=results/asmc_only-%A_%a.out
#SBATCH --error=results/asmc_only-%A_%a.err

# ============================================================
# ASMC Only Experiment (faster, no baselines)
# ============================================================

# Load modules
module purge
module load rhel8/default-amp
module load cuda/11.4
module load python/3.10

# Activate environment
source ~/.bashrc
conda activate samp

# Set environment variables
export HF_HOME=/rds/project/rds-hirjiakbGUV/co-wang1/HF_CACHE
export TRANSFORMERS_CACHE=/rds/project/rds-hirjiakbGUV/co-wang1/HF_CACHE
export HF_DATASETS_CACHE=/rds/project/rds-hirjiakbGUV/co-wang1/HF_CACHE

# Get batch index
BATCH_IDX=${SLURM_ARRAY_TASK_ID:-${1:-0}}

cd /rds-d5/user/co-wang1/hpc-work/LLMsample/reasoning-with-sampling/asmc_standalone

echo "============================================================"
echo "Starting ASMC Only Experiment"
echo "Batch: ${BATCH_IDX}"
echo "Time: $(date)"
echo "============================================================"

python asmc_full_comparison.py \
    --save_str=results/qwen_math \
    --model=qwen_math \
    --dataset=MATH \
    --cot \
    --batch_idx=${BATCH_IDX} \
    --seed=0 \
    --max_tokens=3072 \
    --temp=0.25 \
    --n_particles=32 \
    --block_size=32 \
    --ess_threshold=0.5 \
    --epsilon=0.05 \
    --anneal_tokens=512 \
    --alpha_start=1.5 \
    --anneal_schedule=cosine \
    --early_stop_mass=0.80 \
    --enable_adaptive \
    --use_batched \
    --run_asmc \
    --verbose

echo "============================================================"
echo "Job completed at $(date)"
echo "============================================================"
