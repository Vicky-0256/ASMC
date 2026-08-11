#!/usr/bin/env bash
#SBATCH --job-name=asmc_full
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --output=logs/full_comparison-%A_%a.out
#SBATCH --error=logs/full_comparison-%A_%a.err

# Site-neutral MATH500 comparison runner. Submit with --account/--partition
# arguments appropriate for your cluster, or execute it directly.
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
project_root="$(cd -- "${script_dir}/.." && pwd)"
cd "${project_root}"

batch_idx="${SLURM_ARRAY_TASK_ID:-${1:-0}}"
results_dir="${ASMC_RESULTS_DIR:-results/runs/math500/full_comparison}"
mode="${ASMC_MODE:-fixed}"

case "${mode}" in
    fixed) mode_args=(--fixed) ;;
    adaptive) mode_args=(--adaptive) ;;
    *) echo "ASMC_MODE must be 'fixed' or 'adaptive'" >&2; exit 2 ;;
esac

legacy_args=()
if [[ "${ASMC_LEGACY_STOP_CONSTRAINTS:-0}" == "1" ]]; then
    legacy_args=(--legacy_stop_constraints)
fi

revision_args=()
if [[ -n "${ASMC_MODEL_REVISION:-}" ]]; then
    revision_args=(--model_revision "${ASMC_MODEL_REVISION}")
fi

cap_args=()
if [[ -n "${ASMC_C_INT_CAP:-}" ]]; then
    cap_args=(--c_int_cap "${ASMC_C_INT_CAP}")
fi

if [[ -n "${ASMC_CONDA_ENV:-}" ]]; then
    runner=(conda run --no-capture-output -n "${ASMC_CONDA_ENV}" python)
else
    runner=("${ASMC_PYTHON:-python}")
fi

mkdir -p "${results_dir}" logs

echo "ASMC comparison: batch=${batch_idx}, mode=${mode}, output=${results_dir}"
"${runner[@]}" asmc_full_comparison.py \
    --save_str="${results_dir}" \
    --model="${ASMC_MODEL:-qwen_math}" \
    --dtype="${ASMC_DTYPE:-bfloat16}" \
    --attn_implementation="${ASMC_ATTN_IMPLEMENTATION:-flash_attention_2}" \
    --dataset=MATH \
    --cot \
    --batch_idx="${batch_idx}" \
    --seed="${ASMC_SEED:-0}" \
    --max_tokens="${ASMC_MAX_TOKENS:-3072}" \
    --temp="${ASMC_TEMPERATURE:-0.25}" \
    --n_particles="${ASMC_N_PARTICLES:-64}" \
    --block_size="${ASMC_BLOCK_SIZE:-32}" \
    --ess_threshold="${ASMC_ESS_THRESHOLD:-0.5}" \
    --epsilon="${ASMC_EPSILON:-0.05}" \
    --anneal_tokens="${ASMC_ANNEAL_TOKENS:-512}" \
    --alpha_start="${ASMC_ALPHA_START:-1.5}" \
    --anneal_schedule=cosine \
    --early_stop_mass="${ASMC_EARLY_STOP_MASS:-0.80}" \
    --hard_n_particles="${ASMC_HARD_N_PARTICLES:-${ASMC_N_PARTICLES:-64}}" \
    --asmc_vote_mode="${ASMC_VOTE_MODE:-weighted_no_source}" \
    --use_batched \
    --run_greedy \
    --run_asmc \
    --run_bestofn \
    --bestofn_n="${ASMC_BESTOFN_N:-4}" \
    --bestofn_temp="${ASMC_BESTOFN_TEMPERATURE:-${ASMC_TEMPERATURE:-0.25}}" \
    --bestofn_chunk_size="${ASMC_BESTOFN_CHUNK_SIZE:-8}" \
    --run_naive \
    --run_std \
    --run_mcmc \
    "${mode_args[@]}" \
    "${legacy_args[@]}" \
    "${cap_args[@]}" \
    "${revision_args[@]}"
