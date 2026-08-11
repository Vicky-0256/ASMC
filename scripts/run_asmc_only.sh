#!/usr/bin/env bash
#SBATCH --job-name=asmc_only
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00
#SBATCH --output=logs/asmc_only-%A_%a.out
#SBATCH --error=logs/asmc_only-%A_%a.err

set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
project_root="$(cd -- "${script_dir}/.." && pwd)"
cd "${project_root}"

if [[ $# -gt 1 ]]; then
    echo "usage: $0 [batch-index]" >&2
    exit 2
fi

batch_idx="${SLURM_ARRAY_TASK_ID:-${1:-0}}"
results_dir="${ASMC_RESULTS_DIR:-results/runs/math500/asmc_only}"
mode="${ASMC_MODE:-fixed}"
run_profile="${ASMC_RUN_PROFILE:-full}"

if [[ ! "${batch_idx}" =~ ^[0-4]$ ]]; then
    echo "batch index must be an integer between 0 and 4 for MATH500" >&2
    exit 2
fi

if [[ "${results_dir}" == "/" ]]; then
    echo "ASMC_RESULTS_DIR must not be the filesystem root" >&2
    exit 2
fi

case "${run_profile}" in
    full)
        default_n_problems=""
        default_n_particles="64"
        default_max_tokens="3072"
        default_anneal_tokens="512"
        default_hard_anneal_tokens="768"
        default_attn_implementation="flash_attention_2"
        ;;
    smoke)
        default_n_problems="1"
        default_n_particles="4"
        default_max_tokens="256"
        default_anneal_tokens="64"
        default_hard_anneal_tokens="64"
        default_attn_implementation="sdpa"
        ;;
    *) echo "ASMC_RUN_PROFILE must be 'full' or 'smoke'" >&2; exit 2 ;;
esac

n_problems="${ASMC_N_PROBLEMS-${default_n_problems}}"
n_particles="${ASMC_N_PARTICLES:-${default_n_particles}}"
max_tokens="${ASMC_MAX_TOKENS:-${default_max_tokens}}"
anneal_tokens="${ASMC_ANNEAL_TOKENS:-${default_anneal_tokens}}"
hard_anneal_tokens="${ASMC_HARD_ANNEAL_TOKENS:-${default_hard_anneal_tokens}}"
attn_implementation="${ASMC_ATTN_IMPLEMENTATION:-${default_attn_implementation}}"
n_problem_args=()
if [[ -n "${n_problems}" ]]; then
    if [[ ! "${n_problems}" =~ ^([1-9]|[1-9][0-9]|100)$ ]]; then
        echo "ASMC_N_PROBLEMS must be an integer between 1 and 100" >&2
        exit 2
    fi
    n_problem_args=(--n_problems "${n_problems}")
fi

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

mkdir -p -- "${results_dir}" logs

echo "ASMC-only: profile=${run_profile}, batch=${batch_idx}, mode=${mode}, output=${results_dir}"
# Baseline switches in asmc_full_comparison.py are opt-in. This deliberately
# passes only --run_asmc and exposes no arbitrary extra-argument escape hatch.
# Consequently greedy, naive, standard, MCMC, majority and Best-of-N remain off.
"${runner[@]}" asmc_full_comparison.py \
    --save_str="${results_dir}" \
    --model="${ASMC_MODEL:-qwen_math}" \
    --dtype="${ASMC_DTYPE:-bfloat16}" \
    --attn_implementation="${attn_implementation}" \
    --dataset=MATH \
    --cot \
    --batch_idx="${batch_idx}" \
    "${n_problem_args[@]}" \
    --seed="${ASMC_SEED:-0}" \
    --max_tokens="${max_tokens}" \
    --temp="${ASMC_TEMPERATURE:-0.25}" \
    --n_particles="${n_particles}" \
    --block_size="${ASMC_BLOCK_SIZE:-32}" \
    --ess_threshold="${ASMC_ESS_THRESHOLD:-0.5}" \
    --epsilon="${ASMC_EPSILON:-0.05}" \
    --anneal_tokens="${anneal_tokens}" \
    --alpha_start="${ASMC_ALPHA_START:-1.5}" \
    --hard_anneal_tokens="${hard_anneal_tokens}" \
    --hard_alpha_start="${ASMC_HARD_ALPHA_START:-1.3}" \
    --hard_ess_threshold="${ASMC_HARD_ESS_THRESHOLD:-0.6}" \
    --anneal_schedule="${ASMC_ANNEAL_SCHEDULE:-cosine}" \
    --early_stop_mass="${ASMC_EARLY_STOP_MASS:-0.80}" \
    --fast_mass_threshold="${ASMC_FAST_MASS_THRESHOLD:-0.65}" \
    --hard_n_particles="${ASMC_HARD_N_PARTICLES:-${n_particles}}" \
    --asmc_vote_mode="${ASMC_VOTE_MODE:-weighted_no_source}" \
    --device="${ASMC_DEVICE:-auto}" \
    --use_batched \
    --run_asmc \
    "${mode_args[@]}" \
    "${legacy_args[@]}" \
    "${cap_args[@]}" \
    "${revision_args[@]}"
