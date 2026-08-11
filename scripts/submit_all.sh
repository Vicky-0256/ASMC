#!/usr/bin/env bash
# Submit the public ASMC-only MATH500 entry point. Cluster-specific account and
# partition options can be passed after the optional mode, for example:
#   ./scripts/submit_all.sh asmc_only --account=my-account --partition=gpu
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
project_root="$(cd -- "${script_dir}/.." && pwd)"
cd "${project_root}"

mode="asmc_only"
if [[ $# -gt 0 && "${1}" != -* ]]; then
    mode="${1}"
    shift
fi

if [[ "${mode}" != "asmc_only" ]]; then
    echo "usage: $0 [asmc_only] [sbatch options...]" >&2
    echo "this public submission entry point runs ASMC only" >&2
    exit 2
fi

for argument in "$@"; do
    case "${argument}" in
        --array|--array=*|-a|-a=*|-a[0-9]*)
            echo "pass the batch range through ASMC_BATCH_ARRAY, not sbatch --array/-a" >&2
            exit 2
            ;;
    esac
done

run_profile="${ASMC_RUN_PROFILE:-full}"
case "${run_profile}" in
    full) default_array="0-4" ;;
    smoke) default_array="0" ;;
    *) echo "ASMC_RUN_PROFILE must be 'full' or 'smoke'" >&2; exit 2 ;;
esac

batch_array="${ASMC_BATCH_ARRAY:-${default_array}}"
if [[ "${batch_array}" =~ ^([0-4])(-([0-4]))?(%([1-9][0-9]*))?$ ]]; then
    array_start="${BASH_REMATCH[1]}"
    array_end="${BASH_REMATCH[3]:-${array_start}}"
else
    echo "ASMC_BATCH_ARRAY must be a MATH500 batch or range within 0-4 (optionally %N)" >&2
    exit 2
fi
if (( array_start > array_end )); then
    echo "ASMC_BATCH_ARRAY start must not exceed its end" >&2
    exit 2
fi

mkdir -p -- logs
echo "Submitting ASMC-only profile=${run_profile}, batches=${batch_array}"
sbatch "$@" --array="${batch_array}" scripts/run_asmc_only.sh
