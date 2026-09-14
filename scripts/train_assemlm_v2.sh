#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
RELEASE_ROOT=$(cd "${SCRIPT_DIR}/.." && pwd)
PROJECT_ROOT="${RELEASE_ROOT}"

: "${ASSEMLM_VLM_PATH:?Set ASSEMLM_VLM_PATH to the local Qwen3-VL model directory}"

GPU_IDS=${GPU_IDS:-0}
RUN_ID=${RUN_ID:-assemlm_v2}
RUN_ROOT_DIR_EFFECTIVE=${RUN_ROOT_DIR:-${PWD}/results}
MAIN_PROCESS_PORT=${MAIN_PROCESS_PORT:-29517}
ACCELERATE_BIN=${ASSEMLM_ACCELERATE:-accelerate}
PYTHON_BIN=${ASSEMLM_PYTHON:-python}
WANDB_MODE=${WANDB_MODE:-offline}
DRY_RUN=${DRY_RUN:-false}
ASSEMLM_DATA_ROOT=${ASSEMLM_DATA_ROOT:-${RELEASE_ROOT}/datasets/AssemLM2}
export ASSEMLM_DATA_ROOT

check_dataset_files() {
    local dataset_path found=0
    if [[ ! -d "${ASSEMLM_DATA_ROOT}" ]]; then
        echo "ASSEMLM_DATA_ROOT is not a directory: ${ASSEMLM_DATA_ROOT}" >&2
        exit 1
    fi
    for dataset_path in "${ASSEMLM_DATA_ROOT}"/assemlm_*.hdf5; do
        [[ -f "${dataset_path}" ]] || continue
        printf 'Discovered dataset: %s\n' "${dataset_path}"
        found=$((found + 1))
    done
    if (( found == 0 )); then
        echo "No assemlm_*.hdf5 files found under ${ASSEMLM_DATA_ROOT}" >&2
        exit 1
    fi
}

if ! [[ "${GPU_IDS}" =~ ^[0-9]+(,[0-9]+)*$ ]]; then
    echo "GPU_IDS must contain comma-separated numeric GPU ids, for example 0 or 0,1." >&2
    exit 2
fi
declare -A seen_gpu_ids=()
IFS=',' read -r -a GPU_LIST <<< "${GPU_IDS}"
for gpu_id in "${GPU_LIST[@]}"; do
    if [[ -n "${seen_gpu_ids[${gpu_id}]:-}" ]]; then
        echo "GPU_IDS must not contain duplicate ids: ${GPU_IDS}" >&2
        exit 2
    fi
    seen_gpu_ids["${gpu_id}"]=1
done
if ! [[ "${SEED:-42}" =~ ^[0-9]+$ ]]; then
    echo "SEED must be a non-negative integer, got: ${SEED:-42}" >&2
    exit 2
fi
if ! [[ "${MAIN_PROCESS_PORT}" =~ ^[0-9]+$ ]] || ((
    MAIN_PROCESS_PORT < 1 || MAIN_PROCESS_PORT > 65535
)); then
    echo "MAIN_PROCESS_PORT must be an integer between 1 and 65535." >&2
    exit 2
fi
if [[ ! -d "${ASSEMLM_VLM_PATH}" ]]; then
    echo "ASSEMLM_VLM_PATH is not a directory: ${ASSEMLM_VLM_PATH}" >&2
    exit 1
fi
if [[ "${DRY_RUN}" != "true" ]]; then
    check_dataset_files
fi
if [[ "${DRY_RUN}" != "true" && -e "${RUN_ROOT_DIR_EFFECTIVE}/${RUN_ID}" ]]; then
    echo "Training run already exists: ${RUN_ROOT_DIR_EFFECTIVE}/${RUN_ID}. Set a new RUN_ID; refusing to mix fresh state with an existing run." >&2
    exit 2
fi

if ! PYTHON_RESOLVED=$(command -v "${PYTHON_BIN}"); then
    echo "Python executable not found: ${PYTHON_BIN}" >&2
    exit 1
fi
export PATH="$(dirname "${PYTHON_RESOLVED}"):${PATH}"
if ! ACCELERATE_RESOLVED=$(command -v "${ACCELERATE_BIN}"); then
    echo "Accelerate executable not found: ${ACCELERATE_BIN}" >&2
    exit 1
fi

export CUDA_VISIBLE_DEVICES="${GPU_IDS}"
export PYTHONPATH="${RELEASE_ROOT}:${PYTHONPATH:-}"
export PYTHONHASHSEED="${SEED:-42}"
export WANDB_MODE
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

NUM_PROCESSES=${#GPU_LIST[@]}
if [[ "${NUM_PROCESSES}" -eq 1 ]]; then
    ACCELERATE_CONFIG="${RELEASE_ROOT}/config/deepspeed_zero2_1gpu.yaml"
else
    ACCELERATE_CONFIG="${RELEASE_ROOT}/config/deepspeed_zero2.yaml"
fi

CMD=(
    "${ACCELERATE_RESOLVED}" launch
    --config_file "${ACCELERATE_CONFIG}"
    --num_processes "${NUM_PROCESSES}"
    --main_process_port "${MAIN_PROCESS_PORT}"
    "${RELEASE_ROOT}/assemlm/training/train_assemlm_v2.py"
    --config_yaml "${RELEASE_ROOT}/config/assemlm_v2.yaml"
    --seed "${SEED:-42}"
    --run_root_dir "${RUN_ROOT_DIR_EFFECTIVE}"
    --run_id "${RUN_ID}"
)

printf 'Training command:'
printf ' %q' "${CMD[@]}"
printf '\n'
if [[ "${DRY_RUN}" == "true" ]]; then
    exit 0
fi
cd "${PROJECT_ROOT}"
exec "${CMD[@]}"
