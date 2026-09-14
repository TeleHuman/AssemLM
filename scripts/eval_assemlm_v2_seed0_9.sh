#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
RELEASE_ROOT=$(cd "${SCRIPT_DIR}/.." && pwd)

: "${CHECKPOINT_PATH:?Set CHECKPOINT_PATH to the model checkpoint file}"
: "${CONFIG_PATH:?Set CONFIG_PATH to the matching AssemLM v2 config.yaml}"
: "${ASSEMLM_VLM_PATH:?Set ASSEMLM_VLM_PATH to the local Qwen3-VL model directory}"

GPU_IDS=${GPU_IDS:-0}
EVAL_SEEDS_CSV=${EVAL_SEEDS_CSV:-0,1,2,3,4,5,6,7,8,9}
EVAL_DATASETS_CSV=${EVAL_DATASETS_CSV:-biasassembly,ikea,partnet,twobytwo}
RUN_TIMESTAMP=${RUN_TIMESTAMP:-$(date +%Y%m%d_%H%M%S)}
OUTPUT_ROOT=${OUTPUT_ROOT:-${PWD}/eval_results}
PYTHON_BIN=${ASSEMLM_PYTHON:-python}
ACCELERATE_BIN=${ASSEMLM_ACCELERATE:-accelerate}
DRY_RUN=${DRY_RUN:-false}
MAIN_PROCESS_PORT=${MAIN_PROCESS_PORT:-29518}
TRAIN_NUM_SAMPLES=${TRAIN_NUM_SAMPLES:-100}
TEST_NUM_SAMPLES=${TEST_NUM_SAMPLES:-200}
PARTNET_TEST_NUM_SAMPLES=${PARTNET_TEST_NUM_SAMPLES:-1000}
BIASSEMBLY_TEST_NUM_SAMPLES=${BIASSEMBLY_TEST_NUM_SAMPLES:-1300}
ASSEMLM_DATA_ROOT=${ASSEMLM_DATA_ROOT:-${RELEASE_ROOT}/datasets/AssemLM2}
export ASSEMLM_DATA_ROOT

if ! PYTHON_RESOLVED=$(command -v "${PYTHON_BIN}"); then
    echo "Python executable not found: ${PYTHON_BIN}" >&2
    exit 1
fi
export PATH="$(dirname "${PYTHON_RESOLVED}"):${PATH}"

if ! [[ "${GPU_IDS}" =~ ^[0-9]+(,[0-9]+)*$ ]]; then
    echo "GPU_IDS must contain comma-separated numeric GPU ids." >&2
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

if ! [[ "${EVAL_SEEDS_CSV}" =~ ^[0-9]+(,[0-9]+)*$ ]]; then
    echo "EVAL_SEEDS_CSV must contain comma-separated non-negative integer seeds." >&2
    exit 2
fi
declare -A seen_eval_seeds=()
IFS=',' read -r -a EVAL_SEEDS <<< "${EVAL_SEEDS_CSV}"
for eval_seed in "${EVAL_SEEDS[@]}"; do
    if [[ -n "${seen_eval_seeds[${eval_seed}]:-}" ]]; then
        echo "EVAL_SEEDS_CSV must not contain duplicate seeds: ${EVAL_SEEDS_CSV}" >&2
        exit 2
    fi
    seen_eval_seeds["${eval_seed}"]=1
done

validate_count() {
    local name="$1" value="$2"
    if [[ "${value}" != "all" ]] && { ! [[ "${value}" =~ ^[0-9]+$ ]] || (( value < 1 )); }; then
        echo "${name} must be a positive integer or all, got: ${value}" >&2
        exit 2
    fi
}
validate_count TRAIN_NUM_SAMPLES "${TRAIN_NUM_SAMPLES}"
validate_count TEST_NUM_SAMPLES "${TEST_NUM_SAMPLES}"
validate_count PARTNET_TEST_NUM_SAMPLES "${PARTNET_TEST_NUM_SAMPLES}"
validate_count BIASSEMBLY_TEST_NUM_SAMPLES "${BIASSEMBLY_TEST_NUM_SAMPLES}"

if ! [[ "${MAIN_PROCESS_PORT}" =~ ^[0-9]+$ ]] || (( MAIN_PROCESS_PORT < 1 || MAIN_PROCESS_PORT > 65535 )); then
    echo "MAIN_PROCESS_PORT must be an integer between 1 and 65535." >&2
    exit 2
fi
if [[ ! -f "${CONFIG_PATH}" ]]; then
    echo "CONFIG_PATH does not exist: ${CONFIG_PATH}" >&2
    exit 1
fi
if [[ "${DRY_RUN}" != "true" && ! -f "${CHECKPOINT_PATH}" ]]; then
    echo "CHECKPOINT_PATH does not exist: ${CHECKPOINT_PATH}" >&2
    exit 1
fi
if [[ ! -d "${ASSEMLM_VLM_PATH}" ]]; then
    echo "ASSEMLM_VLM_PATH is not a directory: ${ASSEMLM_VLM_PATH}" >&2
    exit 1
fi
if [[ ! -d "${ASSEMLM_DATA_ROOT}" ]]; then
    echo "ASSEMLM_DATA_ROOT is not a directory: ${ASSEMLM_DATA_ROOT}" >&2
    exit 1
fi

DATASETS=()
if [[ "${EVAL_DATASETS_CSV,,}" == "auto" || "${EVAL_DATASETS_CSV}" == "*" || "${EVAL_DATASETS_CSV,,}" == "all" ]]; then
    shopt -s nullglob
    dataset_paths=("${ASSEMLM_DATA_ROOT}"/assemlm_*.hdf5)
    shopt -u nullglob
    for dataset_path in "${dataset_paths[@]}"; do
        dataset_file=$(basename "${dataset_path}")
        DATASETS+=("${dataset_file#assemlm_}")
        DATASETS[${#DATASETS[@]}-1]="${DATASETS[${#DATASETS[@]}-1]%.hdf5}"
    done
else
    IFS=',' read -r -a requested_datasets <<< "${EVAL_DATASETS_CSV}"
    for dataset_name in "${requested_datasets[@]}"; do
        dataset_name="${dataset_name,,}"
        dataset_name="${dataset_name#assemlm_}"
        dataset_name="${dataset_name%.hdf5}"
        DATASETS+=("${dataset_name}")
    done
fi

if (( ${#DATASETS[@]} == 0 )); then
    echo "No evaluation datasets selected." >&2
    exit 1
fi
for dataset_name in "${DATASETS[@]}"; do
    dataset_path="${ASSEMLM_DATA_ROOT}/assemlm_${dataset_name}.hdf5"
    if [[ ! -f "${dataset_path}" ]]; then
        echo "Missing dataset file: ${dataset_path}" >&2
        exit 1
    fi
    printf 'Evaluation dataset: %s\n' "${dataset_path}"
done

export CUDA_VISIBLE_DEVICES="${GPU_IDS}"
export PYTHONPATH="${RELEASE_ROOT}:${PYTHONPATH:-}"
export WANDB_MODE="disabled"

NUM_PROCESSES=${#GPU_LIST[@]}
if [[ "${NUM_PROCESSES}" -gt 1 ]]; then
    if ! ACCELERATE_RESOLVED=$(command -v "${ACCELERATE_BIN}"); then
        echo "Accelerate executable not found: ${ACCELERATE_BIN}" >&2
        exit 1
    fi
else
    ACCELERATE_RESOLVED="${ACCELERATE_BIN}"
fi
if [[ "${NUM_PROCESSES}" -eq 1 ]]; then
    LAUNCHER=("${PYTHON_BIN}")
else
    LAUNCHER=("${ACCELERATE_RESOLVED}" launch --multi_gpu --num_processes "${NUM_PROCESSES}" --main_process_port "${MAIN_PROCESS_PORT}")
fi

CKPT_LABEL=$(basename "${CHECKPOINT_PATH}")
RUN_ROOT="${OUTPUT_ROOT}/${CKPT_LABEL}/${RUN_TIMESTAMP}"
SEED_RESULT_ARGS=()
if [[ "${DRY_RUN}" != "true" && -e "${RUN_ROOT}" && "${ALLOW_OVERWRITE:-false}" != "true" ]]; then
    echo "Evaluation run already exists: ${RUN_ROOT}. Set a new RUN_TIMESTAMP or ALLOW_OVERWRITE=true explicitly." >&2
    exit 2
fi

test_sample_count() {
    case "$1" in
        twobytwo|ikea) printf 'all' ;;
        biasassembly) printf '%s' "${BIASSEMBLY_TEST_NUM_SAMPLES}" ;;
        partnet) printf '%s' "${PARTNET_TEST_NUM_SAMPLES}" ;;
        *) printf '%s' "${TEST_NUM_SAMPLES}" ;;
    esac
}

run_eval() {
    local seed="$1" label="$2" split="$3" sample_count="$4"
    local output_dir="${RUN_ROOT}/${label}/seed_${seed}/${split}"
    local -a cmd=(
        "${LAUNCHER[@]}"
        "${RELEASE_ROOT}/assemlm/eval/eval_assemlm_v2.py"
        --config_yaml "${CONFIG_PATH}"
        --ckpt_path "${CHECKPOINT_PATH}"
        --seed "${seed}"
        --dataset_name "${label}"
        --split "${split}"
        --output_dir "${output_dir}"
        --run_timestamp "${RUN_TIMESTAMP}"
        --shuffle
        --framework.vlm.base_vlm "${ASSEMLM_VLM_PATH}"
        --framework.point_encoder.vn_dgcnn_patch.fusion_module patch_transformer
        --framework.point_projector.backbone_output_dim 6
        --datasets.assemble_data.train_mix "${label}"
        --datasets.assemble_data.test_mix "${label}"
        --datasets.assemble_data.num_workers 0
    )
    if [[ "${sample_count}" != "all" ]]; then
        cmd+=(--num_samples "${sample_count}")
    fi
    printf '\nseed=%s dataset=%s split=%s command:' "${seed}" "${label}" "${split}"
    printf ' %q' "${cmd[@]}"
    printf '\n'
    if [[ "${DRY_RUN}" != "true" ]]; then
        "${cmd[@]}"
    fi
}

for seed in "${EVAL_SEEDS[@]}"; do
    for label in "${DATASETS[@]}"; do
        run_eval "${seed}" "${label}" train "${TRAIN_NUM_SAMPLES}"
        run_eval "${seed}" "${label}" test "$(test_sample_count "${label}")"
    done

    seed_args=(
        --output_file "${RUN_ROOT}/seed_${seed}/eval_results.json"
        --run_timestamp "${RUN_TIMESTAMP}"
        --seed "${seed}"
    )
    for label in "${DATASETS[@]}"; do
        for split in train test; do
            seed_args+=(--result "${label}:${split}:${RUN_ROOT}/${label}/seed_${seed}/${split}/${RUN_TIMESTAMP}/eval_results.json")
        done
    done
    printf '\nseed=%s aggregate command:' "${seed}"
    printf ' %q' "${PYTHON_BIN}" "${RELEASE_ROOT}/assemlm/eval/aggregate_eval_results.py" "${seed_args[@]}"
    printf '\n'
    if [[ "${DRY_RUN}" != "true" ]]; then
        "${PYTHON_BIN}" "${RELEASE_ROOT}/assemlm/eval/aggregate_eval_results.py" "${seed_args[@]}"
    fi
    SEED_RESULT_ARGS+=(--result "${seed}:${RUN_ROOT}/seed_${seed}/eval_results.json")
done

final_args=(
    --output_file "${RUN_ROOT}/eval_results_mean_variance.json"
    --run_timestamp "${RUN_TIMESTAMP}"
    --expected_seeds "${EVAL_SEEDS_CSV}"
    "${SEED_RESULT_ARGS[@]}"
)
printf '\nCross-seed aggregate command:'
printf ' %q' "${PYTHON_BIN}" "${RELEASE_ROOT}/assemlm/eval/aggregate_seed_results.py" "${final_args[@]}"
printf '\n'
if [[ "${DRY_RUN}" != "true" ]]; then
    "${PYTHON_BIN}" "${RELEASE_ROOT}/assemlm/eval/aggregate_seed_results.py" "${final_args[@]}"
fi
