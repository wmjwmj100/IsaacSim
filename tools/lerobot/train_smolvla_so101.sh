#!/usr/bin/env bash
set -euo pipefail

ENV_NAME="${LEROBOT_CONDA_ENV:-isaacsim-lerobot}"
DATASET_REPO_ID="${DATASET_REPO_ID:-lerobot/svla_so101_pickplace}"
BASE_MODEL_ID="${BASE_MODEL_ID:-lerobot/smolvla_base}"
OUTPUT_DIR="${OUTPUT_DIR:-$(pwd)/outputs/train/so101_smolvla_pickplace}"
JOB_NAME="${JOB_NAME:-so101_smolvla_pickplace}"
POLICY_REPO_ID="${POLICY_REPO_ID:-}"
DEVICE="${DEVICE:-cuda}"
DTYPE="${DTYPE:-bfloat16}"
STEPS="${STEPS:-20000}"
BATCH_SIZE="${BATCH_SIZE:-4}"
GRADIENT_CHECKPOINTING="${GRADIENT_CHECKPOINTING:-true}"

if ! command -v conda >/dev/null 2>&1; then
    echo "conda not found. Install Miniconda/Anaconda first."
    exit 1
fi

CONDA_BASE="$(conda info --base)"
source "${CONDA_BASE}/etc/profile.d/conda.sh"
conda activate "${ENV_NAME}"

cmd=(
    lerobot-train
    "--dataset.repo_id=${DATASET_REPO_ID}"
    "--output_dir=${OUTPUT_DIR}"
    "--job_name=${JOB_NAME}"
    "--policy.path=${BASE_MODEL_ID}"
    "--policy.device=${DEVICE}"
    "--policy.dtype=${DTYPE}"
    "--policy.gradient_checkpointing=${GRADIENT_CHECKPOINTING}"
    "--steps=${STEPS}"
    "--batch_size=${BATCH_SIZE}"
)

if [[ -n "${POLICY_REPO_ID}" ]]; then
    cmd+=("--policy.repo_id=${POLICY_REPO_ID}")
fi

echo "Running training command:"
printf '  %q' "${cmd[@]}"
printf '\n'

exec "${cmd[@]}" "$@"
