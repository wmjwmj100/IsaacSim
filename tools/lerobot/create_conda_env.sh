#!/usr/bin/env bash
set -euo pipefail

ENV_NAME="${1:-isaacsim-lerobot}"
PYTHON_VERSION="${PYTHON_VERSION:-3.11}"
TORCH_VERSION="${TORCH_VERSION:-2.7.1}"
TORCHVISION_VERSION="${TORCHVISION_VERSION:-0.22.1}"
LEROBOT_VERSION="${LEROBOT_VERSION:-0.4.3}"

if ! command -v conda >/dev/null 2>&1; then
    echo "conda not found. Install Miniconda/Anaconda first."
    exit 1
fi

CONDA_BASE="$(conda info --base)"
source "${CONDA_BASE}/etc/profile.d/conda.sh"

if conda env list | awk '{print $1}' | grep -Fxq "${ENV_NAME}"; then
    echo "Conda env already exists: ${ENV_NAME}"
else
    echo "Creating conda env ${ENV_NAME} with Python ${PYTHON_VERSION}"
    conda create -y -n "${ENV_NAME}" "python=${PYTHON_VERSION}"
fi

conda activate "${ENV_NAME}"
python -m pip install --upgrade pip
python -m pip install --upgrade "socksio>=1,<2"
python -m pip install \
    --upgrade \
    "torch==${TORCH_VERSION}" \
    "torchvision==${TORCHVISION_VERSION}"
python -m pip install \
    --upgrade \
    "lerobot[smolvla]==${LEROBOT_VERSION}" \
    "datasets>=4.0.0,<4.2.0" \
    "pillow>=10,<13" \
    "imageio>=2.34,<3"

echo
echo "Environment ready."
echo "Activate it with:"
echo "  source \"${CONDA_BASE}/etc/profile.d/conda.sh\""
echo "  conda activate ${ENV_NAME}"
