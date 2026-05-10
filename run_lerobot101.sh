#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"
DESKTOP_SCRIPT="$SCRIPT_DIR/source/standalone_examples/custom/lerobot_so101_desktop.py"

export PM_PACKAGES_ROOT="${PM_PACKAGES_ROOT:-$SCRIPT_DIR/.cache/packman}"
LEROBOT_CONDA_ENV="${LEROBOT_CONDA_ENV:-deeplearning}"
DEFAULT_SO101_URDF="${HOME}/.cache/robot_descriptions/SO-ARM100/Simulation/SO101/so101_new_calib.urdf"

if [[ -z "${LEROBOT_SO101_URDF:-}" && -f "$DEFAULT_SO101_URDF" ]]; then
    export LEROBOT_SO101_URDF="$DEFAULT_SO101_URDF"
fi

if [[ ! -x "$SCRIPT_DIR/_build/linux-x86_64/release/python.sh" ]]; then
    echo "Isaac Sim is not built yet: $SCRIPT_DIR/_build/linux-x86_64/release/python.sh"
    echo "Run ./build.sh --release first, then retry."
    exit 1
fi

if command -v conda >/dev/null 2>&1; then
    CONDA_BASE="$(conda info --base)"
    # Isaac Sim's setup script must be sourced from the release directory so it can find setup_python_env.sh.
    source "${CONDA_BASE}/etc/profile.d/conda.sh"
    conda activate "${LEROBOT_CONDA_ENV}"
    pushd "$SCRIPT_DIR/_build/linux-x86_64/release" >/dev/null
    export ZSH_VERSION="${ZSH_VERSION-}"
    export PYTHONPATH="${PYTHONPATH-}"
    export LD_LIBRARY_PATH="${LD_LIBRARY_PATH-}"
    source ./setup_conda_env.sh
    popd >/dev/null
    exec python "$DESKTOP_SCRIPT"
fi

exec "$SCRIPT_DIR/_build/linux-x86_64/release/python.sh" \
    "$DESKTOP_SCRIPT"
