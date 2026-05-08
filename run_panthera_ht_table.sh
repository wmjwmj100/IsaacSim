#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

PANTHERA_SCRIPT="$SCRIPT_DIR/source/standalone_examples/custom/panthera_ht_table.py"
DEFAULT_PANTHERA_ROS2_ROOT="$SCRIPT_DIR/external/Panthera-HT-ROS2"
DEFAULT_PANTHERA_URDF="$DEFAULT_PANTHERA_ROS2_ROOT/src/panthera_ht_description_with_finger/urdf/Panthera-HT_description_with_finger.urdf.xacro"

export PM_PACKAGES_ROOT="${PM_PACKAGES_ROOT:-$SCRIPT_DIR/.cache/packman}"
export PYTHONUNBUFFERED="${PYTHONUNBUFFERED:-1}"

if [[ -z "${PANTHERA_ROS2_ROOT:-}" && -d "$DEFAULT_PANTHERA_ROS2_ROOT" ]]; then
    export PANTHERA_ROS2_ROOT="$DEFAULT_PANTHERA_ROS2_ROOT"
fi

if [[ -z "${PANTHERA_URDF:-}" && -f "$DEFAULT_PANTHERA_URDF" ]]; then
    export PANTHERA_URDF="$DEFAULT_PANTHERA_URDF"
fi

if [[ ! -x "$SCRIPT_DIR/_build/linux-x86_64/release/python.sh" ]]; then
    echo "Isaac Sim is not built yet: $SCRIPT_DIR/_build/linux-x86_64/release/python.sh"
    echo "Run ./build.sh --release first, then retry."
    exit 1
fi

if [[ -z "${PANTHERA_URDF:-}" || ! -f "${PANTHERA_URDF:-}" ]]; then
    echo "Panthera-HT URDF not found. Clone the official ROS2 assets first:"
    echo "  mkdir -p external"
    echo "  git clone https://github.com/HighTorque-Robotics/Panthera-HT-ROS2.git external/Panthera-HT-ROS2"
    echo "or set PANTHERA_URDF=/absolute/path/to/Panthera-HT_description_with_finger.urdf.xacro"
    exit 1
fi

echo "[Panthera-HT] Launching Panthera scene with args: ${*:-<none>}"
echo "[Panthera-HT] PANTHERA_HEADLESS=${PANTHERA_HEADLESS:-<unset>} PANTHERA_MAX_FRAMES=${PANTHERA_MAX_FRAMES:-<unset>}"
exec "$SCRIPT_DIR/_build/linux-x86_64/release/python.sh" "$PANTHERA_SCRIPT" "$@"
