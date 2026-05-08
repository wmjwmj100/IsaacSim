#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

PANTHERA_REPO_URL="${PANTHERA_REPO_URL:-https://github.com/HighTorque-Robotics/Panthera-HT-ROS2.git}"
PANTHERA_ROS2_ROOT="${PANTHERA_ROS2_ROOT:-$SCRIPT_DIR/external/Panthera-HT-ROS2}"
PANTHERA_SCRIPT="$SCRIPT_DIR/source/standalone_examples/custom/panthera_ht_table.py"
ISAAC_PYTHON="$SCRIPT_DIR/_build/linux-x86_64/release/python.sh"
DEFAULT_PANTHERA_URDF="$PANTHERA_ROS2_ROOT/src/panthera_ht_description_with_finger/urdf/Panthera-HT_description_with_finger.urdf.xacro"

export PM_PACKAGES_ROOT="${PM_PACKAGES_ROOT:-$SCRIPT_DIR/.cache/packman}"
export PYTHONUNBUFFERED="${PYTHONUNBUFFERED:-1}"

log() {
    printf '\n[Panthera-HT] %s\n' "$*"
}

ensure_git() {
    if ! command -v git >/dev/null 2>&1; then
        echo "git is required to download Panthera-HT assets, but it was not found in PATH."
        exit 1
    fi
}

ensure_panthera_assets() {
    if [[ -n "${PANTHERA_URDF:-}" && -f "$PANTHERA_URDF" ]]; then
        log "Using PANTHERA_URDF=$PANTHERA_URDF"
        return
    fi

    ensure_git
    mkdir -p "$SCRIPT_DIR/external"

    if [[ ! -d "$PANTHERA_ROS2_ROOT/.git" ]]; then
        log "Downloading Panthera-HT ROS2 assets"
        git clone --depth 1 "$PANTHERA_REPO_URL" "$PANTHERA_ROS2_ROOT"
    elif [[ "${PANTHERA_UPDATE_ASSETS:-0}" == "1" ]]; then
        log "Updating Panthera-HT ROS2 assets"
        git -C "$PANTHERA_ROS2_ROOT" pull --ff-only
    else
        log "Using existing Panthera-HT assets at $PANTHERA_ROS2_ROOT"
    fi

    if [[ ! -f "$DEFAULT_PANTHERA_URDF" ]]; then
        echo "Panthera-HT URDF/xacro was not found at:"
        echo "  $DEFAULT_PANTHERA_URDF"
        echo "Set PANTHERA_URDF=/absolute/path/to/Panthera-HT_description_with_finger.urdf.xacro and retry."
        exit 1
    fi

    export PANTHERA_ROS2_ROOT
    export PANTHERA_URDF="$DEFAULT_PANTHERA_URDF"
    log "Using Panthera-HT URDF: $PANTHERA_URDF"
}

ensure_isaac_python() {
    if [[ -x "$ISAAC_PYTHON" ]]; then
        log "Using Isaac Sim Python: $ISAAC_PYTHON"
        return
    fi

    if [[ ! -x "$SCRIPT_DIR/build.sh" ]]; then
        echo "Isaac Sim Python was not found and build.sh is not executable: $SCRIPT_DIR/build.sh"
        exit 1
    fi

    log "Isaac Sim Python not found; building Isaac Sim release first"
    "$SCRIPT_DIR/build.sh" --release

    if [[ ! -x "$ISAAC_PYTHON" ]]; then
        echo "Build finished, but Isaac Sim Python is still missing: $ISAAC_PYTHON"
        exit 1
    fi
}

launch_scene() {
    log "Launching Panthera-HT on an 80cm x 80cm table in Isaac Sim"
    log "Forwarded scene args: ${*:-<none>}"
    log "PANTHERA_HEADLESS=${PANTHERA_HEADLESS:-<unset>} PANTHERA_MAX_FRAMES=${PANTHERA_MAX_FRAMES:-<unset>} PANTHERA_SHOW_LAYOUT_CAMERA_VIEWPORTS=${PANTHERA_SHOW_LAYOUT_CAMERA_VIEWPORTS:-<unset>}"
    exec "$ISAAC_PYTHON" "$PANTHERA_SCRIPT" "$@"
}

ensure_panthera_assets
ensure_isaac_python
launch_scene "$@"
