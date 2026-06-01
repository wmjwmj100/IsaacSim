#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

ROBOCLAW_SCRIPT="$SCRIPT_DIR/source/standalone_examples/custom/roboclaw_smolvla_rollout.py"

export PM_PACKAGES_ROOT="${PM_PACKAGES_ROOT:-$SCRIPT_DIR/.cache/packman}"
export PYTHONUNBUFFERED="${PYTHONUNBUFFERED:-1}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export HF_HUB_DISABLE_XET="${HF_HUB_DISABLE_XET:-1}"

if [[ ! -x "$SCRIPT_DIR/_build/linux-x86_64/release/python.sh" ]]; then
    echo "Isaac Sim is not built yet: $SCRIPT_DIR/_build/linux-x86_64/release/python.sh"
    echo "Run ./build.sh --release first, then retry."
    exit 1
fi

echo "[Roboclaw] Launching SmolVLA rollout with args: ${*:-<none>}"
exec "$SCRIPT_DIR/_build/linux-x86_64/release/python.sh" "$ROBOCLAW_SCRIPT" "$@"
