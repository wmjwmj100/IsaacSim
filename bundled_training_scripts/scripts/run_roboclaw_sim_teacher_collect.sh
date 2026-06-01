#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

PYTHON_SH="$SCRIPT_DIR/_build/linux-x86_64/release/python.sh"
SCRIPT="$SCRIPT_DIR/source/standalone_examples/custom/roboclaw_sim_teacher_collect.py"

export PM_PACKAGES_ROOT="${PM_PACKAGES_ROOT:-$SCRIPT_DIR/.cache/packman}"
export PYTHONUNBUFFERED="${PYTHONUNBUFFERED:-1}"

if [[ ! -x "$PYTHON_SH" ]]; then
    echo "Isaac Sim Python is not built yet: $PYTHON_SH"
    echo "Run ./build.sh --release first, then retry."
    exit 1
fi

echo "[Roboclaw][Teacher] Launching collection with args: ${*:-<none>}"
exec "$PYTHON_SH" "$SCRIPT" "$@"
