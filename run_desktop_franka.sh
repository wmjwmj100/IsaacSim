#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

export PM_PACKAGES_ROOT="${PM_PACKAGES_ROOT:-$SCRIPT_DIR/.cache/packman}"

exec "$SCRIPT_DIR/_build/linux-x86_64/release/python.sh" \
    "$SCRIPT_DIR/source/standalone_examples/custom/franka_desktop_pick_place.py"
