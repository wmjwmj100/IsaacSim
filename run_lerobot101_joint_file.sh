#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

export SO101_JOINT_TARGET_FILE="${SO101_JOINT_TARGET_FILE:-$HOME/.cache/isaacsim/so101_joint_target.json}"
export SO101_JOINT_TARGET_UNITS="${SO101_JOINT_TARGET_UNITS:-rad}"

exec "$SCRIPT_DIR/run_lerobot101_act.sh" \
    --robot-model so101 \
    --enable-joint-target-file \
    --joint-target-file "$SO101_JOINT_TARGET_FILE" \
    --joint-target-units "$SO101_JOINT_TARGET_UNITS" \
    --disable-free-text \
    "$@"
