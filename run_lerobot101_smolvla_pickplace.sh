#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

export SMOLVLA_MODEL_ID="${SMOLVLA_MODEL_ID:-lerobot/smolvla_base}"
export SMOLVLA_DEVICE="${SMOLVLA_DEVICE:-cuda}"
export SMOLVLA_TASK_ID="${SMOLVLA_TASK_ID:-blue_to_left}"
export SMOLVLA_TASK_TEXT="${SMOLVLA_TASK_TEXT:-}"
export SMOLVLA_VLA_IO_DIR="${SMOLVLA_VLA_IO_DIR:-$HOME/.cache/isaacsim/vla_io_so101}"
export SMOLVLA_MAX_CONTROL_FRAMES="${SMOLVLA_MAX_CONTROL_FRAMES:-48}"

exec "$SCRIPT_DIR/run_lerobot101_act.sh" \
    --robot-model so101 \
    --disable-hf-prior \
    --enable-smolvla-prior \
    --smolvla-model-id "$SMOLVLA_MODEL_ID" \
    --smolvla-device "$SMOLVLA_DEVICE" \
    --smolvla-infer-interval 8 \
    --task-id "$SMOLVLA_TASK_ID" \
    --command "$SMOLVLA_TASK_TEXT" \
    --save-vla-io \
    --vla-io-dir "$SMOLVLA_VLA_IO_DIR" \
    --max-control-frames "$SMOLVLA_MAX_CONTROL_FRAMES" \
    --profile-smolvla \
    "$@"
