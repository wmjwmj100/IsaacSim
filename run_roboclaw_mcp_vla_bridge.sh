#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

export ROBOCLAW_ENABLE_MCP_VLA_BRIDGE="${ROBOCLAW_ENABLE_MCP_VLA_BRIDGE:-1}"
export ROBOCLAW_MCP_VLA_BRIDGE_DIR="${ROBOCLAW_MCP_VLA_BRIDGE_DIR:-$SCRIPT_DIR/outputs/isaacsim_vla_bridge}"
export ROBOCLAW_CONTROL_MODE="${ROBOCLAW_CONTROL_MODE:-single-arm}"
export ROBOCLAW_MAX_FRAMES="${ROBOCLAW_MAX_FRAMES:-0}"
export ROBOCLAW_USE_PERSISTENT_HELPER="${ROBOCLAW_USE_PERSISTENT_HELPER:-1}"
export ROBOCLAW_SMOLVLA_MODEL_ID="${ROBOCLAW_SMOLVLA_MODEL_ID:-$SCRIPT_DIR/outputs/train/roboclaw_data613_vp_30ep_smolvla_expert/checkpoints/005000/pretrained_model}"
export ROBOCLAW_DATASET_ROOT="${ROBOCLAW_DATASET_ROOT:-$SCRIPT_DIR/outputs/lerobot_datasets/roboclaw_data613_vp_30ep}"
export ROBOCLAW_TASK_TEXT="${ROBOCLAW_TASK_TEXT:-Pick up the object inside the green box and place it at the location marked by the blue box.}"

echo "[Roboclaw][MCPBridge] Shared bridge dir: $ROBOCLAW_MCP_VLA_BRIDGE_DIR"
echo "[Roboclaw][MCPBridge] SmolVLA checkpoint: $ROBOCLAW_SMOLVLA_MODEL_ID"
echo "[Roboclaw][MCPBridge] Dataset root: $ROBOCLAW_DATASET_ROOT"
echo "[Roboclaw][MCPBridge] Configure MCP with ISAACSIM_VLA_BACKEND=file and ISAACSIM_VLA_BRIDGE_DIR=$ROBOCLAW_MCP_VLA_BRIDGE_DIR"

exec "$SCRIPT_DIR/run_roboclaw_smolvla.sh" \
    --enable-mcp-vla-bridge \
    --mcp-vla-bridge-dir "$ROBOCLAW_MCP_VLA_BRIDGE_DIR" \
    --control-mode "$ROBOCLAW_CONTROL_MODE" \
    --max-frames "$ROBOCLAW_MAX_FRAMES" \
    "$@"
