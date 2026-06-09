#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

export ROBOCLAW_ENABLE_MCP_VLA_BRIDGE="${ROBOCLAW_ENABLE_MCP_VLA_BRIDGE:-1}"
export ROBOCLAW_MCP_VLA_BRIDGE_DIR="${ROBOCLAW_MCP_VLA_BRIDGE_DIR:-$SCRIPT_DIR/outputs/isaacsim_vla_bridge}"
export ROBOCLAW_CONTROL_MODE="${ROBOCLAW_CONTROL_MODE:-single-arm}"
export ROBOCLAW_MAX_FRAMES="${ROBOCLAW_MAX_FRAMES:-0}"
export ROBOCLAW_USE_PERSISTENT_HELPER="${ROBOCLAW_USE_PERSISTENT_HELPER:-1}"
export ROBOCLAW_TASK_TEXT="${ROBOCLAW_TASK_TEXT:-move the object in the red box to the green box}"

echo "[Roboclaw][MCPBridge] Shared bridge dir: $ROBOCLAW_MCP_VLA_BRIDGE_DIR"
echo "[Roboclaw][MCPBridge] Configure MCP with ISAACSIM_VLA_BACKEND=file and ISAACSIM_VLA_BRIDGE_DIR=$ROBOCLAW_MCP_VLA_BRIDGE_DIR"

exec "$SCRIPT_DIR/run_roboclaw_smolvla.sh" \
    --enable-mcp-vla-bridge \
    --mcp-vla-bridge-dir "$ROBOCLAW_MCP_VLA_BRIDGE_DIR" \
    --control-mode "$ROBOCLAW_CONTROL_MODE" \
    --max-frames "$ROBOCLAW_MAX_FRAMES" \
    "$@"
