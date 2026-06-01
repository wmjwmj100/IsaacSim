#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

DEFAULT_MAX_FRAMES="${PANTHERA_MAX_FRAMES:-3}"
DRIVER_VERSION="unknown"
if command -v nvidia-smi >/dev/null 2>&1; then
    DRIVER_VERSION="$(nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null | head -n 1 | tr -d ' ')"
fi

cat <<MSG
[Panthera-HT] Safe smoke mode
[Panthera-HT] NVIDIA driver: $DRIVER_VERSION
[Panthera-HT] Using Isaac Sim no-rendering mode because full RTX/SceneDB currently crashes on this host.
[Panthera-HT] For full RTX rendering with Isaac Sim 5.1, install a supported R580 driver and reboot.
MSG

export PANTHERA_HEADLESS=1
export PANTHERA_NO_RENDERING=1
export PANTHERA_NO_MOTION="${PANTHERA_NO_MOTION:-1}"
export PANTHERA_DISABLE_LAYOUT_CAMERAS=1
export PANTHERA_SKIP_LAYOUT_SCREENSHOTS=1
export PANTHERA_DISABLE_LAYOUT_CAMERA_VIEWPORTS=1
export PANTHERA_MAX_FRAMES="$DEFAULT_MAX_FRAMES"

exec "$SCRIPT_DIR/run_panthera_ht_table.sh" \
    --headless \
    --no-rendering \
    --no-motion \
    --disable-layout-cameras \
    --skip-layout-screenshots \
    --max-frames "$DEFAULT_MAX_FRAMES" \
    "$@"
