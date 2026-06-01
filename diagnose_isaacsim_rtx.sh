#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

required_driver="580.65.06"
installed_driver="unknown"
if command -v nvidia-smi >/dev/null 2>&1; then
    installed_driver="$(nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null | head -n 1 | tr -d ' ')"
fi

printf '[RTX Diagnose] Isaac Sim repo: %s\n' "$SCRIPT_DIR"
printf '[RTX Diagnose] Expected Isaac Sim 5.1 Linux test driver: %s\n' "$required_driver"
printf '[RTX Diagnose] Installed NVIDIA driver: %s\n' "$installed_driver"
printf '[RTX Diagnose] OS: '
(. /etc/os-release && printf '%s\n' "$PRETTY_NAME") 2>/dev/null || uname -a
printf '[RTX Diagnose] Kernel: %s\n' "$(uname -r)"

if [[ "$installed_driver" == 595.* ]]; then
    cat <<MSG
[RTX Diagnose] RESULT: installed driver is R595, while Isaac Sim 5.1 x86_64 docs list Linux R580.65.06 as the tested driver.
[RTX Diagnose] This matches the observed native crash in librtx.scenedb.plugin.so after omni.hydra.rtx startup.
[RTX Diagnose] Local app/user-cache resets and renderer toggles have already been tested and did not change the crash stack.
MSG
elif [[ "$installed_driver" == 580.* ]]; then
    echo '[RTX Diagnose] RESULT: driver major matches Isaac Sim 5.1 R580 test branch; rerun RTX smoke after reboot/driver reload.'
else
    echo '[RTX Diagnose] RESULT: driver major differs from Isaac Sim 5.1 tested R580 branch; prefer R580 before debugging RTX further.'
fi

cat <<'MSG'
[RTX Diagnose] Suggested system-level fix, requires sudo and reboot:
  sudo apt install nvidia-driver-580-open
  sudo reboot
[RTX Diagnose] After reboot, validate:
  cd /work/IsaacSim
  ./diagnose_isaacsim_rtx.sh
  cd _build/linux-x86_64/release && ./isaac-sim.sh --reset-user --portable --/persistent/renderer/startupMessageDisplayed=1
[RTX Diagnose] Safe Panthera smoke while RTX is blocked:
  ./run_panthera_ht_safe.sh
MSG
