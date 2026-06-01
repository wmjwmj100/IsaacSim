#!/usr/bin/env bash
set -euo pipefail

cd /work/IsaacSim

.venv-lerobot/bin/python bundled_training_scripts/tools/lerobot/roboclaw_raw_to_lerobot.py \
  --raw-root outputs/roboclaw_sim_teacher_raw/widepos_new \
  --dataset-root outputs/lerobot_datasets/roboclaw_sim_teacher_widepos_new \
  --repo-id local/roboclaw_sim_teacher_widepos_new \
  --robot-type panthera_ht \
  --task-text "push the object on the table" \
  --only-successful \
  --overwrite
