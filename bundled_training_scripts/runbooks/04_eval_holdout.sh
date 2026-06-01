#!/usr/bin/env bash
set -euo pipefail

cd /work/IsaacSim

.venv-lerobot/bin/python bundled_training_scripts/tools/lerobot/eval_smolvla_action_chunk.py \
  --model-id outputs/train/roboclaw_sim_teacher_widepos_117ep_smolvla_5k_from15k/checkpoints/005000/pretrained_model \
  --dataset-repo-id local/roboclaw_sim_teacher_widepos_117ep \
  --dataset-root outputs/lerobot_datasets/roboclaw_sim_teacher_widepos_117ep \
  --episode-indices 110,111,112,113,114,115,116 \
  --frame-offsets 0,20,50,90 \
  --device cuda \
  --task "push the object on the table" \
  --output-json outputs/offline_smolvla_eval_widepos_117ep_holdout_5k_from15k/repro_chunk_report.json
