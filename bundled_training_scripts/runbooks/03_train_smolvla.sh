#!/usr/bin/env bash
set -euo pipefail

cd /work/IsaacSim

.venv-lerobot/bin/python bundled_training_scripts/tools/lerobot/train_roboclaw_smolvla_widepos_117ep.py \
  --dataset-root outputs/lerobot_datasets/roboclaw_sim_teacher_widepos_117ep \
  --dataset-repo-id local/roboclaw_sim_teacher_widepos_117ep \
  --base-model lerobot/smolvla_base \
  --base-checkpoint outputs/train/roboclaw_sim_teacher_widepos_clean_89ep_smolvla_20k_cont_from5k/checkpoints/015000/pretrained_model \
  --output-dir outputs/train/roboclaw_sim_teacher_widepos_117ep_smolvla_5k_from15k_repro \
  --job-name roboclaw_sim_teacher_widepos_117ep_smolvla_5k_from15k_repro \
  --train-episodes 0:110 \
  --val-episodes 110:117 \
  --steps 5000 \
  --batch-size 1 \
  --num-workers 0 \
  --log-freq 100 \
  --save-freq 1000 \
  --device cuda \
  --overwrite
