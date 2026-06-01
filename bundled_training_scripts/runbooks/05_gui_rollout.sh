#!/usr/bin/env bash
set -euo pipefail

cd /work/IsaacSim

./run_roboclaw_smolvla.sh \
  --usd-path /home/wmj/Documents/roboclaw.usd \
  --smolvla-model-id outputs/train/roboclaw_sim_teacher_widepos_117ep_smolvla_5k_from15k/checkpoints/005000/pretrained_model \
  --dataset-root outputs/lerobot_datasets/roboclaw_sim_teacher_widepos_117ep \
  --control-mode single-arm \
  --active-arm-label front_left \
  --policy-align-initial-state-replay-json outputs/roboclaw_dataset_replay/panthera_episode_000000_replay.json \
  --policy-align-initial-state-index 0 \
  --policy-align-settle-frames 5 \
  --target-initial-x -0.10 \
  --target-initial-y 0.05 \
  --target-initial-z 0.7725 \
  --max-frames 0 \
  --warmup-frames 24 \
  --infer-interval 30 \
  --frame-file-ring-size 120 \
  --max-records-in-memory 500 \
  --keep-open \
  --task-text "push the object on the table" \
  --output-dir outputs/roboclaw_smolvla_rollout_gui_repro
