#!/usr/bin/env bash
set -euo pipefail

cd /work/IsaacSim

./run_roboclaw_sim_teacher_collect.sh \
  --usd-path /home/wmj/Documents/roboclaw.usd \
  --output-dir outputs/roboclaw_sim_teacher_raw \
  --run-name widepos_new \
  --episodes 50 \
  --frames-per-episode 150 \
  --seed 4300 \
  --cube-center-x -0.18 \
  --cube-center-y 0.135 \
  --cube-range-x 0.045 \
  --cube-range-y 0.035 \
  --active-arm-label front_left \
  --headless \
  --overwrite
