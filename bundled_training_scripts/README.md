# Roboclaw SmolVLA Training Bundle

这个文件夹整理了本次 Roboclaw 物块推动任务里和训练相关的脚本、配置和报告。原始文件没有移动，这里是复制件，方便交付或复现。

## 当前模型和数据

- 基座模型: `lerobot/smolvla_base`
- 当前训练数据: `outputs/lerobot_datasets/roboclaw_sim_teacher_widepos_117ep`
- 当前新 checkpoint: `outputs/train/roboclaw_sim_teacher_widepos_117ep_smolvla_5k_from15k/checkpoints/005000/pretrained_model`
- 训练集划分: `110` episodes train, `7` episodes holdout
- 离线 holdout 评估: `reports/holdout_eval_chunk_report_5k_from15k.json`

## 目录

- `scripts/`: shell 启动脚本。
- `source/standalone_examples/custom/`: Isaac Sim 采集和 GUI rollout 脚本。
- `tools/lerobot/`: LeRobot 数据转换、训练、评估、推理脚本。
- `configs/`: split plan、示例 replay。
- `reports/`: 数据集信息、stats、merge report、离线评估结果。
- `runbooks/`: 按步骤复现的命令模板。
- `docs/`: 相关 LeRobot/SmolVLA 格式说明。

## 推荐流程

1. 采集仿真 teacher 数据:

```bash
bash bundled_training_scripts/runbooks/01_collect_teacher.sh
```

2. 转成 LeRobot 数据集:

```bash
bash bundled_training_scripts/runbooks/02_convert_raw_to_lerobot.sh
```

3. 微调 SmolVLA:

```bash
bash bundled_training_scripts/runbooks/03_train_smolvla.sh
```

4. 离线评估 action chunk:

```bash
bash bundled_training_scripts/runbooks/04_eval_holdout.sh
```

5. Isaac Sim GUI 验证:

```bash
bash bundled_training_scripts/runbooks/05_gui_rollout.sh
```

这些脚本默认从仓库根目录 `/work/IsaacSim` 运行。
