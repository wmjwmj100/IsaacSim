# SO-101 + SmolVLA + Isaac Sim 全流程学习笔记

这份文档对应当前仓库里新增的教学脚本，目标不是一次性追求最高成功率，而是带你完整走一遍：

1. 看懂公开数据集长什么样
2. 用公开数据集微调 `SmolVLA`
3. 先做一次离线推理 smoke test
4. 再把模型接回 Isaac Sim 做在线推理

## 0. 当前仓库状态

这台机器当前仓库里还没有 `/_build/linux-x86_64/release/python.sh`，说明 Isaac Sim 源码还没 build 完。

所以建议按两条线并行理解：

- 线 A：先做 `LeRobot/SmolVLA` 数据和训练，马上就能开始
- 线 B：再 build Isaac Sim，最后做在线推理闭环

## 1. 先准备 LeRobot 环境

执行：

```bash
./tools/lerobot/create_conda_env.sh
```

它会创建一个名为 `isaacsim-lerobot` 的 conda 环境，并安装：

- `lerobot[smolvla]`
- `datasets`
- `pillow`
- `imageio`

你体验这一阶段时，应该看到：

- conda 环境被创建
- pip 开始下载 `lerobot` 和 `SmolVLA` 依赖
- 结束时屏幕会打印如何 `conda activate isaacsim-lerobot`

## 2. 看公开 SO-101 数据集

默认我选的是 Hugging Face 官方公开数据集：

- `lerobot/svla_so101_pickplace`

它的好处是：

- 机械臂就是 `SO-101`
- 任务足够简单，适合第一次走流程
- 数据量不大，适合教学

执行：

```bash
conda activate isaacsim-lerobot
python tools/lerobot/inspect_dataset.py
```

这个脚本会做两件事：

- 下载 `meta/info.json`
- 如果本地 `LeRobot` 可用，再抽一帧并把图像保存出来

输出目录默认在：

- `outputs/dataset_inspect/svla_so101_pickplace`

你应该重点看：

- `meta_info.json`
- `sample_summary.json`
- `observation_images_up.png`
- `observation_images_side.png`

你在这一阶段要看明白的是：

- 动作是 6 维
- 状态是 6 维
- 这个公开数据集是 `up + side` 两路相机，不是三路

还有一个需要你特别注意的现实细节：

- 截至 2026-04-22，我核对 Hugging Face 页面时，这个数据集虽然名字叫 `svla_so101_pickplace`，但 `meta/info.json` 里的 `robot_type` 仍显示为 `so100_follower`

所以学习流程里可以先直接用它，但如果你后面要做严格的 `SO-101` 对齐实验，建议你把这个元数据差异单独记下来，不要默认它已经完全清洗好了。

这也是我这次顺手修掉的一个关键兼容点：当前 Isaac 运行时现在已经能按模型配置自动适配 2 路或 3 路相机，不再硬编码三路。

## 3. 先做一次离线 SmolVLA 推理

执行：

```bash
conda activate isaacsim-lerobot
python tools/lerobot/offline_smolvla_smoke.py \
  --model-id lerobot/smolvla_base \
  --dataset-repo-id lerobot/svla_so101_pickplace
```

这个脚本会：

- 加载一个公开数据集样本
- 加载 `lerobot/smolvla_base`
- 对这一个样本做一次前向推理
- 把预测动作、真值动作、均值 L1 误差写到 `report.json`

你体验这一阶段时，重点看：

- 终端里打印出来的 `predicted_action`
- `ground_truth_action`
- `mean_l1_error_vs_gt`
- 保存下来的样本图片

这里不是正式评测，只是确认三件事：

- 模型能加载
- 数据能喂进去
- 推理结果的形状和数值范围是合理的

## 4. 微调 SmolVLA

执行：

```bash
conda activate isaacsim-lerobot
./tools/lerobot/train_smolvla_so101.sh
```

默认配置是偏保守的，适合你这张 `RTX 4070 SUPER 12GB` 先跑起来：

- `DATASET_REPO_ID=lerobot/svla_so101_pickplace`
- `BASE_MODEL_ID=lerobot/smolvla_base`
- `BATCH_SIZE=4`
- `STEPS=20000`
- `policy.gradient_checkpointing=true`

如果你想把模型上传到 Hugging Face，可以额外加：

```bash
POLICY_REPO_ID=<你的用户名>/<你的模型名> ./tools/lerobot/train_smolvla_so101.sh
```

你在这一阶段应该观察：

- 终端会先回显完整训练命令
- `outputs/train/so101_smolvla_pickplace` 下开始产生 checkpoint 和日志
- loss 会逐步下降，但第一次跑不需要过分纠结绝对数值

第一次学习时，你只要先确认：

- 训练流程能通
- checkpoint 确实在落盘
- 没有 OOM

## 5. 再做一次离线推理对比

训练完后，用你的 checkpoint 再跑一遍 smoke test：

```bash
conda activate isaacsim-lerobot
python tools/lerobot/offline_smolvla_smoke.py \
  --model-id outputs/train/so101_smolvla_pickplace/checkpoints/last/pretrained_model
```

你可以把这次的 `report.json` 和 base model 的结果对比着看。

这一阶段你要体会的是：

- “训练”本质上就是让模型对你的任务分布更贴近
- 离线 smoke test 是把问题挡在仿真之外的第一道门

## 6. Build Isaac Sim

回到仓库根目录，先 build：

```bash
./build.sh --release
```

build 完以后，下面两个脚本才真正可运行：

- `./run_lerobot101.sh`
- `./run_lerobot101_act.sh`

我已经给它们补了更明确的提示。如果没 build，它会直接告诉你先跑 `./build.sh --release`。

## 7. 先看 SO-101 机械臂在 Isaac Sim 里动起来

执行：

```bash
./run_lerobot101.sh
```

你应该看到：

- 桌面场景
- 导入后的 `SO-101`
- 机械臂按预设轨迹摆动

这一阶段的目标很单纯：

- 确认 URDF 能导进 Isaac
- 确认关节轴和姿态大体正确
- 确认你已经具备“仿真能跑”的基础

## 8. 在 Isaac Sim 里接 SmolVLA 在线推理

最简单的入口是：

```bash
SMOLVLA_MODEL_ID=<你的模型路径或 Hugging Face 模型名> \
./run_lerobot101_smolvla_pickplace.sh
```

这个封装脚本内部会调用：

- `run_lerobot101_act.sh`

并带上我建议的默认参数：

- `--robot-model so101`
- `--disable-hf-prior`
- `--enable-smolvla-prior`
- `--save-vla-io`
- `--profile-smolvla`

你应该看到：

- Isaac Sim 里桌面、方块、SO-101
- 终端里打印 `SmolVLA loaded`
- 终端周期性打印推理耗时
- `~/.cache/isaacsim/vla_io_so101` 下持续产生图片和 `vla_io.jsonl`

## 9. 最值得你观察的关键节点

### 节点 A：数据集元数据

看 `meta_info.json`。

要看明白：

- action/state 的维度
- 相机 key 的命名
- robot_type 和 fps

### 节点 B：离线 smoke test

看 `report.json`。

要看明白：

- 一次前向输入了什么
- 输出动作是不是 6 维
- 和真值动作差多少

### 节点 C：训练输出目录

看 `outputs/train/so101_smolvla_pickplace`。

要看明白：

- checkpoint 在哪里
- 后面 Isaac 在线推理要拿哪个路径

### 节点 D：Isaac 在线推理快照

看 `vla_io.jsonl` 和配套 png。

要看明白：

- 每次推理时模型看到了哪些相机画面
- 模型输出了什么关节目标
- 这些目标是不是和场景/任务文本匹配

## 10. 你现在可以直接照着跑的最短顺序

```bash
./tools/lerobot/create_conda_env.sh
conda activate isaacsim-lerobot
python tools/lerobot/inspect_dataset.py
python tools/lerobot/offline_smolvla_smoke.py
./tools/lerobot/train_smolvla_so101.sh
./build.sh --release
./run_lerobot101.sh
SMOLVLA_MODEL_ID=<你的checkpoint或HF模型名> ./run_lerobot101_smolvla_pickplace.sh
```

## 11. 这次我实际补了什么

- 修了 Isaac 侧 `SmolVLA` 图像输入硬编码三相机的问题，现在会按 checkpoint 的 image keys 自动适配
- 补了 `LeRobot` conda 环境创建脚本
- 补了数据集检查脚本
- 补了离线 `SmolVLA` smoke test 脚本
- 补了 `SO-101 + SmolVLA` 训练脚本
- 补了一个更省事的 Isaac 在线推理启动脚本

如果你接下来要继续，我建议下一步直接先跑“第 1 到第 4 步”，先把 LeRobot 训练链路完全看懂，再去 build Isaac。这样学习密度最高，也不容易被编译过程打断。
