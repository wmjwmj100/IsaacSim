# SO-101 / SmolVLA 训练数据格式要求

这份文档总结当前仓库中 `SmolVLA` 微调所需的数据格式，方便后续检查公开数据集或制作自己的数据集。

## 1. 当前默认数据集

- 默认数据集：`lerobot/svla_so101_pickplace`
- 训练脚本：`tools/lerobot/train_smolvla_so101.sh`
- 默认模型：`lerobot/smolvla_base`
- 默认训练输出：`outputs/train/so101_smolvla_pickplace`
- 本地缓存示例：`outputs/hf_cache/datasets/svla_so101_pickplace`

训练脚本会把数据集作为 `--dataset.repo_id=<repo_id>` 传给 `lerobot-train`。

## 2. 数据集目录结构

LeRobot 数据集通常包含以下结构：

```text
<dataset_root>/
  meta/
    info.json
    stats.json
    tasks.parquet
    episodes/chunk-000/file-000.parquet
  data/chunk-000/file-000.parquet
  videos/
    observation.images.up/chunk-000/file-000.mp4
    observation.images.side/chunk-000/file-000.mp4
```

其中：

- `meta/info.json`：描述字段、shape、fps、episode 数量等。
- `meta/stats.json`：训练归一化用的统计量。
- `data/*.parquet`：每一帧的状态、动作、索引、时间戳、任务索引等表格数据。
- `videos/*/*.mp4`：每个相机视角对应的视频数据。

## 3. 必需字段

当前 SO-101 / SmolVLA 训练至少需要这些字段：

| 字段 | 类型 | 形状 | 说明 |
| --- | --- | --- | --- |
| `observation.state` | `float32` | `[6]` | 当前机械臂状态 |
| `action` | `float32` | `[6]` | 下一步动作或目标关节值 |
| `observation.images.up` | `video` | `[480, 640, 3]` | 上方相机图像 |
| `observation.images.side` | `video` | `[480, 640, 3]` | 侧方相机图像 |
| `timestamp` | `float32` | `[1]` | 时间戳 |
| `frame_index` | `int64` | `[1]` | 帧索引 |
| `episode_index` | `int64` | `[1]` | episode 索引 |
| `task_index` | `int64` | `[1]` | 任务索引 |

当前缓存数据的 `fps` 是 `30`。

## 4. 6 维状态 / 动作含义

当前公开数据集的 6 维关节顺序为：

```text
[
  shoulder_pan.pos,
  shoulder_lift.pos,
  elbow_flex.pos,
  wrist_flex.pos,
  wrist_roll.pos,
  gripper.pos
]
```

要求：

- `observation.state` 和 `action` 的维度必须一致。
- 每一帧都要有对应的状态、动作、图像和任务信息。
- 自定义数据集的关节顺序要和训练 / 推理脚本保持一致，否则模型会学到错误控制关系。

## 5. 图像字段要求

当前训练数据使用两路相机：

- `observation.images.up`
- `observation.images.side`

而部分 `SmolVLA` 模型配置可能期望：

- `observation.images.camera1`
- `observation.images.camera2`
- `observation.images.camera3`

当前仓库的离线推理脚本会做适配：把 `up` 映射到 `camera1`，把 `side` 映射到 `camera3`，缺失的 `camera2` 用空图像补齐。

## 6. 任务文本要求

每条数据需要能关联到任务文本，例如：

```text
pink lego brick into the transparent box
```

任务文本用于语言条件控制。自定义数据集时，建议保证任务文本简洁、稳定，不要同一个动作目标对应大量风格差异很大的描述。

## 7. 如何查看当前数据

查看元数据：

```bash
cat outputs/hf_cache/datasets/svla_so101_pickplace/meta/info.json
cat outputs/hf_cache/datasets/svla_so101_pickplace/meta/stats.json
```

重新生成默认数据检查输出：

```bash
conda activate isaacsim-lerobot
python tools/lerobot/inspect_dataset.py
```

默认参数等价于：

```text
--repo-id lerobot/svla_so101_pickplace
--dataset-root None
--video-backend pyav
--episode-index 0
--output-dir outputs/dataset_inspect/svla_so101_pickplace
```

生成后查看：

```bash
cat outputs/dataset_inspect/svla_so101_pickplace/meta_info.json
cat outputs/dataset_inspect/svla_so101_pickplace/sample_summary.json
```

查看样本图像：

```bash
outputs/dataset_inspect/svla_so101_pickplace/observation_images_up.png
outputs/dataset_inspect/svla_so101_pickplace/observation_images_side.png
```

当前仓库也已有一次本地缓存 probe 输出，可直接查看：

```bash
cat outputs/dataset_inspect/local_cache_probe_pyav/sample_summary.json
```

已有 probe 的样本图像：

```bash
outputs/dataset_inspect/local_cache_probe_pyav/observation_images_up.png
outputs/dataset_inspect/local_cache_probe_pyav/observation_images_side.png
```

## 8. 自定义数据集检查清单

制作自己的 SO-101 数据集时，至少检查：

- `meta/info.json` 中包含 `action`、`observation.state`、图像字段和索引字段。
- `action` shape 是 `[6]`。
- `observation.state` shape 是 `[6]`。
- 图像是 RGB，形状类似 `[H, W, 3]`。
- 数据帧、视频帧、timestamp 能对齐。
- `stats.json` 存在，并且字段名和训练字段一致。
- 任务文本能准确描述每条轨迹的目标。
- 关节单位、顺序、夹爪开合方向和 Isaac Sim 推理脚本一致。

## 9. 重要注意点

- 当前公开数据集名是 `svla_so101_pickplace`，但缓存元数据里的 `robot_type` 显示为 `so100_follower`；做严格 SO-101 实验时要特别核对硬件/关节定义。
- 如果你换数据集，可以通过环境变量指定：

```bash
DATASET_REPO_ID=<你的数据集repo> ./tools/lerobot/train_smolvla_so101.sh
```
