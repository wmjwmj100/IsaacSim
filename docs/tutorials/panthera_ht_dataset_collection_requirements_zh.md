# Panthera-HT 数据采集要求

这份文档可以直接发给数据采集人员。目标是采集一份能用于训练 VLA / SmolVLA 类模型的数据集。

## 1. 最重要的结论

本项目里的 Panthera-HT 训练数据按 **7 维动作 / 7 维状态** 采集：

```text
[joint1, joint2, joint3, joint4, joint5, joint6, gripper]
```

注意：这里的 `gripper` 是夹爪开合量，不是机械臂第 7 个旋转轴。

## 2. 每一帧必须采集什么

每一帧都必须同时有：

| 字段 | 必须 | 说明 |
| --- | --- | --- |
| `observation.state` | 是 | 当前机器人状态，7 个 float |
| `action` | 是 | 下一步要执行的目标动作，7 个 float |
| `observation.images.up` | 是 | 上方相机 RGB 图像 |
| `observation.images.side` | 是 | 侧方相机 RGB 图像 |
| `timestamp` | 是 | 当前帧时间，单位秒 |
| `frame_index` | 是 | 当前 episode 内的帧编号，从 0 开始 |
| `episode_index` | 是 | 第几个 episode |
| `index` | 是 | 全数据集唯一帧编号，LeRobot 全局索引 |
| `task_index` | 是 | 当前任务编号 |
| `task` | 是 | 当前任务文本，例如“把蓝色方块放到左侧区域” |

## 3. 7 维状态和动作顺序

采集人员必须严格按下面顺序保存，不能换顺序：

| 维度 | 名称 | 单位 | 含义 |
| --- | --- | --- | --- |
| 0 | `joint1.pos` | rad | 机械臂第 1 关节角度 |
| 1 | `joint2.pos` | rad | 机械臂第 2 关节角度 |
| 2 | `joint3.pos` | rad | 机械臂第 3 关节角度 |
| 3 | `joint4.pos` | rad | 机械臂第 4 关节角度 |
| 4 | `joint5.pos` | rad | 机械臂第 5 关节角度 |
| 5 | `joint6.pos` | rad | 机械臂第 6 关节角度 |
| 6 | `gripper.pos` | m 或归一化值 | 夹爪开合量 |

要求：

- `observation.state` 和 `action` 都必须是 `[7]`。
- `observation.state` 是当前帧真实状态。
- `action` 是下一步目标状态，顺序和 `observation.state` 完全一致。
- `joint1` 到 `joint6` 必须用弧度 `rad`，不要用角度 `degree`。
- `gripper` 可以用米制开合量，也可以用 `0~1` 归一化值，但整份数据集必须统一。
- 如果使用 `0~1`，建议 `0=完全闭合`，`1=完全张开`。

## 4. 夹爪怎么采

Panthera-HT 仿真里有左右两个手指关节：

```text
L_finger_joint
R_finger_joint
```

但是训练数据不要保存成 8 维。推荐只保存一个 `gripper.pos`。

推荐规则：

```text
gripper.pos = L_finger_joint
```

或者如果采集系统只能读到左右两个值，也可以用：

```text
gripper.pos = (L_finger_joint + R_finger_joint) / 2
```

但最终交付给训练的数据里只能有一个 `gripper.pos`。

## 5. 图像要求

至少采两路 RGB 相机：

| 字段 | 分辨率 | 说明 |
| --- | --- | --- |
| `observation.images.up` | 建议 `480x640x3` | 上方视角，看清桌面、物体、夹爪 |
| `observation.images.side` | 建议 `480x640x3` | 侧方视角，看清高度、抓取姿态 |

要求：

- 图像必须和彩色 RGB 对齐，不要灰度图。
- 图像帧数必须和状态 / 动作帧数一致。
- 每个 `frame_index` 都能找到对应的两张图像。
- 如果采成视频，视频帧率必须和数据帧率一致。

## 6. 频率要求

推荐采集频率：

```text
30 FPS
```

如果采集端做不到 30 FPS，可以用 10 FPS 或 15 FPS，但必须提前告诉训练人员，并且整份数据集保持固定频率。

不要一会儿 10 FPS，一会儿 30 FPS。

## 7. 任务文本要求

每个 episode 必须有明确任务文本，例如：

```text
把蓝色方块放到左侧区域
把红色方块放到右侧区域
抓起方块并放到目标盘中
```

要求：

- 同一个 episode 内任务文本保持不变。
- 任务文本要和实际动作一致。
- 不要只写“任务1”“测试”“demo”这种无意义文本。

## 8. 推荐交付目录

如果采集人员能直接交付 LeRobot 格式，目录应类似：

```text
panthera_ht_pickplace_dataset/
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

如果暂时不能导出 LeRobot 格式，至少要交付这些原始文件：

```text
raw_dataset/
  episodes/
    episode_000/
      data.csv 或 data.parquet
      up.mp4 或 up/*.png
      side.mp4 或 side/*.png
      task.txt
    episode_001/
      data.csv 或 data.parquet
      up.mp4 或 up/*.png
      side.mp4 或 side/*.png
      task.txt
```

其中 `data.csv` / `data.parquet` 至少包含：

```text
timestamp
frame_index
episode_index
index
task_index
state_joint1
state_joint2
state_joint3
state_joint4
state_joint5
state_joint6
state_gripper
action_joint1
action_joint2
action_joint3
action_joint4
action_joint5
action_joint6
action_gripper
```

## 9. `meta/info.json` 里要写清楚

如果直接做成 LeRobot 数据集，`meta/info.json` 里至少要体现：

```json
{
  "robot_type": "panthera_ht",
  "fps": 30,
  "features": {
    "observation.state": {
      "dtype": "float32",
      "shape": [7],
      "names": [
        "joint1.pos",
        "joint2.pos",
        "joint3.pos",
        "joint4.pos",
        "joint5.pos",
        "joint6.pos",
        "gripper.pos"
      ]
    },
    "action": {
      "dtype": "float32",
      "shape": [7],
      "names": [
        "joint1.pos",
        "joint2.pos",
        "joint3.pos",
        "joint4.pos",
        "joint5.pos",
        "joint6.pos",
        "gripper.pos"
      ]
    },
    "observation.images.up": {
      "dtype": "video",
      "shape": [480, 640, 3]
    },
    "observation.images.side": {
      "dtype": "video",
      "shape": [480, 640, 3]
    },
    "timestamp": {
      "dtype": "float32",
      "shape": [1]
    },
    "frame_index": {
      "dtype": "int64",
      "shape": [1]
    },
    "episode_index": {
      "dtype": "int64",
      "shape": [1]
    },
    "index": {
      "dtype": "int64",
      "shape": [1]
    },
    "task_index": {
      "dtype": "int64",
      "shape": [1]
    }
  }
}
```

## 10. 采集质量要求

每个 episode 要满足：

- 任务开始前能看到机器人、桌面和目标物体。
- 任务过程完整，不要中间断帧。
- 状态、动作、图像必须时间对齐。
- 夹爪开合必须真实变化，不能一直是 0。
- 失败数据可以保留，但需要标记失败原因。
- 不要把不同任务混在同一个 episode 里。

## 11. 训练前验收清单

交付数据前，请逐项检查：

- `observation.state` 是 7 维。
- `action` 是 7 维。
- 7 维顺序是 `joint1` 到 `joint6`，最后是 `gripper`。
- 关节角单位是 `rad`。
- 图像有 `up` 和 `side` 两路。
- 每个 frame 都有状态、动作、图像和全局 `index`。
- 每个 episode 都有任务文本。
- `fps` 固定。
- `gripper` 只有 1 维，不是左右手指 2 维。

## 12. 不能这样交付

不要交付下面这些格式：

```text
[joint1, joint2, joint3, joint4, joint5, joint6]
```

原因：没有夹爪，模型学不会抓取和释放。

```text
[joint1, joint2, joint3, joint4, joint5, joint6, L_finger_joint, R_finger_joint]
```

原因：变成 8 维，左右手指不是两个独立动作。

```text
[joint1, joint2, joint3, joint4, joint5, joint6, joint7]
```

原因：当前仿真使用的 Panthera-HT 资产里没有独立的机械臂 `joint7`，第 7 维应是 `gripper`。

## 13. 给采集人员的一句话版本

请按 30 FPS 采集 Panthera-HT 的两路 RGB 图像、任务文本、7 维当前状态和 7 维下一步动作。7 维顺序固定为：

```text
joint1, joint2, joint3, joint4, joint5, joint6, gripper
```

`joint1` 到 `joint6` 用弧度，`gripper` 是一个夹爪开合量。不要把左右手指保存成两个独立动作维度。
