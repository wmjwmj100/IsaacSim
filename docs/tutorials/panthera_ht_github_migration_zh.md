# Panthera-HT GitHub 轻量迁移与复现指南

这份文档用于把当前 Panthera-HT Isaac Sim 场景维护在自己的 GitHub 仓库中，并保证另一台电脑 clone 后可以方便复现当前状态。

## 推荐仓库形态

推荐使用 **Isaac Sim fork/branch 轻量版**：

- 以官方 `isaac-sim/IsaacSim` 仓库为基础。
- 只提交 Panthera-HT 相关源码、启动脚本和文档。
- 不提交本机构建产物、截图/数据集输出、第三方克隆资产和缓存。

原因：`run_panthera_ht_one_click.sh` 需要在 Isaac Sim 仓库根目录运行，并依赖根目录下的 `build.sh`、`source/` 和 `_build/` 路径约定。fork/branch 模式最容易复现。

## 必须提交的文件

最小可复现 Panthera-HT 负载包括：

```text
run_panthera_ht_one_click.sh
run_panthera_ht_table.sh
source/standalone_examples/custom/panthera_ht_table.py
docs/tutorials/panthera_ht_table.md
docs/tutorials/panthera_ht_dataset_collection_requirements_zh.md
docs/tutorials/panthera_ht_github_migration_zh.md
.gitignore
```

`run_panthera_ht_one_click.sh` 会在缺少 Panthera ROS2 资产时自动克隆 `HighTorque-Robotics/Panthera-HT-ROS2`，并在缺少 Isaac Sim release Python 时自动执行 `./build.sh --release`。

## 不要提交的内容

这些目录/文件是本地生成物或第三方缓存，不应进入 GitHub 仓库：

```text
_build/
outputs/
output/
external/
.cache/
.local-toolchain/
.blackboard/
.eula_accepted
.nvidia-omniverse/
.nvcode/
**/__pycache__/
*.pyc
*.npy
*.npz
*.parquet
*.mp4
```

当前工作区里 `_build`、`outputs` 和 `external` 体积很大；把它们放进 GitHub 会导致仓库难以维护，也可能触发 GitHub/LFS 大文件限制。

## 新机器复现步骤

### 1. 克隆你的 Panthera Isaac Sim 仓库

```bash
git clone <your-github-repo-url> isaacsim-panthera
cd isaacsim-panthera
```

### 2. 准备基础环境

建议环境：

- Linux x86_64
- NVIDIA GPU 与 Isaac Sim 支持的驱动
- `git`、`bash`
- 足够磁盘空间用于 Isaac Sim build、Packman cache 和输出数据
- GUI 实时相机窗口需要本地显示环境；headless 数据导出不需要 GUI

### 3. 一键启动 GUI 场景

```bash
./run_panthera_ht_one_click.sh
```

首次运行会自动完成：

1. 克隆 Panthera ROS2 资产到 `external/Panthera-HT-ROS2`。
2. 如果 `_build/linux-x86_64/release/python.sh` 不存在，则执行 `./build.sh --release`。
3. 启动 `source/standalone_examples/custom/panthera_ht_table.py`。
4. 默认启动主 GUI 场景；如需实时相机 viewport，可额外加 `--show-layout-camera-viewports` 打开 `left_wrist_rgb`、`right_wrist_rgb`、`d435i_rgb`。其中 `d435i_rgb` / `d435i_depth` 现在位于桌子对边中点上方 60 厘米，USD Camera orientation 固定为 X=-27、Y=0、Z=180 度。

### 4. Headless 验证命令

```bash
PANTHERA_HEADLESS=1 \
PANTHERA_MAX_FRAMES=120 \
PANTHERA_DISABLE_LAYOUT_CAMERA_VIEWPORTS=1 \
./run_panthera_ht_one_click.sh
```

这个命令适合服务器或 CI 风格验证：不会打开 GUI 窗口，会导出截图和 VLA sequence，并在指定帧数后退出。

## 常用环境变量

| 变量 | 说明 |
| --- | --- |
| `PANTHERA_HEADLESS=1` | 无头运行，不显示 GUI 窗口 |
| `PANTHERA_MAX_FRAMES=120` | 运行固定帧数后退出 |
| `PANTHERA_DISABLE_LAYOUT_CAMERA_VIEWPORTS=1` | GUI 模式下也禁用实时相机窗口 |
| `PANTHERA_SHOW_LAYOUT_CAMERA_VIEWPORTS=0` | 不自动打开实时相机窗口 |
| `PANTHERA_UPDATE_ASSETS=1` | 更新已有的 Panthera ROS2 资产 |
| `PANTHERA_URDF=/abs/path/file.xacro` | 使用自定义 Panthera URDF/xacro |
| `PANTHERA_TABLE_SIZE=0.80` | 设置桌面尺寸，单位米 |
| `PANTHERA_DISABLE_REALISM=1` | 使用更简洁的旧版桌面场景 |
| `PANTHERA_RANDOMIZE_REALISM=1` | 启用可复现的视觉随机化 |

## 发布到自己的 GitHub

不要推送到当前官方 `origin`。建议增加一个你自己的 remote：

```bash
git remote add panthera <your-github-repo-url>
git push -u panthera <branch-name>
```

如果你是在 GitHub 上 fork 了 `isaac-sim/IsaacSim`，也可以把你的 fork 设为 `origin`，把官方仓库设为 `upstream`：

```bash
git remote rename origin upstream
git remote add origin <your-fork-url>
git push -u origin <branch-name>
```

## 重要注意事项

- 不要把 `external/` 里的 Panthera 资产直接提交；让启动脚本在目标机器上自动拉取，能避免授权和大文件维护问题。
- 不要把 `outputs/` 作为源码提交；如果需要共享示例数据，建议用 GitHub Release 或对象存储发布小样本。
- 当前 UGREEN 腕部相机外参已使用现场手眼标定结果并转换到 USD Camera 坐标轴；内参仍是近似/未标定状态。D435i RGB/depth 参数按当前报告写入。要做严格 sim-to-real，请在真实硬件上继续做相机内参标定和外参复核。
