#!/usr/bin/env python3

import argparse
import csv
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import torch


ACTION_NAMES = [
    "joint1.pos",
    "joint2.pos",
    "joint3.pos",
    "joint4.pos",
    "joint5.pos",
    "joint6.pos",
    "gripper.pos",
]


def _clear_bad_proxy_env() -> None:
    for key in ("ALL_PROXY", "all_proxy", "HTTP_PROXY", "http_proxy", "HTTPS_PROXY", "https_proxy"):
        os.environ.pop(key, None)
    os.environ.pop("HF_HUB_DISABLE_XET", None)


def _to_serializable(value: Any) -> Any:
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    if hasattr(value, "tolist"):
        return value.tolist()
    return value


def _adapt_frame_for_smolvla(frame: dict[str, Any]) -> dict[str, Any]:
    adapted = dict(frame)
    adapted.pop("observation.images.camera3", None)
    for key in list(adapted):
        if key.startswith("observation.images.empty_camera_"):
            adapted.pop(key, None)
    top_key = "observation.images.up" if "observation.images.up" in frame else "observation.images.top"
    wrist_key = "observation.images.side" if "observation.images.side" in frame else "observation.images.wrist"
    if "observation.images.camera1" not in adapted and top_key in frame:
        adapted["observation.images.camera1"] = frame[top_key]
    if "observation.images.camera2" not in adapted and wrist_key in frame:
        adapted["observation.images.camera2"] = frame[wrist_key]
    return adapted


def _force_two_camera_inputs(cfg: Any) -> None:
    input_features = getattr(cfg, "input_features", None)
    if not isinstance(input_features, dict):
        return
    if "observation.images.camera2" not in input_features and "observation.images.camera3" in input_features:
        input_features["observation.images.camera2"] = input_features["observation.images.camera3"]
    for key in list(input_features):
        if key == "observation.images.camera3" or key.startswith("observation.images.empty_camera_"):
            input_features.pop(key, None)
    cfg.empty_cameras = 0


def _as_action_chunk(value: Any, chunk_size: int, action_dim: int | None = None) -> np.ndarray:
    value = _to_serializable(value)
    array = np.asarray(value, dtype=np.float64)
    if array.ndim == 3 and array.shape[0] == 1:
        array = array[0]
    if array.ndim != 2:
        raise RuntimeError(f"Expected action chunk with shape (T, D), got {array.shape}")
    if action_dim is not None:
        if array.shape[1] < action_dim:
            raise RuntimeError(f"Expected action chunk with at least {action_dim} dims, got {array.shape}")
        array = array[:, :action_dim]
    return array[:chunk_size]


def _parse_indices(value: str) -> list[int]:
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def _episode_offsets(episode_length: int, frame_offsets: str, stride: int) -> list[int]:
    if frame_offsets.strip().lower() != "auto":
        return [offset for offset in _parse_indices(frame_offsets) if 0 <= offset < episode_length]
    stride = max(1, int(stride))
    offsets = list(range(0, episode_length, stride))
    if offsets and offsets[-1] != episode_length - 1:
        offsets.append(episode_length - 1)
    return offsets


def _valid_mask(frame: dict[str, Any], chunk_len: int) -> np.ndarray:
    pad_key = "action_is_pad"
    if pad_key not in frame:
        return np.ones(chunk_len, dtype=bool)
    pad = np.asarray(_to_serializable(frame[pad_key]), dtype=bool).reshape(-1)[:chunk_len]
    if pad.shape[0] < chunk_len:
        return np.ones(chunk_len, dtype=bool)
    return ~pad


def _array_stats(array: np.ndarray) -> dict[str, Any]:
    if array.size == 0:
        return {}
    return {
        "min": array.min(axis=0).tolist(),
        "max": array.max(axis=0).tolist(),
        "mean": array.mean(axis=0).tolist(),
        "std": array.std(axis=0).tolist(),
        "q01": np.quantile(array, 0.01, axis=0).tolist(),
        "q50": np.quantile(array, 0.50, axis=0).tolist(),
        "q99": np.quantile(array, 0.99, axis=0).tolist(),
    }


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "episode_index",
        "frame_offset",
        "dataset_index",
        "chunk_step",
        "is_first_step",
        "gt_gripper",
        "pred_gripper",
        "signed_error_gripper",
        "abs_error_gripper",
    ]
    for dim, name in enumerate(ACTION_NAMES):
        fieldnames.extend([f"gt_dim{dim}_{name}", f"pred_dim{dim}_{name}", f"abs_error_dim{dim}_{name}"])
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _maybe_write_plot(path: Path, rows: list[dict[str, Any]]) -> bool:
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return _write_pil_plot(path, rows)

    path.parent.mkdir(parents=True, exist_ok=True)
    chunks: dict[tuple[int, int], list[dict[str, Any]]] = {}
    for row in rows:
        key = (int(row["episode_index"]), int(row["frame_offset"]))
        chunks.setdefault(key, []).append(row)

    max_plots = min(8, len(chunks))
    if max_plots == 0:
        return False
    fig, axes = plt.subplots(max_plots, 1, figsize=(10, max(2.5, 2.0 * max_plots)), sharex=False)
    if max_plots == 1:
        axes = [axes]
    for axis, ((episode_index, frame_offset), chunk_rows) in zip(axes, chunks.items()):
        chunk_rows = sorted(chunk_rows, key=lambda item: int(item["chunk_step"]))
        x = [int(item["chunk_step"]) for item in chunk_rows]
        gt = [float(item["gt_gripper"]) for item in chunk_rows]
        pred = [float(item["pred_gripper"]) for item in chunk_rows]
        axis.plot(x, gt, label="gt gripper", linewidth=1.8)
        axis.plot(x, pred, label="pred gripper", linewidth=1.4)
        axis.set_title(f"episode {episode_index}, frame offset {frame_offset}")
        axis.set_ylabel("gripper")
        axis.grid(True, alpha=0.25)
    axes[-1].set_xlabel("chunk step")
    axes[0].legend(loc="best")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return True


def _write_pil_plot(path: Path, rows: list[dict[str, Any]]) -> bool:
    try:
        from PIL import Image, ImageDraw
    except Exception:
        return False

    chunks: dict[tuple[int, int], list[dict[str, Any]]] = {}
    for row in rows:
        key = (int(row["episode_index"]), int(row["frame_offset"]))
        chunks.setdefault(key, []).append(row)
    if not chunks:
        return False

    selected = list(chunks.items())[:8]
    panel_w = 1000
    panel_h = 190
    left = 70
    right = 20
    top = 34
    bottom = 32
    image = Image.new("RGB", (panel_w, panel_h * len(selected)), "white")
    draw = ImageDraw.Draw(image)
    gt_color = (31, 119, 180)
    pred_color = (214, 39, 40)
    grid_color = (220, 220, 220)
    text_color = (35, 35, 35)

    all_values = [float(row["gt_gripper"]) for row in rows] + [float(row["pred_gripper"]) for row in rows]
    y_min = min(all_values)
    y_max = max(all_values)
    pad = max((y_max - y_min) * 0.08, 1e-4)
    y_min -= pad
    y_max += pad

    def y_to_px(value: float, origin_y: int) -> int:
        scale = (panel_h - top - bottom) / max(y_max - y_min, 1e-9)
        return int(origin_y + panel_h - bottom - (value - y_min) * scale)

    for panel_index, ((episode_index, frame_offset), chunk_rows) in enumerate(selected):
        origin_y = panel_index * panel_h
        chunk_rows = sorted(chunk_rows, key=lambda item: int(item["chunk_step"]))
        x_values = [int(row["chunk_step"]) for row in chunk_rows]
        x_min = min(x_values)
        x_max = max(x_values)
        x_scale = (panel_w - left - right) / max(x_max - x_min, 1)

        for frac in (0.0, 0.25, 0.5, 0.75, 1.0):
            y = origin_y + top + int(frac * (panel_h - top - bottom))
            draw.line((left, y, panel_w - right, y), fill=grid_color)
        draw.rectangle((left, origin_y + top, panel_w - right, origin_y + panel_h - bottom), outline=(150, 150, 150))
        draw.text((left, origin_y + 8), f"episode {episode_index}, frame offset {frame_offset}", fill=text_color)
        draw.text((panel_w - 230, origin_y + 8), "GT", fill=gt_color)
        draw.text((panel_w - 180, origin_y + 8), "Pred", fill=pred_color)
        draw.text((8, origin_y + top - 6), f"{y_max:.3f}", fill=text_color)
        draw.text((8, origin_y + panel_h - bottom - 8), f"{y_min:.3f}", fill=text_color)

        gt_points = []
        pred_points = []
        for row in chunk_rows:
            x = int(left + (int(row["chunk_step"]) - x_min) * x_scale)
            gt_points.append((x, y_to_px(float(row["gt_gripper"]), origin_y)))
            pred_points.append((x, y_to_px(float(row["pred_gripper"]), origin_y)))
        if len(gt_points) >= 2:
            draw.line(gt_points, fill=gt_color, width=2)
            draw.line(pred_points, fill=pred_color, width=2)

    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path)
    return True


def main() -> None:
    _clear_bad_proxy_env()
    parser = argparse.ArgumentParser(description="Diagnose SmolVLA 7th action dimension on LeRobot chunks.")
    parser.add_argument(
        "--model-id",
        default="outputs/train/roboclaw_real_20260529_21ep_smolvla_two_cam_20k_cont_from_two_cam5k/checkpoints/020000/pretrained_model",
    )
    parser.add_argument("--dataset-repo-id", default="local/roboclaw_real_20260529_21ep")
    parser.add_argument("--dataset-root", default="outputs/lerobot_datasets/roboclaw_real_20260529_21ep")
    parser.add_argument("--episode-indices", default="19,20")
    parser.add_argument("--frame-offsets", default="auto")
    parser.add_argument("--auto-offset-stride", type=int, default=50)
    parser.add_argument("--max-chunks", type=int, default=24)
    parser.add_argument("--task", default="push the object on the table")
    parser.add_argument("--video-backend", default="pyav")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--local-files-only", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--use-dataset-stats", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--vlm-model-path", default=None)
    parser.add_argument("--output-dir", default="outputs/gripper_dim_diagnostics/real_20260529_21ep_20k")
    args = parser.parse_args()

    from lerobot.configs.policies import PreTrainedConfig
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    from lerobot.policies.factory import make_pre_post_processors
    from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig as _SmolVLAConfig  # noqa: F401
    from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy

    device = torch.device(args.device)
    cfg = PreTrainedConfig.from_pretrained(args.model_id, local_files_only=args.local_files_only)
    if args.vlm_model_path:
        cfg.vlm_model_name = str(Path(args.vlm_model_path).expanduser())
    _force_two_camera_inputs(cfg)
    chunk_size = int(getattr(cfg, "chunk_size", 50))

    dataset_root = Path(args.dataset_root).expanduser()
    dataset_kwargs: dict[str, Any] = {
        "root": dataset_root,
        "delta_timestamps": {"action": [i / 30 for i in range(chunk_size)]},
    }
    if args.video_backend:
        dataset_kwargs["video_backend"] = args.video_backend
    dataset = LeRobotDataset(args.dataset_repo_id, **dataset_kwargs)

    policy = SmolVLAPolicy.from_pretrained(
        args.model_id,
        config=cfg,
        local_files_only=args.local_files_only,
    ).to(device).eval()
    preprocessor_overrides: dict[str, dict[str, Any]] = {"device_processor": {"device": str(device)}}
    if args.vlm_model_path:
        preprocessor_overrides["tokenizer_processor"] = {"tokenizer_name": str(Path(args.vlm_model_path).expanduser())}
    processor_kwargs: dict[str, Any] = {"preprocessor_overrides": preprocessor_overrides}
    if args.use_dataset_stats:
        processor_kwargs["dataset_stats"] = dataset.meta.stats
    preprocess, postprocess = make_pre_post_processors(cfg, args.model_id, **processor_kwargs)

    rows: list[dict[str, Any]] = []
    chunk_summaries: list[dict[str, Any]] = []
    episode_indices = _parse_indices(args.episode_indices)
    chunk_count = 0

    for episode_index in episode_indices:
        episode_row = dataset.meta.episodes[episode_index]
        from_index = int(episode_row["dataset_from_index"])
        to_index = int(episode_row["dataset_to_index"])
        episode_length = to_index - from_index
        for frame_offset in _episode_offsets(episode_length, args.frame_offsets, args.auto_offset_stride):
            if args.max_chunks > 0 and chunk_count >= args.max_chunks:
                break
            dataset_index = from_index + frame_offset
            frame = _adapt_frame_for_smolvla(dict(dataset[dataset_index]))
            if not isinstance(frame.get("task"), str):
                frame["task"] = args.task
            gt_chunk = _as_action_chunk(frame["action"], chunk_size)
            action_dim = int(gt_chunk.shape[1])
            processed = preprocess(frame)
            policy.reset()
            with torch.inference_mode():
                pred_chunk = postprocess(policy.predict_action_chunk(processed))
            pred_chunk = _as_action_chunk(pred_chunk, chunk_size, action_dim=action_dim)
            valid = _valid_mask(frame, min(len(gt_chunk), len(pred_chunk)))
            gt_valid = gt_chunk[: len(valid)][valid]
            pred_valid = pred_chunk[: len(valid)][valid]
            if gt_valid.size == 0:
                continue
            abs_error = np.abs(pred_valid - gt_valid)
            signed_error = pred_valid - gt_valid
            chunk_summaries.append(
                {
                    "episode_index": int(episode_index),
                    "frame_offset": int(frame_offset),
                    "dataset_index": int(dataset_index),
                    "valid_steps": int(len(gt_valid)),
                    "mean_l1_error": float(abs_error.mean()),
                    "per_dim_mean_abs_error": abs_error.mean(axis=0).tolist(),
                    "per_dim_max_abs_error": abs_error.max(axis=0).tolist(),
                    "gripper_mean_abs_error": float(abs_error[:, 6].mean()),
                    "gripper_max_abs_error": float(abs_error[:, 6].max()),
                    "gripper_signed_error_mean": float(signed_error[:, 6].mean()),
                    "gt_gripper_min": float(gt_valid[:, 6].min()),
                    "gt_gripper_max": float(gt_valid[:, 6].max()),
                    "pred_gripper_min": float(pred_valid[:, 6].min()),
                    "pred_gripper_max": float(pred_valid[:, 6].max()),
                }
            )
            for step, (gt_action, pred_action, err_action, signed_action) in enumerate(
                zip(gt_valid, pred_valid, abs_error, signed_error, strict=False)
            ):
                row: dict[str, Any] = {
                    "episode_index": int(episode_index),
                    "frame_offset": int(frame_offset),
                    "dataset_index": int(dataset_index + step),
                    "chunk_step": int(step),
                    "is_first_step": int(step == 0),
                    "gt_gripper": float(gt_action[6]),
                    "pred_gripper": float(pred_action[6]),
                    "signed_error_gripper": float(signed_action[6]),
                    "abs_error_gripper": float(err_action[6]),
                }
                for dim, name in enumerate(ACTION_NAMES[:action_dim]):
                    row[f"gt_dim{dim}_{name}"] = float(gt_action[dim])
                    row[f"pred_dim{dim}_{name}"] = float(pred_action[dim])
                    row[f"abs_error_dim{dim}_{name}"] = float(err_action[dim])
                rows.append(row)
            chunk_count += 1
        if args.max_chunks > 0 and chunk_count >= args.max_chunks:
            break

    if not rows:
        raise RuntimeError("No diagnostic rows were produced")

    gt_array = np.asarray([[row[f"gt_dim{dim}_{ACTION_NAMES[dim]}"] for dim in range(7)] for row in rows], dtype=np.float64)
    pred_array = np.asarray(
        [[row[f"pred_dim{dim}_{ACTION_NAMES[dim]}"] for dim in range(7)] for row in rows],
        dtype=np.float64,
    )
    abs_errors = np.abs(pred_array - gt_array)
    signed_errors = pred_array - gt_array

    output_dir = Path(args.output_dir).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "gripper_dim_steps.csv"
    plot_path = output_dir / "gripper_dim_plot.png"
    summary_path = output_dir / "summary.json"
    _write_csv(csv_path, rows)
    plot_written = _maybe_write_plot(plot_path, rows)

    gt_gripper = gt_array[:, 6]
    pred_gripper = pred_array[:, 6]
    gripper_range = float(gt_gripper.max() - gt_gripper.min())
    gripper_mae = float(abs_errors[:, 6].mean())
    summary = {
        "success": True,
        "model_id": args.model_id,
        "dataset_repo_id": args.dataset_repo_id,
        "dataset_root": str(dataset_root),
        "episode_indices": episode_indices,
        "frame_offsets": args.frame_offsets,
        "auto_offset_stride": int(args.auto_offset_stride),
        "max_chunks": int(args.max_chunks),
        "evaluated_chunks": len(chunk_summaries),
        "evaluated_steps": int(len(rows)),
        "device": str(device),
        "chunk_size": chunk_size,
        "n_action_steps": getattr(cfg, "n_action_steps", None),
        "use_dataset_stats": bool(args.use_dataset_stats),
        "input_features": list(getattr(cfg, "input_features", {}).keys()),
        "output_action_shape": list(getattr(cfg.output_features["action"], "shape", [])),
        "action_names": ACTION_NAMES,
        "mean_l1_error": float(abs_errors.mean()),
        "max_l1_error": float(abs_errors.max()),
        "per_dim_mean_abs_error": abs_errors.mean(axis=0).tolist(),
        "per_dim_max_abs_error": abs_errors.max(axis=0).tolist(),
        "per_dim_signed_error_mean": signed_errors.mean(axis=0).tolist(),
        "ground_truth_action_stats": _array_stats(gt_array),
        "predicted_action_stats": _array_stats(pred_array),
        "gripper": {
            "index": 6,
            "name": "gripper.pos",
            "mean_abs_error": gripper_mae,
            "max_abs_error": float(abs_errors[:, 6].max()),
            "signed_error_mean": float(signed_errors[:, 6].mean()),
            "ground_truth_min": float(gt_gripper.min()),
            "ground_truth_max": float(gt_gripper.max()),
            "ground_truth_range": gripper_range,
            "ground_truth_std": float(gt_gripper.std()),
            "predicted_min": float(pred_gripper.min()),
            "predicted_max": float(pred_gripper.max()),
            "predicted_range": float(pred_gripper.max() - pred_gripper.min()),
            "predicted_std": float(pred_gripper.std()),
            "mae_as_fraction_of_gt_range": float(gripper_mae / gripper_range) if gripper_range > 0 else None,
            "close_value_in_dataset": float(min(gt_gripper.min(), gt_gripper.max())),
            "open_value_in_dataset": float(max(gt_gripper.min(), gt_gripper.max())),
        },
        "chunk_summaries": chunk_summaries,
        "csv_path": str(csv_path),
        "plot_path": str(plot_path) if plot_written else None,
    }
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    printable = {key: summary[key] for key in summary if key not in ("chunk_summaries",)}
    print(json.dumps(printable, ensure_ascii=False, indent=2))
    print(f"wrote {summary_path}")


if __name__ == "__main__":
    main()
