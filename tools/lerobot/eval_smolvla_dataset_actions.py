#!/usr/bin/env python3

import argparse
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import torch


def _to_serializable(value: Any) -> Any:
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    if hasattr(value, "tolist"):
        return value.tolist()
    return value


def _clear_bad_proxy_env() -> None:
    for key in ("ALL_PROXY", "all_proxy", "HTTP_PROXY", "http_proxy", "HTTPS_PROXY", "https_proxy"):
        os.environ.pop(key, None)
    os.environ.pop("HF_HUB_DISABLE_XET", None)


def _first_action_step(value: Any) -> list[float]:
    value = _to_serializable(value)
    if isinstance(value, list) and value and isinstance(value[0], list):
        value = value[0]
    if not isinstance(value, list):
        raise RuntimeError(f"Expected list action, got {type(value).__name__}")
    return [float(item) for item in value]


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


def _as_vector(value: Any, key: str) -> list[float]:
    value = _to_serializable(value)
    array = np.asarray(value, dtype=np.float64).reshape(-1)
    return [float(item) for item in array.tolist()]


def main() -> None:
    _clear_bad_proxy_env()

    parser = argparse.ArgumentParser(description="Evaluate a SmolVLA checkpoint on real LeRobot frames.")
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--dataset-repo-id", default="local/panthera_ht_merged")
    parser.add_argument(
        "--dataset-root",
        default="outputs/lerobot_datasets/panthera_ht_merged",
        help="Local dataset root. Pass an empty string to load the dataset by repo id only.",
    )
    parser.add_argument("--vlm-model-path", default=None)
    parser.add_argument("--episode-index", type=int, default=0)
    parser.add_argument("--frame-offsets", default="0,60,120,180,240,300,360,420,480,540,600,660,720")
    parser.add_argument("--video-backend", default="pyav")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--task", default="push the object on the table")
    parser.add_argument("--local-files-only", action="store_true", default=True)
    parser.add_argument(
        "--use-dataset-stats",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use dataset stats for SmolVLA state/action normalization and action unnormalization.",
    )
    parser.add_argument("--output-json", default="outputs/offline_smolvla_eval_20k/multiframe_report.json")
    args = parser.parse_args()

    from lerobot.configs.policies import PreTrainedConfig
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    from lerobot.policies.factory import make_pre_post_processors
    from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy

    device = torch.device(args.device)
    dataset_kwargs: dict[str, Any] = {}
    dataset_root = Path(args.dataset_root).expanduser() if args.dataset_root else None
    if dataset_root:
        dataset_kwargs["root"] = dataset_root
    if args.video_backend:
        dataset_kwargs["video_backend"] = args.video_backend
    dataset = LeRobotDataset(args.dataset_repo_id, **dataset_kwargs)
    episode_row = dataset.meta.episodes[args.episode_index]
    from_index = int(episode_row["dataset_from_index"])
    to_index = int(episode_row["dataset_to_index"])
    episode_length = to_index - from_index
    offsets = [int(item.strip()) for item in args.frame_offsets.split(",") if item.strip()]
    offsets = [offset for offset in offsets if 0 <= offset < episode_length]
    if not offsets:
        raise RuntimeError("No valid frame offsets selected")

    cfg = PreTrainedConfig.from_pretrained(args.model_id, local_files_only=args.local_files_only)
    if args.vlm_model_path:
        cfg.vlm_model_name = str(Path(args.vlm_model_path).expanduser())
    _force_two_camera_inputs(cfg)
    policy = SmolVLAPolicy.from_pretrained(
        args.model_id,
        config=cfg,
        local_files_only=args.local_files_only,
    ).to(device).eval()
    preprocessor_overrides: dict[str, dict[str, Any]] = {"device_processor": {"device": str(device)}}
    if args.vlm_model_path:
        preprocessor_overrides["tokenizer_processor"] = {
            "tokenizer_name": str(Path(args.vlm_model_path).expanduser())
        }
    processor_kwargs: dict[str, Any] = {"preprocessor_overrides": preprocessor_overrides}
    if args.use_dataset_stats:
        processor_kwargs["dataset_stats"] = dataset.meta.stats
    preprocess, postprocess = make_pre_post_processors(
        cfg,
        args.model_id,
        **processor_kwargs,
    )

    records: list[dict[str, Any]] = []
    for offset in offsets:
        dataset_index = from_index + offset
        frame = _adapt_frame_for_smolvla(dict(dataset[dataset_index]))
        if not isinstance(frame.get("task"), str):
            frame["task"] = args.task
        gt_action = _as_vector(frame["action"], "action")
        obs_state = _as_vector(frame["observation.state"], "observation.state")
        processed = preprocess(frame)
        policy.reset()
        with torch.inference_mode():
            pred = policy.select_action(processed)
            pred = postprocess(pred)
        pred_action = _first_action_step(pred)
        pred_action = pred_action[: len(gt_action)]
        error = np.asarray(pred_action, dtype=np.float64) - np.asarray(gt_action, dtype=np.float64)
        records.append(
            {
                "episode_index": args.episode_index,
                "frame_offset": offset,
                "dataset_index": dataset_index,
                "task": frame.get("task"),
                "observation_state": obs_state,
                "ground_truth_action": gt_action,
                "predicted_action_step0": pred_action,
                "abs_error": np.abs(error).tolist(),
                "mean_l1_error": float(np.abs(error).mean()),
                "max_abs_error": float(np.abs(error).max()),
            }
        )

    pred_array = np.asarray([record["predicted_action_step0"] for record in records], dtype=np.float64)
    gt_array = np.asarray([record["ground_truth_action"] for record in records], dtype=np.float64)
    abs_errors = np.abs(pred_array - gt_array)
    payload = {
        "success": True,
        "model_id": args.model_id,
        "dataset_repo_id": args.dataset_repo_id,
        "dataset_root": str(dataset_root) if dataset_root else None,
        "episode_index": args.episode_index,
        "episode_length": episode_length,
        "evaluated_offsets": offsets,
        "device": str(device),
        "chunk_size": getattr(cfg, "chunk_size", None),
        "n_action_steps": getattr(cfg, "n_action_steps", None),
        "use_dataset_stats": bool(args.use_dataset_stats),
        "mean_l1_error": float(abs_errors.mean()),
        "per_dim_mean_abs_error": abs_errors.mean(axis=0).tolist(),
        "per_dim_max_abs_error": abs_errors.max(axis=0).tolist(),
        "predicted_min": pred_array.min(axis=0).tolist(),
        "predicted_max": pred_array.max(axis=0).tolist(),
        "ground_truth_min": gt_array.min(axis=0).tolist(),
        "ground_truth_max": gt_array.max(axis=0).tolist(),
        "records": records,
    }
    output_path = Path(args.output_json).expanduser()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({key: payload[key] for key in payload if key != "records"}, ensure_ascii=False, indent=2))
    print(f"wrote {output_path}")


if __name__ == "__main__":
    main()
