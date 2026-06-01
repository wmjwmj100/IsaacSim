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
        raise RuntimeError(f"Expected action chunk with shape (T, >=7), got {array.shape}")
    if action_dim is not None:
        if array.shape[1] < action_dim:
            raise RuntimeError(f"Expected action chunk with at least {action_dim} dims, got {array.shape}")
        return array[:chunk_size, :action_dim]
    return array[:chunk_size]


def main() -> None:
    _clear_bad_proxy_env()

    parser = argparse.ArgumentParser(description="Evaluate a SmolVLA checkpoint on full action chunks.")
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--dataset-repo-id", default="local/roboclaw_sim_teacher_widepos_clean_89ep")
    parser.add_argument(
        "--dataset-root",
        default="outputs/lerobot_datasets/roboclaw_sim_teacher_widepos_clean_89ep",
        help="Local dataset root. Pass an empty string to load the dataset by repo id only.",
    )
    parser.add_argument("--vlm-model-path", default=None)
    parser.add_argument("--episode-indices", default="0,20,60")
    parser.add_argument("--frame-offsets", default="0,20,50,90")
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
    parser.add_argument("--output-json", default="outputs/offline_smolvla_eval_widepos/chunk_report.json")
    args = parser.parse_args()

    from lerobot.configs.policies import PreTrainedConfig
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    from lerobot.policies.factory import make_pre_post_processors
    from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
    from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig as _SmolVLAConfig  # noqa: F401

    device = torch.device(args.device)
    cfg = PreTrainedConfig.from_pretrained(args.model_id, local_files_only=args.local_files_only)
    chunk_size = int(getattr(cfg, "chunk_size", 50))
    if args.vlm_model_path:
        cfg.vlm_model_name = str(Path(args.vlm_model_path).expanduser())
    _force_two_camera_inputs(cfg)

    delta_timestamps = {"action": [i / 30 for i in range(chunk_size)]}
    dataset_kwargs: dict[str, Any] = {
        "delta_timestamps": delta_timestamps,
    }
    dataset_root = Path(args.dataset_root).expanduser() if args.dataset_root else None
    if dataset_root:
        dataset_kwargs["root"] = dataset_root
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

    episode_indices = [int(item.strip()) for item in args.episode_indices.split(",") if item.strip()]
    frame_offsets = [int(item.strip()) for item in args.frame_offsets.split(",") if item.strip()]
    records: list[dict[str, Any]] = []
    for episode_index in episode_indices:
        episode_row = dataset.meta.episodes[episode_index]
        from_index = int(episode_row["dataset_from_index"])
        to_index = int(episode_row["dataset_to_index"])
        episode_length = to_index - from_index
        valid_offsets = [offset for offset in frame_offsets if 0 <= offset < episode_length]
        for offset in valid_offsets:
            dataset_index = from_index + offset
            frame = _adapt_frame_for_smolvla(dict(dataset[dataset_index]))
            if not isinstance(frame.get("task"), str):
                frame["task"] = args.task
            gt_chunk = _as_action_chunk(frame["action"], chunk_size)
            action_dim = int(gt_chunk.shape[1])
            processed = preprocess(frame)
            policy.reset()
            with torch.inference_mode():
                pred_chunk = policy.predict_action_chunk(processed)
                pred_chunk = postprocess(pred_chunk)
            pred_chunk_array = _as_action_chunk(pred_chunk, chunk_size, action_dim=action_dim)
            valid_mask = np.ones(gt_chunk.shape[0], dtype=bool)
            pad_key = "action_is_pad"
            if pad_key in frame:
                valid_mask = ~np.asarray(_to_serializable(frame[pad_key]), dtype=bool)[: gt_chunk.shape[0]]
            gt_valid = gt_chunk[valid_mask]
            pred_valid = pred_chunk_array[valid_mask]
            abs_error = np.abs(pred_valid - gt_valid)
            records.append(
                {
                    "episode_index": episode_index,
                    "frame_offset": offset,
                    "dataset_index": dataset_index,
                    "valid_steps": int(valid_mask.sum()),
                    "mean_l1_error": float(abs_error.mean()),
                    "first_step_mean_l1_error": float(abs_error[0].mean()),
                    "last_step_mean_l1_error": float(abs_error[-1].mean()),
                    "per_dim_mean_abs_error": abs_error.mean(axis=0).tolist(),
                    "per_step_mean_abs_error": abs_error.mean(axis=1).tolist(),
                    "predicted_first_action": pred_valid[0].tolist(),
                    "ground_truth_first_action": gt_valid[0].tolist(),
                    "predicted_last_action": pred_valid[-1].tolist(),
                    "ground_truth_last_action": gt_valid[-1].tolist(),
                }
            )

    all_errors = []
    for record in records:
        all_errors.extend(record["per_step_mean_abs_error"])
    payload = {
        "success": True,
        "model_id": args.model_id,
        "dataset_repo_id": args.dataset_repo_id,
        "dataset_root": str(dataset_root) if dataset_root else None,
        "episode_indices": episode_indices,
        "frame_offsets": frame_offsets,
        "device": str(device),
        "chunk_size": chunk_size,
        "n_action_steps": getattr(cfg, "n_action_steps", None),
        "use_dataset_stats": bool(args.use_dataset_stats),
        "num_chunks": len(records),
        "mean_step_l1_error": float(np.mean(all_errors)) if all_errors else None,
        "max_step_l1_error": float(np.max(all_errors)) if all_errors else None,
        "records": records,
    }
    output_path = Path(args.output_json).expanduser()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({key: payload[key] for key in payload if key != "records"}, ensure_ascii=False, indent=2))
    print(f"wrote {output_path}")


if __name__ == "__main__":
    main()
