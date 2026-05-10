#!/usr/bin/env python3

import argparse
import json
from pathlib import Path
from typing import Any

import torch
import numpy as np


def _safe_name(text: str) -> str:
    return "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in text)


def _to_serializable(value: Any) -> Any:
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    if hasattr(value, "tolist"):
        return value.tolist()
    return value


def _first_action_step(value: Any) -> Any:
    value = _to_serializable(value)
    if isinstance(value, list) and value and isinstance(value[0], list):
        return value[0]
    return value


def _to_pil_image(value: Any):
    from PIL import Image

    if isinstance(value, Image.Image):
        return value
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    elif hasattr(value, "numpy"):
        value = value.numpy()
    array = np.asarray(value)
    if array.ndim == 3 and array.shape[0] in (1, 3, 4) and array.shape[-1] not in (1, 3, 4):
        array = np.transpose(array, (1, 2, 0))
    if array.dtype != np.uint8:
        array = np.clip(array, 0.0, 1.0) if np.issubdtype(array.dtype, np.floating) else array
        if np.issubdtype(array.dtype, np.floating):
            array = (array * 255.0).round().astype(np.uint8)
        else:
            array = array.astype(np.uint8)
    if array.ndim == 3 and array.shape[2] == 1:
        array = array[:, :, 0]
    return Image.fromarray(array)


def _save_images(frame: dict[str, Any], output_dir: Path) -> list[str]:
    saved = []
    for key, value in frame.items():
        if not key.startswith("observation.images."):
            continue
        pil_image = _to_pil_image(value)
        filename = f"{_safe_name(key)}.png"
        pil_image.save(output_dir / filename)
        saved.append(filename)
    return saved


def _make_empty_like_image(value: Any) -> Any:
    if hasattr(value, "detach"):
        value = value.detach().clone()
        value.zero_()
        return value
    if hasattr(value, "numpy"):
        value = value.copy()
        value[...] = 0
        return value
    array = np.asarray(value).copy()
    array[...] = 0
    return array


def _adapt_frame_for_smolvla(frame: dict[str, Any]) -> dict[str, Any]:
    adapted = dict(frame)
    up_key = "observation.images.up"
    side_key = "observation.images.side"
    camera_key_map = {
        "observation.images.camera1": up_key,
        "observation.images.camera2": None,
        "observation.images.camera3": side_key,
    }
    reference_image = frame.get(up_key)
    if reference_image is None:
        reference_image = frame.get(side_key)
    for dst_key, src_key in camera_key_map.items():
        if dst_key in adapted:
            continue
        if src_key and src_key in frame:
            adapted[dst_key] = frame[src_key]
            continue
        if reference_image is not None:
            adapted[dst_key] = _make_empty_like_image(reference_image)
    return adapted


def _select_action(policy: Any, frame: dict[str, Any], processed: dict[str, Any]) -> Any:
    try:
        return policy.select_action(frame)
    except Exception as raw_exc:
        try:
            return policy.select_action(processed)
        except Exception as processed_exc:
            raise RuntimeError(
                "policy.select_action failed for both raw frame and preprocessed batch. "
                f"raw_error={raw_exc!r}, processed_error={processed_exc!r}"
            ) from processed_exc


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-id", default="lerobot/smolvla_base")
    parser.add_argument("--dataset-repo-id", default="lerobot/svla_so101_pickplace")
    parser.add_argument("--dataset-root", default=None)
    parser.add_argument("--vlm-model-path", default=None)
    parser.add_argument("--video-backend", default="pyav")
    parser.add_argument("--episode-index", type=int, default=0)
    parser.add_argument("--frame-offset", type=int, default=0)
    parser.add_argument("--task", default="Pick up the cube and place it in the target area.")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument(
        "--output-dir",
        default=str(Path("outputs") / "offline_smolvla_smoke"),
    )
    args = parser.parse_args()

    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    from lerobot.configs.policies import PreTrainedConfig
    from lerobot.policies.factory import make_pre_post_processors
    from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    output_dir = Path(args.output_dir).expanduser() / _safe_name(args.model_id)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading dataset: {args.dataset_repo_id}")
    dataset_root = Path(args.dataset_root).expanduser() if args.dataset_root else None
    dataset_kwargs: dict[str, Any] = {}
    if dataset_root:
        dataset_kwargs["root"] = dataset_root
    if args.video_backend:
        dataset_kwargs["video_backend"] = args.video_backend
    dataset = LeRobotDataset(args.dataset_repo_id, **dataset_kwargs)
    from_idx = int(dataset.meta.episodes["dataset_from_index"][args.episode_index])
    frame_index = from_idx + int(args.frame_offset)
    frame = _adapt_frame_for_smolvla(dict(dataset[frame_index]))
    if not isinstance(frame.get("task"), str):
        frame["task"] = args.task

    print(f"Loading model: {args.model_id} on {device}")
    model_config = PreTrainedConfig.from_pretrained(args.model_id, local_files_only=args.local_files_only)
    if args.vlm_model_path:
        model_config.vlm_model_name = str(Path(args.vlm_model_path).expanduser())
    policy = SmolVLAPolicy.from_pretrained(
        args.model_id,
        config=model_config,
        local_files_only=args.local_files_only,
    ).to(device).eval()
    preprocessor_overrides: dict[str, dict[str, Any]] = {"device_processor": {"device": str(device)}}
    if args.vlm_model_path:
        preprocessor_overrides["tokenizer_processor"] = {
            "tokenizer_name": str(Path(args.vlm_model_path).expanduser())
        }
    preprocess, postprocess = make_pre_post_processors(
        model_config,
        args.model_id,
        preprocessor_overrides=preprocessor_overrides,
    )

    processed = preprocess(frame)
    with torch.inference_mode():
        pred_action = _select_action(policy, frame, processed)
        pred_action = postprocess(pred_action)

    pred_action = _to_serializable(pred_action)
    pred_action_step0 = _first_action_step(pred_action)
    gt_action = _to_serializable(frame.get("action"))
    obs_state = _to_serializable(frame.get("observation.state"))

    l1_error = None
    if isinstance(pred_action_step0, list) and isinstance(gt_action, list):
        n = min(len(pred_action_step0), len(gt_action))
        if n > 0:
            l1_error = sum(abs(float(pred_action_step0[i]) - float(gt_action[i])) for i in range(n)) / n

    saved_images = _save_images(frame, output_dir)
    report = {
        "success": True,
        "status": "ok",
        "model_id": args.model_id,
        "dataset_repo_id": args.dataset_repo_id,
        "dataset_root": str(dataset_root) if dataset_root else None,
        "vlm_model_path": str(Path(args.vlm_model_path).expanduser()) if args.vlm_model_path else None,
        "video_backend": args.video_backend,
        "episode_index": args.episode_index,
        "frame_index": frame_index,
        "task": frame.get("task"),
        "observation_state": obs_state,
        "ground_truth_action": gt_action,
        "predicted_action": pred_action,
        "predicted_action_step0": pred_action_step0,
        "mean_l1_error_step0_vs_gt": l1_error,
        "saved_images": saved_images,
    }
    with open(output_dir / "report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print(json.dumps(report, ensure_ascii=False, indent=2))
    print()
    print(f"Saved smoke-test report to: {output_dir / 'report.json'}")


if __name__ == "__main__":
    main()
