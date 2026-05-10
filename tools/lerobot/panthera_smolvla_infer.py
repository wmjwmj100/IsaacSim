#!/usr/bin/env python3

import argparse
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image


def _to_tensor_image(path: Path) -> torch.Tensor:
    image = Image.open(path).convert("RGB")
    array = np.asarray(image, dtype=np.float32) / 255.0
    if array.ndim != 3 or array.shape[2] != 3:
        raise RuntimeError(f"Expected RGB image at {path}, got shape={array.shape}")
    array = np.transpose(array, (2, 0, 1))
    return torch.from_numpy(array)


def _first_action_step(value: Any) -> Any:
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    if hasattr(value, "tolist"):
        value = value.tolist()
    if isinstance(value, list) and value and isinstance(value[0], list):
        return value[0]
    return value


def _to_serializable(value: Any) -> Any:
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    if hasattr(value, "tolist"):
        return value.tolist()
    return value


def _feature_shape(feature: Any) -> list[int]:
    if isinstance(feature, dict):
        return list(feature.get("shape", []))
    return list(getattr(feature, "shape", []))


def _config_type(cfg: Any) -> str:
    value = getattr(cfg, "type", None)
    if value is not None:
        return str(value)
    return cfg.__class__.__name__


def build_frame(args: argparse.Namespace) -> dict[str, Any]:
    frame_path = Path(args.frame_json).expanduser()
    payload = json.loads(frame_path.read_text())
    state = torch.as_tensor(payload["state7"], dtype=torch.float32)
    frame: dict[str, Any] = {
        "observation.state": state,
        "task": payload.get("task", args.task_text),
    }
    camera1_path = payload.get("camera1_path")
    camera2_path = payload.get("camera2_path")
    camera3_path = payload.get("camera3_path")
    camera_keys = {
        "observation.images.camera1": camera1_path,
        "observation.images.camera2": camera2_path,
        "observation.images.camera3": camera3_path,
    }
    for key, image_path in camera_keys.items():
        if image_path:
            frame[key] = _to_tensor_image(Path(image_path))
            continue
        if key == "observation.images.camera2":
            frame[key] = torch.zeros((3, 256, 256), dtype=torch.float32)
        else:
            raise RuntimeError(f"Missing required camera path for {key}")
    return frame


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--frame-json", required=True)
    parser.add_argument("--output-json", default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--task-text", default="operate the Panthera arm on the table")
    parser.add_argument("--control-mode", default=None)
    parser.add_argument("--frame-index", type=int, default=None)
    parser.add_argument("--local-files-only", action="store_true", default=True)
    parser.add_argument("--vlm-model-path", default=None)
    args = parser.parse_args()

    from lerobot.configs.policies import PreTrainedConfig
    from lerobot.policies.factory import make_pre_post_processors
    from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy

    frame_payload = json.loads(Path(args.frame_json).expanduser().read_text())
    frame_index = int(args.frame_index if args.frame_index is not None else frame_payload.get("frame_index", 0))
    control_mode = args.control_mode or frame_payload.get("control_mode", "log-only")
    camera1_source = frame_payload.get("camera1_source", "d435i_rgb")
    camera3_source = frame_payload.get("camera3_source", "left_wrist_rgb")
    state_source = frame_payload.get("state_source", frame_payload.get("active_arm_label", "front_left") + "_7d")

    device = torch.device(args.device)
    print(
        f"[Panthera-HT][SmolVLA] Loading checkpoint: {args.model_id} device={device} control_mode={control_mode}",
        flush=True,
    )
    cfg = PreTrainedConfig.from_pretrained(args.model_id, local_files_only=args.local_files_only)
    if args.vlm_model_path:
        cfg.vlm_model_name = str(Path(args.vlm_model_path).expanduser())

    policy = SmolVLAPolicy.from_pretrained(args.model_id, config=cfg, local_files_only=args.local_files_only).to(device).eval()
    preprocessor_overrides: dict[str, dict[str, Any]] = {"device_processor": {"device": str(device)}}
    if args.vlm_model_path:
        preprocessor_overrides["tokenizer_processor"] = {"tokenizer_name": str(Path(args.vlm_model_path).expanduser())}
    preprocess, postprocess = make_pre_post_processors(cfg, args.model_id, preprocessor_overrides=preprocessor_overrides)

    print(
        f"[Panthera-HT][SmolVLA] Checkpoint loaded: type={_config_type(cfg)} action_shape={_feature_shape(cfg.output_features['action'])} "
        f"state_shape={_feature_shape(cfg.input_features['observation.state'])} image_keys={list(cfg.input_features.keys())} "
        f"chunk_size={getattr(cfg, 'chunk_size', None)} n_action_steps={getattr(cfg, 'n_action_steps', None)}",
        flush=True,
    )

    frame = build_frame(args)
    camera2_zero_filler = isinstance(frame.get("observation.images.camera2"), torch.Tensor) and torch.count_nonzero(frame["observation.images.camera2"]) == 0
    print(
        f"[Panthera-HT][SmolVLA] Input mapping: camera1={camera1_source} camera2=zeros_3x256x256 "
        f"camera3={camera3_source} state={state_source}",
        flush=True,
    )
    print(
        f"[Panthera-HT][SmolVLA] camera2_zero_filler={str(bool(camera2_zero_filler)).lower()}",
        flush=True,
    )

    processed = preprocess(frame)
    t0 = torch.cuda.Event(enable_timing=True) if device.type == "cuda" else None
    t1 = torch.cuda.Event(enable_timing=True) if device.type == "cuda" else None
    if t0 is not None and t1 is not None:
        t0.record()
    with torch.inference_mode():
        action = policy.select_action(processed)
        action = postprocess(action)
    if t0 is not None and t1 is not None:
        t1.record()
        torch.cuda.synchronize()
        latency_ms = float(t0.elapsed_time(t1))
    else:
        latency_ms = None

    action_serialized = _to_serializable(action)
    action_step0 = _first_action_step(action_serialized)
    print(
        f"[Panthera-HT][SmolVLA] Inference frame={frame_index} state7={_to_serializable(frame['observation.state'])} task={frame['task']!r}",
        flush=True,
    )
    print(
        f"[Panthera-HT][SmolVLA] Predicted action7 frame={frame_index} action={action_step0} latency_ms={latency_ms}",
        flush=True,
    )

    result = {
        "success": True,
        "model_id": args.model_id,
        "device": str(device),
        "task": frame["task"],
        "frame_index": frame_index,
        "control_mode": control_mode,
        "camera2_zero_filler": bool(camera2_zero_filler),
        "checkpoint_type": _config_type(cfg),
        "state_shape": _feature_shape(cfg.input_features["observation.state"]),
        "action_shape": _feature_shape(cfg.output_features["action"]),
        "chunk_size": getattr(cfg, "chunk_size", None),
        "n_action_steps": getattr(cfg, "n_action_steps", None),
        "input_mapping": {
            "camera1": camera1_source,
            "camera2": "zeros_3x256x256",
            "camera3": camera3_source,
            "state": state_source,
        },
        "predicted_action": action_serialized,
        "predicted_action_step0": action_step0,
        "latency_ms": latency_ms,
    }
    if args.output_json:
        output_path = Path(args.output_json).expanduser()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(result, ensure_ascii=False, indent=2))
    print(json.dumps(result, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
    main()
