#!/usr/bin/env python3

import argparse
import copy
import json
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image

DEFAULT_DATASET_ROOT = Path("outputs/lerobot_datasets/panthera_ht_merged")


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


def force_two_camera_inputs(cfg: Any) -> None:
    input_features = getattr(cfg, "input_features", None)
    if not isinstance(input_features, dict):
        return
    if "observation.images.camera2" not in input_features and "observation.images.camera3" in input_features:
        input_features["observation.images.camera2"] = input_features["observation.images.camera3"]
    for key in list(input_features):
        if key == "observation.images.camera3" or key.startswith("observation.images.empty_camera_"):
            input_features.pop(key, None)
    cfg.empty_cameras = 0


def load_dataset_stats(dataset_root: str) -> dict[str, dict[str, Any]] | None:
    if not dataset_root:
        return None
    from lerobot.datasets.utils import load_stats

    root = Path(dataset_root).expanduser()
    if not root.is_absolute():
        root = Path.cwd() / root
    return load_stats(root)


def _iter_stat_values(stats: dict[str, dict[str, Any]] | None) -> list[Any]:
    if not isinstance(stats, dict):
        return []
    values: list[Any] = []
    for key in ("observation.state", "action"):
        feature_stats = stats.get(key)
        if not isinstance(feature_stats, dict):
            continue
        values.extend(feature_stats.values())
    return values


def infer_joint_unit_mode(stats: dict[str, dict[str, Any]] | None, requested_mode: str) -> str:
    if requested_mode != "auto":
        return requested_mode
    max_abs = 0.0
    for value in _iter_stat_values(stats):
        if value is None:
            continue
        array = np.asarray(value, dtype=np.float32)
        if array.size == 0:
            continue
        max_abs = max(max_abs, float(np.abs(array).max()))
    return "degrees" if max_abs > 10.0 else "radians"


def build_frame(args: argparse.Namespace, cfg: Any | None = None, joint_unit_mode: str = "radians") -> dict[str, Any]:
    frame_path = Path(args.frame_json).expanduser()
    payload = json.loads(frame_path.read_text())
    state_values = payload["state7"]
    state_shape = _feature_shape(cfg.input_features["observation.state"]) if cfg is not None else []
    expected_state_dim = int(state_shape[0]) if state_shape else len(state_values)
    if len(state_values) > expected_state_dim:
        if expected_state_dim == 6 and len(state_values) >= 7:
            state_values = list(state_values[:5]) + [state_values[-1]]
        else:
            state_values = list(state_values[:expected_state_dim])
    if joint_unit_mode == "degrees":
        state_values = np.rad2deg(np.asarray(state_values, dtype=np.float32)).tolist()
    state = torch.as_tensor(state_values, dtype=torch.float32)
    frame: dict[str, Any] = {
        "observation.state": state,
        "task": payload.get("task", args.task_text),
    }
    camera1_path = payload.get("camera1_path")
    if not camera1_path:
        raise RuntimeError("Missing required camera path for observation.images.camera1")
    frame["observation.images.camera1"] = _to_tensor_image(Path(camera1_path))

    wrist_path = payload.get("camera2_path") or payload.get("camera3_path")
    if not wrist_path:
        raise RuntimeError("Missing required wrist camera path: expected camera2_path or legacy camera3_path")
    frame["observation.images.camera2"] = _to_tensor_image(Path(wrist_path))
    return frame


def _log(args: argparse.Namespace, message: str) -> None:
    stream = sys.stderr if getattr(args, "server", False) else sys.stdout
    print(message, file=stream, flush=True)


def load_runtime(args: argparse.Namespace) -> dict[str, Any]:
    from lerobot.configs.policies import PreTrainedConfig
    from lerobot.policies.factory import make_pre_post_processors
    from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy

    device = torch.device(args.device)
    _log(
        args,
        f"[Panthera-HT][SmolVLA] Loading checkpoint: {args.model_id} device={device} control_mode={args.control_mode}",
    )
    cfg = PreTrainedConfig.from_pretrained(args.model_id, local_files_only=args.local_files_only)
    if args.vlm_model_path:
        cfg.vlm_model_name = str(Path(args.vlm_model_path).expanduser())
    force_two_camera_inputs(cfg)

    policy = SmolVLAPolicy.from_pretrained(args.model_id, config=cfg, local_files_only=args.local_files_only).to(device).eval()
    preprocessor_overrides: dict[str, dict[str, Any]] = {"device_processor": {"device": str(device)}}
    if args.vlm_model_path:
        preprocessor_overrides["tokenizer_processor"] = {"tokenizer_name": str(Path(args.vlm_model_path).expanduser())}
    dataset_stats = load_dataset_stats(args.dataset_root)
    joint_unit_mode = infer_joint_unit_mode(dataset_stats, args.joint_unit_mode)
    preprocess, postprocess = make_pre_post_processors(
        cfg,
        args.model_id,
        dataset_stats=dataset_stats,
        preprocessor_overrides=preprocessor_overrides,
    )

    _log(
        args,
        f"[Panthera-HT][SmolVLA] Checkpoint loaded: type={_config_type(cfg)} action_shape={_feature_shape(cfg.output_features['action'])} "
        f"state_shape={_feature_shape(cfg.input_features['observation.state'])} image_keys={list(cfg.input_features.keys())} "
        f"chunk_size={getattr(cfg, 'chunk_size', None)} n_action_steps={getattr(cfg, 'n_action_steps', None)} "
        f"dataset_stats={'loaded' if dataset_stats else 'missing'} dataset_root={args.dataset_root} "
        f"joint_unit_mode={joint_unit_mode}",
    )
    return {
        "device": device,
        "cfg": cfg,
        "policy": policy,
        "preprocess": preprocess,
        "postprocess": postprocess,
        "dataset_stats_loaded": bool(dataset_stats),
        "joint_unit_mode": joint_unit_mode,
    }


def run_inference(args: argparse.Namespace, runtime: dict[str, Any]) -> dict[str, Any]:
    frame_payload = json.loads(Path(args.frame_json).expanduser().read_text())
    frame_index = int(args.frame_index if args.frame_index is not None else frame_payload.get("frame_index", 0))
    control_mode = args.control_mode or frame_payload.get("control_mode", "log-only")
    camera1_source = frame_payload.get("camera1_source", "d435i_rgb")
    camera2_source = frame_payload.get("camera2_source", frame_payload.get("camera3_source", "left_wrist_rgb"))
    state_source = frame_payload.get("state_source", frame_payload.get("active_arm_label", "front_left") + "_7d")

    device = runtime["device"]
    cfg = runtime["cfg"]
    policy = runtime["policy"]
    preprocess = runtime["preprocess"]
    postprocess = runtime["postprocess"]

    joint_unit_mode = str(runtime.get("joint_unit_mode", "radians"))
    frame = build_frame(args, cfg, joint_unit_mode)
    image_input_keys = [key for key in frame if key.startswith("observation.images.")]
    input_mapping: dict[str, str] = {
        "camera1": camera1_source,
        "camera2": camera2_source,
        "state": state_source,
    }
    _log(
        args,
        f"[Panthera-HT][SmolVLA] Input mapping: {input_mapping}",
    )
    _log(
        args,
        f"[Panthera-HT][SmolVLA] image_input_keys={image_input_keys}",
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
    _log(
        args,
        f"[Panthera-HT][SmolVLA] Inference frame={frame_index} state={_to_serializable(frame['observation.state'])} task={frame['task']!r}",
    )
    _log(
        args,
        f"[Panthera-HT][SmolVLA] Predicted action frame={frame_index} action={action_step0} latency_ms={latency_ms}",
    )

    result = {
        "success": True,
        "model_id": args.model_id,
        "device": str(device),
        "task": frame["task"],
        "frame_index": frame_index,
        "control_mode": control_mode,
        "checkpoint_type": _config_type(cfg),
        "state_shape": _feature_shape(cfg.input_features["observation.state"]),
        "action_shape": _feature_shape(cfg.output_features["action"]),
        "chunk_size": getattr(cfg, "chunk_size", None),
        "n_action_steps": getattr(cfg, "n_action_steps", None),
        "dataset_stats_loaded": bool(runtime.get("dataset_stats_loaded")),
        "joint_unit_mode": joint_unit_mode,
        "image_input_keys": image_input_keys,
        "input_mapping": input_mapping,
        "predicted_action": action_serialized,
        "predicted_action_step0": action_step0,
        "latency_ms": latency_ms,
    }
    if args.output_json:
        output_path = Path(args.output_json).expanduser()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(result, ensure_ascii=False, indent=2))
    return result


def server_loop(args: argparse.Namespace, runtime: dict[str, Any]) -> None:
    ready = {
        "type": "ready",
        "model_id": args.model_id,
        "device": str(runtime["device"]),
        "checkpoint_type": _config_type(runtime["cfg"]),
        "chunk_size": getattr(runtime["cfg"], "chunk_size", None),
        "n_action_steps": getattr(runtime["cfg"], "n_action_steps", None),
        "dataset_stats_loaded": bool(runtime.get("dataset_stats_loaded")),
        "joint_unit_mode": runtime.get("joint_unit_mode", "radians"),
    }
    print(json.dumps(ready, ensure_ascii=False), flush=True)
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        request: dict[str, Any] = {}
        try:
            request = json.loads(line)
            if request.get("type") == "shutdown":
                print(json.dumps({"type": "shutdown_ack"}, ensure_ascii=False), flush=True)
                return
            request_args = copy.copy(args)
            request_args.frame_json = request["frame_json"]
            request_args.output_json = request.get("output_json")
            request_args.frame_index = request.get("frame_index", args.frame_index)
            request_args.control_mode = request.get("control_mode", args.control_mode)
            request_args.task_text = request.get("task_text", args.task_text)
            result = run_inference(request_args, runtime)
            print(
                json.dumps(
                    {"type": "result", "request_id": request.get("request_id"), "result": result},
                    ensure_ascii=False,
                ),
                flush=True,
            )
        except Exception as exc:
            print(
                json.dumps({"type": "error", "request_id": request.get("request_id"), "error": str(exc)}, ensure_ascii=False),
                flush=True,
            )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--frame-json", default=None)
    parser.add_argument("--output-json", default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--task-text", default="operate the Panthera arm on the table")
    parser.add_argument("--control-mode", default=None)
    parser.add_argument("--frame-index", type=int, default=None)
    parser.add_argument("--local-files-only", action="store_true", default=True)
    parser.add_argument("--vlm-model-path", default=None)
    parser.add_argument("--dataset-root", default=str(DEFAULT_DATASET_ROOT))
    parser.add_argument("--joint-unit-mode", choices=("auto", "radians", "degrees"), default="auto")
    parser.add_argument("--server", action="store_true")
    args = parser.parse_args()

    if not args.server and not args.frame_json:
        parser.error("--frame-json is required unless --server is set")
    runtime = load_runtime(args)
    if args.server:
        server_loop(args, runtime)
        return
    result = run_inference(args, runtime)
    print(json.dumps(result, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
    main()
