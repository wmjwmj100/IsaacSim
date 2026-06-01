#!/usr/bin/env python3

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from panthera_smolvla_infer import build_frame, load_dataset_stats


def _json_ready(value: Any) -> Any:
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    return value


def _as_action_chunk(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    array = np.asarray(value, dtype=np.float64)
    if array.ndim == 3 and array.shape[0] == 1:
        array = array[0]
    if array.ndim != 2 or array.shape[1] < 7:
        raise RuntimeError(f"Expected action chunk shape (T, >=7), got {array.shape}")
    return array[:, :7]


def _vector_summary(vectors: np.ndarray, prefix: str) -> dict[str, Any]:
    return {
        f"first_{prefix}7": vectors[0].tolist(),
        f"last_{prefix}7": vectors[-1].tolist(),
        f"min_{prefix}7": vectors.min(axis=0).tolist(),
        f"max_{prefix}7": vectors.max(axis=0).tolist(),
        f"mean_{prefix}7": vectors.mean(axis=0).tolist(),
    }


def _load_teacher_actions(path: Path, count: int, start_index: int) -> np.ndarray | None:
    if not path:
        return None
    payload = json.loads(path.read_text())
    records = payload.get("records")
    if not isinstance(records, list):
        raise RuntimeError(f"Teacher replay has no records: {path}")
    selected = records[start_index : start_index + count]
    if len(selected) != count:
        raise RuntimeError(f"Teacher replay only has {len(selected)} records from start_index={start_index}")
    return np.asarray([record["action7"] for record in selected], dtype=np.float64)


def main() -> None:
    parser = argparse.ArgumentParser(description="Export one SmolVLA predicted action chunk as replay JSON.")
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--frame-json", required=True)
    parser.add_argument("--output-path", required=True)
    parser.add_argument("--dataset-root", default="outputs/lerobot_datasets/roboclaw_sim_teacher_widepos_clean_89ep")
    parser.add_argument("--vlm-model-path", default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--task-text", default="push the object on the table")
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--local-files-only", action="store_true", default=True)
    parser.add_argument("--teacher-replay-json", default="")
    parser.add_argument("--teacher-start-index", type=int, default=0)
    args = parser.parse_args()

    from lerobot.configs.policies import PreTrainedConfig
    from lerobot.policies.factory import make_pre_post_processors
    from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy

    device = torch.device(args.device)
    cfg = PreTrainedConfig.from_pretrained(args.model_id, local_files_only=args.local_files_only)
    if args.vlm_model_path:
        cfg.vlm_model_name = str(Path(args.vlm_model_path).expanduser())

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
    preprocess, postprocess = make_pre_post_processors(
        cfg,
        args.model_id,
        dataset_stats=load_dataset_stats(args.dataset_root),
        preprocessor_overrides=preprocessor_overrides,
    )

    frame_args = argparse.Namespace(
        frame_json=args.frame_json,
        task_text=args.task_text,
    )
    frame_payload = json.loads(Path(args.frame_json).expanduser().read_text())
    frame = build_frame(frame_args)
    processed = preprocess(frame)
    policy.reset()
    with torch.inference_mode():
        predicted = postprocess(policy.predict_action_chunk(processed))
    actions = _as_action_chunk(predicted)
    state7 = [float(value) for value in frame_payload["state7"][:7]]

    teacher_actions = None
    comparison = None
    if args.teacher_replay_json:
        teacher_actions = _load_teacher_actions(
            Path(args.teacher_replay_json).expanduser(),
            len(actions),
            int(args.teacher_start_index),
        )
        abs_error = np.abs(actions - teacher_actions)
        comparison = {
            "teacher_replay_json": args.teacher_replay_json,
            "teacher_start_index": int(args.teacher_start_index),
            "mean_abs_error": float(abs_error.mean()),
            "max_abs_error": float(abs_error.max()),
            "per_dim_mean_abs_error": abs_error.mean(axis=0).tolist(),
            "per_step_mean_abs_error": abs_error.mean(axis=1).tolist(),
            "first_step_abs_error": abs_error[0].tolist(),
            "last_step_abs_error": abs_error[-1].tolist(),
        }

    records = [
        {
            "dataset_index": i,
            "episode_index": 0,
            "frame_index": i,
            "timestamp": float(i / max(1, int(args.fps))),
            "task_index": 0,
            "state7": state7,
            "action7": actions[i].tolist(),
        }
        for i in range(len(actions))
    ]
    payload = {
        "format": "panthera_dataset_action_replay_v1",
        "source": "smolvla_predicted_chunk",
        "model_id": args.model_id,
        "frame_json": args.frame_json,
        "dataset_root": args.dataset_root,
        "task": frame.get("task", args.task_text),
        "fps": int(args.fps),
        "episode_index": 0,
        "episode_length": len(records),
        "action_dim": 7,
        "state_dim": 7,
        "action_names": [
            "joint1.pos",
            "joint2.pos",
            "joint3.pos",
            "joint4.pos",
            "joint5.pos",
            "joint6.pos",
            "gripper.pos",
        ],
        "state_names": [
            "joint1.pos",
            "joint2.pos",
            "joint3.pos",
            "joint4.pos",
            "joint5.pos",
            "joint6.pos",
            "gripper.pos",
        ],
        "num_actions": len(records),
        **_vector_summary(actions, "action"),
        "first_state7": state7,
        "last_state7": state7,
        "min_state7": state7,
        "max_state7": state7,
        "comparison_to_teacher": comparison,
        "records": records,
    }

    output_path = Path(args.output_path).expanduser()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(_json_ready(payload), ensure_ascii=False, indent=2) + "\n")
    summary = {
        "output_path": str(output_path),
        "num_actions": len(records),
        "first_action7": payload["first_action7"],
        "last_action7": payload["last_action7"],
        "comparison_to_teacher": comparison,
    }
    print(json.dumps(_json_ready(summary), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
