#!/usr/bin/env python3

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np


def _json_ready(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    return value


def _as_float_list(value: Any, key: str) -> list[float]:
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    elif hasattr(value, "numpy"):
        value = value.numpy()
    array = np.asarray(value, dtype=np.float64).reshape(-1)
    if array.shape[0] != 7:
        raise ValueError(f"{key} must be 7D, got shape {array.shape}")
    if not np.isfinite(array).all():
        raise ValueError(f"{key} contains non-finite values")
    return [float(item) for item in array.tolist()]


def _frame_to_record(frame: dict[str, Any], dataset_index: int) -> dict[str, Any]:
    action = _as_float_list(frame["action"], "action")
    state = _as_float_list(frame["observation.state"], "observation.state")
    return {
        "dataset_index": int(dataset_index),
        "episode_index": int(np.asarray(frame.get("episode_index", -1)).reshape(-1)[0]),
        "frame_index": int(np.asarray(frame.get("frame_index", dataset_index)).reshape(-1)[0]),
        "timestamp": float(np.asarray(frame.get("timestamp", 0.0)).reshape(-1)[0]),
        "task_index": int(np.asarray(frame.get("task_index", -1)).reshape(-1)[0]),
        "state7": state,
        "action7": action,
    }


def export_actions(
    repo_id: str,
    dataset_root: Path,
    episode_index: int,
    output_path: Path,
    start_frame: int,
    max_frames: int,
    stride: int,
    video_backend: str | None,
) -> dict[str, Any]:
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    dataset_kwargs: dict[str, Any] = {"root": dataset_root}
    if video_backend:
        dataset_kwargs["video_backend"] = video_backend
    dataset = LeRobotDataset(repo_id, **dataset_kwargs)
    episode_row = dataset.meta.episodes[episode_index]
    from_index = int(episode_row["dataset_from_index"])
    to_index = int(episode_row["dataset_to_index"])
    selected_start = from_index + int(start_frame)
    if selected_start < from_index or selected_start >= to_index:
        raise ValueError(f"start_frame={start_frame} is outside episode {episode_index} length {to_index - from_index}")
    selected_indices = list(range(selected_start, to_index, max(1, int(stride))))
    if max_frames > 0:
        selected_indices = selected_indices[: int(max_frames)]
    if not selected_indices:
        raise ValueError("No frames selected for export")

    records = [_frame_to_record(dict(dataset[index]), index) for index in selected_indices]
    actions = np.asarray([record["action7"] for record in records], dtype=np.float64)
    states = np.asarray([record["state7"] for record in records], dtype=np.float64)
    task = None
    tasks = episode_row.get("tasks")
    if isinstance(tasks, list) and tasks:
        task = tasks[0]
    elif isinstance(tasks, str):
        task = tasks

    payload = {
        "format": "panthera_dataset_action_replay_v1",
        "repo_id": repo_id,
        "dataset_root": str(dataset_root),
        "episode_index": int(episode_index),
        "episode_dataset_from_index": from_index,
        "episode_dataset_to_index": to_index,
        "episode_length": to_index - from_index,
        "start_frame": int(start_frame),
        "max_frames": int(max_frames),
        "stride": int(stride),
        "fps": int(getattr(dataset.meta, "fps", 30) or 30),
        "task": task or "push the object on the table",
        "action_dim": 7,
        "state_dim": 7,
        "action_names": ["joint1.pos", "joint2.pos", "joint3.pos", "joint4.pos", "joint5.pos", "joint6.pos", "gripper.pos"],
        "state_names": ["joint1.pos", "joint2.pos", "joint3.pos", "joint4.pos", "joint5.pos", "joint6.pos", "gripper.pos"],
        "num_actions": len(records),
        "first_action7": records[0]["action7"],
        "last_action7": records[-1]["action7"],
        "min_action7": actions.min(axis=0).tolist(),
        "max_action7": actions.max(axis=0).tolist(),
        "mean_action7": actions.mean(axis=0).tolist(),
        "first_state7": records[0]["state7"],
        "last_state7": records[-1]["state7"],
        "min_state7": states.min(axis=0).tolist(),
        "max_state7": states.max(axis=0).tolist(),
        "records": records,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(_json_ready(payload), ensure_ascii=False, indent=2) + "\n")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description="Export a LeRobot episode's 7D action stream for Panthera open-loop replay.")
    parser.add_argument("--repo-id", default="local/panthera_ht_merged")
    parser.add_argument("--dataset-root", default="outputs/hf_cache/datasets/panthera_ht_merged")
    parser.add_argument("--episode-index", type=int, default=0)
    parser.add_argument("--start-frame", type=int, default=0)
    parser.add_argument("--max-frames", type=int, default=180)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--video-backend", default=None)
    parser.add_argument(
        "--output-path",
        default="outputs/panthera_ht/replay_actions/panthera_episode_000000_actions.json",
    )
    args = parser.parse_args()

    payload = export_actions(
        repo_id=args.repo_id,
        dataset_root=Path(args.dataset_root).expanduser(),
        episode_index=args.episode_index,
        output_path=Path(args.output_path).expanduser(),
        start_frame=args.start_frame,
        max_frames=args.max_frames,
        stride=args.stride,
        video_backend=args.video_backend,
    )
    summary = {key: payload[key] for key in ("format", "repo_id", "dataset_root", "episode_index", "num_actions", "first_action7", "last_action7", "task")}
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
