#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from typing import Any

import numpy as np


JOINT7_NAMES = ["joint1", "joint2", "joint3", "joint4", "joint5", "joint6", "gripper"]
FINGER_NAMES = ("L_finger_joint", "R_finger_joint", "L_finger", "R_finger")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Replay a 7D LeRobot trajectory in Isaac Sim and diagnose joint mapping/tracking."
    )
    parser.add_argument("--usd-path", default="/home/wmj/Documents/roboclaw.usd")
    parser.add_argument("--articulation-path", default="/Panthera_HT_FrontLeft/root_joint")
    parser.add_argument("--replay-json", required=True)
    parser.add_argument("--source", choices=("state", "action"), default="state")
    parser.add_argument("--apply-mode", choices=("teleport", "target"), default="teleport")
    parser.add_argument(
        "--gripper-mode",
        choices=("clip-0-04", "raw"),
        default="clip-0-04",
        help="clip-0-04 matches the current rollout code; raw tests whether USD can execute dataset gripper values.",
    )
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--max-records", type=int, default=0)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument(
        "--settle-steps",
        type=int,
        default=1,
        help="Physics steps after each command before measuring actual joint positions.",
    )
    parser.add_argument("--warmup-steps", type=int, default=5)
    parser.add_argument("--renderer", default="RaytracedLighting")
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-csv", default="")
    return parser.parse_args()


def json_ready(value: Any) -> Any:
    if isinstance(value, Path):
        return value.as_posix()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(item) for item in value]
    return value


def load_replay(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text())
    records = payload.get("records")
    if not isinstance(records, list) or not records:
        raise RuntimeError(f"Replay file has no records: {path}")
    return payload


def replay_vector(record: dict[str, Any], source: str) -> list[float]:
    key = "state7" if source == "state" else "action7"
    value = np.asarray(record[key], dtype=np.float64).reshape(-1)
    if value.shape[0] != 7 or not np.isfinite(value).all():
        raise RuntimeError(f"{key} must be finite 7D, got {record.get(key)!r}")
    return [float(item) for item in value.tolist()]


def selected_records(payload: dict[str, Any], start_index: int, max_records: int, stride: int) -> list[tuple[int, dict[str, Any]]]:
    records = payload["records"]
    stride = max(1, int(stride))
    start_index = max(0, int(start_index))
    selected = list(enumerate(records[start_index:], start=start_index))[::stride]
    if max_records > 0:
        selected = selected[: int(max_records)]
    if not selected:
        raise RuntimeError("No replay records selected")
    return selected


def action_target_for_dofs(dof_names: list[str], action7: list[float], gripper_mode: str) -> np.ndarray:
    gripper = float(action7[6])
    if gripper_mode == "clip-0-04":
        gripper = float(np.clip(gripper, 0.0, 0.04))
    targets = {
        "joint1": float(action7[0]),
        "joint2": float(action7[1]),
        "joint3": float(action7[2]),
        "joint4": float(action7[3]),
        "joint5": float(action7[4]),
        "joint6": float(action7[5]),
        "L_finger": gripper,
        "R_finger": gripper,
        "L_finger_joint": gripper,
        "R_finger_joint": gripper,
    }
    pose = np.zeros((1, len(dof_names)), dtype=np.float32)
    for dof_index, dof_name in enumerate(dof_names):
        pose[0, dof_index] = targets.get(dof_name, 0.0)
    return pose


def positions_by_name(articulation: Any) -> dict[str, float]:
    positions = np.asarray(articulation.get_joint_positions(), dtype=np.float64).reshape(-1)
    return {
        str(dof_name): float(positions[dof_index])
        for dof_index, dof_name in enumerate(articulation.dof_names[: len(positions)])
    }


def state7_from_positions(positions: dict[str, float]) -> list[float]:
    finger_values = [positions.get(name) for name in FINGER_NAMES]
    valid_fingers = [float(value) for value in finger_values if value is not None]
    gripper = float(sum(valid_fingers) / len(valid_fingers)) if valid_fingers else 0.0
    return [float(positions.get(f"joint{joint_index}", 0.0)) for joint_index in range(1, 7)] + [gripper]


def target_by_name(dof_names: list[str], target: np.ndarray) -> dict[str, float]:
    values = np.asarray(target, dtype=np.float64).reshape(-1)
    return {str(name): float(values[index]) for index, name in enumerate(dof_names[: len(values)])}


def finite_corrcoef(x: np.ndarray, y: np.ndarray) -> float | None:
    if x.size < 2 or y.size < 2:
        return None
    if float(np.std(x)) < 1e-9 or float(np.std(y)) < 1e-9:
        return None
    value = float(np.corrcoef(x, y)[0, 1])
    return value if math.isfinite(value) else None


def fit_line(x: np.ndarray, y: np.ndarray) -> dict[str, float | None]:
    if x.size < 2 or float(np.std(x)) < 1e-9:
        return {"slope": None, "intercept": None, "r2": None}
    slope, intercept = np.polyfit(x, y, 1)
    pred = slope * x + intercept
    ss_res = float(np.sum((y - pred) ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    r2 = None if ss_tot < 1e-12 else 1.0 - ss_res / ss_tot
    return {"slope": float(slope), "intercept": float(intercept), "r2": None if r2 is None else float(r2)}


def summarize(records: list[dict[str, Any]], dof_names: list[str]) -> dict[str, Any]:
    source7 = np.asarray([record["source7"] for record in records], dtype=np.float64)
    actual7 = np.asarray([record["actual_after_step_state7"] for record in records], dtype=np.float64)
    target7 = np.asarray([record["commanded_target_state7"] for record in records], dtype=np.float64)
    after_set7 = np.asarray([record["actual_after_set_state7"] for record in records], dtype=np.float64)

    source_to_actual_corr = []
    for i in range(6):
        row = []
        for j in range(6):
            row.append(finite_corrcoef(source7[:, i], actual7[:, j]))
        source_to_actual_corr.append(row)

    source_to_actual_fit = {}
    target_tracking = {}
    after_set_tracking = {}
    for index, name in enumerate(JOINT7_NAMES):
        source_to_actual_fit[name] = fit_line(source7[:, index], actual7[:, index])
        err = actual7[:, index] - target7[:, index]
        target_tracking[name] = {
            "mean_signed_error": float(np.mean(err)),
            "mean_abs_error": float(np.mean(np.abs(err))),
            "max_abs_error": float(np.max(np.abs(err))),
            "p95_abs_error": float(np.quantile(np.abs(err), 0.95)),
        }
        set_err = after_set7[:, index] - target7[:, index]
        after_set_tracking[name] = {
            "mean_signed_error": float(np.mean(set_err)),
            "mean_abs_error": float(np.mean(np.abs(set_err))),
            "max_abs_error": float(np.max(np.abs(set_err))),
            "p95_abs_error": float(np.quantile(np.abs(set_err), 0.95)),
        }

    best_actual_for_source = {}
    for source_index, source_name in enumerate(JOINT7_NAMES[:6]):
        row = source_to_actual_corr[source_index]
        finite = [(actual_index, value) for actual_index, value in enumerate(row) if value is not None]
        if finite:
            actual_index, value = max(finite, key=lambda item: abs(item[1]))
            best_actual_for_source[source_name] = {
                "actual_joint": JOINT7_NAMES[actual_index],
                "corr": float(value),
            }
        else:
            best_actual_for_source[source_name] = {"actual_joint": None, "corr": None}

    return {
        "num_records": len(records),
        "dof_names": dof_names,
        "source7_min": source7.min(axis=0).tolist(),
        "source7_max": source7.max(axis=0).tolist(),
        "target7_min": target7.min(axis=0).tolist(),
        "target7_max": target7.max(axis=0).tolist(),
        "actual_after_step7_min": actual7.min(axis=0).tolist(),
        "actual_after_step7_max": actual7.max(axis=0).tolist(),
        "source_to_actual_corr_joint1_to_joint6": source_to_actual_corr,
        "best_actual_for_each_source_joint": best_actual_for_source,
        "source_to_actual_same_dim_fit": source_to_actual_fit,
        "target_tracking_after_step": target_tracking,
        "target_tracking_immediate_after_set": after_set_tracking,
    }


def write_csv(path: Path, records: list[dict[str, Any]]) -> None:
    headers = ["record_i", "dataset_frame_index"]
    for name in JOINT7_NAMES:
        headers.append(f"source_{name}")
    for name in JOINT7_NAMES:
        headers.append(f"target_{name}")
    for name in JOINT7_NAMES:
        headers.append(f"actual_after_set_{name}")
    for name in JOINT7_NAMES:
        headers.append(f"actual_after_step_{name}")
    for name in JOINT7_NAMES:
        headers.append(f"error_after_step_{name}")

    lines = [",".join(headers)]
    for record in records:
        source = record["source7"]
        target = record["commanded_target_state7"]
        after_set = record["actual_after_set_state7"]
        after_step = record["actual_after_step_state7"]
        errors = [after_step[i] - target[i] for i in range(7)]
        values: list[Any] = [record["replay_index"], record.get("dataset_frame_index")]
        values.extend(source)
        values.extend(target)
        values.extend(after_set)
        values.extend(after_step)
        values.extend(errors)
        lines.append(",".join("" if value is None else str(value) for value in values))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n")


def main() -> None:
    args = parse_args()

    from isaacsim import SimulationApp

    simulation_app = SimulationApp({"headless": True, "renderer": args.renderer})

    from isaacsim.core.api import World
    from isaacsim.core.prims import Articulation
    from isaacsim.core.utils.stage import is_stage_loading, open_stage

    replay_path = Path(args.replay_json).expanduser()
    payload = load_replay(replay_path)
    selected = selected_records(payload, args.start_index, args.max_records, args.stride)
    started_at = time.time()

    try:
        if not open_stage(str(Path(args.usd_path).expanduser())):
            raise RuntimeError(f"Failed to open USD: {args.usd_path}")
        for _ in range(600):
            if not is_stage_loading():
                break
            simulation_app.update()

        world = World(stage_units_in_meters=1.0, physics_prim_path="/physicsScene")
        articulation = world.scene.add(
            Articulation(prim_paths_expr=args.articulation_path, name="roboclaw_joint_diagnostics")
        )
        world.reset()
        world.play()
        for _ in range(max(0, int(args.warmup_steps))):
            world.step(render=bool(args.render))

        dof_names = [str(name) for name in articulation.dof_names]
        first_vector = replay_vector(selected[0][1], args.source)
        first_target = action_target_for_dofs(dof_names, first_vector, args.gripper_mode)
        articulation.set_joint_positions(first_target)
        articulation.set_joint_position_targets(first_target)
        for _ in range(max(0, int(args.settle_steps))):
            world.step(render=bool(args.render))

        records: list[dict[str, Any]] = []
        for replay_index, source_record in selected:
            source7 = replay_vector(source_record, args.source)
            target = action_target_for_dofs(dof_names, source7, args.gripper_mode)
            before = positions_by_name(articulation)
            if args.apply_mode == "teleport":
                articulation.set_joint_positions(target)
            else:
                articulation.set_joint_position_targets(target)
            after_set = positions_by_name(articulation)
            for _ in range(max(1, int(args.settle_steps))):
                world.step(render=bool(args.render))
            after_step = positions_by_name(articulation)
            records.append(
                {
                    "replay_index": int(replay_index),
                    "dataset_index": source_record.get("dataset_index"),
                    "dataset_frame_index": source_record.get("frame_index"),
                    "timestamp": source_record.get("timestamp"),
                    "source": args.source,
                    "apply_mode": args.apply_mode,
                    "gripper_mode": args.gripper_mode,
                    "source7": source7,
                    "commanded_target_by_dof": target_by_name(dof_names, target),
                    "commanded_target_state7": state7_from_positions(target_by_name(dof_names, target)),
                    "actual_before_by_dof": before,
                    "actual_before_state7": state7_from_positions(before),
                    "actual_after_set_by_dof": after_set,
                    "actual_after_set_state7": state7_from_positions(after_set),
                    "actual_after_step_by_dof": after_step,
                    "actual_after_step_state7": state7_from_positions(after_step),
                }
            )

        summary = summarize(records, dof_names)
        result = {
            "success": True,
            "usd_path": str(Path(args.usd_path).expanduser()),
            "articulation_path": args.articulation_path,
            "replay_json": str(replay_path),
            "replay_payload": {
                "repo_id": payload.get("repo_id"),
                "dataset_root": payload.get("dataset_root"),
                "episode_index": payload.get("episode_index"),
                "episode_length": payload.get("episode_length"),
                "fps": payload.get("fps"),
                "task": payload.get("task"),
            },
            "source": args.source,
            "apply_mode": args.apply_mode,
            "gripper_mode": args.gripper_mode,
            "start_index": int(args.start_index),
            "max_records": int(args.max_records),
            "stride": int(args.stride),
            "settle_steps": int(args.settle_steps),
            "duration_s": time.time() - started_at,
            "summary": summary,
            "records": records,
        }
        output_json = Path(args.output_json).expanduser()
        output_json.parent.mkdir(parents=True, exist_ok=True)
        output_json.write_text(json.dumps(json_ready(result), ensure_ascii=False, indent=2) + "\n")
        if args.output_csv:
            write_csv(Path(args.output_csv).expanduser(), records)
        print(json.dumps({"success": True, "output_json": str(output_json), "summary": summary}, ensure_ascii=False, indent=2))
    finally:
        simulation_app.close()


if __name__ == "__main__":
    main()
