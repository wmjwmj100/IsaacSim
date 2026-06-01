#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw


ACTION_NAMES = [
    "joint1.pos",
    "joint2.pos",
    "joint3.pos",
    "joint4.pos",
    "joint5.pos",
    "joint6.pos",
    "gripper.pos",
]
IMAGE_SHAPE = (480, 640, 3)


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    records = []
    with path.open("r", encoding="utf-8") as file:
        for line in file:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def as_vector7(value: Any, label: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.float32).reshape(-1)
    if array.shape != (7,):
        raise ValueError(f"{label} must be 7D, got shape {array.shape}")
    if not np.isfinite(array).all():
        raise ValueError(f"{label} contains non-finite values: {array}")
    return array


def load_rgb(path: Path) -> np.ndarray:
    image = Image.open(path).convert("RGB")
    array = np.asarray(image, dtype=np.uint8)
    if array.shape != IMAGE_SHAPE:
        raise ValueError(f"Expected image shape {IMAGE_SHAPE} at {path}, got {array.shape}")
    return array


def feature_spec() -> dict[str, dict[str, Any]]:
    return {
        "observation.state": {
            "dtype": "float32",
            "shape": (7,),
            "names": ACTION_NAMES,
        },
        "action": {
            "dtype": "float32",
            "shape": (7,),
            "names": ACTION_NAMES,
        },
        "observation.images.up": {
            "dtype": "image",
            "shape": IMAGE_SHAPE,
            "names": ["height", "width", "channels"],
        },
        "observation.images.side": {
            "dtype": "image",
            "shape": IMAGE_SHAPE,
            "names": ["height", "width", "channels"],
        },
    }


def write_preview(raw_root: Path, output_path: Path, max_frames: int = 6) -> None:
    episode_dirs = sorted(path for path in raw_root.glob("episode_*") if path.is_dir())
    if not episode_dirs:
        return
    records = [record for record in load_jsonl(episode_dirs[0] / "records.jsonl") if record.get("saved", True)]
    if not records:
        return
    if len(records) <= max_frames:
        selected = records
    else:
        selected = [records[int(round(i))] for i in np.linspace(0, len(records) - 1, max_frames)]

    thumb_w, thumb_h = 240, 180
    label_h = 22
    sheet = Image.new("RGB", (thumb_w * len(selected), (thumb_h + label_h) * 2), "white")
    draw = ImageDraw.Draw(sheet)
    for column, record in enumerate(selected):
        wrist_key = "camera2_path" if "camera2_path" in record else "camera3_path"
        for row, key in enumerate(("camera1_path", wrist_key)):
            image = Image.open(record[key]).convert("RGB").resize((thumb_w, thumb_h))
            y = row * (thumb_h + label_h)
            sheet.paste(image, (column * thumb_w, y + label_h))
            label = f"f{int(record['frame_index']):03d} {record.get('teacher_phase', '')}"
            draw.text((column * thumb_w + 4, y + 4), label, fill=(0, 0, 0))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output_path)


def convert(args: argparse.Namespace) -> dict[str, Any]:
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    raw_root = Path(args.raw_root).expanduser()
    if not raw_root.is_absolute():
        raw_root = Path.cwd() / raw_root
    if not (raw_root / "summary.json").exists():
        raise FileNotFoundError(f"Raw summary not found: {raw_root / 'summary.json'}")
    raw_summary = load_json(raw_root / "summary.json")

    dataset_root = Path(args.dataset_root).expanduser()
    if not dataset_root.is_absolute():
        dataset_root = Path.cwd() / dataset_root
    if dataset_root.exists():
        if not args.overwrite:
            raise FileExistsError(f"Dataset root exists; pass --overwrite to replace it: {dataset_root}")
        shutil.rmtree(dataset_root)

    dataset = LeRobotDataset.create(
        repo_id=args.repo_id,
        fps=int(raw_summary.get("fps") or args.fps),
        features=feature_spec(),
        root=dataset_root,
        robot_type=args.robot_type,
        use_videos=False,
        image_writer_processes=0,
        image_writer_threads=int(args.image_writer_threads),
    )

    episode_reports = []
    total_frames = 0
    for episode_dir in sorted(path for path in raw_root.glob("episode_*") if path.is_dir()):
        episode_summary = load_json(episode_dir / "summary.json")
        records = [record for record in load_jsonl(episode_dir / "records.jsonl") if record.get("saved", True)]
        if args.only_successful and not episode_summary.get("success"):
            continue
        displacement = episode_summary.get("cube_xy_displacement_m")
        if args.min_displacement is not None and (displacement is None or float(displacement) < args.min_displacement):
            continue
        if args.max_displacement is not None and (displacement is None or float(displacement) > args.max_displacement):
            continue
        if not records:
            continue
        for record in records:
            camera1_path = Path(record["camera1_path"])
            wrist_camera_path = Path(record.get("camera2_path") or record["camera3_path"])
            frame = {
                "observation.state": as_vector7(record["state7"], "observation.state"),
                "action": as_vector7(record["action7"], "action"),
                "observation.images.up": load_rgb(camera1_path),
                "observation.images.side": load_rgb(wrist_camera_path),
                "task": record.get("task") or raw_summary.get("task") or args.task_text,
            }
            dataset.add_frame(frame)
        dataset.save_episode(parallel_encoding=False)
        episode_reports.append(
            {
                "episode_dir": str(episode_dir),
                "raw_episode_index": episode_summary.get("episode_index"),
                "frames": len(records),
                "raw_success": episode_summary.get("success"),
                "cube_xy_displacement_m": episode_summary.get("cube_xy_displacement_m"),
            }
        )
        total_frames += len(records)

    preview_path = dataset_root / "preview_contact_sheet.jpg"
    write_preview(raw_root, preview_path)

    report = {
        "schema": "roboclaw_lerobot_conversion_v1",
        "raw_root": str(raw_root),
        "dataset_root": str(dataset_root),
        "repo_id": args.repo_id,
        "robot_type": args.robot_type,
        "episodes": len(episode_reports),
        "frames": total_frames,
        "only_successful": bool(args.only_successful),
        "min_displacement": args.min_displacement,
        "max_displacement": args.max_displacement,
        "raw_success_count": raw_summary.get("success_count"),
        "raw_success_rate": raw_summary.get("success_rate"),
        "episode_reports": episode_reports,
        "preview_contact_sheet": str(preview_path),
    }
    (dataset_root / "conversion_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert Roboclaw Isaac teacher raw data into a LeRobot dataset.")
    parser.add_argument("--raw-root", required=True)
    parser.add_argument("--dataset-root", default="outputs/lerobot_datasets/roboclaw_sim_teacher")
    parser.add_argument("--repo-id", default="local/roboclaw_sim_teacher")
    parser.add_argument("--robot-type", default="panthera_ht")
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--task-text", default="push the object on the table")
    parser.add_argument("--image-writer-threads", type=int, default=0)
    parser.add_argument("--only-successful", action="store_true")
    parser.add_argument("--min-displacement", type=float, default=None)
    parser.add_argument("--max-displacement", type=float, default=None)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    report = convert(args)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
