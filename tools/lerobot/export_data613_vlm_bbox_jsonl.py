#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


PROMPT = "Locate the object inside the green box and the target location marked by the blue box."


def _norm_bbox_xyxy(bbox: list[float], image_size: list[int]) -> list[int]:
    width, height = image_size
    x1, y1, x2, y2 = bbox
    return [
        round(x1 / width * 1000),
        round(y1 / height * 1000),
        round(x2 / width * 1000),
        round(y2 / height * 1000),
    ]


def _record_from_annotation(annotation_path: Path, merged_episode_index: int) -> dict[str, Any]:
    episode_root = annotation_path.parents[1]
    annotation = json.loads(annotation_path.read_text(encoding="utf-8"))
    image_size = annotation["start"]["image_size"]
    start_bbox = annotation["start"]["bbox_xyxy"]
    end_bbox = annotation["end"]["bbox_xyxy"]
    start_center = annotation["start"]["center_xy"]
    end_center = annotation["end"]["center_xy"]
    top_image = episode_root / "images/up/episode_000000/frame_000000.png"
    side_image = episode_root / "images/side/episode_000000/frame_000000.png"
    annotated_top_image = (
        episode_root
        / "start_end_region_annotations/frame_images/up/episode_000000/frame_000000.png"
    )

    return {
        "source_episode_root": str(episode_root),
        "source_episode_index": int(annotation.get("episode_index", 0)),
        "merged_episode_index": merged_episode_index,
        "target_label": annotation.get("target_label", "object"),
        "camera": annotation.get("camera", "up"),
        "image": str(top_image),
        "side_image": str(side_image) if side_image.exists() else None,
        "annotated_image": str(annotated_top_image) if annotated_top_image.exists() else None,
        "instruction": PROMPT,
        "answer_json": {
            "object_bbox_xyxy_1000": _norm_bbox_xyxy(start_bbox, image_size),
            "target_bbox_xyxy_1000": _norm_bbox_xyxy(end_bbox, image_size),
        },
        "bbox_xyxy": {
            "object": start_bbox,
            "target": end_bbox,
        },
        "center_xy": {
            "object": start_center,
            "target": end_center,
        },
        "image_size": image_size,
        "frame_index": 0,
        "prompt_for_action_training": (
            "Pick up the object inside the green box and place it at the location marked by the blue box."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export data613 start/end box annotations as JSONL supervision for VLM localization."
    )
    parser.add_argument("--data-root", default="/work/IsaacSim/data/data613")
    parser.add_argument(
        "--merge-report",
        default="/work/IsaacSim/outputs/lerobot_datasets/roboclaw_data613_vp_30ep/merge_report.json",
    )
    parser.add_argument(
        "--output",
        default="/work/IsaacSim/outputs/lerobot_datasets/roboclaw_data613_vp_30ep/vlm_bbox_supervision.jsonl",
    )
    args = parser.parse_args()

    data_root = Path(args.data_root).expanduser()
    merge_report_path = Path(args.merge_report).expanduser()
    output_path = Path(args.output).expanduser()

    merge_report = json.loads(merge_report_path.read_text(encoding="utf-8"))
    merged_indices_by_source = {
        item["source_root"]: int(item["merged_episode_index"])
        for item in merge_report["source_episodes"]
    }

    records = []
    for annotation_path in sorted(data_root.glob("**/start_end_region_annotations/episode_000000.json")):
        episode_root = str(annotation_path.parents[1])
        if episode_root not in merged_indices_by_source:
            raise KeyError(f"Annotation root is absent from merge_report.json: {episode_root}")
        records.append(_record_from_annotation(annotation_path, merged_indices_by_source[episode_root]))

    records.sort(key=lambda item: item["merged_episode_index"])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as file:
        for record in records:
            file.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"Wrote {len(records)} VLM bbox records to {output_path}")


if __name__ == "__main__":
    main()
