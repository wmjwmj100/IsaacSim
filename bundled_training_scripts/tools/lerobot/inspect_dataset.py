#!/usr/bin/env python3

import argparse
import json
from pathlib import Path
from typing import Any

from huggingface_hub import hf_hub_download
import numpy as np


def _safe_name(text: str) -> str:
    return "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in text)


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


def _print_meta(meta: dict[str, Any]) -> list[str]:
    features = meta.get("features", {})
    image_keys = [key for key in features.keys() if key.startswith("observation.images.")]
    print(f"robot_type: {meta.get('robot_type')}")
    print(f"episodes: {meta.get('total_episodes')}")
    print(f"frames: {meta.get('total_frames')}")
    print(f"fps: {meta.get('fps')}")
    print(f"image_keys: {image_keys}")
    print("feature_shapes:")
    for key, value in features.items():
        print(f"  {key}: dtype={value.get('dtype')}, shape={value.get('shape')}")
    return image_keys


def _save_frame_artifacts(repo_id: str, episode_index: int, output_dir: Path, video_backend: str | None = None) -> None:
    try:
        from PIL import Image
        from lerobot.datasets.lerobot_dataset import LeRobotDataset
    except Exception as exc:
        print()
        print("Skipping frame sampling because LeRobot is not available.")
        print(f"Reason: {exc}")
        return

    dataset_kwargs: dict[str, Any] = {}
    if video_backend:
        dataset_kwargs["video_backend"] = video_backend
    dataset = LeRobotDataset(repo_id, **dataset_kwargs)
    from_idx = int(dataset.meta.episodes["dataset_from_index"][episode_index])
    frame = dict(dataset[from_idx])

    output_dir.mkdir(parents=True, exist_ok=True)
    summary: dict[str, Any] = {
        "repo_id": repo_id,
        "episode_index": episode_index,
        "frame_index": from_idx,
        "keys": sorted(frame.keys()),
    }

    for key, value in frame.items():
        if not key.startswith("observation.images."):
            continue
        pil_image = _to_pil_image(value)
        filename = f"{_safe_name(key)}.png"
        pil_image.save(output_dir / filename)
        summary.setdefault("saved_images", []).append(filename)

    for key in ("observation.state", "action", "task"):
        if key in frame:
            value = frame[key]
            if hasattr(value, "tolist"):
                value = value.tolist()
            summary[key] = value

    with open(output_dir / "sample_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print()
    print(f"Saved sample frame artifacts to: {output_dir}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-id", default="lerobot/svla_so101_pickplace")
    parser.add_argument("--dataset-root", default=None)
    parser.add_argument("--video-backend", default="pyav")
    parser.add_argument("--episode-index", type=int, default=0)
    parser.add_argument(
        "--output-dir",
        default=str(Path("outputs") / "dataset_inspect" / "svla_so101_pickplace"),
    )
    args = parser.parse_args()

    dataset_root = Path(args.dataset_root).expanduser() if args.dataset_root else None
    if dataset_root:
        info_path = dataset_root / "meta" / "info.json"
    else:
        info_path = hf_hub_download(repo_id=args.repo_id, repo_type="dataset", filename="meta/info.json")
    with open(info_path, "r", encoding="utf-8") as f:
        meta = json.load(f)

    print(f"dataset: {args.repo_id}")
    image_keys = _print_meta(meta)

    output_dir = Path(args.output_dir).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / "meta_info.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    print()
    print(f"Saved dataset metadata to: {output_dir / 'meta_info.json'}")
    if not image_keys:
        print("Dataset exposes no image keys, so frame preview was skipped.")
        return

    if dataset_root:
        try:
            from lerobot.datasets.lerobot_dataset import LeRobotDataset
            dataset_kwargs: dict[str, Any] = {"root": dataset_root}
            if args.video_backend:
                dataset_kwargs["video_backend"] = args.video_backend
            dataset = LeRobotDataset(args.repo_id, **dataset_kwargs)
            from_idx = int(dataset.meta.episodes["dataset_from_index"][args.episode_index])
            frame = dict(dataset[from_idx])

            summary: dict[str, Any] = {
                "repo_id": args.repo_id,
                "dataset_root": str(dataset_root),
                "episode_index": args.episode_index,
                "frame_index": from_idx,
                "keys": sorted(frame.keys()),
            }

            for key, value in frame.items():
                if not key.startswith("observation.images."):
                    continue
                pil_image = _to_pil_image(value)
                filename = f"{_safe_name(key)}.png"
                pil_image.save(output_dir / filename)
                summary.setdefault("saved_images", []).append(filename)

            for key in ("observation.state", "action", "task"):
                if key in frame:
                    value = frame[key]
                    if hasattr(value, "tolist"):
                        value = value.tolist()
                    summary[key] = value

            with open(output_dir / "sample_summary.json", "w", encoding="utf-8") as f:
                json.dump(summary, f, ensure_ascii=False, indent=2)

            print()
            print(f"Saved sample frame artifacts to: {output_dir}")
        except Exception as exc:
            print()
            print("Skipping frame sampling because LeRobot local load failed.")
            print(f"Reason: {exc}")
    else:
        _save_frame_artifacts(args.repo_id, args.episode_index, output_dir, video_backend=args.video_backend)


if __name__ == "__main__":
    main()
