#!/usr/bin/env python3

import argparse
import json
import shutil
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq


DATA_PATH = "data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet"
EPISODES_PATH = "meta/episodes/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet"


def _to_jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): _to_jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_jsonable(item) for item in value]
    return value


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(_to_jsonable(payload), file, ensure_ascii=False, indent=2)
        file.write("\n")


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def _stats_from_array(array: np.ndarray) -> dict[str, np.ndarray]:
    array = np.asarray(array, dtype=np.float64)
    return {
        "min": np.min(array, axis=0),
        "max": np.max(array, axis=0),
        "mean": np.mean(array, axis=0),
        "std": np.std(array, axis=0),
        "count": np.array([array.shape[0]], dtype=np.int64),
        "q01": np.quantile(array, 0.01, axis=0),
        "q10": np.quantile(array, 0.10, axis=0),
        "q50": np.quantile(array, 0.50, axis=0),
        "q90": np.quantile(array, 0.90, axis=0),
        "q99": np.quantile(array, 0.99, axis=0),
    }


def _load_tables(root: Path) -> list[pa.Table]:
    info = _read_json(root / "meta" / "info.json")
    pattern = info.get("data_path", DATA_PATH)
    tables = []
    for path in sorted((root / "data").glob("chunk-*/*.parquet")):
        tables.append(pq.read_table(path))
    if not tables:
        raise RuntimeError(f"No parquet data files found under {root / 'data'} for pattern {pattern!r}")
    return tables


def _session_roots(input_root: Path) -> list[Path]:
    roots = [path for path in sorted(input_root.iterdir()) if (path / "meta" / "info.json").is_file()]
    if (input_root / "meta" / "info.json").is_file():
        roots = [input_root]
    if not roots:
        raise RuntimeError(f"No LeRobot roots found in {input_root}")
    return roots


def _copy_collection_files(roots: list[Path], output_root: Path) -> None:
    collections = []
    for episode_index, root in enumerate(roots):
        collection_path = root / "collection.json"
        if not collection_path.is_file():
            continue
        collection = _read_json(collection_path)
        collection["source_root"] = str(root)
        collection["merged_episode_index"] = episode_index
        collections.append(collection)
    if collections:
        _write_json(output_root / "collection.json", {"merged_collections": collections})


def merge_roots(input_root: Path, output_root: Path, repo_id: str, overwrite: bool = False) -> dict[str, Any]:
    roots = _session_roots(input_root)
    if output_root.exists():
        if not overwrite:
            raise RuntimeError(f"Output root already exists: {output_root}. Pass --overwrite to replace it.")
        shutil.rmtree(output_root)

    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "data" / "chunk-000").mkdir(parents=True, exist_ok=True)
    (output_root / "meta" / "episodes" / "chunk-000").mkdir(parents=True, exist_ok=True)

    first_info = _read_json(roots[0] / "meta" / "info.json")
    features = first_info["features"]
    robot_type = first_info.get("robot_type", "panthera_ht")
    fps = int(first_info.get("fps", 30))

    episode_rows: list[dict[str, Any]] = []
    task_texts: list[str] = []
    frame_offset = 0
    all_state = []
    all_action = []

    for episode_index, root in enumerate(roots):
        info = _read_json(root / "meta" / "info.json")
        if info.get("features") != features:
            raise RuntimeError(f"Feature schema mismatch in {root}")
        if int(info.get("fps", fps)) != fps:
            raise RuntimeError(f"FPS mismatch in {root}: {info.get('fps')} != {fps}")

        tables = _load_tables(root)
        table = pa.concat_tables(tables, promote_options="default") if len(tables) > 1 else tables[0]
        row_count = table.num_rows

        task = ""
        collection_path = root / "collection.json"
        if collection_path.is_file():
            task = str(_read_json(collection_path).get("task") or "")
        if not task:
            tasks_table = pq.read_table(root / "meta" / "tasks.parquet")
            task = str(tasks_table.to_pylist()[0]["task"])
        if task not in task_texts:
            task_texts.append(task)
        task_index = task_texts.index(task)

        table = table.set_column(
            table.schema.get_field_index("episode_index"),
            "episode_index",
            pa.array(np.full(row_count, episode_index, dtype=np.int64)),
        )
        table = table.set_column(
            table.schema.get_field_index("frame_index"),
            "frame_index",
            pa.array(np.arange(row_count, dtype=np.int64)),
        )
        table = table.set_column(
            table.schema.get_field_index("index"),
            "index",
            pa.array(np.arange(frame_offset, frame_offset + row_count, dtype=np.int64)),
        )
        table = table.set_column(
            table.schema.get_field_index("task_index"),
            "task_index",
            pa.array(np.full(row_count, task_index, dtype=np.int64)),
        )
        pq.write_table(table, output_root / DATA_PATH.format(chunk_index=0, file_index=episode_index))

        state = np.asarray(table.column("observation.state").to_pylist(), dtype=np.float32)
        action = np.asarray(table.column("action").to_pylist(), dtype=np.float32)
        all_state.append(state)
        all_action.append(action)

        episode_rows.append(
            {
                "episode_index": episode_index,
                "tasks": [task],
                "length": row_count,
                "data/chunk_index": 0,
                "data/file_index": episode_index,
                "dataset_from_index": frame_offset,
                "dataset_to_index": frame_offset + row_count,
                "meta/episodes/chunk_index": 0,
                "meta/episodes/file_index": 0,
            }
        )
        frame_offset += row_count

    tasks = pd.DataFrame({"task_index": list(range(len(task_texts)))}, index=task_texts)
    tasks.to_parquet(output_root / "meta" / "tasks.parquet")

    state_stats = _stats_from_array(np.concatenate(all_state, axis=0))
    action_stats = _stats_from_array(np.concatenate(all_action, axis=0))
    stats = _read_json(roots[0] / "meta" / "stats.json")
    stats["observation.state"] = state_stats
    stats["action"] = action_stats

    for row in episode_rows:
        start = row["dataset_from_index"]
        end = row["dataset_to_index"]
        episode_state = np.concatenate(all_state, axis=0)[start:end]
        episode_action = np.concatenate(all_action, axis=0)[start:end]
        row.update({f"stats/observation.state/{key}": value for key, value in _stats_from_array(episode_state).items()})
        row.update({f"stats/action/{key}": value for key, value in _stats_from_array(episode_action).items()})

    first_episode_meta = pq.read_table(roots[0] / EPISODES_PATH.format(chunk_index=0, file_index=0)).to_pylist()[0]
    for row in episode_rows:
        for key, value in first_episode_meta.items():
            if key.startswith("stats/") and key not in row:
                row[key] = value

    episodes_table = pa.Table.from_pylist(_to_jsonable(episode_rows))
    pq.write_table(episodes_table, output_root / EPISODES_PATH.format(chunk_index=0, file_index=0))

    info = dict(first_info)
    info.update(
        {
            "robot_type": robot_type,
            "total_episodes": len(roots),
            "total_frames": frame_offset,
            "total_tasks": len(task_texts),
            "fps": fps,
            "splits": {"train": f"0:{len(roots)}"},
            "data_path": DATA_PATH,
            "video_path": None,
            "features": features,
        }
    )
    _write_json(output_root / "meta" / "info.json", info)
    _write_json(output_root / "meta" / "stats.json", stats)
    _copy_collection_files(roots, output_root)

    report = {
        "repo_id": repo_id,
        "input_root": str(input_root),
        "output_root": str(output_root),
        "source_roots": [str(root) for root in roots],
        "total_episodes": len(roots),
        "total_frames": frame_offset,
        "tasks": task_texts,
        "features": list(features.keys()),
    }
    _write_json(output_root / "merge_report.json", report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Merge multiple local LeRobot v3 roots into one local root.")
    parser.add_argument("--input-root", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--repo-id", default="local/panthera_ht_merged")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    report = merge_roots(
        input_root=Path(args.input_root).expanduser(),
        output_root=Path(args.output_root).expanduser(),
        repo_id=args.repo_id,
        overwrite=args.overwrite,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
