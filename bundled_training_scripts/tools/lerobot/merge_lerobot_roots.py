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


def _stats_from_array(array: np.ndarray, min_std: float = 1e-6) -> dict[str, np.ndarray]:
    array = np.asarray(array, dtype=np.float64)
    std = np.std(array, axis=0)
    std = np.maximum(std, min_std)
    return {
        "min": np.min(array, axis=0),
        "max": np.max(array, axis=0),
        "mean": np.mean(array, axis=0),
        "std": std,
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


def _read_episode_rows(root: Path) -> list[dict[str, Any]]:
    episode_tables = [pq.read_table(path) for path in sorted((root / "meta" / "episodes").glob("chunk-*/*.parquet"))]
    if not episode_tables:
        raise RuntimeError(f"No episode metadata parquet files found under {root / 'meta' / 'episodes'}")
    table = pa.concat_tables(episode_tables, promote_options="default") if len(episode_tables) > 1 else episode_tables[0]
    return table.to_pylist()


def _load_episode_table(root: Path, info: dict[str, Any], episode_row: dict[str, Any]) -> pa.Table:
    data_path = info.get("data_path", DATA_PATH).format(
        chunk_index=int(episode_row["data/chunk_index"]),
        file_index=int(episode_row["data/file_index"]),
    )
    table = pq.read_table(root / data_path)
    old_episode_index = int(episode_row["episode_index"])
    if "episode_index" in table.column_names:
        mask = pa.compute.equal(table.column("episode_index"), pa.scalar(old_episode_index, type=table["episode_index"].type))
        table = table.filter(mask)
    if table.num_rows == 0:
        raise RuntimeError(f"Episode {old_episode_index} in {root} has no rows in {data_path}")
    return table


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
    source_episodes: list[dict[str, Any]] = []

    for root in roots:
        info = _read_json(root / "meta" / "info.json")
        if info.get("features") != features:
            raise RuntimeError(f"Feature schema mismatch in {root}")
        if int(info.get("fps", fps)) != fps:
            raise RuntimeError(f"FPS mismatch in {root}: {info.get('fps')} != {fps}")

        root_tasks = pq.read_table(root / "meta" / "tasks.parquet").to_pandas()
        root_tasks.index.name = "task"
        for source_episode_row in _read_episode_rows(root):
            episode_index = len(episode_rows)
            table = _load_episode_table(root, info, source_episode_row)
            row_count = table.num_rows

            source_tasks = source_episode_row.get("tasks") or []
            task = str(source_tasks[0]) if source_tasks else ""
            if not task:
                source_task_index = int(np.asarray(table.column("task_index").to_pylist(), dtype=np.int64).reshape(-1)[0])
                task = str(root_tasks[root_tasks["task_index"] == source_task_index].index[0])
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

            episode_row = dict(source_episode_row)
            episode_row.update(
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
            episode_rows.append(episode_row)
            source_episodes.append(
                {
                    "source_root": str(root),
                    "source_episode_index": source_episode_row.get("episode_index"),
                    "merged_episode_index": episode_index,
                    "frames": row_count,
                    "task": task,
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
            "total_episodes": len(episode_rows),
            "total_frames": frame_offset,
            "total_tasks": len(task_texts),
            "fps": fps,
            "splits": {"train": f"0:{len(episode_rows)}"},
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
        "source_episodes": source_episodes,
        "total_episodes": len(episode_rows),
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
