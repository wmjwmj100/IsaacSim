#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch


def _clear_bad_proxy_env() -> None:
    for key in ("ALL_PROXY", "all_proxy", "HTTP_PROXY", "http_proxy", "HTTPS_PROXY", "https_proxy"):
        os.environ.pop(key, None)
    os.environ.pop("HF_HUB_DISABLE_XET", None)


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _resolve_local_hf_snapshot(repo_id: str, required_files: tuple[str, ...]) -> Path | None:
    cache_root = Path(os.environ.get("HF_HUB_CACHE", Path.home() / ".cache" / "huggingface" / "hub"))
    snapshot_root = cache_root / f"models--{repo_id.replace('/', '--')}" / "snapshots"
    if not snapshot_root.is_dir():
        return None
    candidates = sorted(path for path in snapshot_root.iterdir() if path.is_dir())
    for candidate in reversed(candidates):
        if all((candidate / rel_path).exists() for rel_path in required_files):
            return candidate
    return None


def _force_two_camera_inputs(cfg: Any) -> None:
    input_features = getattr(cfg, "input_features", None)
    if not isinstance(input_features, dict):
        return
    if "observation.images.camera2" not in input_features and "observation.images.camera3" in input_features:
        input_features["observation.images.camera2"] = input_features["observation.images.camera3"]
    for key in list(input_features):
        if key == "observation.images.camera3" or key.startswith("observation.images.empty_camera_"):
            input_features.pop(key, None)
    cfg.empty_cameras = 0


def _load_parquet_rows(path: Path) -> list[dict[str, Any]]:
    import pyarrow.parquet as pq

    return pq.read_table(path).to_pylist()


def _read_video_frame(path: Path, frame_index: int) -> Any:
    import av

    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        stream.thread_type = "AUTO"
        average_rate = float(stream.average_rate) if stream.average_rate else 30.0
        pts = int(round((frame_index / average_rate) / float(stream.time_base)))
        try:
            container.seek(pts, stream=stream, backward=True)
        except Exception:
            container.seek(0, stream=stream)
        for frame in container.decode(stream):
            current = int(round(frame.time * average_rate)) if frame.time is not None else int(frame.pts or 0)
            if current >= frame_index:
                array = frame.to_ndarray(format="rgb24")
                return torch.from_numpy(array).permute(2, 0, 1).float() / 255.0
    raise RuntimeError(f"Could not decode frame {frame_index} from {path}")


def _action_chunk(rows: list[dict[str, Any]], start: int, chunk_size: int, action_dim: int) -> np.ndarray:
    chunk: list[list[float]] = []
    episode_index = rows[start]["episode_index"]
    index = start
    while index < len(rows) and len(chunk) < chunk_size and rows[index]["episode_index"] == episode_index:
        chunk.append([float(v) for v in rows[index]["action"][:action_dim]])
        index += 1
    if not chunk:
        raise RuntimeError(f"No action rows at index {start}")
    while len(chunk) < chunk_size:
        chunk.append(chunk[-1])
    return np.asarray(chunk, dtype=np.float64)


def main() -> None:
    _clear_bad_proxy_env()

    parser = argparse.ArgumentParser(description="Fast cached official SmolVLA baseline eval.")
    parser.add_argument("--model-id", default="lerobot/smolvla_base")
    parser.add_argument("--dataset-root", default="outputs/hf_cache/datasets/svla_so100_pickplace")
    parser.add_argument("--episode-indices", default="0,10,20,30,40")
    parser.add_argument("--frame-offsets", default="0,150,300")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--local-files-only", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--output-json", default="outputs/offline_smolvla_eval_official_base/smolvla_base_svla_so100_cached_report.json")
    parser.add_argument("--verbose-timing", action="store_true")
    args = parser.parse_args()

    from lerobot.configs.policies import PreTrainedConfig
    from lerobot.policies.factory import make_pre_post_processors
    from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig as _SmolVLAConfig  # noqa: F401
    from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy

    dataset_root = Path(args.dataset_root).expanduser()
    data_rows = _load_parquet_rows(dataset_root / "data/chunk-000/file-000.parquet")
    episode_rows = _load_parquet_rows(dataset_root / "meta/episodes/chunk-000/file-000.parquet")
    dataset_stats = _load_json(dataset_root / "meta/stats.json")

    local_base_model = _resolve_local_hf_snapshot(
        args.model_id,
        ("config.json", "model.safetensors"),
    )
    local_vlm_model = _resolve_local_hf_snapshot(
        "HuggingFaceTB/SmolVLM2-500M-Video-Instruct",
        ("config.json", "model.safetensors", "processor_config.json"),
    )
    model_source = str(local_base_model or args.model_id)

    device = torch.device(args.device)
    cfg_t0 = time.perf_counter()
    cfg = PreTrainedConfig.from_pretrained(model_source, local_files_only=args.local_files_only)
    if args.verbose_timing:
        print(json.dumps({"stage": "config_load", "seconds": time.perf_counter() - cfg_t0}, ensure_ascii=False), flush=True)
    if local_vlm_model:
        cfg.vlm_model_name = str(local_vlm_model)
    _force_two_camera_inputs(cfg)
    chunk_size = int(getattr(cfg, "chunk_size", 50))
    action_dim = int(cfg.output_features["action"].shape[0])

    policy_t0 = time.perf_counter()
    policy = SmolVLAPolicy.from_pretrained(
        model_source,
        config=cfg,
        local_files_only=args.local_files_only,
    ).to(device).eval()
    if args.verbose_timing:
        print(json.dumps({"stage": "policy_load", "seconds": time.perf_counter() - policy_t0}, ensure_ascii=False), flush=True)
    proc_t0 = time.perf_counter()
    preprocess, postprocess = make_pre_post_processors(
        cfg,
        model_source,
        dataset_stats=dataset_stats,
        preprocessor_overrides={"device_processor": {"device": str(device)}},
    )
    if args.verbose_timing:
        print(json.dumps({"stage": "processors_load", "seconds": time.perf_counter() - proc_t0}, ensure_ascii=False), flush=True)

    top_video = dataset_root / "videos/observation.images.top/chunk-000/file-000.mp4"
    wrist_video = dataset_root / "videos/observation.images.wrist/chunk-000/file-000.mp4"
    episode_indices = [int(item) for item in args.episode_indices.split(",") if item.strip()]
    frame_offsets = [int(item) for item in args.frame_offsets.split(",") if item.strip()]

    records: list[dict[str, Any]] = []
    for episode_index in episode_indices:
        episode = episode_rows[episode_index]
        from_index = int(episode["dataset_from_index"])
        to_index = int(episode["dataset_to_index"])
        episode_length = to_index - from_index
        task = episode["tasks"][0] if episode.get("tasks") else "Pick up the cube and place it in the box."
        top_from_timestamp = float(episode["videos/observation.images.top/from_timestamp"])
        wrist_from_timestamp = float(episode["videos/observation.images.wrist/from_timestamp"])
        for offset in frame_offsets:
            if offset < 0 or offset >= episode_length:
                continue
            chunk_t0 = time.perf_counter()
            dataset_index = from_index + offset
            row = data_rows[dataset_index]
            timestamp = float(row["timestamp"])
            top_global_frame = int(round((top_from_timestamp + timestamp) * 30.0))
            wrist_global_frame = int(round((wrist_from_timestamp + timestamp) * 30.0))
            if args.verbose_timing:
                print(
                    json.dumps(
                        {
                            "stage": "start_chunk",
                            "episode_index": episode_index,
                            "frame_offset": offset,
                            "top_global_frame": top_global_frame,
                            "wrist_global_frame": wrist_global_frame,
                        },
                        ensure_ascii=False,
                    ),
                    flush=True,
                )
            image_t0 = time.perf_counter()
            top_image = _read_video_frame(top_video, top_global_frame)
            wrist_image = _read_video_frame(wrist_video, wrist_global_frame)
            if args.verbose_timing:
                print(json.dumps({"stage": "images", "seconds": time.perf_counter() - image_t0}, ensure_ascii=False), flush=True)
            frame = {
                "observation.state": torch.tensor(row["observation.state"], dtype=torch.float32),
                "observation.images.camera1": top_image,
                "observation.images.camera2": wrist_image,
                "task": task,
            }
            gt_chunk = _action_chunk(data_rows, dataset_index, chunk_size, action_dim)
            preprocess_t0 = time.perf_counter()
            processed = preprocess(frame)
            if args.verbose_timing:
                print(json.dumps({"stage": "preprocess", "seconds": time.perf_counter() - preprocess_t0}, ensure_ascii=False), flush=True)
            policy.reset()
            forward_t0 = time.perf_counter()
            with torch.inference_mode():
                pred_chunk = postprocess(policy.predict_action_chunk(processed))
            if args.verbose_timing:
                print(json.dumps({"stage": "forward_postprocess", "seconds": time.perf_counter() - forward_t0}, ensure_ascii=False), flush=True)
            pred_array = pred_chunk.detach().cpu().numpy()
            if pred_array.ndim == 3 and pred_array.shape[0] == 1:
                pred_array = pred_array[0]
            pred_array = pred_array[: chunk_size, :action_dim].astype(np.float64)
            abs_error = np.abs(pred_array - gt_chunk)
            records.append(
                {
                    "episode_index": episode_index,
                    "frame_offset": offset,
                    "dataset_index": dataset_index,
                    "valid_steps": int(gt_chunk.shape[0]),
                    "mean_l1_error": float(abs_error.mean()),
                    "first_step_mean_l1_error": float(abs_error[0].mean()),
                    "last_step_mean_l1_error": float(abs_error[-1].mean()),
                    "per_dim_mean_abs_error": abs_error.mean(axis=0).tolist(),
                    "per_step_mean_abs_error": abs_error.mean(axis=1).tolist(),
                    "predicted_first_action": pred_array[0].tolist(),
                    "ground_truth_first_action": gt_chunk[0].tolist(),
                    "predicted_last_action": pred_array[-1].tolist(),
                    "ground_truth_last_action": gt_chunk[-1].tolist(),
                    "wall_time_s": float(time.perf_counter() - chunk_t0),
                }
            )
            print(
                json.dumps(
                    {
                        "episode_index": episode_index,
                        "frame_offset": offset,
                        "mean_l1_error": records[-1]["mean_l1_error"],
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )

    all_errors = [value for record in records for value in record["per_step_mean_abs_error"]]
    payload = {
        "success": True,
        "model_id": model_source,
        "requested_model_id": args.model_id,
        "dataset_root": str(dataset_root),
        "episode_indices": episode_indices,
        "frame_offsets": frame_offsets,
        "device": str(device),
        "chunk_size": chunk_size,
        "num_chunks": len(records),
        "mean_step_l1_error": float(np.mean(all_errors)) if all_errors else None,
        "max_step_l1_error": float(np.max(all_errors)) if all_errors else None,
        "records": records,
    }
    output_path = Path(args.output_json).expanduser()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in payload.items() if key != "records"}, ensure_ascii=False, indent=2))
    print(f"wrote {output_path}")


if __name__ == "__main__":
    main()
