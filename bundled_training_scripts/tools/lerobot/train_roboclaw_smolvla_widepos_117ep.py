#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any


def _clear_bad_proxy_env() -> None:
    import os

    for key in ("ALL_PROXY", "all_proxy", "HTTP_PROXY", "http_proxy", "HTTPS_PROXY", "https_proxy"):
        os.environ.pop(key, None)
    os.environ.pop("HF_HUB_DISABLE_XET", None)


def _resolve_local_hf_snapshot(repo_id: str, required_files: tuple[str, ...]) -> Path | None:
    import os

    cache_root = Path(os.environ.get("HF_HUB_CACHE", Path.home() / ".cache" / "huggingface" / "hub"))
    snapshot_root = cache_root / f"models--{repo_id.replace('/', '--')}" / "snapshots"
    if not snapshot_root.is_dir():
        return None

    candidates = sorted(path for path in snapshot_root.iterdir() if path.is_dir())
    for candidate in reversed(candidates):
        if all((candidate / rel_path).exists() for rel_path in required_files):
            return candidate
    return None


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _feature_shape(dataset_root: str, feature_key: str) -> tuple[int, ...]:
    info_path = Path(dataset_root).expanduser() / "meta" / "info.json"
    info = _load_json(info_path)
    return tuple(info["features"][feature_key]["shape"])


def _resolve_image_feature_key(dataset_root: str, candidates: tuple[str, ...]) -> str:
    info_path = Path(dataset_root).expanduser() / "meta" / "info.json"
    info = _load_json(info_path)
    features = info["features"]
    for key in candidates:
        if key in features:
            return key
    raise KeyError(f"None of the expected image keys exist in {info_path}: {candidates}")


def _image_policy_shape(dataset_root: str, feature_key: str) -> tuple[int, int, int]:
    shape = _feature_shape(dataset_root, feature_key)
    if len(shape) != 3:
        raise ValueError(f"Expected image feature {feature_key!r} to have HWC shape, got {shape}")
    return (shape[2], shape[0], shape[1])


def _list_episode_indices(dataset_root: Path) -> list[int]:
    import pyarrow.parquet as pq

    episodes_path = dataset_root / "meta" / "episodes" / "chunk-000" / "file-000.parquet"
    table = pq.read_table(episodes_path)
    return [int(item["episode_index"]) for item in table.to_pylist()]


def _write_split_plan(path: Path, train_episodes: list[int], val_episodes: list[int]) -> None:
    payload = {
        "train_episodes": train_episodes,
        "val_episodes": val_episodes,
        "train_count": len(train_episodes),
        "val_count": len(val_episodes),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Fine-tune SmolVLA on the Roboclaw wide-position dataset.")
    parser.add_argument(
        "--dataset-root",
        default="outputs/lerobot_datasets/roboclaw_sim_teacher_widepos_117ep",
    )
    parser.add_argument(
        "--dataset-repo-id",
        default="local/roboclaw_sim_teacher_widepos_117ep",
    )
    parser.add_argument(
        "--train-episodes",
        default="0:110",
        help="Train episode selection. Use '0:110' for the first 110 episodes.",
    )
    parser.add_argument(
        "--val-episodes",
        default="110:117",
        help="Hold-out episode selection. Use '110:117' for the last 7 episodes.",
    )
    parser.add_argument("--base-model", default="lerobot/smolvla_base")
    parser.add_argument(
        "--base-checkpoint",
        default="outputs/train/roboclaw_sim_teacher_widepos_clean_89ep_smolvla_20k_cont_from5k/checkpoints/015000/pretrained_model",
        help="Checkpoint used as initialization. This starts a new run from these weights.",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/train/roboclaw_sim_teacher_widepos_117ep_smolvla_20k_cont_from15k",
    )
    parser.add_argument("--job-name", default="roboclaw_sim_teacher_widepos_117ep_smolvla_20k_cont_from15k")
    parser.add_argument("--steps", type=int, default=5000)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--log-freq", type=int, default=100)
    parser.add_argument("--save-freq", type=int, default=5000)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    _clear_bad_proxy_env()

    dataset_root = Path(args.dataset_root).expanduser()
    train_episodes = _expand_episode_spec(args.train_episodes)
    val_episodes = _expand_episode_spec(args.val_episodes)
    output_dir = Path(args.output_dir).expanduser()
    if args.overwrite and output_dir.exists():
        shutil.rmtree(output_dir)
    split_plan_path = output_dir.parent / f"{output_dir.name}_split_plan.json"
    _write_split_plan(split_plan_path, train_episodes, val_episodes)

    # Import after env cleanup to avoid HF proxy issues.
    from lerobot.configs.default import DatasetConfig
    from lerobot.configs.policies import PreTrainedConfig
    from lerobot.configs.train import TrainPipelineConfig
    from lerobot.configs.types import FeatureType, PolicyFeature
    from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig as _SmolVLAConfig  # noqa: F401
    from lerobot.scripts.lerobot_train import train

    local_base_model = _resolve_local_hf_snapshot(args.base_model, ("config.json", "model.safetensors"))
    local_vlm_model = _resolve_local_hf_snapshot(
        "HuggingFaceTB/SmolVLM2-500M-Video-Instruct",
        ("config.json", "model.safetensors", "processor_config.json"),
    )
    if local_base_model:
        print(f"Using local SmolVLA base snapshot: {local_base_model}")
    if local_vlm_model:
        print(f"Using local VLM snapshot: {local_vlm_model}")

    policy = PreTrainedConfig.from_pretrained(str(local_base_model or args.base_model), local_files_only=False)
    top_image_key = _resolve_image_feature_key(
        args.dataset_root,
        ("observation.images.up", "observation.images.top"),
    )
    wrist_image_key = _resolve_image_feature_key(
        args.dataset_root,
        ("observation.images.side", "observation.images.wrist"),
    )
    print(f"Image mapping: OBS_IMAGE_1={top_image_key} -> camera1, OBS_IMAGE_2={wrist_image_key} -> camera2")
    policy.pretrained_path = Path(args.base_checkpoint).expanduser()
    policy.device = args.device
    policy.use_amp = False
    policy.push_to_hub = False
    policy.repo_id = None
    policy.n_action_steps = 50
    policy.chunk_size = 50
    policy.empty_cameras = 0
    policy.train_expert_only = True
    policy.freeze_vision_encoder = True
    policy.train_state_proj = True
    if local_vlm_model:
        policy.vlm_model_name = str(local_vlm_model)
    policy.input_features = {
        "observation.state": PolicyFeature(
            type=FeatureType.STATE,
            shape=_feature_shape(args.dataset_root, "observation.state"),
        ),
        "observation.images.camera1": PolicyFeature(
            type=FeatureType.VISUAL,
            shape=_image_policy_shape(args.dataset_root, top_image_key),
        ),
        "observation.images.camera2": PolicyFeature(
            type=FeatureType.VISUAL,
            shape=_image_policy_shape(args.dataset_root, wrist_image_key),
        ),
    }
    policy.output_features["action"] = PolicyFeature(
        type=FeatureType.ACTION,
        shape=_feature_shape(args.dataset_root, "action"),
    )

    cfg = TrainPipelineConfig(
        dataset=DatasetConfig(
            repo_id=args.dataset_repo_id,
            root=args.dataset_root,
            episodes=train_episodes,
            video_backend="pyav",
            use_imagenet_stats=False,
        ),
        output_dir=output_dir,
        job_name=args.job_name,
        seed=1000,
        num_workers=args.num_workers,
        batch_size=args.batch_size,
        steps=args.steps,
        eval_freq=0,
        log_freq=args.log_freq,
        save_checkpoint=True,
        save_freq=args.save_freq,
        use_policy_training_preset=True,
        policy=policy,
        rename_map={
            top_image_key: "observation.images.camera1",
            wrist_image_key: "observation.images.camera2",
        },
    )
    train(cfg)
    if output_dir.exists():
        shutil.copy2(split_plan_path, output_dir / "split_plan.json")


def _expand_episode_spec(spec: str) -> list[int]:
    spec = spec.strip()
    if not spec:
        return []
    if ":" in spec:
        start_s, end_s = spec.split(":", 1)
        return list(range(int(start_s), int(end_s)))
    return [int(item.strip()) for item in spec.split(",") if item.strip()]


if __name__ == "__main__":
    main()
