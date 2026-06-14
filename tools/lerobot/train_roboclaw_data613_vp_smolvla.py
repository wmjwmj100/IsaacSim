#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path
from typing import Any


DEFAULT_DATASET_ROOT = "/work/IsaacSim/outputs/lerobot_datasets/roboclaw_data613_vp_30ep"
DEFAULT_DATASET_REPO_ID = "local/roboclaw_data613_vp_30ep"
DEFAULT_BASE_CHECKPOINT = (
    "/work/IsaacSim/outputs/train/"
    "roboclaw_real_20260529_21ep_smolvla_two_cam_20k_cont_from_two_cam5k/"
    "checkpoints/020000/pretrained_model"
)
DEFAULT_OUTPUT_DIR = "/work/IsaacSim/outputs/train/roboclaw_data613_vp_30ep_smolvla_expert"
DEFAULT_JOB_NAME = "roboclaw_data613_vp_30ep_smolvla_expert"


def _clear_bad_proxy_env() -> None:
    for key in ("ALL_PROXY", "all_proxy", "HTTP_PROXY", "http_proxy", "HTTPS_PROXY", "https_proxy"):
        os.environ.pop(key, None)
    os.environ.pop("HF_HUB_DISABLE_XET", None)


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


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _feature_shape(dataset_root: str | Path, feature_key: str) -> tuple[int, ...]:
    info_path = Path(dataset_root).expanduser() / "meta" / "info.json"
    info = _load_json(info_path)
    return tuple(info["features"][feature_key]["shape"])


def _resolve_image_feature_key(dataset_root: str | Path, candidates: tuple[str, ...]) -> str:
    info_path = Path(dataset_root).expanduser() / "meta" / "info.json"
    info = _load_json(info_path)
    features = info["features"]
    for key in candidates:
        if key in features:
            return key
    raise KeyError(f"None of the expected image keys exist in {info_path}: {candidates}")


def _image_policy_shape(dataset_root: str | Path, feature_key: str) -> tuple[int, int, int]:
    shape = _feature_shape(dataset_root, feature_key)
    if len(shape) != 3:
        raise ValueError(f"Expected image feature {feature_key!r} to have HWC shape, got {shape}")
    return (shape[2], shape[0], shape[1])


def _expand_episode_spec(spec: str) -> list[int] | None:
    spec = spec.strip()
    if not spec:
        return None
    if ":" in spec:
        start_s, end_s = spec.split(":", 1)
        return list(range(int(start_s), int(end_s)))
    return [int(item.strip()) for item in spec.split(",") if item.strip()]


def _resolve_resume_config(path: str | Path) -> Path:
    resume_path = Path(path).expanduser()
    if resume_path.is_file():
        if resume_path.name != "train_config.json":
            raise ValueError(f"Expected a train_config.json file for resume, got: {resume_path}")
        return resume_path

    if not resume_path.is_dir():
        raise FileNotFoundError(f"Resume checkpoint path does not exist: {resume_path}")

    candidates = (
        resume_path / "train_config.json",
        resume_path / "pretrained_model" / "train_config.json",
    )
    for candidate in candidates:
        if candidate.exists():
            return candidate

    raise FileNotFoundError(
        "Could not find train_config.json under resume path. Expected either "
        f"{candidates[0]} or {candidates[1]}"
    )


def _ensure_lerobot_resume_arg(config_path: Path) -> None:
    config_arg = f"--config_path={config_path}"
    if not any(arg.startswith("--config_path=") for arg in sys.argv[1:]):
        sys.argv.append(config_arg)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Train SmolVLA System 1/action expert on data613 with boxed top view, side view, "
            "new prompt, and 7D actions."
        )
    )
    parser.add_argument("--dataset-root", default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--dataset-repo-id", default=DEFAULT_DATASET_REPO_ID)
    parser.add_argument(
        "--episodes",
        default="",
        help="Optional episode selection, e.g. '0:30' or '0,1,2'. Empty means all episodes.",
    )
    parser.add_argument("--base-model", default="lerobot/smolvla_base")
    parser.add_argument(
        "--base-checkpoint",
        default=DEFAULT_BASE_CHECKPOINT,
        help="Existing two-camera 7D SmolVLA checkpoint used to initialize this run.",
    )
    parser.add_argument(
        "--resume-from",
        default="",
        help=(
            "Resume an existing LeRobot checkpoint. Accepts either a checkpoint directory, "
            "a pretrained_model directory, or its train_config.json. When set, --base-checkpoint "
            "is ignored and --steps is interpreted as the total target step count."
        ),
    )
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--job-name", default=DEFAULT_JOB_NAME)
    parser.add_argument("--steps", type=int, default=5000)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--log-freq", type=int, default=50)
    parser.add_argument("--save-freq", type=int, default=1000)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    _clear_bad_proxy_env()

    dataset_root = Path(args.dataset_root).expanduser()
    base_checkpoint = Path(args.base_checkpoint).expanduser()
    output_dir = Path(args.output_dir).expanduser()
    resume_config = _resolve_resume_config(args.resume_from) if args.resume_from else None
    episodes = _expand_episode_spec(args.episodes)

    if not (dataset_root / "meta" / "info.json").exists():
        raise FileNotFoundError(f"LeRobot dataset root is missing meta/info.json: {dataset_root}")
    if resume_config is None and not (base_checkpoint / "config.json").exists():
        raise FileNotFoundError(f"Base checkpoint is missing config.json: {base_checkpoint}")
    if args.overwrite and output_dir.exists():
        shutil.rmtree(output_dir)

    # Import SmolVLA before reading pretrained configs so draccus registers the policy type.
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
        dataset_root,
        ("observation.images.up", "observation.images.top", "observation.images.camera1"),
    )
    side_image_key = _resolve_image_feature_key(
        dataset_root,
        ("observation.images.side", "observation.images.wrist", "observation.images.camera2"),
    )
    print(f"Image mapping: {top_image_key} -> observation.images.camera1")
    print(f"Image mapping: {side_image_key} -> observation.images.camera2")

    if resume_config is not None:
        print(f"Resuming training from: {resume_config}")
        _ensure_lerobot_resume_arg(resume_config)
        cfg = TrainPipelineConfig.from_pretrained(resume_config)
        cfg.resume = True
        cfg.output_dir = output_dir
        cfg.job_name = args.job_name
        cfg.steps = args.steps
        cfg.save_freq = args.save_freq
        cfg.log_freq = args.log_freq
        cfg.num_workers = args.num_workers
        cfg.batch_size = args.batch_size
        cfg.eval_freq = 0
        cfg.save_checkpoint = True
        cfg.dataset = DatasetConfig(
            repo_id=args.dataset_repo_id,
            root=str(dataset_root),
            episodes=episodes,
            video_backend="pyav",
            use_imagenet_stats=False,
        )
        cfg.rename_map = {
            top_image_key: "observation.images.camera1",
            side_image_key: "observation.images.camera2",
        }
        cfg.policy.device = args.device
        cfg.policy.use_amp = False
        cfg.policy.push_to_hub = False
        cfg.policy.repo_id = None
        train(cfg)
        return

    policy.pretrained_path = base_checkpoint
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
            shape=_feature_shape(dataset_root, "observation.state"),
        ),
        "observation.images.camera1": PolicyFeature(
            type=FeatureType.VISUAL,
            shape=_image_policy_shape(dataset_root, top_image_key),
        ),
        "observation.images.camera2": PolicyFeature(
            type=FeatureType.VISUAL,
            shape=_image_policy_shape(dataset_root, side_image_key),
        ),
    }
    policy.output_features["action"] = PolicyFeature(
        type=FeatureType.ACTION,
        shape=_feature_shape(dataset_root, "action"),
    )

    cfg = TrainPipelineConfig(
        dataset=DatasetConfig(
            repo_id=args.dataset_repo_id,
            root=str(dataset_root),
            episodes=episodes,
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
            side_image_key: "observation.images.camera2",
        },
    )
    train(cfg)


if __name__ == "__main__":
    main()
