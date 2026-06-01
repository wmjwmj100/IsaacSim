from __future__ import annotations

import argparse
import json
import math
import os
import select
import struct
import subprocess
import sys
import time
import zlib
from pathlib import Path

from isaacsim import SimulationApp


REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_USD_PATH = "/home/wmj/Documents/roboclaw.usd"
DEFAULT_CHECKPOINT = (
    Path.home()
    / ".cache/huggingface/hub/models--lerobot--smolvla_base/snapshots/c83c3163b8ca9b7e67c509fffd9121e66cb96205"
)
DEFAULT_SMOLVLA_PYTHON = REPO_ROOT / ".venv-lerobot/bin/python"
DEFAULT_DATASET_ROOT = REPO_ROOT / "outputs/lerobot_datasets/panthera_ht_merged"
DEFAULT_VLM_MODEL_PATH = Path(
    "/home/wmj/.cache/huggingface/hub/models--HuggingFaceTB--SmolVLM2-500M-Video-Instruct/"
    "snapshots/7b375e1b73b11138ff12fe22c8f2822d8fe03467"
)
DEFAULT_OUTPUT_DIR = REPO_ROOT / "outputs/roboclaw_smolvla_rollout"
DEFAULT_TARGET_PRIM_PATH = "/World/Table/TargetCube"
DEFAULT_REPLAY_JSON = REPO_ROOT / "outputs/roboclaw_dataset_replay/panthera_episode_000000_replay.json"

CAMERA_PATHS = {
    "overhead": "/World/RealSenseD435i/DepthCamera",
    "left_wrist": "/Panthera_HT_FrontLeft/link6/WristRGBCamera",
    "right_wrist": "/Panthera_HT_FrontRight/link6/WristRGBCamera",
}
ARTICULATION_PATHS = {
    "front_left": "/Panthera_HT_FrontLeft/root_joint",
    "front_right": "/Panthera_HT_FrontRight/root_joint",
}
END_EFFECTOR_PATHS = {
    "front_left": "/Panthera_HT_FrontLeft/link6",
    "front_right": "/Panthera_HT_FrontRight/link6",
}


def env_flag(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def env_int(name: str, default: int) -> int:
    value = os.getenv(name)
    return int(value) if value else default


def env_float(name: str, default: float) -> float:
    value = os.getenv(name)
    return float(value) if value else default


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the fine-tuned SmolVLA checkpoint against the Roboclaw USD scene."
    )
    parser.add_argument("--usd-path", default=os.getenv("ROBOCLAW_USD_PATH", DEFAULT_USD_PATH))
    parser.add_argument("--headless", action="store_true", default=env_flag("ROBOCLAW_HEADLESS", False))
    parser.add_argument("--renderer", default=os.getenv("ROBOCLAW_RENDERER", "RaytracedLighting"))
    parser.add_argument("--max-frames", type=int, default=env_int("ROBOCLAW_MAX_FRAMES", 60))
    parser.add_argument("--warmup-frames", type=int, default=env_int("ROBOCLAW_WARMUP_FRAMES", 24))
    parser.add_argument("--infer-interval", type=int, default=env_int("ROBOCLAW_INFER_INTERVAL", 10))
    parser.add_argument(
        "--frame-file-ring-size",
        type=int,
        default=env_int("ROBOCLAW_FRAME_FILE_RING_SIZE", 0),
        help="If >0, reuse frame image/result filenames modulo this size to avoid unlimited disk growth.",
    )
    parser.add_argument(
        "--max-records-in-memory",
        type=int,
        default=env_int("ROBOCLAW_MAX_RECORDS_IN_MEMORY", 0),
        help="If >0, keep only the latest N records in memory. Useful for unbounded GUI rollouts.",
    )
    parser.add_argument(
        "--keep-open",
        action="store_true",
        default=env_flag("ROBOCLAW_KEEP_OPEN", False),
        help="Keep Isaac Sim open after rollout so the scene remains visible.",
    )
    parser.add_argument(
        "--use-persistent-helper",
        action="store_true",
        default=env_flag("ROBOCLAW_USE_PERSISTENT_HELPER", True),
        help="Load SmolVLA once and send per-frame requests over stdin/stdout.",
    )
    parser.add_argument(
        "--active-arm-label",
        choices=("front_left", "front_right"),
        default=os.getenv("ROBOCLAW_ACTIVE_ARM_LABEL", "front_left"),
    )
    parser.add_argument(
        "--control-mode",
        choices=("log-only", "single-arm"),
        default=os.getenv("ROBOCLAW_CONTROL_MODE", "log-only"),
        help="log-only only records policy actions; single-arm sends policy targets to the selected arm.",
    )
    parser.add_argument(
        "--policy-align-initial-state-replay-json",
        default=os.getenv("ROBOCLAW_POLICY_ALIGN_INITIAL_STATE_REPLAY_JSON", ""),
        help="For SmolVLA rollout, initialize the active arm from state7 in this replay JSON before inference starts.",
    )
    parser.add_argument(
        "--policy-align-initial-state-index",
        type=int,
        default=env_int("ROBOCLAW_POLICY_ALIGN_INITIAL_STATE_INDEX", 0),
        help="Record index to use with --policy-align-initial-state-replay-json.",
    )
    parser.add_argument(
        "--policy-align-settle-frames",
        type=int,
        default=env_int("ROBOCLAW_POLICY_ALIGN_SETTLE_FRAMES", 3),
        help="Physics/render frames to step after policy initial-state alignment.",
    )
    parser.add_argument("--smolvla-model-id", default=os.getenv("ROBOCLAW_SMOLVLA_MODEL_ID", str(DEFAULT_CHECKPOINT)))
    parser.add_argument("--smolvla-python", default=os.getenv("ROBOCLAW_SMOLVLA_PYTHON", str(DEFAULT_SMOLVLA_PYTHON)))
    parser.add_argument("--smolvla-device", default=os.getenv("ROBOCLAW_SMOLVLA_DEVICE", "cuda"))
    parser.add_argument("--dataset-root", default=os.getenv("ROBOCLAW_DATASET_ROOT", str(DEFAULT_DATASET_ROOT)))
    parser.add_argument("--vlm-model-path", default=os.getenv("ROBOCLAW_VLM_MODEL_PATH", str(DEFAULT_VLM_MODEL_PATH)))
    parser.add_argument("--task-text", default=os.getenv("ROBOCLAW_TASK_TEXT", "push the object on the table"))
    parser.add_argument(
        "--gripper-clip-lower",
        type=float,
        default=None,
        help="Lower bound for gripper execution. Defaults to dataset stats action gripper min.",
    )
    parser.add_argument(
        "--gripper-clip-upper",
        type=float,
        default=None,
        help="Upper bound for gripper execution. Defaults to dataset stats action gripper max.",
    )
    parser.add_argument(
        "--six-dof-action-layout",
        choices=("so100_5j_gripper", "joints_only"),
        default=os.getenv("ROBOCLAW_SIX_DOF_ACTION_LAYOUT", "so100_5j_gripper"),
        help=(
            "How to execute a 6D policy action. so100_5j_gripper maps action[0:5] to joint1..joint5, "
            "locks one robot joint, and maps action[5] to the gripper."
        ),
    )
    parser.add_argument(
        "--policy-joint-unit-mode",
        choices=("auto", "radians", "degrees"),
        default=os.getenv("ROBOCLAW_POLICY_JOINT_UNIT_MODE", "auto"),
        help="Units used by the policy action/state stats. auto infers degrees when dataset stats exceed normal radian ranges.",
    )
    parser.add_argument(
        "--locked-joint-name",
        default=os.getenv("ROBOCLAW_LOCKED_JOINT_NAME", "joint6"),
        help="Robot DOF to hold fixed when executing a 6D so100_5j_gripper action.",
    )
    parser.add_argument(
        "--locked-joint-value",
        type=float,
        default=env_float("ROBOCLAW_LOCKED_JOINT_VALUE", float("nan")),
        help="Fixed value for --locked-joint-name. Defaults to that joint's current startup value.",
    )
    parser.add_argument("--output-dir", default=os.getenv("ROBOCLAW_OUTPUT_DIR", str(DEFAULT_OUTPUT_DIR)))
    parser.add_argument("--target-prim-path", default=os.getenv("ROBOCLAW_TARGET_PRIM_PATH", DEFAULT_TARGET_PRIM_PATH))
    parser.add_argument("--target-initial-x", type=float, default=env_float("ROBOCLAW_TARGET_INITIAL_X", float("nan")))
    parser.add_argument("--target-initial-y", type=float, default=env_float("ROBOCLAW_TARGET_INITIAL_Y", float("nan")))
    parser.add_argument("--target-initial-z", type=float, default=env_float("ROBOCLAW_TARGET_INITIAL_Z", float("nan")))
    parser.add_argument(
        "--replay-json",
        default=os.getenv("ROBOCLAW_REPLAY_JSON", ""),
        help="LeRobot 7D replay JSON exported by tools/lerobot/export_lerobot_actions.py. If set, SmolVLA is skipped.",
    )
    parser.add_argument(
        "--replay-source",
        choices=("state", "action"),
        default=os.getenv("ROBOCLAW_REPLAY_SOURCE", "state"),
        help="Replay observation.state for calibration checks, or action for open-loop command checks.",
    )
    parser.add_argument(
        "--replay-apply-mode",
        choices=("teleport", "target"),
        default=os.getenv("ROBOCLAW_REPLAY_APPLY_MODE", "teleport"),
        help="teleport sets joint positions exactly; target sends joint position targets to the articulation controller.",
    )
    parser.add_argument(
        "--replay-start-index",
        type=int,
        default=env_int("ROBOCLAW_REPLAY_START_INDEX", 0),
        help="Start index inside the exported replay records.",
    )
    parser.add_argument(
        "--replay-stride",
        type=int,
        default=env_int("ROBOCLAW_REPLAY_STRIDE", 1),
        help="Record stride when replaying the exported trajectory.",
    )
    parser.add_argument(
        "--replay-hold-last",
        action=argparse.BooleanOptionalAction,
        default=env_flag("ROBOCLAW_REPLAY_HOLD_LAST", True),
        help="Hold the final replay sample when max-frames exceeds the replay length.",
    )
    parser.add_argument(
        "--replay-loop",
        action=argparse.BooleanOptionalAction,
        default=env_flag("ROBOCLAW_REPLAY_LOOP", False),
        help="Loop the replay trajectory indefinitely. Useful for a visible GUI check.",
    )
    parser.add_argument(
        "--replay-align-initial-state",
        action=argparse.BooleanOptionalAction,
        default=env_flag("ROBOCLAW_REPLAY_ALIGN_INITIAL_STATE", True),
        help="Initialize the active arm from the first replay state7 before replay starts.",
    )
    parser.add_argument(
        "--replay-log-interval",
        type=int,
        default=env_int("ROBOCLAW_REPLAY_LOG_INTERVAL", 30),
        help="Print replay diagnostics every N frames.",
    )
    parser.add_argument(
        "--success-displacement-threshold",
        type=float,
        default=env_float("ROBOCLAW_SUCCESS_DISPLACEMENT_THRESHOLD", 0.05),
        help="XY displacement in meters required to count the target object as moved.",
    )
    parser.add_argument("--overhead-camera-path", default=os.getenv("ROBOCLAW_OVERHEAD_CAMERA_PATH", CAMERA_PATHS["overhead"]))
    parser.add_argument(
        "--left-wrist-camera-path", default=os.getenv("ROBOCLAW_LEFT_WRIST_CAMERA_PATH", CAMERA_PATHS["left_wrist"])
    )
    parser.add_argument(
        "--right-wrist-camera-path", default=os.getenv("ROBOCLAW_RIGHT_WRIST_CAMERA_PATH", CAMERA_PATHS["right_wrist"])
    )
    parser.add_argument(
        "--left-articulation-path",
        default=os.getenv("ROBOCLAW_LEFT_ARTICULATION_PATH", ARTICULATION_PATHS["front_left"]),
    )
    parser.add_argument(
        "--right-articulation-path",
        default=os.getenv("ROBOCLAW_RIGHT_ARTICULATION_PATH", ARTICULATION_PATHS["front_right"]),
    )
    parser.add_argument(
        "--end-effector-prim-path",
        default=os.getenv("ROBOCLAW_END_EFFECTOR_PRIM_PATH", ""),
        help="Prim used for distance-to-target diagnostics. Defaults to the active arm link6.",
    )
    return parser.parse_args()


args = parse_args()
simulation_app = SimulationApp({"headless": args.headless, "renderer": args.renderer})

import carb
import numpy as np
import omni.timeline
from isaacsim.core.api import World
from isaacsim.core.prims import Articulation, SingleRigidPrim
from isaacsim.core.utils.prims import is_prim_path_valid
from isaacsim.core.utils.stage import get_current_stage, is_stage_loading, open_stage
from isaacsim.core.utils.viewports import set_active_viewport_camera, set_camera_view
from isaacsim.core.utils.xforms import get_world_pose
from isaacsim.sensors.camera import Camera
from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics, UsdShade


def log(message: str) -> None:
    print(message, flush=True)
    carb.log_info(message)


def json_ready(value):
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


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(json_ready(payload), ensure_ascii=False, indent=2, sort_keys=True) + "\n")


def write_png_rgb(path: Path, rgb: np.ndarray) -> None:
    image = np.asarray(rgb)
    if image.dtype != np.uint8:
        image = np.clip(image, 0, 255).astype(np.uint8)
    if image.ndim != 3 or image.shape[2] < 3:
        raise RuntimeError(f"Expected HxWx3 RGB image for {path}, got shape {image.shape}")
    image = image[:, :, :3]
    height, width = image.shape[:2]

    def png_chunk(chunk_type: bytes, chunk_data: bytes) -> bytes:
        return (
            struct.pack(">I", len(chunk_data))
            + chunk_type
            + chunk_data
            + struct.pack(">I", zlib.crc32(chunk_type + chunk_data) & 0xFFFFFFFF)
        )

    raw_rows = b"".join(b"\x00" + image[row].tobytes() for row in range(height))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(
        b"\x89PNG\r\n\x1a\n"
        + png_chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0))
        + png_chunk(b"IDAT", zlib.compress(raw_rows, level=6))
        + png_chunk(b"IEND", b"")
    )


def camera_rgb(camera_name: str, camera: Camera) -> np.ndarray:
    rgb = camera.get_rgb()
    if rgb is None or getattr(rgb, "size", 0) == 0:
        raise RuntimeError(f"camera {camera_name} returned no RGB data")
    array = np.asarray(rgb)
    if array.ndim != 3 or array.shape[2] < 3:
        raise RuntimeError(f"camera {camera_name} returned unexpected shape {array.shape}")
    return array[:, :, :3]


def active_wrist_name(active_arm_label: str) -> str:
    return "left_wrist" if active_arm_label == "front_left" else "right_wrist"


def active_articulation_path(parsed_args: argparse.Namespace) -> str:
    if parsed_args.active_arm_label == "front_left":
        return parsed_args.left_articulation_path
    return parsed_args.right_articulation_path


def active_end_effector_path(parsed_args: argparse.Namespace) -> str:
    if parsed_args.end_effector_prim_path:
        return parsed_args.end_effector_prim_path
    return END_EFFECTOR_PATHS[parsed_args.active_arm_label]


def camera_paths(parsed_args: argparse.Namespace) -> dict[str, str]:
    return {
        "overhead": parsed_args.overhead_camera_path,
        "left_wrist": parsed_args.left_wrist_camera_path,
        "right_wrist": parsed_args.right_wrist_camera_path,
    }


def actual_joint_positions_by_name(articulation: Articulation) -> dict[str, float]:
    positions = np.asarray(articulation.get_joint_positions(), dtype=np.float32).reshape(-1)
    return {
        dof_name: float(positions[dof_index])
        for dof_index, dof_name in enumerate(articulation.dof_names[: len(positions)])
    }


def state7(articulation: Articulation) -> list[float]:
    positions = actual_joint_positions_by_name(articulation)
    finger_values = [positions.get(name) for name in ("L_finger_joint", "R_finger_joint", "L_finger", "R_finger")]
    valid_fingers = [float(value) for value in finger_values if value is not None]
    gripper = float(sum(valid_fingers) / len(valid_fingers)) if valid_fingers else 0.0
    return [float(positions.get(f"joint{joint_index}", 0.0)) for joint_index in range(1, 7)] + [gripper]


def action_target_for_dofs(
    dof_names: list[str],
    action: list[float],
    gripper_clip_bounds: tuple[float, float],
    six_dof_action_layout: str = "so100_5j_gripper",
    locked_joint_name: str = "joint6",
    locked_joint_value: float = 0.0,
) -> np.ndarray:
    if len(action) < 6:
        raise RuntimeError(f"SmolVLA action is too short: {action}")
    gripper_lower, gripper_upper = gripper_clip_bounds
    if len(action) == 6 and six_dof_action_layout == "so100_5j_gripper":
        gripper = float(np.clip(action[5], gripper_lower, gripper_upper))
        targets = {
            "joint1": float(action[0]),
            "joint2": float(action[1]),
            "joint3": float(action[2]),
            "joint4": float(action[3]),
            "joint5": float(action[4]),
            locked_joint_name: float(locked_joint_value),
            "L_finger": gripper,
            "R_finger": gripper,
            "L_finger_joint": gripper,
            "R_finger_joint": gripper,
        }
    else:
        targets = {
            "joint1": float(action[0]),
            "joint2": float(action[1]),
            "joint3": float(action[2]),
            "joint4": float(action[3]),
            "joint5": float(action[4]),
            "joint6": float(action[5]),
        }
    if len(action) >= 7:
        gripper = float(np.clip(action[6], gripper_lower, gripper_upper))
        targets.update(
            {
                "L_finger": gripper,
                "R_finger": gripper,
                "L_finger_joint": gripper,
                "R_finger_joint": gripper,
            }
        )
    pose = np.zeros((1, len(dof_names)), dtype=np.float32)
    for dof_index, dof_name in enumerate(dof_names):
        pose[0, dof_index] = targets.get(dof_name, 0.0)
    return pose


def action_mapping_summary(
    action: list[float],
    six_dof_action_layout: str,
    locked_joint_name: str,
    locked_joint_value: float,
) -> dict[str, object]:
    if len(action) == 6 and six_dof_action_layout == "so100_5j_gripper":
        return {
            "action_dim": 6,
            "layout": six_dof_action_layout,
            "arm_action_indices": {
                "joint1": 0,
                "joint2": 1,
                "joint3": 2,
                "joint4": 3,
                "joint5": 4,
            },
            "locked_joint_name": locked_joint_name,
            "locked_joint_value": float(locked_joint_value),
            "gripper_action_index": 5,
        }
    if len(action) == 6:
        return {
            "action_dim": 6,
            "layout": six_dof_action_layout,
            "arm_action_indices": {
                "joint1": 0,
                "joint2": 1,
                "joint3": 2,
                "joint4": 3,
                "joint5": 4,
                "joint6": 5,
            },
            "gripper_action_index": None,
        }
    return {
        "action_dim": len(action),
        "layout": "7d_joint1_to_joint6_plus_gripper",
        "arm_action_indices": {
            "joint1": 0,
            "joint2": 1,
            "joint3": 2,
            "joint4": 3,
            "joint5": 4,
            "joint6": 5,
        },
        "gripper_action_index": 6 if len(action) >= 7 else None,
    }


def load_dataset_stats_json(dataset_root: str | Path) -> dict[str, object] | None:
    root = Path(dataset_root).expanduser()
    if not root.is_absolute():
        root = REPO_ROOT / root
    stats_path = root / "meta" / "stats.json"
    if not stats_path.exists():
        return None
    try:
        return json.loads(stats_path.read_text())
    except Exception as exc:
        log(f"[Roboclaw][WARN] Failed to load dataset stats from {stats_path}: {exc}")
        return None


def resolve_policy_joint_unit_mode(parsed_args: argparse.Namespace) -> tuple[str, str]:
    requested_mode = parsed_args.policy_joint_unit_mode
    if requested_mode != "auto":
        return requested_mode, "explicit"
    stats = load_dataset_stats_json(parsed_args.dataset_root)
    max_abs = 0.0
    if isinstance(stats, dict):
        for key in ("observation.state", "action"):
            feature_stats = stats.get(key)
            if not isinstance(feature_stats, dict):
                continue
            for value in feature_stats.values():
                if value is None:
                    continue
                array = np.asarray(value, dtype=np.float32)
                if array.size:
                    max_abs = max(max_abs, float(np.abs(array).max()))
    return ("degrees", "dataset_stats_auto") if max_abs > 10.0 else ("radians", "dataset_stats_auto")


def action_from_model_units(action: list[float], policy_joint_unit_mode: str) -> list[float]:
    action_array = np.asarray(action, dtype=np.float32)
    if policy_joint_unit_mode == "degrees":
        action_array = np.deg2rad(action_array)
    return [float(value) for value in action_array.tolist()]


def execution_gripper_clip_bounds(
    gripper_clip_bounds: tuple[float, float],
    gripper_clip_source: str,
    policy_joint_unit_mode: str,
) -> tuple[tuple[float, float], str]:
    if gripper_clip_source == "dataset_stats" and policy_joint_unit_mode == "degrees":
        bounds = np.deg2rad(np.asarray(gripper_clip_bounds, dtype=np.float32))
        return (float(bounds[0]), float(bounds[1])), "radians_from_dataset_degrees"
    return gripper_clip_bounds, "execution_units"


def resolve_locked_joint_value(parsed_args: argparse.Namespace, articulation: Articulation) -> tuple[float, str]:
    explicit_value = float(parsed_args.locked_joint_value)
    if math.isfinite(explicit_value):
        return explicit_value, "explicit"
    positions = actual_joint_positions_by_name(articulation)
    if parsed_args.locked_joint_name in positions:
        return float(positions[parsed_args.locked_joint_name]), "current_joint_position"
    return 0.0, "fallback_zero_missing_joint"


def joint_abs_error_for_target(articulation: Articulation, target: np.ndarray) -> dict[str, object]:
    actual = np.asarray(articulation.get_joint_positions(), dtype=np.float64).reshape(-1)
    expected = np.asarray(target, dtype=np.float64).reshape(-1)
    count = min(len(actual), len(expected), len(articulation.dof_names))
    if count <= 0:
        return {
            "max_abs_error": None,
            "mean_abs_error": None,
            "by_name": {},
        }
    delta = np.abs(actual[:count] - expected[:count])
    return {
        "max_abs_error": float(delta.max()),
        "mean_abs_error": float(delta.mean()),
        "by_name": {
            articulation.dof_names[index]: float(delta[index])
            for index in range(count)
        },
    }


def _validate_replay_vector(record: dict[str, object], key: str, record_index: int) -> list[float]:
    value = record.get(key)
    if not isinstance(value, list) or len(value) != 7:
        raise RuntimeError(f"Replay record {record_index} {key} must be a 7D list, got {value}")
    vector = [float(item) for item in value]
    if not all(math.isfinite(item) for item in vector):
        raise RuntimeError(f"Replay record {record_index} {key} has non-finite values: {vector}")
    return vector


def _vector_summary(vectors: list[list[float]], prefix: str) -> dict[str, object]:
    if not vectors:
        return {}
    array = np.asarray(vectors, dtype=np.float64)
    return {
        f"first_{prefix}7": array[0].tolist(),
        f"last_{prefix}7": array[-1].tolist(),
        f"min_{prefix}7": array.min(axis=0).tolist(),
        f"max_{prefix}7": array.max(axis=0).tolist(),
        f"mean_{prefix}7": array.mean(axis=0).tolist(),
    }


def load_dataset_gripper_bounds(dataset_root: str | Path) -> tuple[float, float] | None:
    root = Path(dataset_root).expanduser()
    if not root.is_absolute():
        root = REPO_ROOT / root
    stats_path = root / "meta" / "stats.json"
    if not stats_path.exists():
        return None
    try:
        payload = json.loads(stats_path.read_text())
        action_stats = payload.get("action")
        if not isinstance(action_stats, dict):
            return None
        action_min = np.asarray(action_stats["min"], dtype=np.float64).reshape(-1)
        action_max = np.asarray(action_stats["max"], dtype=np.float64).reshape(-1)
        if len(action_min) == 0 or len(action_max) == 0:
            return None
        lower = float(action_min[-1])
        upper = float(action_max[-1])
        if not (math.isfinite(lower) and math.isfinite(upper)):
            return None
        if lower > upper:
            lower, upper = upper, lower
        return lower, upper
    except Exception as exc:
        log(f"[Roboclaw][WARN] Failed to load gripper bounds from {stats_path}: {exc}")
        return None


def resolve_gripper_clip_bounds(parsed_args: argparse.Namespace) -> tuple[tuple[float, float], str]:
    lower = parsed_args.gripper_clip_lower
    upper = parsed_args.gripper_clip_upper
    if (lower is None) ^ (upper is None):
        raise RuntimeError("--gripper-clip-lower and --gripper-clip-upper must be set together")
    if lower is not None and upper is not None:
        return (float(lower), float(upper)), "explicit"
    dataset_bounds = load_dataset_gripper_bounds(parsed_args.dataset_root)
    if dataset_bounds is not None:
        return dataset_bounds, "dataset_stats"
    return (-math.inf, math.inf), "fallback_unbounded"


def load_dataset_replay(replay_path: str, gripper_clip_bounds: tuple[float, float]) -> dict[str, object]:
    path = Path(replay_path).expanduser()
    if not path.is_absolute():
        path = REPO_ROOT / path
    if not path.exists():
        raise FileNotFoundError(f"Replay JSON does not exist: {path}")
    payload = json.loads(path.read_text())
    records = payload.get("records")
    if not isinstance(records, list) or not records:
        raise RuntimeError(f"Dataset replay file has no records: {path}")

    states: list[list[float]] = []
    actions: list[list[float]] = []
    normalized_records: list[dict[str, object]] = []
    for record_index, record in enumerate(records):
        if not isinstance(record, dict):
            raise RuntimeError(f"Replay record {record_index} is not an object")
        state7 = _validate_replay_vector(record, "state7", record_index)
        action7 = _validate_replay_vector(record, "action7", record_index)
        states.append(state7)
        actions.append(action7)
        normalized_records.append(
            {
                "dataset_index": record.get("dataset_index"),
                "episode_index": record.get("episode_index"),
                "frame_index": record.get("frame_index"),
                "timestamp": record.get("timestamp"),
                "task_index": record.get("task_index"),
                "state7": state7,
                "action7": action7,
            }
        )

    return {
        "path": path,
        "payload": payload,
        "records": normalized_records,
        "states": states,
        "actions": actions,
        "state_summary": _vector_summary(states, "state"),
        "action_summary": _vector_summary(actions, "action"),
        "gripper_clip_count_state": int(
            np.count_nonzero((np.asarray(states)[:, 6] < gripper_clip_bounds[0]) | (np.asarray(states)[:, 6] > gripper_clip_bounds[1]))
        ),
        "gripper_clip_count_action": int(
            np.count_nonzero((np.asarray(actions)[:, 6] < gripper_clip_bounds[0]) | (np.asarray(actions)[:, 6] > gripper_clip_bounds[1]))
        ),
    }


def replay_record_for_frame(
    replay: dict[str, object],
    frame_index: int,
    start_index: int,
    stride: int,
    hold_last: bool,
    loop: bool,
) -> tuple[int, dict[str, object]] | None:
    records = replay["records"]
    if not isinstance(records, list) or not records:
        return None
    start_index = max(0, int(start_index))
    stride = max(1, int(stride))
    replay_index = start_index + frame_index * stride
    if loop:
        usable_count = max(1, len(records) - start_index)
        replay_index = start_index + ((frame_index * stride) % usable_count)
    elif replay_index >= len(records):
        if not hold_last:
            return None
        replay_index = len(records) - 1
    record = records[replay_index]
    if not isinstance(record, dict):
        raise RuntimeError(f"Replay record {replay_index} is not an object")
    return replay_index, record


def replay_vector_from_record(record: dict[str, object], source: str) -> list[float]:
    key = "state7" if source == "state" else "action7"
    value = record.get(key)
    if not isinstance(value, list) or len(value) != 7:
        raise RuntimeError(f"Replay record missing valid {key}: {record}")
    return [float(item) for item in value]


def gripper_clip_count(values: np.ndarray, bounds: tuple[float, float]) -> int:
    lower, upper = bounds
    return int(np.count_nonzero((values < lower) | (values > upper)))


def world_position(prim_path: str) -> list[float] | None:
    if not prim_path or not is_prim_path_valid(prim_path):
        return None
    rigid_prim = getattr(world_position, "_rigid_prim", None)
    rigid_prim_path = getattr(world_position, "_rigid_prim_path", None)
    if rigid_prim_path != prim_path:
        try:
            rigid_prim = SingleRigidPrim(prim_path=prim_path, name="roboclaw_target_object", reset_xform_properties=False)
            rigid_prim.initialize()
            world_position._rigid_prim = rigid_prim
            world_position._rigid_prim_path = prim_path
        except Exception:
            rigid_prim = None
            world_position._rigid_prim = None
            world_position._rigid_prim_path = prim_path
    if rigid_prim is not None:
        try:
            position, _orientation = rigid_prim.get_world_pose()
            return [float(value) for value in np.asarray(position, dtype=np.float64).reshape(-1)[:3]]
        except Exception:
            pass
    position, _orientation = get_world_pose(prim_path)
    return [float(value) for value in np.asarray(position, dtype=np.float64).reshape(-1)[:3]]


def target_initial_override(parsed_args: argparse.Namespace) -> list[float] | None:
    values = [
        float(parsed_args.target_initial_x),
        float(parsed_args.target_initial_y),
        float(parsed_args.target_initial_z),
    ]
    if any(math.isfinite(value) for value in values) and not all(math.isfinite(value) for value in values):
        raise RuntimeError("--target-initial-x, --target-initial-y, and --target-initial-z must be set together")
    if not all(math.isfinite(value) for value in values):
        return None
    return values


def set_target_initial_pose(prim_path: str, position: list[float]) -> None:
    target = SingleRigidPrim(prim_path=prim_path, name="roboclaw_target_initial_pose", reset_xform_properties=False)
    target.initialize()
    target.set_world_pose(
        position=np.asarray(position, dtype=np.float32),
        orientation=np.asarray([1.0, 0.0, 0.0, 0.0], dtype=np.float32),
    )
    target.set_linear_velocity(np.zeros(3, dtype=np.float32))
    target.set_angular_velocity(np.zeros(3, dtype=np.float32))
    world_position._rigid_prim = target
    world_position._rigid_prim_path = prim_path


def set_target_red_appearance(prim_path: str) -> None:
    stage = get_current_stage()
    target_prim = stage.GetPrimAtPath(prim_path)
    if not target_prim.IsValid():
        raise RuntimeError(f"target object prim not found at {prim_path}")

    red = Gf.Vec3f(0.9, 0.02, 0.01)
    UsdGeom.Scope.Define(stage, Sdf.Path("/World/Looks"))
    material = UsdShade.Material.Define(stage, Sdf.Path("/World/Looks/RoboclawTargetRed"))
    shader = UsdShade.Shader.Define(stage, Sdf.Path("/World/Looks/RoboclawTargetRed/Shader"))
    shader.CreateIdAttr("UsdPreviewSurface")
    shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(red)
    shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.45)
    shader.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(0.0)
    material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
    UsdShade.MaterialBindingAPI.Apply(target_prim).Bind(
        material, bindingStrength=UsdShade.Tokens.strongerThanDescendants
    )

    for prim in Usd.PrimRange(target_prim):
        if prim.IsA(UsdGeom.Gprim):
            UsdGeom.Gprim(prim).CreateDisplayColorAttr().Set([red])


def xy_displacement_m(initial_position: list[float] | None, final_position: list[float] | None) -> float | None:
    if initial_position is None or final_position is None:
        return None
    delta = np.asarray(final_position[:2], dtype=np.float64) - np.asarray(initial_position[:2], dtype=np.float64)
    return float(np.linalg.norm(delta))


def distance_m(position_a: list[float] | None, position_b: list[float] | None) -> float | None:
    if position_a is None or position_b is None:
        return None
    return float(np.linalg.norm(np.asarray(position_a[:3], dtype=np.float64) - np.asarray(position_b[:3], dtype=np.float64)))


def target_metrics(
    target_prim_path: str,
    initial_position: list[float] | None,
    threshold_m: float,
) -> dict[str, object]:
    final_position = world_position(target_prim_path)
    displacement = xy_displacement_m(initial_position, final_position)
    return {
        "target_prim_path": target_prim_path,
        "initial_world_position": initial_position,
        "final_world_position": final_position,
        "xy_displacement_m": displacement,
        "success_displacement_threshold_m": float(threshold_m),
        "success": bool(displacement is not None and displacement >= threshold_m),
    }


def validate_paths(parsed_args: argparse.Namespace) -> None:
    required_files = {
        "usd": Path(parsed_args.usd_path).expanduser(),
    }
    if parsed_args.replay_json:
        required_files["replay_json"] = Path(parsed_args.replay_json).expanduser()
    else:
        required_files["smolvla_python"] = Path(parsed_args.smolvla_python).expanduser()
        required_files["smolvla_model_id"] = Path(parsed_args.smolvla_model_id).expanduser()
        required_files["dataset_root"] = Path(parsed_args.dataset_root).expanduser()
        if parsed_args.policy_align_initial_state_replay_json:
            required_files["policy_align_initial_state_replay_json"] = Path(
                parsed_args.policy_align_initial_state_replay_json
            ).expanduser()
        if parsed_args.vlm_model_path:
            required_files["vlm_model_path"] = Path(parsed_args.vlm_model_path).expanduser()
    for label, path in required_files.items():
        if not path.exists():
            raise FileNotFoundError(f"{label} does not exist: {path}")


def wait_for_stage_load() -> None:
    for _ in range(2):
        simulation_app.update()
    while is_stage_loading():
        simulation_app.update()


def summarize_stage_prims() -> dict[str, object]:
    stage = get_current_stage()
    root_prims = [str(prim.GetPath()) for prim in stage.GetPseudoRoot().GetChildren()]
    camera_prims = []
    articulation_roots = []
    for prim in stage.Traverse():
        if prim.IsA(UsdGeom.Camera):
            camera_prims.append(str(prim.GetPath()))
        if prim.HasAPI(UsdPhysics.ArticulationRootAPI):
            articulation_roots.append(str(prim.GetPath()))
    return {
        "root_prims": root_prims,
        "camera_prims": camera_prims,
        "articulation_roots": articulation_roots,
    }


def require_prim(path: str, label: str) -> None:
    if not is_prim_path_valid(path):
        raise RuntimeError(f"{label} prim not found at {path}")


def build_camera(name: str, prim_path: str) -> Camera:
    require_prim(prim_path, f"{name} camera")
    return Camera(prim_path=prim_path, name=f"roboclaw_{name}", resolution=(640, 480), annotator_device="cpu")


def inference_env() -> dict[str, str]:
    env = os.environ.copy()
    for key in (
        "ALL_PROXY",
        "HTTP_PROXY",
        "HTTPS_PROXY",
        "NO_PROXY",
        "all_proxy",
        "http_proxy",
        "https_proxy",
        "no_proxy",
        "PYTHONHOME",
        "PYTHONPATH",
    ):
        env.pop(key, None)
    env.setdefault("PYTHONUNBUFFERED", "1")
    env.setdefault("HF_HOME", str(REPO_ROOT / "outputs/hf_runtime_cache"))
    env.setdefault("HF_DATASETS_CACHE", str(REPO_ROOT / "outputs/hf_runtime_cache/datasets"))
    env.setdefault("TRANSFORMERS_CACHE", str(REPO_ROOT / "outputs/hf_runtime_cache/transformers"))
    env.setdefault("HF_HUB_OFFLINE", "1")
    env.setdefault("TRANSFORMERS_OFFLINE", "1")
    env.setdefault("HF_HUB_DISABLE_XET", "1")
    return env


def run_smolvla_inference(
    frame_index: int,
    frame_state7: list[float],
    cameras: dict[str, Camera],
    parsed_args: argparse.Namespace,
    output_dir: Path,
) -> dict[str, object]:
    wrist_name = active_wrist_name(parsed_args.active_arm_label)
    frame_dir = output_dir / "frames"
    file_index = frame_index
    ring_size = max(0, int(parsed_args.frame_file_ring_size))
    if ring_size > 0:
        file_index = frame_index % ring_size
    camera1_path = frame_dir / f"frame_{file_index:06d}_camera1_overhead.png"
    camera2_path = frame_dir / f"frame_{file_index:06d}_camera2_{wrist_name}.png"
    frame_json_path = frame_dir / f"frame_{file_index:06d}.json"
    result_json_path = frame_dir / f"frame_{file_index:06d}_result.json"

    write_png_rgb(camera1_path, camera_rgb("overhead", cameras["overhead"]))
    write_png_rgb(camera2_path, camera_rgb(wrist_name, cameras[wrist_name]))
    frame_payload = {
        "frame_index": frame_index,
        "state7": frame_state7,
        "task": parsed_args.task_text,
        "camera1_path": str(camera1_path),
        "camera2_path": str(camera2_path),
        "camera1_source": "overhead",
        "camera2_source": wrist_name,
        "active_arm_label": parsed_args.active_arm_label,
        "state_source": f"{parsed_args.active_arm_label}_7d",
        "control_mode": parsed_args.control_mode,
    }
    write_json(frame_json_path, frame_payload)

    command = [
        str(Path(parsed_args.smolvla_python).expanduser()),
        str(REPO_ROOT / "tools/lerobot/panthera_smolvla_infer.py"),
        "--model-id",
        str(Path(parsed_args.smolvla_model_id).expanduser()),
        "--frame-json",
        str(frame_json_path),
        "--output-json",
        str(result_json_path),
        "--device",
        parsed_args.smolvla_device,
        "--dataset-root",
        str(Path(parsed_args.dataset_root).expanduser()),
        "--task-text",
        parsed_args.task_text,
        "--control-mode",
        parsed_args.control_mode,
        "--joint-unit-mode",
        parsed_args.policy_joint_unit_mode,
        "--frame-index",
        str(frame_index),
    ]
    if parsed_args.vlm_model_path:
        command.extend(["--vlm-model-path", str(Path(parsed_args.vlm_model_path).expanduser())])

    started_at = time.time()
    completed = subprocess.run(
        command,
        cwd=REPO_ROOT,
        env=inference_env(),
        text=True,
        capture_output=True,
        check=False,
    )
    helper_latency_s = time.time() - started_at
    for line in completed.stdout.splitlines():
        log(line)
    for line in completed.stderr.splitlines():
        log(f"[Roboclaw][SmolVLA][stderr] {line}")
    if completed.returncode != 0:
        raise RuntimeError(f"SmolVLA helper failed with exit code {completed.returncode}")

    result = json.loads(result_json_path.read_text())
    action = result.get("predicted_action_step0")
    if not isinstance(action, list) or len(action) < 6:
        raise RuntimeError(f"SmolVLA helper did not return a valid action with at least 6 values: {action}")
    action = [float(value) for value in action[:7]]
    result["predicted_action_step0"] = action
    result["predicted_action_dim"] = len(action)
    result["helper_latency_s"] = helper_latency_s
    result["frame_json_path"] = str(frame_json_path)
    result["result_json_path"] = str(result_json_path)
    result["camera1_path"] = str(camera1_path)
    result["camera2_path"] = str(camera2_path)
    return result


class PersistentSmolVLAHelper:
    def __init__(self, parsed_args: argparse.Namespace) -> None:
        self._parsed_args = parsed_args
        self._process: subprocess.Popen[str] | None = None
        self._request_index = 0
        self.ready_payload: dict[str, object] | None = None

    def __enter__(self) -> "PersistentSmolVLAHelper":
        command = [
            str(Path(self._parsed_args.smolvla_python).expanduser()),
            str(REPO_ROOT / "tools/lerobot/panthera_smolvla_infer.py"),
            "--model-id",
            str(Path(self._parsed_args.smolvla_model_id).expanduser()),
            "--device",
            self._parsed_args.smolvla_device,
            "--dataset-root",
            str(Path(self._parsed_args.dataset_root).expanduser()),
            "--task-text",
            self._parsed_args.task_text,
            "--control-mode",
            self._parsed_args.control_mode,
            "--joint-unit-mode",
            self._parsed_args.policy_joint_unit_mode,
            "--server",
        ]
        if self._parsed_args.vlm_model_path:
            command.extend(["--vlm-model-path", str(Path(self._parsed_args.vlm_model_path).expanduser())])
        started_at = time.time()
        self._process = subprocess.Popen(
            command,
            cwd=REPO_ROOT,
            env=inference_env(),
            text=True,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            bufsize=1,
        )
        ready = self._read_stdout_json(timeout_s=120.0)
        if ready.get("type") != "ready":
            raise RuntimeError(f"Unexpected SmolVLA helper ready payload: {ready}")
        ready["startup_latency_s"] = time.time() - started_at
        self.ready_payload = ready
        log(f"[Roboclaw][SmolVLA] Persistent helper ready: {ready}")
        self._drain_stderr()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        process = self._process
        if process is None:
            return
        if process.poll() is None and process.stdin is not None:
            try:
                process.stdin.write(json.dumps({"type": "shutdown"}) + "\n")
                process.stdin.flush()
            except BrokenPipeError:
                pass
        try:
            process.wait(timeout=5.0)
        except subprocess.TimeoutExpired:
            process.terminate()
            try:
                process.wait(timeout=5.0)
            except subprocess.TimeoutExpired:
                process.kill()
        self._drain_stderr()

    def _read_stdout_json(self, timeout_s: float) -> dict[str, object]:
        if self._process is None or self._process.stdout is None:
            raise RuntimeError("SmolVLA helper is not running")
        deadline = time.time() + timeout_s
        while time.time() < deadline:
            if self._process.poll() is not None:
                self._drain_stderr()
                raise RuntimeError(f"SmolVLA helper exited early with code {self._process.returncode}")
            ready, _unused, _unused2 = select.select([self._process.stdout], [], [], 0.1)
            if not ready:
                self._drain_stderr()
                continue
            line = self._process.stdout.readline()
            if not line:
                continue
            line = line.strip()
            if not line:
                continue
            try:
                return json.loads(line)
            except json.JSONDecodeError:
                log(f"[Roboclaw][SmolVLA][server] {line}")
        self._drain_stderr()
        raise RuntimeError("Timed out waiting for SmolVLA helper response")

    def _drain_stderr(self) -> None:
        process = self._process
        if process is None or process.stderr is None:
            return
        while True:
            ready, _unused, _unused2 = select.select([process.stderr], [], [], 0)
            if not ready:
                return
            line = process.stderr.readline()
            if not line:
                return
            log(f"[Roboclaw][SmolVLA][stderr] {line.rstrip()}")

    def infer(
        self,
        frame_index: int,
        frame_json_path: Path,
        result_json_path: Path,
        parsed_args: argparse.Namespace,
    ) -> dict[str, object]:
        if self._process is None or self._process.stdin is None:
            raise RuntimeError("SmolVLA helper is not running")
        self._request_index += 1
        request_id = self._request_index
        started_at = time.time()
        request = {
            "type": "infer",
            "request_id": request_id,
            "frame_index": frame_index,
            "frame_json": str(frame_json_path),
            "output_json": str(result_json_path),
            "control_mode": parsed_args.control_mode,
            "task_text": parsed_args.task_text,
        }
        self._process.stdin.write(json.dumps(request, ensure_ascii=False) + "\n")
        self._process.stdin.flush()
        while True:
            response = self._read_stdout_json(timeout_s=120.0)
            if response.get("type") == "error":
                raise RuntimeError(f"SmolVLA helper error: {response.get('error')}")
            if response.get("type") == "result" and response.get("request_id") == request_id:
                result = response["result"]
                if not isinstance(result, dict):
                    raise RuntimeError(f"Unexpected SmolVLA result payload: {response}")
                result["helper_latency_s"] = time.time() - started_at
                self._drain_stderr()
                return result


def write_policy_frame(
    frame_index: int,
    frame_state7: list[float],
    cameras: dict[str, Camera],
    parsed_args: argparse.Namespace,
    output_dir: Path,
) -> tuple[Path, Path]:
    wrist_name = active_wrist_name(parsed_args.active_arm_label)
    frame_dir = output_dir / "frames"
    file_index = frame_index
    ring_size = max(0, int(parsed_args.frame_file_ring_size))
    if ring_size > 0:
        file_index = frame_index % ring_size
    camera1_path = frame_dir / f"frame_{file_index:06d}_camera1_overhead.png"
    camera2_path = frame_dir / f"frame_{file_index:06d}_camera2_{wrist_name}.png"
    frame_json_path = frame_dir / f"frame_{file_index:06d}.json"
    result_json_path = frame_dir / f"frame_{file_index:06d}_result.json"

    write_png_rgb(camera1_path, camera_rgb("overhead", cameras["overhead"]))
    write_png_rgb(camera2_path, camera_rgb(wrist_name, cameras[wrist_name]))
    write_json(
        frame_json_path,
        {
            "frame_index": frame_index,
            "state7": frame_state7,
            "task": parsed_args.task_text,
            "camera1_path": str(camera1_path),
            "camera2_path": str(camera2_path),
            "camera1_source": "overhead",
            "camera2_source": wrist_name,
            "active_arm_label": parsed_args.active_arm_label,
            "state_source": f"{parsed_args.active_arm_label}_7d",
            "control_mode": parsed_args.control_mode,
        },
    )
    return frame_json_path, result_json_path


def main() -> None:
    validate_paths(args)
    output_dir = Path(args.output_dir).expanduser() / time.strftime("%Y%m%d_%H%M%S")
    output_dir.mkdir(parents=True, exist_ok=True)
    helper_ready_payload = None
    gripper_clip_bounds, gripper_clip_source = resolve_gripper_clip_bounds(args)
    policy_joint_unit_mode, policy_joint_unit_mode_source = resolve_policy_joint_unit_mode(args)
    gripper_clip_bounds_execution, gripper_clip_execution_units = execution_gripper_clip_bounds(
        gripper_clip_bounds,
        gripper_clip_source,
        policy_joint_unit_mode,
    )

    usd_path = str(Path(args.usd_path).expanduser())
    log(f"[Roboclaw] Opening USD: {usd_path}")
    if not open_stage(usd_path):
        raise RuntimeError(f"Failed to open USD stage: {usd_path}")
    wait_for_stage_load()

    paths = camera_paths(args)
    wrist_name = active_wrist_name(args.active_arm_label)
    articulation_path = active_articulation_path(args)
    end_effector_path = active_end_effector_path(args)
    stage_summary = summarize_stage_prims()
    require_prim(articulation_path, f"{args.active_arm_label} articulation")
    if args.target_prim_path:
        require_prim(args.target_prim_path, "target object")
        set_target_red_appearance(args.target_prim_path)
        log(f"[Roboclaw] Target object appearance override: red material bound to {args.target_prim_path}")
    if end_effector_path:
        require_prim(end_effector_path, "end-effector diagnostic")
    for camera_name in ("overhead", wrist_name):
        require_prim(paths[camera_name], f"{camera_name} camera")
    if not args.headless:
        try:
            set_active_viewport_camera(paths["overhead"])
            log(f"[Roboclaw] GUI active viewport camera set to DepthCamera: {paths['overhead']}")
        except Exception as exc:
            log(f"[Roboclaw][WARN] Failed to set active viewport camera to {paths['overhead']}: {exc}")
            set_camera_view(eye=[1.25, 1.1, 1.15], target=[0.0, 0.0, 0.78], camera_prim_path="/OmniverseKit_Persp")

    world = World(stage_units_in_meters=1.0, physics_prim_path="/physicsScene")
    active_articulation = world.scene.add(
        Articulation(prim_paths_expr=articulation_path, name=f"roboclaw_{args.active_arm_label}")
    )
    world.reset()
    world.play()
    target_initial_override_position = target_initial_override(args)
    if target_initial_override_position is not None:
        set_target_initial_pose(args.target_prim_path, target_initial_override_position)
        log(
            f"[Roboclaw] Target initial pose override: {args.target_prim_path} "
            f"position={target_initial_override_position}"
        )
        world.step(render=True)

    cameras = {
        "overhead": build_camera("overhead", paths["overhead"]),
        wrist_name: build_camera(wrist_name, paths[wrist_name]),
    }
    for camera in cameras.values():
        camera.initialize()

    warmup_frames = max(1, int(args.warmup_frames))
    for _ in range(warmup_frames):
        world.step(render=True)

    policy_initial_state_alignment: dict[str, object] | None = None
    if args.policy_align_initial_state_replay_json and not args.replay_json:
        alignment_replay = load_dataset_replay(args.policy_align_initial_state_replay_json, gripper_clip_bounds)
        alignment_records = alignment_replay["records"]
        if not isinstance(alignment_records, list) or not alignment_records:
            raise RuntimeError(f"No records in policy initial-state replay: {args.policy_align_initial_state_replay_json}")
        alignment_index = min(max(0, int(args.policy_align_initial_state_index)), len(alignment_records) - 1)
        alignment_record = alignment_records[alignment_index]
        if not isinstance(alignment_record, dict):
            raise RuntimeError(f"Policy alignment record {alignment_index} is not an object")
        alignment_state7 = replay_vector_from_record(alignment_record, "state")
        alignment_target = action_target_for_dofs(
            active_articulation.dof_names,
            alignment_state7,
            gripper_clip_bounds_execution,
        )
        active_articulation.set_joint_positions(alignment_target)
        active_articulation.set_joint_position_targets(alignment_target)
        settle_frames = max(0, int(args.policy_align_settle_frames))
        for _ in range(settle_frames):
            world.step(render=True)
        policy_initial_state_alignment = {
            "replay_json": str(alignment_replay["path"]),
            "replay_index": alignment_index,
            "state7": alignment_state7,
            "joint_target_abs_error": joint_abs_error_for_target(active_articulation, alignment_target),
            "settle_frames": settle_frames,
        }
        log(
            f"[Roboclaw][SmolVLA] Policy initial state aligned from replay={alignment_replay['path']} "
            f"index={alignment_index} state7={alignment_state7}"
        )

    locked_joint_value, locked_joint_value_source = resolve_locked_joint_value(args, active_articulation)
    target_initial_position = world_position(args.target_prim_path)
    end_effector_initial_position = world_position(end_effector_path)
    log(f"[Roboclaw] Active arm: {args.active_arm_label} articulation={articulation_path}")
    log(f"[Roboclaw] DOF names: {active_articulation.dof_names}")
    log(f"[Roboclaw] Camera mapping: camera1=overhead:{paths['overhead']} camera2={wrist_name}:{paths[wrist_name]}")
    log(f"[Roboclaw] Control mode: {args.control_mode}")
    log(
        f"[Roboclaw] Policy joint unit mode: {policy_joint_unit_mode} "
        f"source={policy_joint_unit_mode_source} gripper_execution_bounds={gripper_clip_bounds_execution} "
        f"bounds_units={gripper_clip_execution_units}"
    )
    log(
        f"[Roboclaw] Six-DOF action layout: {args.six_dof_action_layout} "
        f"locked_joint={args.locked_joint_name} locked_value={locked_joint_value} source={locked_joint_value_source}"
    )
    log(f"[Roboclaw] Target object: {args.target_prim_path} initial_world_position={target_initial_position}")
    log(
        f"[Roboclaw] End effector diagnostic: {end_effector_path} initial_world_position={end_effector_initial_position} "
        f"initial_distance_to_target_m={distance_m(end_effector_initial_position, target_initial_position)}"
    )

    max_frames = int(args.max_frames)
    infer_interval = max(1, int(args.infer_interval))
    replay: dict[str, object] | None = None
    replay_apply_count = 0
    replay_last_index: int | None = None
    replay_last_vector7: list[float] | None = None
    replay_last_target_error: dict[str, object] | None = None
    replay_max_joint_target_abs_error: float | None = None
    replay_mean_joint_target_abs_error_sum = 0.0
    replay_joint_error_samples = 0
    replay_initial_state7: list[float] | None = None
    if args.replay_json:
        replay = load_dataset_replay(args.replay_json, gripper_clip_bounds)
        if args.replay_align_initial_state:
            records_for_alignment = replay["records"]
            if isinstance(records_for_alignment, list) and records_for_alignment:
                alignment_index = min(max(0, int(args.replay_start_index)), len(records_for_alignment) - 1)
                first_record = records_for_alignment[alignment_index]
                if isinstance(first_record, dict):
                    replay_initial_state7 = replay_vector_from_record(first_record, "state")
                    active_articulation.set_joint_positions(
                        action_target_for_dofs(
                            active_articulation.dof_names,
                            replay_initial_state7,
                            gripper_clip_bounds_execution,
                        )
                    )
                    world.step(render=True)
        log(
            f"[Roboclaw][Replay] Enabled replay_json={replay['path']} source={args.replay_source} "
            f"apply_mode={args.replay_apply_mode} records={len(replay['records'])} "
            f"episode={replay['payload'].get('episode_index')} task={replay['payload'].get('task')!r} "
            f"align_initial_state={bool(args.replay_align_initial_state)}"
        )
        log(
            f"[Roboclaw][Replay] Source validation first_state7={replay['state_summary'].get('first_state7')} "
            f"last_state7={replay['state_summary'].get('last_state7')} "
            f"first_action7={replay['action_summary'].get('first_action7')} "
            f"last_action7={replay['action_summary'].get('last_action7')}"
        )
    inference_count = 0
    apply_count = 0
    records: list[dict[str, object]] = []
    exit_reason = f"max_frames reached ({max_frames})"
    started_at = time.time()
    persistent_helper: PersistentSmolVLAHelper | None = None
    min_ee_to_target_distance_m: float | None = None
    final_ee_to_target_distance_m: float | None = None

    try:
        if args.use_persistent_helper and replay is None:
            persistent_helper = PersistentSmolVLAHelper(args)
            persistent_helper.__enter__()
            helper_ready_payload = persistent_helper.ready_payload
        frame_index = 0
        while max_frames <= 0 or frame_index < max_frames:
            if not simulation_app.is_running():
                exit_reason = "simulation_app requested exit"
                break
            world.step(render=True)
            frame_state7 = state7(active_articulation)
            target_position = world_position(args.target_prim_path)
            end_effector_position = world_position(end_effector_path)
            ee_to_target_distance = distance_m(end_effector_position, target_position)
            if ee_to_target_distance is not None:
                min_ee_to_target_distance_m = (
                    ee_to_target_distance
                    if min_ee_to_target_distance_m is None
                    else min(min_ee_to_target_distance_m, ee_to_target_distance)
                )
                final_ee_to_target_distance_m = ee_to_target_distance
            record: dict[str, object] = {
                "frame_index": frame_index,
                "state7": frame_state7,
                "joint_positions_by_name": actual_joint_positions_by_name(active_articulation),
                "target_world_position": target_position,
                "end_effector_world_position": end_effector_position,
                "ee_to_target_distance_m": ee_to_target_distance,
            }
            if replay is not None:
                replay_item = replay_record_for_frame(
                    replay,
                    frame_index,
                    args.replay_start_index,
                    args.replay_stride,
                    bool(args.replay_hold_last),
                    bool(args.replay_loop),
                )
                if replay_item is None:
                    exit_reason = f"dataset replay exhausted at frame {frame_index}"
                    break
                replay_index, source_record = replay_item
                replay_vector7 = replay_vector_from_record(source_record, args.replay_source)
                replay_target = action_target_for_dofs(
                    active_articulation.dof_names,
                    replay_vector7,
                    gripper_clip_bounds_execution,
                )
                if args.replay_apply_mode == "teleport":
                    active_articulation.set_joint_positions(replay_target)
                else:
                    active_articulation.set_joint_position_targets(replay_target)
                replay_apply_count += 1
                replay_last_index = replay_index
                replay_last_vector7 = replay_vector7
                replay_last_target_error = joint_abs_error_for_target(active_articulation, replay_target)
                max_error = replay_last_target_error.get("max_abs_error")
                mean_error = replay_last_target_error.get("mean_abs_error")
                if max_error is not None:
                    replay_max_joint_target_abs_error = max(replay_max_joint_target_abs_error or 0.0, float(max_error))
                if mean_error is not None:
                    replay_mean_joint_target_abs_error_sum += float(mean_error)
                    replay_joint_error_samples += 1
                record.update(
                    {
                        "replay_index": replay_index,
                        "replay_source": args.replay_source,
                        "replay_source_record": source_record,
                        "replay_vector7": replay_vector7,
                        "applied_target": replay_target,
                        "joint_target_abs_error": replay_last_target_error,
                    }
                )
                log_interval = max(1, int(args.replay_log_interval))
                if frame_index == 0 or frame_index % log_interval == 0:
                    log(
                        f"[Roboclaw][Replay] frame={frame_index} replay_index={replay_index} "
                        f"{args.replay_source}7={replay_vector7} "
                        f"actual_joint_positions={actual_joint_positions_by_name(active_articulation)} "
                        f"joint_target_error={replay_last_target_error} "
                        f"target_world_position={world_position(args.target_prim_path)} "
                        f"ee_to_target_distance_m={ee_to_target_distance}"
                    )
            elif frame_index % infer_interval == 0:
                log(f"[Roboclaw][SmolVLA] Inference frame={frame_index} state7={frame_state7}")
                if persistent_helper is not None:
                    frame_json_path, result_json_path = write_policy_frame(
                        frame_index, frame_state7, cameras, args, output_dir
                    )
                    result = persistent_helper.infer(frame_index, frame_json_path, result_json_path, args)
                    result["frame_json_path"] = str(frame_json_path)
                    result["result_json_path"] = str(result_json_path)
                else:
                    result = run_smolvla_inference(frame_index, frame_state7, cameras, args, output_dir)
                raw_model_action = [float(value) for value in result["predicted_action_step0"]]
                action = action_from_model_units(raw_model_action, policy_joint_unit_mode)
                result["predicted_action_step0_model_units"] = raw_model_action
                result["predicted_action_step0_execution_units"] = action
                result["execution_joint_unit_mode"] = "radians"
                inference_count += 1
                record["smolvla_result"] = result
                record["model_action"] = raw_model_action
                record["action"] = action
                record["predicted_action_dim"] = len(action)
                if len(action) >= 7:
                    record["action7"] = action
                log(
                    f"[Roboclaw][SmolVLA] Predicted action_dim={len(action)} frame={frame_index} "
                    f"model_units={raw_model_action} execution_radians={action}"
                )
                if args.control_mode == "single-arm":
                    target = action_target_for_dofs(
                        active_articulation.dof_names,
                        action,
                        gripper_clip_bounds_execution,
                        args.six_dof_action_layout,
                        args.locked_joint_name,
                        locked_joint_value,
                    )
                    active_articulation.set_joint_position_targets(target)
                    apply_count += 1
                    record["applied_target"] = target
                    record["action_mapping"] = action_mapping_summary(
                        action,
                        args.six_dof_action_layout,
                        args.locked_joint_name,
                        locked_joint_value,
                    )
                    log(f"[Roboclaw][SmolVLA] Applied single-arm target frame={frame_index}")
                else:
                    log("[Roboclaw][SmolVLA] log-only: action recorded but not applied")
            records.append(record)
            max_records_in_memory = max(0, int(args.max_records_in_memory))
            if max_records_in_memory > 0 and len(records) > max_records_in_memory:
                del records[: len(records) - max_records_in_memory]
            frame_index += 1
    except KeyboardInterrupt:
        exit_reason = "keyboard interrupt"
    finally:
        if persistent_helper is not None:
            persistent_helper.__exit__(None, None, None)
        duration_s = time.time() - started_at
        final_target_metrics = target_metrics(
            args.target_prim_path, target_initial_position, args.success_displacement_threshold
        )
        task_success = bool(
            inference_count > 0 and apply_count > 0 and args.control_mode == "single-arm" and final_target_metrics["success"]
        )
        if replay is not None:
            task_success = bool(replay_apply_count > 0)
        summary = {
            "success": task_success,
            "policy_pipeline_success": inference_count > 0,
            "replay_enabled": replay is not None,
            "exit_reason": exit_reason,
            "duration_s": duration_s,
            "usd_path": usd_path,
            "output_dir": str(output_dir),
            "active_arm_label": args.active_arm_label,
            "active_articulation_path": articulation_path,
            "end_effector_path": end_effector_path,
            "dof_names": active_articulation.dof_names,
            "camera_paths": {"overhead": paths["overhead"], wrist_name: paths[wrist_name]},
            "control_mode": args.control_mode,
            "max_frames": max_frames,
            "warmup_frames": warmup_frames,
            "infer_interval": infer_interval,
            "frame_file_ring_size": max(0, int(args.frame_file_ring_size)),
            "max_records_in_memory": max(0, int(args.max_records_in_memory)),
            "keep_open": bool(args.keep_open),
            "use_persistent_helper": bool(args.use_persistent_helper),
            "gripper_clip_bounds": list(gripper_clip_bounds),
            "gripper_clip_source": gripper_clip_source,
            "gripper_clip_bounds_execution": list(gripper_clip_bounds_execution),
            "gripper_clip_execution_units": gripper_clip_execution_units,
            "policy_joint_unit_mode": policy_joint_unit_mode,
            "policy_joint_unit_mode_source": policy_joint_unit_mode_source,
            "six_dof_action_layout": args.six_dof_action_layout,
            "locked_joint_name": args.locked_joint_name,
            "locked_joint_value": locked_joint_value,
            "locked_joint_value_source": locked_joint_value_source,
            "policy_initial_state_alignment": policy_initial_state_alignment,
            "inference_count": inference_count,
            "apply_count": apply_count,
            "replay_apply_count": replay_apply_count,
            "replay_source": args.replay_source if replay is not None else None,
            "replay_apply_mode": args.replay_apply_mode if replay is not None else None,
            "replay_json": str(replay["path"]) if replay is not None else None,
            "replay_record_count": len(replay["records"]) if replay is not None else None,
            "replay_start_index": int(args.replay_start_index) if replay is not None else None,
            "replay_stride": int(args.replay_stride) if replay is not None else None,
            "replay_hold_last": bool(args.replay_hold_last) if replay is not None else None,
            "replay_loop": bool(args.replay_loop) if replay is not None else None,
            "replay_align_initial_state": bool(args.replay_align_initial_state) if replay is not None else None,
            "replay_initial_state7": replay_initial_state7,
            "replay_last_index": replay_last_index,
            "replay_last_vector7": replay_last_vector7,
            "replay_last_joint_target_abs_error": replay_last_target_error,
            "replay_max_joint_target_abs_error": replay_max_joint_target_abs_error,
            "replay_mean_joint_target_abs_error": (
                replay_mean_joint_target_abs_error_sum / replay_joint_error_samples
                if replay_joint_error_samples
                else None
            ),
            "replay_payload_summary": (
                {
                    "format": replay["payload"].get("format"),
                    "repo_id": replay["payload"].get("repo_id"),
                    "dataset_root": replay["payload"].get("dataset_root"),
                    "episode_index": replay["payload"].get("episode_index"),
                    "episode_length": replay["payload"].get("episode_length"),
                    "fps": replay["payload"].get("fps"),
                    "task": replay["payload"].get("task"),
                    **replay["state_summary"],
                    **replay["action_summary"],
                    "gripper_clip_count_state": replay["gripper_clip_count_state"],
                    "gripper_clip_count_action": replay["gripper_clip_count_action"],
                }
                if replay is not None
                else None
            ),
            "task_text": args.task_text,
            "target_initial_override": target_initial_override_position,
            "target_metrics": final_target_metrics,
            "end_effector_metrics": {
                "initial_world_position": end_effector_initial_position,
                "final_world_position": world_position(end_effector_path),
                "initial_distance_to_target_m": distance_m(end_effector_initial_position, target_initial_position),
                "final_distance_to_target_m": final_ee_to_target_distance_m,
                "min_distance_to_target_m": min_ee_to_target_distance_m,
            },
            "smolvla_model_id": str(Path(args.smolvla_model_id).expanduser()),
            "smolvla_device": args.smolvla_device,
            "dataset_root": str(Path(args.dataset_root).expanduser()),
            "vlm_model_path": str(Path(args.vlm_model_path).expanduser()) if args.vlm_model_path else None,
            "helper_ready": helper_ready_payload,
            "stage_summary": stage_summary,
            "records": records,
        }
        write_json(output_dir / "summary.json", summary)
        log(f"[Roboclaw] Summary written: {output_dir / 'summary.json'}")
        log(
            f"[Roboclaw] Target displacement xy={final_target_metrics['xy_displacement_m']}m "
            f"threshold={final_target_metrics['success_displacement_threshold_m']}m success={final_target_metrics['success']}"
        )
        if args.keep_open:
            log("[Roboclaw] --keep-open is set; Isaac Sim will remain open until you close the window or press Ctrl+C.")
            try:
                while simulation_app.is_running():
                    world.step(render=True)
            except KeyboardInterrupt:
                log("[Roboclaw] Closing after Ctrl+C.")
        omni.timeline.get_timeline_interface().stop()
        simulation_app.close()


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        log(f"[Roboclaw][ERROR] {exc}")
        simulation_app.close()
        raise
