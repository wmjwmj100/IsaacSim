from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import struct
import time
import zlib
from pathlib import Path
from typing import Any

from isaacsim import SimulationApp


REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_USD_PATH = "/home/wmj/Documents/roboclaw.usd"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "outputs/roboclaw_sim_teacher_raw"
DEFAULT_TARGET_PRIM_PATH = "/World/Table/TargetCube"
CAMERA_PATHS = {
    "overhead": "/World/RealSenseD435i/DepthCamera",
    "left_wrist": "/Panthera_HT_FrontLeft/link6/WristRGBCamera",
    "right_wrist": "/Panthera_HT_FrontRight/link6/WristRGBCamera",
}
ARTICULATION_PATHS = {
    "front_left": "/Panthera_HT_FrontLeft/root_joint",
    "front_right": "/Panthera_HT_FrontRight/root_joint",
}
FINGER_PATHS = {
    "front_left": (
        "/Panthera_HT_FrontLeft/L_finger",
        "/Panthera_HT_FrontLeft/R_finger",
    ),
    "front_right": (
        "/Panthera_HT_FrontRight/L_finger",
        "/Panthera_HT_FrontRight/R_finger",
    ),
}
HOME_ACTION7 = [
    -0.6163809895515442,
    1.063742995262146,
    0.38138899207115173,
    0.35688498616218567,
    -0.0031419999431818724,
    -0.02010600082576275,
    0.018,
]
JOINT_LOWER = [-1.8, -0.15, -0.35, -1.45, -1.15, -1.25]
JOINT_UPPER = [0.45, 2.65, 2.05, 1.65, 1.20, 1.35]
CUBE_SIZE_M = [0.10, 0.055, 0.045]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Collect Isaac Sim teacher demonstrations for Roboclaw using differential IK."
    )
    parser.add_argument("--usd-path", default=os.getenv("ROBOCLAW_USD_PATH", DEFAULT_USD_PATH))
    parser.add_argument("--output-dir", default=os.getenv("ROBOCLAW_TEACHER_OUTPUT_DIR", str(DEFAULT_OUTPUT_DIR)))
    parser.add_argument("--run-name", default=os.getenv("ROBOCLAW_TEACHER_RUN_NAME", ""))
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--headless", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--renderer", default=os.getenv("ROBOCLAW_RENDERER", "RaytracedLighting"))
    parser.add_argument("--episodes", type=int, default=int(os.getenv("ROBOCLAW_TEACHER_EPISODES", "2")))
    parser.add_argument(
        "--frames-per-episode",
        type=int,
        default=int(os.getenv("ROBOCLAW_TEACHER_FRAMES_PER_EPISODE", "150")),
    )
    parser.add_argument("--fps", type=int, default=int(os.getenv("ROBOCLAW_TEACHER_FPS", "30")))
    parser.add_argument("--seed", type=int, default=int(os.getenv("ROBOCLAW_TEACHER_SEED", "1000")))
    parser.add_argument("--warmup-frames", type=int, default=int(os.getenv("ROBOCLAW_TEACHER_WARMUP_FRAMES", "24")))
    parser.add_argument("--settle-frames", type=int, default=int(os.getenv("ROBOCLAW_TEACHER_SETTLE_FRAMES", "12")))
    parser.add_argument("--active-arm-label", choices=("front_left", "front_right"), default="front_left")
    parser.add_argument("--task-text", default="push the object on the table")
    parser.add_argument("--target-prim-path", default=DEFAULT_TARGET_PRIM_PATH)
    parser.add_argument("--cube-center-x", type=float, default=-0.18)
    parser.add_argument("--cube-center-y", type=float, default=0.135)
    parser.add_argument("--cube-range-x", type=float, default=0.045)
    parser.add_argument("--cube-range-y", type=float, default=0.035)
    parser.add_argument("--success-displacement-threshold", type=float, default=0.035)
    parser.add_argument("--push-distance", type=float, default=0.12)
    parser.add_argument("--approach-distance", type=float, default=0.16)
    parser.add_argument("--contact-margin", type=float, default=0.035)
    parser.add_argument("--contact-z-offset", type=float, default=0.018)
    parser.add_argument("--hover-z-offset", type=float, default=0.16)
    parser.add_argument("--ik-damping", type=float, default=0.08)
    parser.add_argument("--ik-gain", type=float, default=0.85)
    parser.add_argument("--max-target-step-m", type=float, default=0.045)
    parser.add_argument("--max-joint-step-rad", type=float, default=0.075)
    parser.add_argument("--apply-mode", choices=("target", "teleport"), default="target")
    parser.add_argument("--save-every", type=int, default=1)
    parser.add_argument("--overhead-camera-path", default=CAMERA_PATHS["overhead"])
    parser.add_argument("--left-wrist-camera-path", default=CAMERA_PATHS["left_wrist"])
    parser.add_argument("--right-wrist-camera-path", default=CAMERA_PATHS["right_wrist"])
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
from isaacsim.core.utils.viewports import set_camera_view
from isaacsim.core.utils.xforms import get_world_pose
from isaacsim.sensors.camera import Camera
from pxr import UsdGeom, UsdPhysics


def log(message: str) -> None:
    print(message, flush=True)
    carb.log_info(message)


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


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(json_ready(payload), ensure_ascii=False, indent=2, sort_keys=True) + "\n")


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        for record in records:
            file.write(json.dumps(json_ready(record), ensure_ascii=False, sort_keys=True) + "\n")


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


def require_prim(path: str, label: str) -> None:
    if not is_prim_path_valid(path):
        raise RuntimeError(f"{label} prim not found at {path}")


def build_camera(name: str, prim_path: str) -> Camera:
    require_prim(prim_path, f"{name} camera")
    return Camera(prim_path=prim_path, name=f"roboclaw_teacher_{name}", resolution=(640, 480), annotator_device="cpu")


def active_articulation_path() -> str:
    return ARTICULATION_PATHS[args.active_arm_label]


def active_wrist_name() -> str:
    return "left_wrist" if args.active_arm_label == "front_left" else "right_wrist"


def active_wrist_camera_path() -> str:
    return args.left_wrist_camera_path if args.active_arm_label == "front_left" else args.right_wrist_camera_path


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


def action_target_for_dofs(dof_names: list[str], action7: list[float]) -> np.ndarray:
    gripper = float(np.clip(action7[6], 0.0, 0.04))
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


def world_position(prim_path: str) -> np.ndarray:
    position, _orientation = get_world_pose(prim_path)
    return np.asarray(position, dtype=np.float64).reshape(-1)[:3]


def finger_midpoint() -> np.ndarray:
    left_path, right_path = FINGER_PATHS[args.active_arm_label]
    return 0.5 * (world_position(left_path) + world_position(right_path))


def xy_displacement(initial_position: np.ndarray, final_position: np.ndarray) -> float:
    return float(np.linalg.norm(np.asarray(final_position[:2]) - np.asarray(initial_position[:2])))


def summarize_stage_prims() -> dict[str, Any]:
    stage = get_current_stage()
    camera_prims = []
    articulation_roots = []
    for prim in stage.Traverse():
        if prim.IsA(UsdGeom.Camera):
            camera_prims.append(str(prim.GetPath()))
        if prim.HasAPI(UsdPhysics.ArticulationRootAPI):
            articulation_roots.append(str(prim.GetPath()))
    return {"camera_prims": camera_prims, "articulation_roots": articulation_roots}


def solve_position_ik(
    articulation: Articulation,
    goal_position: np.ndarray,
    gripper: float,
) -> tuple[list[float], dict[str, Any]]:
    current_position = finger_midpoint()
    error = np.asarray(goal_position, dtype=np.float64) - current_position
    error_norm = float(np.linalg.norm(error))
    max_target_step = max(1e-6, float(args.max_target_step_m))
    clipped_error = error.copy()
    if error_norm > max_target_step:
        clipped_error *= max_target_step / error_norm

    jacobians = np.asarray(articulation.get_jacobians(), dtype=np.float64)
    left_index = articulation.get_link_index("L_finger") - 1
    right_index = articulation.get_link_index("R_finger") - 1
    j_left = jacobians[0, left_index, :3, :6]
    j_right = jacobians[0, right_index, :3, :6]
    jacobian = 0.5 * (j_left + j_right)
    damping = float(args.ik_damping)
    regularizer = np.eye(3, dtype=np.float64) * (damping**2)
    delta = jacobian.T @ np.linalg.solve(jacobian @ jacobian.T + regularizer, clipped_error)
    delta *= float(args.ik_gain)
    max_joint_step = max(1e-6, float(args.max_joint_step_rad))
    delta = np.clip(delta, -max_joint_step, max_joint_step)

    current_joints = np.asarray(articulation.get_joint_positions(), dtype=np.float64).reshape(-1)
    target6 = current_joints[:6] + delta
    target6 = np.clip(target6, np.asarray(JOINT_LOWER, dtype=np.float64), np.asarray(JOINT_UPPER, dtype=np.float64))
    action7 = [float(value) for value in target6.tolist()] + [float(np.clip(gripper, 0.0, 0.04))]
    diagnostics = {
        "finger_goal_position": goal_position,
        "finger_current_position": current_position,
        "finger_error_m": error,
        "finger_error_norm_m": error_norm,
        "finger_error_clipped_m": clipped_error,
        "ik_delta6": delta,
    }
    return action7, diagnostics


def smoothstep(value: float) -> float:
    value = float(np.clip(value, 0.0, 1.0))
    return value * value * (3.0 - 2.0 * value)


def lerp(a: np.ndarray, b: np.ndarray, t: float) -> np.ndarray:
    return (1.0 - t) * a + t * b


def teacher_goal(
    frame_index: int,
    frames_per_episode: int,
    cube_initial: np.ndarray,
    push_direction_xy: np.ndarray,
) -> tuple[np.ndarray, str, float]:
    progress = frame_index / max(1, frames_per_episode - 1)
    cube_half_along_push = CUBE_SIZE_M[1] * 0.5
    contact_distance = cube_half_along_push + float(args.contact_margin)
    approach_distance = float(args.approach_distance)
    push_distance = float(args.push_distance)
    contact_z = cube_initial[2] + float(args.contact_z_offset)
    hover_z = cube_initial[2] + float(args.hover_z_offset)

    push3 = np.array([push_direction_xy[0], push_direction_xy[1], 0.0], dtype=np.float64)
    approach_high = cube_initial - push3 * approach_distance
    approach_high[2] = hover_z
    approach_low = cube_initial - push3 * approach_distance
    approach_low[2] = contact_z
    contact = cube_initial - push3 * contact_distance
    contact[2] = contact_z
    push_end = cube_initial + push3 * push_distance
    push_end[2] = contact_z
    lift = push_end.copy()
    lift[2] = hover_z

    if progress < 0.18:
        phase = "approach_high"
        goal = lerp(finger_midpoint(), approach_high, smoothstep(progress / 0.18))
    elif progress < 0.34:
        phase = "descend"
        goal = lerp(approach_high, approach_low, smoothstep((progress - 0.18) / 0.16))
    elif progress < 0.48:
        phase = "contact"
        goal = lerp(approach_low, contact, smoothstep((progress - 0.34) / 0.14))
    elif progress < 0.84:
        phase = "push"
        goal = lerp(contact, push_end, smoothstep((progress - 0.48) / 0.36))
    else:
        phase = "lift"
        goal = lerp(push_end, lift, smoothstep((progress - 0.84) / 0.16))

    gripper = 0.018 + 0.003 * math.sin(2.0 * math.pi * progress)
    return goal, phase, gripper


def prepare_output_dir() -> Path:
    root = Path(args.output_dir).expanduser()
    run_name = args.run_name.strip() or time.strftime("%Y%m%d_%H%M%S")
    output_dir = root / run_name
    if output_dir.exists():
        if not args.overwrite:
            raise RuntimeError(f"Output directory exists; pass --overwrite or choose another --run-name: {output_dir}")
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def reset_episode(
    world: World,
    articulation: Articulation,
    target: SingleRigidPrim,
    cube_position: np.ndarray,
) -> None:
    world.reset()
    world.play()
    home = action_target_for_dofs(articulation.dof_names, HOME_ACTION7)
    articulation.set_joint_positions(home)
    articulation.set_joint_position_targets(home)
    target.set_world_pose(position=cube_position, orientation=np.array([1.0, 0.0, 0.0, 0.0]))
    target.set_linear_velocity(np.zeros(3, dtype=np.float32))
    target.set_angular_velocity(np.zeros(3, dtype=np.float32))
    for _ in range(max(1, int(args.settle_frames))):
        world.step(render=True)


def run_collection() -> dict[str, Any]:
    output_dir = prepare_output_dir()
    usd_path = str(Path(args.usd_path).expanduser())
    if not Path(usd_path).exists():
        raise FileNotFoundError(f"USD path does not exist: {usd_path}")

    log(f"[Roboclaw][Teacher] Opening USD: {usd_path}")
    if not open_stage(usd_path):
        raise RuntimeError(f"Failed to open USD stage: {usd_path}")
    for _ in range(2):
        simulation_app.update()
    while is_stage_loading():
        simulation_app.update()
    if not args.headless:
        set_camera_view(eye=[1.25, 1.1, 1.15], target=[0.0, 0.0, 0.78], camera_prim_path="/OmniverseKit_Persp")

    articulation_path = active_articulation_path()
    wrist_name = active_wrist_name()
    wrist_camera_path = active_wrist_camera_path()
    required = {
        "articulation": articulation_path,
        "target": args.target_prim_path,
        "overhead_camera": args.overhead_camera_path,
        "wrist_camera": wrist_camera_path,
        "left_finger": FINGER_PATHS[args.active_arm_label][0],
        "right_finger": FINGER_PATHS[args.active_arm_label][1],
    }
    for label, prim_path in required.items():
        require_prim(prim_path, label)

    world = World(stage_units_in_meters=1.0, physics_prim_path="/physicsScene")
    articulation = world.scene.add(Articulation(prim_paths_expr=articulation_path, name="roboclaw_teacher_arm"))
    world.reset()
    world.play()
    target = SingleRigidPrim(prim_path=args.target_prim_path, name="roboclaw_teacher_target", reset_xform_properties=False)
    target.initialize()

    cameras = {
        "camera1": build_camera("overhead", args.overhead_camera_path),
        "camera3": build_camera(wrist_name, wrist_camera_path),
    }
    for camera in cameras.values():
        camera.initialize()
    for _ in range(max(1, int(args.warmup_frames))):
        world.step(render=True)

    rng = np.random.default_rng(int(args.seed))
    stage_summary = summarize_stage_prims()
    initial_cube_position = world_position(args.target_prim_path)
    cube_z = float(initial_cube_position[2])
    records_total = 0
    episode_summaries = []

    for episode_index in range(int(args.episodes)):
        cube_position = np.array(
            [
                float(args.cube_center_x) + rng.uniform(-float(args.cube_range_x), float(args.cube_range_x)),
                float(args.cube_center_y) + rng.uniform(-float(args.cube_range_y), float(args.cube_range_y)),
                cube_z,
            ],
            dtype=np.float64,
        )
        direction_angle = rng.uniform(math.radians(-12.0), math.radians(12.0))
        push_direction = np.array([math.sin(direction_angle), math.cos(direction_angle)], dtype=np.float64)
        push_direction /= np.linalg.norm(push_direction)
        reset_episode(world, articulation, target, cube_position)

        episode_dir = output_dir / f"episode_{episode_index:06d}"
        frame_dir = episode_dir / "frames"
        frame_dir.mkdir(parents=True, exist_ok=True)
        episode_records: list[dict[str, Any]] = []
        episode_start_cube = world_position(args.target_prim_path)
        started_at = time.time()
        log(
            f"[Roboclaw][Teacher] episode={episode_index} cube_start={episode_start_cube.tolist()} "
            f"push_direction={push_direction.tolist()}"
        )

        for frame_index in range(int(args.frames_per_episode)):
            if not simulation_app.is_running():
                raise RuntimeError("Simulation app stopped during collection")
            current_state7 = state7(articulation)
            current_cube = world_position(args.target_prim_path)
            goal, phase, gripper = teacher_goal(frame_index, int(args.frames_per_episode), episode_start_cube, push_direction)
            action7, ik_info = solve_position_ik(articulation, goal, gripper)

            should_save = frame_index % max(1, int(args.save_every)) == 0
            camera1_path = frame_dir / f"frame_{frame_index:06d}_camera1_overhead.png"
            camera3_path = frame_dir / f"frame_{frame_index:06d}_camera3_{wrist_name}.png"
            if should_save:
                write_png_rgb(camera1_path, camera_rgb("camera1", cameras["camera1"]))
                write_png_rgb(camera3_path, camera_rgb("camera3", cameras["camera3"]))

            target_pose = action_target_for_dofs(articulation.dof_names, action7)
            if args.apply_mode == "teleport":
                articulation.set_joint_positions(target_pose)
            else:
                articulation.set_joint_position_targets(target_pose)

            record = {
                "episode_index": episode_index,
                "frame_index": frame_index,
                "timestamp": frame_index / float(args.fps),
                "task": args.task_text,
                "state7": current_state7,
                "action7": action7,
                "camera1_path": str(camera1_path),
                "camera3_path": str(camera3_path),
                "camera1_source": args.overhead_camera_path,
                "camera3_source": wrist_camera_path,
                "active_arm_label": args.active_arm_label,
                "teacher_phase": phase,
                "teacher_goal_finger_position": ik_info["finger_goal_position"],
                "teacher_current_finger_position": ik_info["finger_current_position"],
                "teacher_finger_error_norm_m": ik_info["finger_error_norm_m"],
                "cube_world_position": current_cube,
                "cube_xy_displacement_m": xy_displacement(episode_start_cube, current_cube),
                "push_direction_xy": push_direction,
                "saved": bool(should_save),
            }
            episode_records.append(record)
            records_total += 1 if should_save else 0
            world.step(render=True)

        final_cube = world_position(args.target_prim_path)
        final_displacement = xy_displacement(episode_start_cube, final_cube)
        success = bool(final_displacement >= float(args.success_displacement_threshold))
        episode_summary = {
            "episode_index": episode_index,
            "frames": int(args.frames_per_episode),
            "saved_frames": sum(1 for record in episode_records if record["saved"]),
            "duration_s": time.time() - started_at,
            "cube_initial_position": episode_start_cube,
            "cube_final_position": final_cube,
            "cube_xy_displacement_m": final_displacement,
            "success_displacement_threshold_m": float(args.success_displacement_threshold),
            "success": success,
            "push_direction_xy": push_direction,
            "mean_teacher_finger_error_norm_m": float(
                np.mean([record["teacher_finger_error_norm_m"] for record in episode_records])
            ),
            "max_teacher_finger_error_norm_m": float(
                np.max([record["teacher_finger_error_norm_m"] for record in episode_records])
            ),
        }
        write_jsonl(episode_dir / "records.jsonl", episode_records)
        write_json(episode_dir / "summary.json", episode_summary)
        episode_summaries.append(episode_summary)
        log(
            f"[Roboclaw][Teacher] episode={episode_index} displacement_xy={final_displacement:.4f}m "
            f"success={str(success).lower()} summary={episode_dir / 'summary.json'}"
        )

    successful = [summary for summary in episode_summaries if summary["success"]]
    summary = {
        "schema": "roboclaw_sim_teacher_raw_v1",
        "usd_path": usd_path,
        "output_dir": output_dir,
        "active_arm_label": args.active_arm_label,
        "active_articulation_path": articulation_path,
        "target_prim_path": args.target_prim_path,
        "camera_paths": {"camera1": args.overhead_camera_path, "camera3": wrist_camera_path},
        "task": args.task_text,
        "fps": int(args.fps),
        "episodes": int(args.episodes),
        "frames_per_episode": int(args.frames_per_episode),
        "saved_frame_count": int(records_total),
        "success_count": len(successful),
        "success_rate": len(successful) / max(1, len(episode_summaries)),
        "episode_summaries": episode_summaries,
        "settings": vars(args),
        "stage_summary": stage_summary,
    }
    write_json(output_dir / "summary.json", summary)
    log(f"[Roboclaw][Teacher] Collection summary written: {output_dir / 'summary.json'}")
    log(
        f"[Roboclaw][Teacher] success_count={len(successful)}/{len(episode_summaries)} "
        f"saved_frames={records_total} output_dir={output_dir}"
    )
    return summary


def main() -> None:
    try:
        run_collection()
    finally:
        omni.timeline.get_timeline_interface().stop()
        simulation_app.close()


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        log(f"[Roboclaw][Teacher][ERROR] {exc}")
        simulation_app.close()
        raise
