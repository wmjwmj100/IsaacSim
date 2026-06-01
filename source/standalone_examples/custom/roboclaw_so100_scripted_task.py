from __future__ import annotations

import argparse
import math
import time
from pathlib import Path

from isaacsim import SimulationApp


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a visible scripted SO100 block-push task in roboclaw.usd.")
    parser.add_argument("--usd-path", default="/home/wmj/Documents/roboclaw.usd")
    parser.add_argument("--renderer", default="RaytracedLighting")
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--target-prim-path", default="/World/Table/TargetCube")
    parser.add_argument("--replacement-urdf-path", default=str(Path.home() / ".cache/robot_descriptions/SO-ARM100/Simulation/SO100/so100.urdf"))
    parser.add_argument("--hide-original-arm", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--frames", type=int, default=420)
    parser.add_argument("--warmup-frames", type=int, default=24)
    parser.add_argument("--push-distance", type=float, default=0.10)
    parser.add_argument("--robot-offset-y", type=float, default=-0.22)
    parser.add_argument("--robot-offset-x", type=float, default=0.0)
    parser.add_argument("--keep-open", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


args = parse_args()
simulation_app = SimulationApp({"headless": args.headless, "renderer": args.renderer})

import carb
import numpy as np
import omni.kit.app
import omni.kit.commands
import omni.timeline
from isaacsim.core.api import World
from isaacsim.core.prims import Articulation, SingleRigidPrim
from isaacsim.core.utils.prims import is_prim_path_valid
from isaacsim.core.utils.stage import get_current_stage, is_stage_loading, open_stage
from isaacsim.core.utils.viewports import set_active_viewport_camera, set_camera_view
from isaacsim.core.utils.xforms import get_world_pose
from pxr import Gf, Usd, UsdGeom, UsdPhysics


ORIGINAL_ARM_ROOT = "/Panthera_HT_FrontLeft"
ORIGINAL_ARM_ARTICULATION = "/Panthera_HT_FrontLeft/root_joint"
SO100_LOWER = np.array([-2.0, 0.0, -3.14158, -2.5, -3.14158, -0.2], dtype=np.float64)
SO100_UPPER = np.array([2.0, 3.5, 0.0, 1.2, 3.14158, 2.0], dtype=np.float64)
SO100_HOME = np.array([0.0, 1.25, -1.55, -0.55, 0.0, 0.65], dtype=np.float64)


def log(message: str) -> None:
    print(message, flush=True)
    carb.log_info(message)


def wait_for_stage_load() -> None:
    for _ in range(2):
        simulation_app.update()
    while is_stage_loading():
        simulation_app.update()


def require_prim(path: str, label: str) -> None:
    if not is_prim_path_valid(path):
        raise RuntimeError(f"{label} prim not found at {path}")


def world_position(prim_path: str) -> np.ndarray:
    position, _orientation = get_world_pose(prim_path)
    return np.asarray(position, dtype=np.float64).reshape(-1)[:3]


def ensure_urdf_importer_enabled() -> None:
    ext_name = "isaacsim.asset.importer.urdf"
    ext_manager = omni.kit.app.get_app().get_extension_manager()
    if not ext_manager.is_extension_enabled(ext_name):
        ext_manager.set_extension_enabled_immediate(ext_name, True)
        simulation_app.update()


def sanitize_so100_urdf_for_isaac(urdf_path: str) -> str:
    source = Path(urdf_path).expanduser()
    content = source.read_text(encoding="utf-8")
    patched = content
    for i in range(1, 7):
        patched = patched.replace(f'<joint name="{i}" type="revolute">', f'<joint name="joint_{i}" type="revolute">')
        patched = patched.replace(f'<joint name="{i}">', f'<joint name="joint_{i}">')
        patched = patched.replace(f'<transmission name="{i}_trans">', f'<transmission name="joint_{i}_trans">')
    output = source.with_name(source.stem + "_isaacsim.urdf")
    if not output.exists() or output.read_text(encoding="utf-8") != patched:
        output.write_text(patched, encoding="utf-8")
    return str(output)


def import_so100_to_stage(urdf_path: str) -> str:
    ensure_urdf_importer_enabled()
    status, import_config = omni.kit.commands.execute("URDFCreateImportConfig")
    if not status:
        raise RuntimeError("Failed to create URDF import config")
    import_config.merge_fixed_joints = False
    import_config.fix_base = True
    import_config.make_default_prim = False
    import_config.create_physics_scene = False
    import_config.convex_decomp = False
    import_config.import_inertia_tensor = True
    import_config.distance_scale = 1.0
    status, articulation_path = omni.kit.commands.execute(
        "URDFParseAndImportFile",
        urdf_path=urdf_path,
        import_config=import_config,
        get_articulation_root=True,
    )
    if not status or not articulation_path:
        raise RuntimeError(f"Failed to import SO100 URDF: {urdf_path}")
    return str(articulation_path)


def set_prim_translation(prim_path: str, xyz: np.ndarray) -> None:
    stage = get_current_stage()
    prim = stage.GetPrimAtPath(prim_path)
    if not prim or not prim.IsValid():
        raise RuntimeError(f"Prim not found: {prim_path}")
    xform = UsdGeom.Xformable(prim)
    xform.ClearXformOpOrder()
    xform.AddTranslateOp(UsdGeom.XformOp.PrecisionDouble).Set(Gf.Vec3d(float(xyz[0]), float(xyz[1]), float(xyz[2])))


def prim_world_bbox(prim_path: str) -> dict[str, list[float]] | None:
    stage = get_current_stage()
    prim = stage.GetPrimAtPath(prim_path)
    if not prim or not prim.IsValid():
        return None
    bbox_cache = UsdGeom.BBoxCache(
        Usd.TimeCode.Default(),
        [UsdGeom.Tokens.default_, UsdGeom.Tokens.render, UsdGeom.Tokens.proxy],
        useExtentsHint=True,
    )
    aligned = bbox_cache.ComputeWorldBound(prim).ComputeAlignedBox()
    if aligned.IsEmpty():
        return None
    minimum = aligned.GetMin()
    maximum = aligned.GetMax()
    center = [(float(minimum[i]) + float(maximum[i])) * 0.5 for i in range(3)]
    size = [float(maximum[i]) - float(minimum[i]) for i in range(3)]
    return {
        "min": [float(minimum[i]) for i in range(3)],
        "max": [float(maximum[i]) for i in range(3)],
        "center": center,
        "size": size,
    }


def hide_original_arm() -> None:
    stage = get_current_stage()
    prim = stage.GetPrimAtPath(ORIGINAL_ARM_ROOT)
    if prim and prim.IsValid():
        prim.SetActive(False)


def find_end_effector_path(root_path: str) -> str:
    stage = get_current_stage()
    root_prefix = root_path.rstrip("/") + "/"
    candidates: list[str] = []
    for prim in stage.Traverse():
        path = str(prim.GetPath())
        if path.startswith(root_prefix) and any(token in path.lower() for token in ("jaw", "gripper", "wrist")):
            candidates.append(path)
    return candidates[-1] if candidates else root_path


def clipped_target(q: np.ndarray) -> np.ndarray:
    return np.clip(np.asarray(q, dtype=np.float64), SO100_LOWER, SO100_UPPER)


def set_robot_q(robot: Articulation, q: np.ndarray) -> None:
    target = clipped_target(q).astype(np.float32).reshape(1, -1)
    robot.set_joint_positions(target)
    robot.set_joint_position_targets(target)


def smoothstep(value: float) -> float:
    value = float(np.clip(value, 0.0, 1.0))
    return value * value * (3.0 - 2.0 * value)


def lerp(a: np.ndarray, b: np.ndarray, t: float) -> np.ndarray:
    return (1.0 - t) * a + t * b


def solve_position_ik(robot: Articulation, ee_path: str, goal: np.ndarray) -> tuple[np.ndarray, float]:
    current = world_position(ee_path)
    error = np.asarray(goal, dtype=np.float64) - current
    error_norm = float(np.linalg.norm(error))
    if error_norm < 1e-4:
        return np.asarray(robot.get_joint_positions(), dtype=np.float64).reshape(-1)[:6], error_norm
    max_step = 0.035
    if error_norm > max_step:
        error *= max_step / error_norm
    jacobians = np.asarray(robot.get_jacobians(), dtype=np.float64)
    link_name = ee_path.rstrip("/").split("/")[-2] if ee_path.endswith("/collisions") else ee_path.rstrip("/").split("/")[-1]
    link_index = robot.get_link_index(link_name)
    j = jacobians[0, max(0, int(link_index) - 1), :3, :6]
    damping = 0.06
    delta = j.T @ np.linalg.solve(j @ j.T + np.eye(3) * (damping**2), error)
    delta = np.clip(delta * 0.85, -0.08, 0.08)
    current_q = np.asarray(robot.get_joint_positions(), dtype=np.float64).reshape(-1)[:6]
    return clipped_target(current_q + delta), error_norm


def scripted_goal(frame: int, total: int, cube_start: np.ndarray) -> tuple[str, np.ndarray]:
    progress = frame / max(1, total - 1)
    contact_z = cube_start[2] + 0.04
    hover_z = cube_start[2] + 0.22
    behind = np.array([cube_start[0] + 0.12, cube_start[1], contact_z], dtype=np.float64)
    behind_high = behind.copy()
    behind_high[2] = hover_z
    contact = np.array([cube_start[0] + 0.055, cube_start[1], contact_z], dtype=np.float64)
    push_end = np.array([cube_start[0] - float(args.push_distance), cube_start[1], contact_z], dtype=np.float64)
    retreat = push_end.copy()
    retreat[2] = hover_z

    if progress < 0.18:
        return "home", behind_high
    if progress < 0.34:
        return "descend", lerp(behind_high, behind, smoothstep((progress - 0.18) / 0.16))
    if progress < 0.48:
        return "contact", lerp(behind, contact, smoothstep((progress - 0.34) / 0.14))
    if progress < 0.84:
        return "push_left_10cm", lerp(contact, push_end, smoothstep((progress - 0.48) / 0.36))
    return "retreat", lerp(push_end, retreat, smoothstep((progress - 0.84) / 0.16))


def run() -> None:
    usd_path = str(Path(args.usd_path).expanduser())
    log(f"[ScriptedSO100] Opening USD: {usd_path}")
    if not open_stage(usd_path):
        raise RuntimeError(f"Failed to open USD: {usd_path}")
    wait_for_stage_load()
    require_prim(args.target_prim_path, "target cube")
    require_prim(ORIGINAL_ARM_ARTICULATION, "original active arm")

    cube_initial = world_position(args.target_prim_path)
    root_pos = np.array(
        [cube_initial[0] + float(args.robot_offset_x), cube_initial[1] + float(args.robot_offset_y), cube_initial[2] + 0.0025],
        dtype=np.float64,
    )
    articulation_path = import_so100_to_stage(sanitize_so100_urdf_for_isaac(args.replacement_urdf_path))
    root_path = articulation_path.rsplit("/", 1)[0]
    set_prim_translation(root_path, root_pos)
    if args.hide_original_arm:
        hide_original_arm()
    ee_path = find_end_effector_path(root_path)
    bbox = prim_world_bbox(root_path)
    view_target = bbox["center"] if bbox else root_pos.tolist()
    set_camera_view(
        eye=[view_target[0] + 0.45, view_target[1] - 0.75, view_target[2] + 0.45],
        target=view_target,
        camera_prim_path="/OmniverseKit_Persp",
    )
    try:
        set_active_viewport_camera("/OmniverseKit_Persp")
    except Exception:
        pass

    world = World(stage_units_in_meters=1.0, physics_prim_path="/physicsScene")
    robot = world.scene.add(Articulation(prim_paths_expr=articulation_path, name="scripted_so100"))
    cube = SingleRigidPrim(prim_path=args.target_prim_path, name="scripted_target_cube", reset_xform_properties=False)
    world.reset()
    world.play()
    cube.initialize()
    set_robot_q(robot, SO100_HOME)
    for _ in range(max(1, int(args.warmup_frames))):
        world.step(render=True)

    log(
        "[ScriptedSO100] Task: move target cube left by 0.10m. "
        f"cube_start={cube_initial.tolist()} robot_root={root_pos.tolist()} ee_path={ee_path} bbox={bbox}"
    )
    last_phase = ""
    for frame in range(int(args.frames)):
        if not simulation_app.is_running():
            break
        phase, goal = scripted_goal(frame, int(args.frames), cube_initial)
        q, err = solve_position_ik(robot, ee_path, goal)
        set_robot_q(robot, q)
        if phase != last_phase or frame % 30 == 0:
            cube_now = world_position(args.target_prim_path)
            displacement_xy = float(np.linalg.norm(cube_now[:2] - cube_initial[:2]))
            log(
                f"[ScriptedSO100] frame={frame} phase={phase} goal={goal.tolist()} "
                f"ee={world_position(ee_path).tolist()} err={err:.4f} "
                f"cube={cube_now.tolist()} cube_displacement_xy={displacement_xy:.4f}"
            )
            last_phase = phase
        world.step(render=True)

    cube_final = world_position(args.target_prim_path)
    displacement = float(np.linalg.norm(cube_final[:2] - cube_initial[:2]))
    success = displacement >= 0.05
    log(
        f"[ScriptedSO100] Finished cube_start={cube_initial.tolist()} cube_final={cube_final.tolist()} "
        f"xy_displacement={displacement:.4f} success={success}"
    )
    if args.keep_open:
        log("[ScriptedSO100] Keeping GUI open.")
        while simulation_app.is_running():
            world.step(render=True)


if __name__ == "__main__":
    try:
        run()
    finally:
        omni.timeline.get_timeline_interface().stop()
        simulation_app.close()
