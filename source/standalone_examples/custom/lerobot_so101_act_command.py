# SPDX-FileCopyrightText: Copyright (c) 2020-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from isaacsim import SimulationApp

simulation_app = SimulationApp({"headless": False, "renderer": "RaytracedLighting"})

import argparse
from dataclasses import dataclass
import json
import math
import os
import time
from pathlib import Path
from typing import Any, Callable

import carb
import numpy as np
import omni.kit.app
import omni.kit.commands
import omni.timeline
import omni.usd
import torch
import torch.nn.functional as F
from isaacsim.core.api import World
from isaacsim.core.api.objects import DynamicCuboid, FixedCuboid
from isaacsim.core.prims import Articulation, XFormPrim
from isaacsim.core.utils.render_product import get_camera_prim_path
from isaacsim.core.utils.viewports import set_camera_view
from isaacsim.sensors.camera import Camera
from lerobot.configs.types import FeatureType, NormalizationMode, PolicyFeature
try:
    from lerobot.utils.constants import ACTION, OBS_ENV_STATE, OBS_STATE
except ImportError:
    from lerobot.constants import ACTION, OBS_ENV_STATE, OBS_STATE
from lerobot.policies.act.configuration_act import ACTConfig
from lerobot.policies.act.modeling_act import ACTPolicy
from pxr import Gf, Sdf, Usd, UsdGeom, UsdLux


def ensure_urdf_importer_enabled() -> None:
    ext_name = "isaacsim.asset.importer.urdf"
    ext_manager = omni.kit.app.get_app().get_extension_manager()
    if not ext_manager.is_extension_enabled(ext_name):
        carb.log_info(f"Enabling extension: {ext_name}")
        ext_manager.set_extension_enabled_immediate(ext_name, True)
        simulation_app.update()


ROBOT_PROFILES = {
    "so100": {
        "label": "SO100",
        "env_keys": ["LEROBOT_SO100_URDF", "LEROBOT_URDF"],
        "robot_description_module": "robot_descriptions.so_arm100_description",
        "fallback_urdf": "~/.cache/robot_descriptions/SO-ARM100/Simulation/SO100/so100.urdf",
        "joint_lower": np.array([-2.0, 0.0, -3.14158, -2.5, -3.14158, -0.2], dtype=np.float32),
        "joint_upper": np.array([2.0, 3.5, 0.0, 1.2, 3.14158, 2.0], dtype=np.float32),
    },
    "so101": {
        "label": "SO101",
        "env_keys": ["LEROBOT_SO101_URDF", "LEROBOT_URDF"],
        "robot_description_module": "robot_descriptions.so_arm101_description",
        "fallback_urdf": "~/.cache/robot_descriptions/SO-ARM100/Simulation/SO101/so101_new_calib.urdf",
        "joint_lower": np.array([-1.91986, -1.74533, -1.74533, -1.65806, -2.79253, -0.174533], dtype=np.float32),
        "joint_upper": np.array([1.91986, 1.74533, 1.5708, 1.65806, 2.79253, 1.74533], dtype=np.float32),
    },
}


def _get_robot_profile(robot_model: str) -> dict[str, Any]:
    key = robot_model.strip().lower()
    if key not in ROBOT_PROFILES:
        raise RuntimeError(f"Unsupported robot-model={robot_model!r}. Choose one of: {', '.join(ROBOT_PROFILES.keys())}")
    return ROBOT_PROFILES[key]


def resolve_robot_urdf_path(robot_model: str) -> str:
    profile = _get_robot_profile(robot_model)
    for env_key in profile["env_keys"]:
        env_path = os.getenv(env_key)
        if not env_path:
            continue
        path = os.path.expanduser(env_path)
        if os.path.isfile(path):
            return path
        raise RuntimeError(f"URDF path from {env_key} does not exist: {path}")

    module_name = profile["robot_description_module"]
    try:
        module = __import__(module_name, fromlist=["URDF_PATH"])
        urdf_path = getattr(module, "URDF_PATH", "")
        if urdf_path and os.path.isfile(urdf_path):
            return urdf_path
    except Exception as exc:
        carb.log_warn(f"Import {module_name} failed: {exc}")

    fallback = os.path.expanduser(profile["fallback_urdf"])
    if os.path.isfile(fallback):
        return fallback

    raise RuntimeError(
        f"Cannot find {profile['label']} URDF. Install with: "
        "_build/linux-x86_64/release/python.sh -m pip install robot_descriptions"
    )


def sanitize_urdf_for_isaac(urdf_path: str) -> str:
    with open(urdf_path, "r", encoding="utf-8") as f:
        content = f.read()

    patched = content
    for i in range(1, 7):
        patched = patched.replace(f'<joint name="{i}" type="revolute">', f'<joint name="joint_{i}" type="revolute">')
        patched = patched.replace(f'<joint name="{i}">', f'<joint name="joint_{i}">')
        patched = patched.replace(f'<transmission name="{i}_trans">', f'<transmission name="joint_{i}_trans">')

    output_path = os.path.join(
        os.path.dirname(urdf_path),
        os.path.splitext(os.path.basename(urdf_path))[0] + "_isaacsim.urdf",
    )
    output_needs_update = True
    if os.path.isfile(output_path):
        with open(output_path, "r", encoding="utf-8") as f:
            output_needs_update = f.read() != patched
    if output_needs_update:
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(patched)
    return output_path


@dataclass(frozen=True)
class TabletopObjectSpec:
    object_name: str
    prim_path: str
    position_xy: tuple[float, float]
    color: tuple[float, float, float]
    size: float = 0.050


@dataclass(frozen=True)
class TargetZoneSpec:
    zone_name: str
    prim_path: str
    center_xy: tuple[float, float]
    color: tuple[float, float, float]
    scale_xy: tuple[float, float] = (0.12, 0.12)


@dataclass(frozen=True)
class TabletopTaskSpec:
    task_id: str
    task_text: str
    object_name: str
    target_zone: str
    base_command: str = "home"


TABLETOP_OBJECT_SPECS: dict[str, TabletopObjectSpec] = {
    "blue_cube": TabletopObjectSpec(
        object_name="blue_cube",
        prim_path="/World/Objects/BlueCube",
        position_xy=(0.36, 0.10),
        color=(0.10, 0.40, 0.95),
    ),
    "red_cube": TabletopObjectSpec(
        object_name="red_cube",
        prim_path="/World/Objects/RedCube",
        position_xy=(0.47, -0.03),
        color=(0.92, 0.18, 0.22),
    ),
    "green_cube": TabletopObjectSpec(
        object_name="green_cube",
        prim_path="/World/Objects/GreenCube",
        position_xy=(0.57, 0.12),
        color=(0.12, 0.70, 0.28),
    ),
}

TARGET_ZONE_SPECS: dict[str, TargetZoneSpec] = {
    "left_zone": TargetZoneSpec(
        zone_name="left_zone",
        prim_path="/World/Targets/LeftZone",
        center_xy=(0.20, 0.25),
        color=(0.95, 0.82, 0.22),
    ),
    "right_zone": TargetZoneSpec(
        zone_name="right_zone",
        prim_path="/World/Targets/RightZone",
        center_xy=(0.66, -0.22),
        color=(0.92, 0.48, 0.14),
    ),
    "front_zone": TargetZoneSpec(
        zone_name="front_zone",
        prim_path="/World/Targets/FrontZone",
        center_xy=(0.70, 0.22),
        color=(0.22, 0.78, 0.95),
    ),
    "home_zone": TargetZoneSpec(
        zone_name="home_zone",
        prim_path="/World/Targets/HomeZone",
        center_xy=(0.19, -0.22),
        color=(0.70, 0.70, 0.72),
    ),
}

TABLETOP_TASK_SPECS: dict[str, TabletopTaskSpec] = {
    "blue_to_left": TabletopTaskSpec(
        task_id="blue_to_left",
        task_text="Pick up the blue cube and place it in the left target zone.",
        object_name="blue_cube",
        target_zone="left_zone",
        base_command="left",
    ),
    "blue_to_right": TabletopTaskSpec(
        task_id="blue_to_right",
        task_text="Pick up the blue cube and place it in the right target zone.",
        object_name="blue_cube",
        target_zone="right_zone",
        base_command="right",
    ),
    "blue_to_front": TabletopTaskSpec(
        task_id="blue_to_front",
        task_text="Pick up the blue cube and place it in the front target zone.",
        object_name="blue_cube",
        target_zone="front_zone",
        base_command="home",
    ),
    "blue_to_home": TabletopTaskSpec(
        task_id="blue_to_home",
        task_text="Pick up the blue cube and return it to the home target zone.",
        object_name="blue_cube",
        target_zone="home_zone",
        base_command="home",
    ),
    "red_to_left": TabletopTaskSpec(
        task_id="red_to_left",
        task_text="Pick up the red cube and place it in the left target zone.",
        object_name="red_cube",
        target_zone="left_zone",
        base_command="left",
    ),
    "red_to_right": TabletopTaskSpec(
        task_id="red_to_right",
        task_text="Pick up the red cube and place it in the right target zone.",
        object_name="red_cube",
        target_zone="right_zone",
        base_command="right",
    ),
    "red_to_front": TabletopTaskSpec(
        task_id="red_to_front",
        task_text="Pick up the red cube and place it in the front target zone.",
        object_name="red_cube",
        target_zone="front_zone",
        base_command="home",
    ),
    "green_to_left": TabletopTaskSpec(
        task_id="green_to_left",
        task_text="Pick up the green cube and place it in the left target zone.",
        object_name="green_cube",
        target_zone="left_zone",
        base_command="left",
    ),
    "green_to_right": TabletopTaskSpec(
        task_id="green_to_right",
        task_text="Pick up the green cube and place it in the right target zone.",
        object_name="green_cube",
        target_zone="right_zone",
        base_command="right",
    ),
    "green_to_home": TabletopTaskSpec(
        task_id="green_to_home",
        task_text="Pick up the green cube and return it to the home target zone.",
        object_name="green_cube",
        target_zone="home_zone",
        base_command="home",
    ),
}

DEFAULT_TASK_ID = "blue_to_left"


def resolve_tabletop_task_spec(task_id: str) -> TabletopTaskSpec:
    key = task_id.strip().lower()
    if key not in TABLETOP_TASK_SPECS:
        raise RuntimeError(
            f"Unsupported task-id={task_id!r}. Choose one of: {', '.join(TABLETOP_TASK_SPECS.keys())}"
        )
    return TABLETOP_TASK_SPECS[key]


def task_id_from_text(command_text: str) -> str | None:
    if not command_text:
        return None
    text = command_text.strip().lower()
    if text in TABLETOP_TASK_SPECS:
        return text
    for prefix in ("task:", "task=", "任务:", "任务="):
        if text.startswith(prefix):
            candidate = text[len(prefix) :].strip()
            if candidate in TABLETOP_TASK_SPECS:
                return candidate
    return None


def list_tabletop_tasks() -> list[TabletopTaskSpec]:
    return [TABLETOP_TASK_SPECS[key] for key in TABLETOP_TASK_SPECS.keys()]


def _tabletop_object_position(tabletop_z: float, spec: TabletopObjectSpec) -> np.ndarray:
    return np.array(
        [
            float(spec.position_xy[0]),
            float(spec.position_xy[1]),
            tabletop_z + spec.size * 0.5 + 0.003,
        ],
        dtype=np.float32,
    )


def add_tabletop_objects(world: World, tabletop_z: float) -> dict[str, str]:
    object_paths: dict[str, str] = {}
    for spec in TABLETOP_OBJECT_SPECS.values():
        cube_pos = _tabletop_object_position(tabletop_z, spec)
        world.scene.add(
            DynamicCuboid(
                name=spec.object_name,
                prim_path=spec.prim_path,
                position=cube_pos,
                size=float(spec.size),
                color=np.array(spec.color, dtype=np.float32),
            )
        )
        object_paths[spec.object_name] = spec.prim_path
        carb.log_info(
            f"Added task object: {spec.prim_path} at {cube_pos.tolist()}, color={list(spec.color)}, size={spec.size:.3f}m"
        )
    return object_paths


def add_target_zones(world: World, tabletop_z: float) -> dict[str, np.ndarray]:
    zone_centers: dict[str, np.ndarray] = {}
    zone_height = 0.004
    for spec in TARGET_ZONE_SPECS.values():
        zone_center = np.array(
            [
                float(spec.center_xy[0]),
                float(spec.center_xy[1]),
                tabletop_z + zone_height * 0.5 + 0.001,
            ],
            dtype=np.float32,
        )
        world.scene.add(
            FixedCuboid(
                name=spec.zone_name,
                prim_path=spec.prim_path,
                position=zone_center,
                scale=np.array([spec.scale_xy[0], spec.scale_xy[1], zone_height], dtype=np.float32),
                size=1.0,
                color=np.array(spec.color, dtype=np.float32),
            )
        )
        zone_centers[spec.zone_name] = zone_center
        carb.log_info(
            f"Added target zone: {spec.prim_path} at {zone_center.tolist()}, scale={[float(spec.scale_xy[0]), float(spec.scale_xy[1]), zone_height]}"
        )
    return zone_centers


def add_desk_scene(world: World) -> float:
    tabletop_z = 0.75
    top_size = np.array([1.00, 0.80, 0.06])
    top_center = np.array([0.45, 0.00, tabletop_z - top_size[2] / 2.0])

    world.scene.add(
        FixedCuboid(
            prim_path="/World/Desk/Top",
            name="desk_top",
            position=top_center,
            scale=top_size,
            size=1.0,
            color=np.array([0.45, 0.30, 0.20]),
        )
    )

    leg_size = np.array([0.06, 0.06, tabletop_z - top_size[2]])
    leg_z = leg_size[2] / 2.0
    leg_offsets = [
        np.array([0.08, 0.34, leg_z]),
        np.array([0.08, -0.34, leg_z]),
        np.array([0.82, 0.34, leg_z]),
        np.array([0.82, -0.34, leg_z]),
    ]
    for i, leg_pos in enumerate(leg_offsets):
        world.scene.add(
            FixedCuboid(
                prim_path=f"/World/Desk/Leg_{i+1}",
                name=f"desk_leg_{i+1}",
                position=leg_pos,
                scale=leg_size,
                size=1.0,
                color=np.array([0.20, 0.20, 0.20]),
            )
        )

    return tabletop_z


def add_lights() -> None:
    stage = omni.usd.get_context().get_stage()

    dome = UsdLux.DomeLight.Define(stage, Sdf.Path("/World/DomeLight"))
    dome.CreateIntensityAttr(2200.0)
    dome.CreateExposureAttr(0.0)
    dome.CreateColorAttr(Gf.Vec3f(1.0, 0.98, 0.95))

    sun = UsdLux.DistantLight.Define(stage, Sdf.Path("/World/SunLight"))
    sun.CreateIntensityAttr(3500.0)
    sun.CreateAngleAttr(0.53)
    sun_xf = UsdGeom.Xformable(sun.GetPrim())
    sun_xf.AddRotateXYZOp().Set(Gf.Vec3f(-40.0, 20.0, 0.0))


def get_prim_world_transform(prim_path: str) -> Gf.Matrix4d | None:
    stage = omni.usd.get_context().get_stage()
    prim = stage.GetPrimAtPath(prim_path)
    if not prim or not prim.IsValid():
        return None
    return UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(0.0)


def get_prim_world_pos(prim_path: str) -> np.ndarray | None:
    mat = get_prim_world_transform(prim_path)
    if mat is None:
        return None
    t = mat.ExtractTranslation()
    return np.array([float(t[0]), float(t[1]), float(t[2])], dtype=np.float32)


def get_prim_world_aabb(prim_path: str) -> tuple[np.ndarray, np.ndarray] | None:
    stage = omni.usd.get_context().get_stage()
    prim = stage.GetPrimAtPath(prim_path)
    if not prim or not prim.IsValid() or not prim.IsA(UsdGeom.Imageable):
        return None
    bbox_cache = UsdGeom.BBoxCache(
        Usd.TimeCode.Default(),
        [UsdGeom.Tokens.default_, UsdGeom.Tokens.render],
        useExtentsHint=True,
    )
    aligned = bbox_cache.ComputeWorldBound(prim).ComputeAlignedBox()
    min_v = aligned.GetMin()
    max_v = aligned.GetMax()
    aabb_min = np.array([float(min_v[0]), float(min_v[1]), float(min_v[2])], dtype=np.float32)
    aabb_max = np.array([float(max_v[0]), float(max_v[1]), float(max_v[2])], dtype=np.float32)
    if not np.all(np.isfinite(aabb_min)) or not np.all(np.isfinite(aabb_max)):
        return None
    if bool(np.any(aabb_max < aabb_min)):
        return None
    return aabb_min, aabb_max


def aabb_axis_gap(aabb_a: tuple[np.ndarray, np.ndarray] | None, aabb_b: tuple[np.ndarray, np.ndarray] | None) -> np.ndarray | None:
    if aabb_a is None or aabb_b is None:
        return None
    min_a, max_a = aabb_a
    min_b, max_b = aabb_b
    return np.maximum(np.maximum(min_a - max_b, min_b - max_a), 0.0)


def aabb_distance(aabb_a: tuple[np.ndarray, np.ndarray] | None, aabb_b: tuple[np.ndarray, np.ndarray] | None) -> float | None:
    axis_gap = aabb_axis_gap(aabb_a, aabb_b)
    if axis_gap is None:
        return None
    return float(np.linalg.norm(axis_gap))


def aabb_signed_z_gap(aabb_a: tuple[np.ndarray, np.ndarray] | None, aabb_b: tuple[np.ndarray, np.ndarray] | None) -> float | None:
    if aabb_a is None or aabb_b is None:
        return None
    min_a, max_a = aabb_a
    min_b, max_b = aabb_b
    if max_a[2] < min_b[2]:
        return float(max_a[2] - min_b[2])
    if max_b[2] < min_a[2]:
        return float(min_a[2] - max_b[2])
    return 0.0


def aabb_union(*aabbs: tuple[np.ndarray, np.ndarray] | None) -> tuple[np.ndarray, np.ndarray] | None:
    valid = [aabb for aabb in aabbs if aabb is not None]
    if not valid:
        return None
    mins = np.stack([aabb[0] for aabb in valid], axis=0)
    maxs = np.stack([aabb[1] for aabb in valid], axis=0)
    return np.min(mins, axis=0), np.max(maxs, axis=0)


def jsonify_aabb(aabb: tuple[np.ndarray, np.ndarray] | None) -> dict[str, np.ndarray] | None:
    if aabb is None:
        return None
    return {"min": aabb[0], "max": aabb[1], "center": 0.5 * (aabb[0] + aabb[1])}


def aabb_size(aabb: tuple[np.ndarray, np.ndarray] | None) -> np.ndarray | None:
    if aabb is None:
        return None
    return aabb[1] - aabb[0]


def known_cube_aabb_from_center(center: np.ndarray | None, size: float | None) -> tuple[np.ndarray, np.ndarray] | None:
    if center is None or size is None or not math.isfinite(float(size)) or float(size) <= 0.0:
        return None
    half_extent = np.full(3, float(size) * 0.5, dtype=np.float32)
    return center - half_extent, center + half_extent


def aabb_size_matches(size_vec: np.ndarray | None, expected_size: float | None) -> bool | None:
    if size_vec is None or expected_size is None or not math.isfinite(float(expected_size)):
        return None
    tolerance = max(0.005, float(expected_size) * 0.20)
    return bool(np.all(np.abs(size_vec - float(expected_size)) <= tolerance))


def evaluate_tabletop_task_success(
    task_spec: TabletopTaskSpec,
    object_prim_paths: dict[str, str],
    tabletop_z: float,
) -> tuple[bool, dict[str, Any]]:
    object_spec = TABLETOP_OBJECT_SPECS[task_spec.object_name]
    zone_spec = TARGET_ZONE_SPECS[task_spec.target_zone]
    object_prim_path = object_prim_paths[task_spec.object_name]
    object_pos = get_prim_world_pos(object_prim_path)
    if object_pos is None:
        return False, {"reason": f"missing_prim:{object_prim_path}"}

    target_center = np.array(
        [
            float(zone_spec.center_xy[0]),
            float(zone_spec.center_xy[1]),
            tabletop_z + object_spec.size * 0.5 + 0.003,
        ],
        dtype=np.float32,
    )
    delta_xy = np.abs(object_pos[:2] - target_center[:2])
    zone_half_extent = 0.45 * np.array(zone_spec.scale_xy, dtype=np.float32)
    in_zone = bool(np.all(delta_xy <= zone_half_extent))
    landed_on_table = float(object_pos[2]) <= float(target_center[2] + 0.045)
    success = bool(in_zone and landed_on_table)
    details = {
        "object_pos": object_pos,
        "target_center": target_center,
        "delta_xy": delta_xy,
        "zone_half_extent": zone_half_extent,
        "in_zone": in_zone,
        "landed_on_table": landed_on_table,
    }
    return success, details


def build_scripted_grasp_sequence(
    task_spec: TabletopTaskSpec,
    joint_lower: np.ndarray,
    joint_upper: np.ndarray,
) -> list[tuple[str, int, np.ndarray]]:
    templates = build_command_templates("so101", joint_lower, joint_upper)
    zone = TARGET_ZONE_SPECS[task_spec.target_zone]
    object_spec = TABLETOP_OBJECT_SPECS[task_spec.object_name]
    object_xy = np.array(object_spec.position_xy, dtype=np.float32)
    target_xy = np.array(zone.center_xy, dtype=np.float32)
    delta_xy = target_xy - object_xy

    approach = templates["left"].copy()
    approach[5] = templates["open"][5]
    pre_grasp = approach.copy()
    pre_grasp[1] -= 0.08
    pre_grasp[2] += 0.10
    pre_grasp[3] -= 0.05

    close = pre_grasp.copy()
    close[5] = templates["close"][5]
    lift = close.copy()
    lift[1] += 0.22
    lift[2] -= 0.32
    lift[3] += 0.18
    transport = lift.copy()
    transport[0] += float(np.clip(-3.2 * delta_xy[1], -0.75, 0.75))
    transport[4] += float(np.clip(2.0 * delta_xy[0], -0.45, 0.45))
    release = transport.copy()
    release[5] = templates["open"][5]

    phases = [
        ("open", 24, approach),
        ("pre_grasp", 42, pre_grasp),
        ("close", 36, close),
        ("lift", 48, lift),
        ("transport", 64, transport),
        ("release", 36, release),
        ("retreat", 48, templates["home"]),
    ]
    return [(name, frames, np.clip(target, joint_lower, joint_upper).astype(np.float32)) for name, frames, target in phases]


def build_reach_calibration_sequence(
    joint_lower: np.ndarray,
    joint_upper: np.ndarray,
    hold_frames: int,
) -> list[tuple[str, int, np.ndarray]]:
    templates = build_command_templates("so101", joint_lower, joint_upper)
    hold = max(4, int(hold_frames))
    open_q = float(templates["open"][5])
    candidates: list[tuple[str, np.ndarray]] = []

    def add(name: str, target: list[float]) -> None:
        candidates.append((name, np.array(target, dtype=np.float32)))

    # Sweep base/yaw first around a stable arm shape to locate the cube in table XY.
    for base in [-0.85, -0.65, -0.45, -0.25, -0.05, 0.15, 0.35, 0.55, 0.75, 0.95]:
        add(f"base_{base:+.2f}", [base, -0.55, 1.10, -0.55, 0.0, open_q])

    # Then sweep lower approach shapes near the likely negative-base side of the table.
    for base in [-0.75, -0.55, -0.35, -0.15, 0.05]:
        for shoulder, elbow, wrist in [
            (-0.65, 1.10, -0.70),
            (-0.80, 1.25, -0.85),
            (-0.95, 1.40, -1.00),
            (-1.10, 1.50, -1.15),
        ]:
            add(f"reach_b{base:+.2f}_s{shoulder:+.2f}_e{elbow:+.2f}", [base, shoulder, elbow, wrist, 0.0, open_q])

    # Small wrist-roll sweep around the most likely target-side base values.
    for base in [-0.65, -0.45, -0.25]:
        for wrist_roll in [-0.60, -0.30, 0.0, 0.30, 0.60]:
            add(f"roll_b{base:+.2f}_r{wrist_roll:+.2f}", [base, -0.95, 1.40, -1.00, wrist_roll, open_q])

    # The first reach sweep showed good XY alignment but the measured gap center stayed
    # ~17 cm above the cube. Add targeted low-Z and jaw-angle probes before changing
    # robot placement or blaming the VLA model.
    close_q = float(templates["close"][5])
    for base in [-0.85, -0.65, -0.45, -0.25]:
        for jaw_q in [open_q, 0.90, 0.60, 0.30, close_q]:
            add(f"jaw_b{base:+.2f}_q{jaw_q:+.2f}", [base, -0.55, 1.10, -0.55, 0.0, jaw_q])

    low_z_shapes = [
        (-0.15, 0.55, -1.55),
        (-0.15, 0.55, -0.95),
        (-0.15, 0.55, -0.15),
        (-0.35, 0.75, -1.55),
        (-0.35, 0.75, -0.95),
        (-0.35, 0.75, -0.15),
        (-0.55, 0.95, -1.55),
        (-0.55, 0.95, -0.95),
        (-0.55, 0.95, -0.15),
        (-0.75, 1.15, -1.55),
        (-0.75, 1.15, -0.95),
        (-0.75, 1.15, -0.15),
        (-0.95, 1.35, -1.55),
        (-0.95, 1.35, -0.95),
        (-0.95, 1.35, -0.15),
        (-1.15, 1.55, -1.55),
        (-1.15, 1.55, -0.95),
        (-1.15, 1.55, -0.15),
        (0.15, 0.35, -1.55),
        (0.15, 0.35, -0.55),
        (0.45, 0.10, -1.55),
        (0.45, 0.10, -0.55),
    ]
    for base in [-0.85, -0.65, -0.45, -0.25, -0.05]:
        for idx, (shoulder, elbow, wrist) in enumerate(low_z_shapes):
            add(
                f"lowz_b{base:+.2f}_{idx:02d}",
                [base, shoulder, elbow, wrist, 0.0, open_q],
            )

    # Corrected known-cube bounds show lowz_*_17 overlaps the cube in X/Y but
    # sits about 3.3 cm above it. Probe only the local neighborhood before
    # changing robot placement or switching VLA models.
    local_contact_idx = 0
    for base in [-0.75, -0.65, -0.55, -0.45, -0.35]:
        for shoulder in [-1.30, -1.20, -1.10, -1.00]:
            for elbow in [1.45, 1.55, 1.62]:
                for wrist in [-0.35, -0.25, -0.15, -0.05, 0.05]:
                    add(
                        f"contact_b{base:+.2f}_{local_contact_idx:03d}",
                        [base, shoulder, elbow, wrist, 0.0, open_q],
                    )
                    local_contact_idx += 1

    phases: list[tuple[str, int, np.ndarray]] = []
    phases.append(("home", hold, templates["home"].copy()))
    for name, target in candidates:
        phases.append((name, hold, np.clip(target, joint_lower, joint_upper).astype(np.float32)))
    phases.append(("home_final", hold, templates["home"].copy()))
    return phases



def build_contact_lift_validation_sequence(
    joint_lower: np.ndarray,
    joint_upper: np.ndarray,
) -> list[tuple[str, int, np.ndarray]]:
    templates = build_command_templates("so101", joint_lower, joint_upper)
    open_q = float(templates["open"][5])
    close_q = float(templates["close"][5])

    contact_open = np.array([-0.55, -1.00, 1.55, 0.05, 0.0, open_q], dtype=np.float32)
    close_75 = contact_open.copy()
    close_75[5] = 0.75
    close_45 = contact_open.copy()
    close_45[5] = 0.45
    close_full = contact_open.copy()
    close_full[5] = close_q

    lift_small = close_full.copy()
    lift_small[1] = -0.85
    lift_small[2] = 1.35
    lift_small[3] = 0.18
    lift_mid = close_full.copy()
    lift_mid[1] = -0.70
    lift_mid[2] = 1.15
    lift_mid[3] = 0.28
    release = lift_mid.copy()
    release[5] = open_q

    phases = [
        ("home", 24, templates["home"].copy()),
        ("contact_open", 60, contact_open),
        ("close_75", 36, close_75),
        ("close_45", 36, close_45),
        ("close_full", 48, close_full),
        ("lift_small", 60, lift_small),
        ("lift_mid", 72, lift_mid),
        ("release", 48, release),
        ("retreat", 48, templates["home"].copy()),
    ]
    return [(name, frames, np.clip(target, joint_lower, joint_upper).astype(np.float32)) for name, frames, target in phases]


def get_scripted_phase(frame_idx: int, phases: list[tuple[str, int, np.ndarray]]) -> tuple[str, np.ndarray, bool]:
    elapsed = 0
    for name, frames, target in phases:
        if frame_idx < elapsed + frames:
            return name, target, False
        elapsed += frames
    return "done", phases[-1][2], True


def scripted_grasp_debug_details(
    robot_root_path: str,
    active_object_prim_path: str,
    object_size: float | None,
    phase_name: str,
    phase_target: np.ndarray,
    frame_idx: int,
    task_success_details: dict[str, Any],
) -> dict[str, Any]:
    object_pos = get_prim_world_pos(active_object_prim_path)
    gripper_pos = get_prim_world_pos(f"{robot_root_path}/gripper")
    jaw_pos = get_prim_world_pos(f"{robot_root_path}/jaw")
    wrist_pos = get_prim_world_pos(f"{robot_root_path}/wrist")
    object_aabb = get_prim_world_aabb(active_object_prim_path)
    known_object_aabb = known_cube_aabb_from_center(object_pos, object_size)
    object_aabb_size = aabb_size(object_aabb)
    gripper_aabb = get_prim_world_aabb(f"{robot_root_path}/gripper")
    jaw_aabb = get_prim_world_aabb(f"{robot_root_path}/jaw")
    wrist_aabb = get_prim_world_aabb(f"{robot_root_path}/wrist")
    gripper_jaw_aabb = aabb_union(gripper_aabb, jaw_aabb)
    object_aabb_size_matches_spec = aabb_size_matches(object_aabb_size, object_size)
    object_aabb_to_known_object = aabb_distance(object_aabb, known_object_aabb)
    object_aabb_axis_gap_to_known_object = aabb_axis_gap(object_aabb, known_object_aabb)
    object_aabb_matches_known_object = None
    if object_aabb is not None and known_object_aabb is not None:
        object_aabb_matches_known_object = bool(
            np.allclose(object_aabb[0], known_object_aabb[0], atol=0.0025)
            and np.allclose(object_aabb[1], known_object_aabb[1], atol=0.0025)
        )
    gripper_aabb_to_object = aabb_distance(gripper_aabb, object_aabb)
    jaw_aabb_to_object = aabb_distance(jaw_aabb, object_aabb)
    wrist_aabb_to_object = aabb_distance(wrist_aabb, object_aabb)
    gripper_jaw_aabb_to_object = aabb_distance(gripper_jaw_aabb, object_aabb)
    actual_object_aabb_contact = None
    if gripper_jaw_aabb_to_object is not None:
        contact_tolerance = 1e-5
        if object_size is not None and math.isfinite(float(object_size)):
            contact_tolerance = max(contact_tolerance, float(object_size) * 1e-3)
        actual_object_aabb_contact = bool(gripper_jaw_aabb_to_object <= contact_tolerance)
    gap_center = None
    gap_to_object = None
    jaw_to_object = None
    gripper_to_object = None
    if object_pos is not None and jaw_pos is not None:
        jaw_to_object = float(np.linalg.norm(jaw_pos - object_pos))
    if object_pos is not None and gripper_pos is not None:
        gripper_to_object = float(np.linalg.norm(gripper_pos - object_pos))
    if object_pos is not None and jaw_pos is not None and gripper_pos is not None:
        gap_center = 0.5 * (jaw_pos + gripper_pos)
        gap_to_object = float(np.linalg.norm(gap_center - object_pos))
    return {
        "scripted_phase": phase_name,
        "scripted_frame": int(frame_idx),
        "scripted_target_6": phase_target,
        "object_pos": object_pos,
        "wrist_pos": wrist_pos,
        "gripper_pos": gripper_pos,
        "jaw_pos": jaw_pos,
        "gap_center": gap_center,
        "gap_to_object": gap_to_object,
        "jaw_to_object": jaw_to_object,
        "gripper_to_object": gripper_to_object,
        "object_aabb": jsonify_aabb(object_aabb),
        "object_aabb_size": object_aabb_size,
        "object_aabb_size_matches_spec": object_aabb_size_matches_spec,
        "known_object_size": object_size,
        "known_object_aabb": jsonify_aabb(known_object_aabb),
        "object_aabb_to_known_object": object_aabb_to_known_object,
        "object_aabb_axis_gap_to_known_object": object_aabb_axis_gap_to_known_object,
        "object_aabb_matches_known_object": object_aabb_matches_known_object,
        "gripper_aabb": jsonify_aabb(gripper_aabb),
        "jaw_aabb": jsonify_aabb(jaw_aabb),
        "wrist_aabb": jsonify_aabb(wrist_aabb),
        "gripper_jaw_aabb": jsonify_aabb(gripper_jaw_aabb),
        "gripper_aabb_to_object": gripper_aabb_to_object,
        "jaw_aabb_to_object": jaw_aabb_to_object,
        "wrist_aabb_to_object": wrist_aabb_to_object,
        "gripper_jaw_aabb_to_object": gripper_jaw_aabb_to_object,
        "actual_object_aabb_contact": actual_object_aabb_contact,
        "gripper_aabb_z_gap_to_object": aabb_signed_z_gap(gripper_aabb, object_aabb),
        "jaw_aabb_z_gap_to_object": aabb_signed_z_gap(jaw_aabb, object_aabb),
        "wrist_aabb_z_gap_to_object": aabb_signed_z_gap(wrist_aabb, object_aabb),
        "gripper_jaw_aabb_z_gap_to_object": aabb_signed_z_gap(gripper_jaw_aabb, object_aabb),
        "gripper_aabb_axis_gap_to_known_object": aabb_axis_gap(gripper_aabb, known_object_aabb),
        "jaw_aabb_axis_gap_to_known_object": aabb_axis_gap(jaw_aabb, known_object_aabb),
        "wrist_aabb_axis_gap_to_known_object": aabb_axis_gap(wrist_aabb, known_object_aabb),
        "gripper_jaw_aabb_axis_gap_to_known_object": aabb_axis_gap(gripper_jaw_aabb, known_object_aabb),
        "gripper_aabb_to_known_object": aabb_distance(gripper_aabb, known_object_aabb),
        "jaw_aabb_to_known_object": aabb_distance(jaw_aabb, known_object_aabb),
        "wrist_aabb_to_known_object": aabb_distance(wrist_aabb, known_object_aabb),
        "gripper_jaw_aabb_to_known_object": aabb_distance(gripper_jaw_aabb, known_object_aabb),
        "gripper_aabb_z_gap_to_known_object": aabb_signed_z_gap(gripper_aabb, known_object_aabb),
        "jaw_aabb_z_gap_to_known_object": aabb_signed_z_gap(jaw_aabb, known_object_aabb),
        "wrist_aabb_z_gap_to_known_object": aabb_signed_z_gap(wrist_aabb, known_object_aabb),
        "gripper_jaw_aabb_z_gap_to_known_object": aabb_signed_z_gap(gripper_jaw_aabb, known_object_aabb),
        "task_success_details": task_success_details,
    }


def quat_wxyz_from_yaw_deg(yaw_deg: float) -> np.ndarray:
    half = math.radians(yaw_deg) * 0.5
    return np.array([math.cos(half), 0.0, 0.0, math.sin(half)], dtype=np.float32)


def import_robot_to_stage(urdf_path: str) -> str:
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
        raise RuntimeError(f"Failed to import URDF: {urdf_path}")

    return articulation_path


JOINT_LOWER = ROBOT_PROFILES["so100"]["joint_lower"].copy()
JOINT_UPPER = ROBOT_PROFILES["so100"]["joint_upper"].copy()


def build_command_templates(robot_model: str, joint_lower: np.ndarray, joint_upper: np.ndarray) -> dict[str, np.ndarray]:
    key = robot_model.strip().lower()
    if key == "so100":
        templates = {
            "home": np.array([0.0, 1.15, -1.25, -0.55, 0.0, 1.20], dtype=np.float32),
            "left": np.array([0.75, 1.20, -1.25, -0.55, 0.0, 1.20], dtype=np.float32),
            "right": np.array([-0.75, 1.20, -1.25, -0.55, 0.0, 1.20], dtype=np.float32),
            "up": np.array([0.0, 0.85, -0.95, -0.40, 0.0, 1.30], dtype=np.float32),
            "down": np.array([0.0, 1.55, -1.70, -0.85, 0.0, 1.00], dtype=np.float32),
            "open": np.array([0.0, 1.10, -1.20, -0.55, 0.0, 1.75], dtype=np.float32),
            "close": np.array([0.0, 1.10, -1.20, -0.55, 0.0, 0.05], dtype=np.float32),
        }
    else:
        templates = {
            "home": np.array([0.0, -0.55, 1.10, -0.55, 0.0, 0.20], dtype=np.float32),
            "left": np.array([0.85, -0.35, 1.10, -0.60, 0.0, 0.30], dtype=np.float32),
            "right": np.array([-0.85, -0.35, 1.10, -0.60, 0.0, 0.30], dtype=np.float32),
            "up": np.array([0.0, -0.15, 0.60, -0.20, 0.0, 0.55], dtype=np.float32),
            "down": np.array([0.0, -0.85, 1.35, -0.75, 0.0, 0.10], dtype=np.float32),
            "open": np.array([0.0, -0.45, 0.95, -0.45, 0.0, 1.20], dtype=np.float32),
            "close": np.array([0.0, -0.45, 0.95, -0.45, 0.0, 0.05], dtype=np.float32),
        }

    clipped_templates: dict[str, np.ndarray] = {}
    for name, value in templates.items():
        clipped_templates[name] = np.clip(value, joint_lower, joint_upper).astype(np.float32)
    return clipped_templates


COMMAND_TEMPLATES = build_command_templates("so100", JOINT_LOWER, JOINT_UPPER)
COMMAND_ORDER = list(COMMAND_TEMPLATES.keys())

COMMAND_KEYWORDS = {
    "home": ["home", "reset", "归位", "回零", "初始"],
    "left": ["left", "左", "向左"],
    "right": ["right", "右", "向右"],
    "up": ["up", "raise", "抬", "上"],
    "down": ["down", "lower", "下", "放下"],
    "open": ["open", "松开", "张开", "放开"],
    "close": ["close", "grip", "抓", "夹紧"],
}


def command_to_id(command_text: str) -> int | None:
    if not command_text:
        return None
    text = command_text.strip().lower()
    for cmd_name, keywords in COMMAND_KEYWORDS.items():
        for kw in keywords:
            if kw in text:
                return COMMAND_ORDER.index(cmd_name)
    return None


def _find_first_keyword_pos(text: str, keywords: list[str]) -> int:
    pos = -1
    for kw in keywords:
        i = text.find(kw)
        if i >= 0 and (pos < 0 or i < pos):
            pos = i
    return pos


def _extract_text_intensity(text: str) -> float:
    low_words = ["一点点", "稍微", "轻轻", "轻微", "slight", "slightly", "a bit", "little"]
    high_words = ["大幅", "很多", "明显", "快速", "快点", "strong", "hard", "more", "faster"]
    scale = 1.0
    if any(w in text for w in low_words):
        scale *= 0.6
    if any(w in text for w in high_words):
        scale *= 1.5
    return float(np.clip(scale, 0.35, 2.0))


def parse_free_text_command(command_text: str, fallback_cmd_id: int) -> tuple[int, np.ndarray, str]:
    if not command_text:
        return fallback_cmd_id, np.zeros(6, dtype=np.float32), "empty"

    text = command_text.strip().lower()

    parse_keywords = {
        "home": ["home", "reset", "归位", "回零", "初始", "回中", "中位"],
        "left": ["向左", "往左", "左移", "left", "move left", "左"],
        "right": ["向右", "往右", "右移", "right", "move right", "右"],
        "up": ["向上", "往上", "抬", "升", "up", "raise", "lift"],
        "down": ["向下", "往下", "下压", "下降", "放低", "放下", "down", "lower"],
        "open": ["张开", "松开", "放开", "open", "release"],
        "close": ["夹紧", "抓紧", "抓取", "闭合", "close", "grip"],
    }

    hits: list[tuple[str, int]] = []
    for cmd_name, keywords in parse_keywords.items():
        pos = _find_first_keyword_pos(text, keywords)
        if pos >= 0:
            hits.append((cmd_name, pos))

    if not hits:
        legacy_id = command_to_id(text)
        if legacy_id is not None:
            cmd_name = COMMAND_ORDER[legacy_id]
            return legacy_id, np.zeros(6, dtype=np.float32), f"legacy={cmd_name}"
        return fallback_cmd_id, np.zeros(6, dtype=np.float32), "no-intent"

    hits.sort(key=lambda x: x[1])
    pos_map = {name: pos for name, pos in hits}
    motion_priority = ["left", "right", "up", "down"]
    gripper_priority = ["open", "close"]

    base_cmd_name = "home"
    motion_hits = [name for name in motion_priority if name in pos_map]
    gripper_hits = [name for name in gripper_priority if name in pos_map]
    if motion_hits:
        base_cmd_name = min(motion_hits, key=lambda name: pos_map[name])
    elif gripper_hits:
        base_cmd_name = min(gripper_hits, key=lambda name: pos_map[name])
    elif "home" in pos_map:
        base_cmd_name = "home"
    else:
        base_cmd_name = COMMAND_ORDER[fallback_cmd_id]
    base_cmd_id = COMMAND_ORDER.index(base_cmd_name)

    selected = [name for name, _ in hits]
    selected_non_home = [name for name in selected if name != "home"]

    home = COMMAND_TEMPLATES["home"]
    delta = np.zeros(6, dtype=np.float32)
    for name in selected_non_home:
        primitive = COMMAND_TEMPLATES[name] - home
        if name in ("open", "close"):
            primitive = primitive * 0.7
        delta += primitive

    if selected_non_home:
        delta /= math.sqrt(float(len(selected_non_home)))

    delta *= _extract_text_intensity(text)
    delta_limit = 0.45 * (JOINT_UPPER - JOINT_LOWER)
    delta = np.clip(delta, -delta_limit, delta_limit).astype(np.float32)

    summary = ",".join(selected)
    return base_cmd_id, delta, summary


def make_command_onehot(cmd_id: int, device: torch.device) -> torch.Tensor:
    vec = torch.zeros((len(COMMAND_ORDER),), dtype=torch.float32, device=device)
    vec[cmd_id] = 1.0
    return vec


def normalize_proxy_env_for_hf() -> None:
    # httpx used by huggingface_hub rejects 'socks://' and expects 'socks5://'.
    proxy_keys = ["ALL_PROXY", "all_proxy", "HTTP_PROXY", "http_proxy", "HTTPS_PROXY", "https_proxy"]
    for key in proxy_keys:
        value = os.getenv(key)
        if value and value.startswith("socks://"):
            os.environ[key] = "socks5://" + value[len("socks://") :]


def load_hf_pretrained_act_policy(
    model_id: str, device: torch.device, local_files_only: bool = False
) -> tuple[Any, Any]:
    normalize_proxy_env_for_hf()

    # Explicit import to ensure "act" is registered in draccus choice registry.
    from lerobot.configs.policies import PreTrainedConfig
    from lerobot.policies.act.configuration_act import ACTConfig as _ACTConfig  # noqa: F401
    from lerobot.policies.factory import get_policy_class

    cfg = PreTrainedConfig.from_pretrained(model_id, local_files_only=local_files_only)
    if cfg.type != "act":
        raise RuntimeError(f"Model {model_id} is not ACT (got type={cfg.type})")

    cfg.device = str(device)
    policy_cls = get_policy_class(cfg.type)
    policy = policy_cls.from_pretrained(model_id, config=cfg, local_files_only=local_files_only)
    policy.eval()
    return policy, cfg


def load_hf_pretrained_smolvla_policy(
    model_id: str, device: torch.device, local_files_only: bool = False
) -> tuple[Any, Any]:
    normalize_proxy_env_for_hf()

    # Explicit import to ensure "smolvla" is registered in draccus choice registry.
    from lerobot.configs.policies import PreTrainedConfig
    from lerobot.policies.factory import get_policy_class
    from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig as _SmolVLAConfig  # noqa: F401

    cfg = PreTrainedConfig.from_pretrained(model_id, local_files_only=local_files_only)
    if cfg.type != "smolvla":
        raise RuntimeError(f"Model {model_id} is not SmolVLA (got type={cfg.type})")

    cfg.device = str(device)
    policy_cls = get_policy_class(cfg.type)
    policy = policy_cls.from_pretrained(model_id, config=cfg, local_files_only=local_files_only)
    policy.eval()
    if hasattr(policy, "reset"):
        policy.reset()
    fixed = sanitize_invalid_normalization_stats(policy)
    if fixed > 0:
        carb.log_warn(f"SmolVLA normalization stats patched: {fixed} invalid buffers fixed")
    return policy, cfg


def sanitize_invalid_normalization_stats(policy: Any) -> int:
    fixed = 0
    for name, buf in policy.named_buffers():
        if not torch.is_floating_point(buf):
            continue
        if "normalize_" not in name and "unnormalize_" not in name:
            continue

        has_nan = bool(torch.isnan(buf).any().item())
        has_inf = bool(torch.isinf(buf).any().item())
        if not (has_nan or has_inf):
            continue

        fill_value = 0.0 if "mean" in name else 1.0
        buf.data = torch.nan_to_num(buf.data, nan=fill_value, posinf=fill_value, neginf=fill_value)
        fixed += 1
    return fixed


class HFActPrior:
    def __init__(self, policy: Any, config: Any, device: torch.device):
        self.policy = policy
        self.config = config
        self.device = device

        image_features = self.config.image_features
        if not image_features:
            raise RuntimeError("HF ACT policy has no image feature, unsupported for this runtime prior.")
        self.image_key = next(iter(image_features.keys()))
        c, h, w = image_features[self.image_key].shape
        self.static_image = torch.zeros((1, c, h, w), dtype=torch.float32, device=self.device)

        state_feature = self.config.robot_state_feature
        if state_feature is None:
            raise RuntimeError("HF ACT policy has no observation.state feature, unsupported for this runtime prior.")
        self.state_key = OBS_STATE
        self.state_dim = int(state_feature.shape[0])

    def predict_target(self, joint_state_6: np.ndarray) -> np.ndarray:
        state = torch.zeros((1, self.state_dim), dtype=torch.float32, device=self.device)
        n = min(self.state_dim, 6)
        state[0, :n] = torch.from_numpy(joint_state_6[:n]).to(self.device)
        batch = {
            self.state_key: state,
            self.image_key: self.static_image,
        }
        with torch.no_grad():
            action = self.policy.select_action(batch)
        action_np = action.squeeze(0).detach().cpu().numpy().astype(np.float32)
        if action_np.shape[0] < 6:
            raise RuntimeError(f"HF ACT action dim is too small: {action_np.shape[0]}")
        return action_np[:6]


class SmolVLAPrior:
    def __init__(self, policy: Any, config: Any, device: torch.device, pretrained_path: str):
        self.policy = policy
        self.config = config
        self.device = device
        self.pretrained_path = pretrained_path

        image_features = self.config.image_features
        if not image_features:
            raise RuntimeError("SmolVLA policy has no image feature, unsupported for this runtime prior.")

        self.image_keys = list(image_features.keys())
        self.static_images: dict[str, torch.Tensor] = {}
        for key, feature in image_features.items():
            c, h, w = feature.shape
            self.static_images[key] = torch.zeros((c, h, w), dtype=torch.float32, device=self.device)

        state_feature = self.config.robot_state_feature
        if state_feature is None:
            raise RuntimeError("SmolVLA policy has no observation.state feature, unsupported for this runtime prior.")
        self.state_dim = int(state_feature.shape[0])

        from lerobot.policies.factory import make_pre_post_processors

        preprocessor_overrides: dict[str, dict[str, Any]] = {
            "device_processor": {"device": str(self.device)},
        }
        tokenizer_name = getattr(self.config, "vlm_model_name", None)
        if tokenizer_name:
            preprocessor_overrides["tokenizer_processor"] = {"tokenizer_name": tokenizer_name}

        resolved_pretrained_path = os.path.expanduser(pretrained_path)
        if not os.path.exists(resolved_pretrained_path):
            resolved_pretrained_path = pretrained_path

        self.preprocess, self.postprocess = make_pre_post_processors(
            self.config,
            pretrained_path=resolved_pretrained_path,
            preprocessor_overrides=preprocessor_overrides,
            postprocessor_overrides={"device_processor": {"device": "cpu"}},
        )
        self.joint_unit_mode = self._infer_joint_unit_mode()
        if self.joint_unit_mode == "degrees":
            carb.log_info("SmolVLA joint units inferred as degrees. Converting Isaac radians <-> SmolVLA degrees.")
        else:
            carb.log_info("SmolVLA joint units inferred as radians. No runtime joint-unit conversion needed.")

        # Some SmolVLA checkpoints miss normalized stats required by LeRobot's normalize modules.
        # For this runtime adapter, use raw values to keep inference stable.
        self.policy.normalize_inputs = _BatchIdentityModule()
        self.policy.unnormalize_outputs = _BatchIdentityModule()

    def reset(self) -> None:
        if hasattr(self.policy, "reset"):
            self.policy.reset()

    def _to_single_image(self, image: Any, key: str) -> torch.Tensor:
        tensor = torch.as_tensor(image, dtype=torch.float32, device=self.device)
        if tensor.ndim == 4:
            if tensor.shape[0] != 1:
                raise RuntimeError(f"SmolVLA image {key!r} expected batch size 1, got shape={tuple(tensor.shape)}")
            tensor = tensor[0]
        elif tensor.ndim == 5:
            if tensor.shape[0] != 1:
                raise RuntimeError(f"SmolVLA image {key!r} expected batch size 1, got shape={tuple(tensor.shape)}")
            tensor = tensor[0, -1]
        if tensor.ndim != 3:
            raise RuntimeError(f"SmolVLA image {key!r} expected CHW tensor, got shape={tuple(tensor.shape)}")
        return tensor

    def _iter_joint_stats(self) -> list[Any]:
        values: list[Any] = []
        for pipeline in (self.preprocess, self.postprocess):
            for step in getattr(pipeline, "steps", []):
                stats = getattr(step, "stats", None)
                if not isinstance(stats, dict):
                    continue
                for key in (OBS_STATE, ACTION):
                    if key not in stats or not isinstance(stats[key], dict):
                        continue
                    values.extend(stats[key].values())
        return values

    def _infer_joint_unit_mode(self) -> str:
        max_abs = 0.0
        for value in self._iter_joint_stats():
            if value is None:
                continue
            tensor = torch.as_tensor(value, dtype=torch.float32)
            if tensor.numel() == 0:
                continue
            current = float(tensor.abs().max().item())
            if current > max_abs:
                max_abs = current
        return "degrees" if max_abs > 10.0 else "radians"

    def _state_to_model_units(self, joint_state_6: np.ndarray) -> np.ndarray:
        state = np.asarray(joint_state_6, dtype=np.float32).copy()
        if self.joint_unit_mode == "degrees":
            state = np.rad2deg(state)
        return state

    def _action_from_model_units(self, action_6: np.ndarray) -> np.ndarray:
        action = np.asarray(action_6, dtype=np.float32).copy()
        if self.joint_unit_mode == "degrees":
            action = np.deg2rad(action)
        return action

    def _build_frame(
        self, joint_state_6: np.ndarray, task_text: str, image_tensors: dict[str, torch.Tensor] | None
    ) -> dict[str, Any]:
        state = torch.zeros((self.state_dim,), dtype=torch.float32, device=self.device)
        n = min(self.state_dim, 6)
        state_input = self._state_to_model_units(joint_state_6)
        state[:n] = torch.from_numpy(state_input[:n]).to(self.device)

        frame: dict[str, Any] = {
            OBS_STATE: state,
            "task": task_text if task_text else "move robot arm",
        }
        if image_tensors is None:
            frame.update(self.static_images)
        else:
            for key in self.image_keys:
                if key in image_tensors:
                    frame[key] = self._to_single_image(image_tensors[key], key)
                else:
                    frame[key] = self.static_images[key]
        return frame

    def predict_target(
        self, joint_state_6: np.ndarray, task_text: str, image_tensors: dict[str, torch.Tensor] | None = None
    ) -> np.ndarray:
        frame = self._build_frame(joint_state_6, task_text, image_tensors)
        batch = self.preprocess(frame)

        with torch.no_grad():
            self.reset()
            action = self.policy.select_action(batch)
            action = self.postprocess(action)

        if not isinstance(action, torch.Tensor):
            action = torch.as_tensor(action, dtype=torch.float32)
        if action.ndim == 1:
            action = action.unsqueeze(0)
        elif action.ndim >= 3:
            action = action[:, 0, :]

        action_np = action.squeeze(0).detach().cpu().numpy().astype(np.float32)
        if action_np.shape[0] < 6:
            raise RuntimeError(f"SmolVLA action dim is too small: {action_np.shape[0]}")
        return self._action_from_model_units(action_np[:6])


class _BatchIdentityModule(torch.nn.Module):
    def forward(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        return batch


def train_act_command_policy(device: torch.device) -> ACTPolicy:
    cfg = ACTConfig(
        n_obs_steps=1,
        chunk_size=1,
        n_action_steps=1,
        use_vae=False,
        dim_model=128,
        n_heads=4,
        dim_feedforward=512,
        n_encoder_layers=2,
        n_decoder_layers=1,
        dropout=0.0,
        input_features={
            OBS_STATE: PolicyFeature(type=FeatureType.STATE, shape=(6,)),
            OBS_ENV_STATE: PolicyFeature(type=FeatureType.ENV, shape=(len(COMMAND_ORDER),)),
        },
        output_features={
            ACTION: PolicyFeature(type=FeatureType.ACTION, shape=(6,)),
        },
        normalization_mapping={
            FeatureType.STATE: NormalizationMode.IDENTITY,
            FeatureType.ENV: NormalizationMode.IDENTITY,
            FeatureType.ACTION: NormalizationMode.IDENTITY,
            FeatureType.VISUAL: NormalizationMode.IDENTITY,
        },
        device=str(device),
    )

    policy = ACTPolicy(cfg).to(device)
    policy.train()
    optimizer = torch.optim.Adam(policy.parameters(), lr=4e-4)

    templates = torch.from_numpy(np.stack([COMMAND_TEMPLATES[c] for c in COMMAND_ORDER], axis=0)).to(device)
    lower = torch.from_numpy(JOINT_LOWER).to(device)
    upper = torch.from_numpy(JOINT_UPPER).to(device)

    n_steps = 800
    batch_size = 256
    for step in range(n_steps):
        cmd_ids = torch.randint(0, len(COMMAND_ORDER), (batch_size,), device=device)
        cmd_onehot = F.one_hot(cmd_ids, len(COMMAND_ORDER)).to(torch.float32)

        joint_states = lower + torch.rand((batch_size, 6), device=device) * (upper - lower)
        targets = templates[cmd_ids]

        blend = torch.rand((batch_size, 1), device=device) * 0.25 + 0.15
        next_step_target = joint_states + blend * (targets - joint_states)

        pred_chunk = policy.model({OBS_STATE: joint_states, OBS_ENV_STATE: cmd_onehot})[0]
        pred = pred_chunk[:, 0, :]
        loss = F.smooth_l1_loss(pred, next_step_target)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        if step % 200 == 0 or step == n_steps - 1:
            carb.log_info(f"ACT warmup train step={step}, loss={loss.item():.6f}")

    policy.eval()
    policy.reset()
    return policy


class CommandFileWatcher:
    def __init__(self, command_file: str):
        self.command_file = Path(command_file).expanduser()
        self.command_file.parent.mkdir(parents=True, exist_ok=True)
        self.last_mtime_ns = 0

    def read_if_changed(self) -> str | None:
        if not self.command_file.exists():
            return None
        stat = self.command_file.stat()
        if stat.st_mtime_ns == self.last_mtime_ns:
            return None
        self.last_mtime_ns = stat.st_mtime_ns
        return self.command_file.read_text(encoding="utf-8").strip()


def parse_joint_target_file_text(text: str, default_units: str = "rad") -> tuple[np.ndarray, str]:
    stripped = text.strip()
    if not stripped:
        raise ValueError("joint target file is empty")

    units = default_units.lower()
    values: Any
    try:
        decoded = json.loads(stripped)
    except json.JSONDecodeError:
        cleaned_lines: list[str] = []
        for line in stripped.splitlines():
            cleaned_line = line.split("#", 1)[0].strip()
            if cleaned_line:
                cleaned_lines.append(cleaned_line)
        flat_text = " ".join(cleaned_lines).replace(",", " ")
        values = [float(part) for part in flat_text.split()]
    else:
        if isinstance(decoded, dict):
            units = str(decoded.get("unit", decoded.get("units", units))).lower()
            for key in ("target", "joints", "positions", "q"):
                if key in decoded:
                    values = decoded[key]
                    break
            else:
                raise ValueError("JSON joint target object must contain one of: target, joints, positions, q")
        else:
            values = decoded

    target = np.asarray(values, dtype=np.float32).reshape(-1)
    if target.shape[0] < 6:
        raise ValueError(f"expected at least 6 joint values, got {target.shape[0]}")
    target = target[:6]

    if units in ("deg", "degree", "degrees"):
        target = np.deg2rad(target).astype(np.float32)
        normalized_units = "deg"
    elif units in ("rad", "radian", "radians"):
        normalized_units = "rad"
    else:
        raise ValueError(f"unsupported joint target units: {units!r}. Use rad or deg")
    return target, normalized_units


def ensure_joint_target_file(path: Path, initial_target: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        return
    record = {
        "unit": "rad",
        "target": [float(value) for value in initial_target[:6]],
        "joint_order": ["base", "shoulder", "elbow", "wrist", "wrist_roll", "gripper"],
        "usage": "Edit target while Isaac Sim is running, then save this file.",
    }
    path.write_text(json.dumps(record, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _adapt_image_channels(chw: torch.Tensor, channels: int) -> torch.Tensor:
    if chw.shape[0] == channels:
        return chw
    if channels == 1:
        return chw[:1, :, :] if chw.shape[0] >= 1 else torch.zeros((1, chw.shape[1], chw.shape[2]), dtype=chw.dtype)
    if chw.shape[0] == 1 and channels >= 3:
        return chw.repeat(channels, 1, 1)
    if chw.shape[0] > channels:
        return chw[:channels, :, :]
    pad = torch.zeros((channels - chw.shape[0], chw.shape[1], chw.shape[2]), dtype=chw.dtype)
    return torch.cat([chw, pad], dim=0)


def _save_image_u8(path: Path, image_u8: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        from PIL import Image

        Image.fromarray(image_u8).save(str(path))
    except Exception:
        np.save(str(path.with_suffix(".npy")), image_u8)


def _tensor_batch_to_u8_hwc(tensor: torch.Tensor) -> np.ndarray:
    t = tensor.detach().to("cpu")
    if t.ndim == 4:
        t = t[0]
    if t.ndim != 3:
        raise RuntimeError(f"Unexpected image tensor shape: {tuple(t.shape)}")
    arr = t.float().numpy()
    if arr.shape[0] == 1:
        arr = np.repeat(arr, 3, axis=0)
    arr = np.transpose(arr[:3, :, :], (1, 2, 0))
    if float(arr.max()) <= 1.0 + 1e-6:
        arr = arr * 255.0
    return np.clip(arr, 0.0, 255.0).astype(np.uint8)


def _jsonify_record_value(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return [_jsonify_record_value(v) for v in value.tolist()]
    if isinstance(value, dict):
        return {str(k): _jsonify_record_value(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonify_record_value(v) for v in value]
    if isinstance(value, (np.floating, np.integer, np.bool_)):
        return value.item()
    return value


def _safe_name(text: str) -> str:
    safe = []
    for ch in text:
        if ch.isalnum() or ch in ("-", "_"):
            safe.append(ch)
        else:
            safe.append("_")
    return "".join(safe).strip("_") or "item"


def _rotmat_to_quat_wxyz(rot: np.ndarray) -> np.ndarray:
    m00, m01, m02 = float(rot[0, 0]), float(rot[0, 1]), float(rot[0, 2])
    m10, m11, m12 = float(rot[1, 0]), float(rot[1, 1]), float(rot[1, 2])
    m20, m21, m22 = float(rot[2, 0]), float(rot[2, 1]), float(rot[2, 2])
    tr = m00 + m11 + m22
    if tr > 0.0:
        s = math.sqrt(tr + 1.0) * 2.0
        w = 0.25 * s
        x = (m21 - m12) / s
        y = (m02 - m20) / s
        z = (m10 - m01) / s
    elif m00 > m11 and m00 > m22:
        s = math.sqrt(1.0 + m00 - m11 - m22) * 2.0
        w = (m21 - m12) / s
        x = 0.25 * s
        y = (m01 + m10) / s
        z = (m02 + m20) / s
    elif m11 > m22:
        s = math.sqrt(1.0 + m11 - m00 - m22) * 2.0
        w = (m02 - m20) / s
        x = (m01 + m10) / s
        y = 0.25 * s
        z = (m12 + m21) / s
    else:
        s = math.sqrt(1.0 + m22 - m00 - m11) * 2.0
        w = (m10 - m01) / s
        x = (m02 + m20) / s
        y = (m12 + m21) / s
        z = 0.25 * s
    quat = np.array([w, x, y, z], dtype=np.float32)
    quat_norm = float(np.linalg.norm(quat))
    if quat_norm > 1e-8:
        quat /= quat_norm
    else:
        quat = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
    return quat


def _look_at_quat_wxyz(eye: np.ndarray, target: np.ndarray, up_hint: np.ndarray) -> np.ndarray:
    forward = target - eye
    f_norm = float(np.linalg.norm(forward))
    if f_norm < 1e-6:
        forward = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    else:
        forward = forward / f_norm

    up = up_hint.astype(np.float32).copy()
    up_norm = float(np.linalg.norm(up))
    if up_norm < 1e-6:
        up = np.array([0.0, 0.0, 1.0], dtype=np.float32)
    else:
        up = up / up_norm

    # Camera world axes convention: +X forward, +Z up.
    y_axis = np.cross(up, forward)
    y_norm = float(np.linalg.norm(y_axis))
    if y_norm < 1e-6:
        up = np.array([0.0, 1.0, 0.0], dtype=np.float32)
        y_axis = np.cross(up, forward)
        y_norm = float(np.linalg.norm(y_axis))
        if y_norm < 1e-6:
            return np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
    y_axis = y_axis / y_norm
    z_axis = np.cross(forward, y_axis)
    z_norm = float(np.linalg.norm(z_axis))
    if z_norm > 1e-6:
        z_axis = z_axis / z_norm
    else:
        z_axis = up

    rot = np.array(
        [
            [forward[0], y_axis[0], z_axis[0]],
            [forward[1], y_axis[1], z_axis[1]],
            [forward[2], y_axis[2], z_axis[2]],
        ],
        dtype=np.float32,
    )
    return _rotmat_to_quat_wxyz(rot)


def _quat_wxyz_to_rotmat(quat: np.ndarray) -> np.ndarray:
    w, x, y, z = float(quat[0]), float(quat[1]), float(quat[2]), float(quat[3])
    return np.array(
        [
            [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
            [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
            [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
        ],
        dtype=np.float32,
    )


class SmolVLAImageProvider:
    """Capture Isaac camera frames and convert them into the image tensors expected by a SmolVLA checkpoint."""

    def __init__(
        self,
        image_features: dict[str, Any],
        robot_root_path: str,
        pick_object_prim_path: str,
        refresh_render_fn: Callable[[], None] | None = None,
    ):
        self.shapes: dict[str, tuple[int, int, int]] = {}
        max_h, max_w = 0, 0
        for key, feature in image_features.items():
            c, h, w = tuple(int(v) for v in feature.shape)
            self.shapes[key] = (c, h, w)
            max_h = max(max_h, h)
            max_w = max(max_w, w)
        if not self.shapes:
            raise RuntimeError("SmolVLA checkpoint exposes no image keys.")
        self.resolution = (max_w, max_h)
        self.key_to_slot: dict[str, str] = {}
        self.slot_to_camera: dict[str, Camera] = {}
        self.required_slots: list[str] = []
        self.slot_names = ["up", "wrist", "side"]
        self.fallback_slot_order = ["up", "side", "wrist"] if len(self.shapes) <= 2 else list(self.slot_names)
        self.robot_root_path = robot_root_path
        self.base_prim_path = f"{self.robot_root_path}/base"
        self.shoulder_prim_path = f"{self.robot_root_path}/shoulder"
        self.upper_arm_prim_path = f"{self.robot_root_path}/upper_arm"
        self.lower_arm_prim_path = f"{self.robot_root_path}/lower_arm"
        self.wrist_prim_path = f"{self.robot_root_path}/wrist"
        self.gripper_prim_path = f"{self.robot_root_path}/gripper"
        self.jaw_prim_path = f"{self.robot_root_path}/jaw"
        self.pick_object_prim_path = pick_object_prim_path
        self.wrist_camera_prim_path = "/World/VLACameras/wrist"
        # External cameras should frame the robot body plus the pick object, not just the end-effector.
        self.slot_eye_offsets = {
            "up": np.array([0.10, -0.08, 1.12], dtype=np.float32),
            "side": np.array([0.22, -0.56, 0.12], dtype=np.float32),
        }
        self.slot_target_offsets = {
            "up": np.array([0.02, 0.03, -0.14], dtype=np.float32),
            "side": np.array([0.02, 0.04, -0.20], dtype=np.float32),
        }
        self.wrist_fallback_eye = np.array([0.82, -0.20, 0.90], dtype=np.float32)
        self.look_target = np.array([0.44, 0.02, 0.79], dtype=np.float32)
        self.wrist_cam_backoff = 0.08
        self.wrist_cam_up = 0.035
        self.wrist_cam_side = 0.010
        self.wrist_cam_look_ahead = 0.28
        self.wrist_cam_look_down = 0.06
        self.wrist_debug_print_every = 60
        self._wrist_debug_counter = 0
        self.scene_debug_print_every = 16
        self._scene_debug_counters: dict[str, int] = {}
        self._last_side_camera_debug: dict[str, np.ndarray] = {}
        self.refresh_render_fn = refresh_render_fn
        self.post_pose_render_updates = 2
        self.cached_batch: dict[str, torch.Tensor] | None = None
        self.cached_slot_rgb_chw: dict[str, torch.Tensor] = {}
        self.latest_slot_rgb_u8: dict[str, np.ndarray] = {}
        self.warned_empty_frame = False

    def set_pick_object_prim_path(self, prim_path: str) -> None:
        if prim_path == self.pick_object_prim_path:
            return
        self.pick_object_prim_path = prim_path
        self.cached_batch = None
        self.cached_slot_rgb_chw.clear()
        self.latest_slot_rgb_u8.clear()

    def _slot_for_key(self, key: str, fallback_index: int) -> str:
        low = key.lower()
        slot_map = {
            "camera1": "up",
            "camera2": "wrist",
            "camera3": "side",
            "top": "up",
            "up": "up",
            "wrist": "wrist",
            "side": "side",
        }
        for token, slot in slot_map.items():
            if token in low:
                if slot not in self.required_slots:
                    return slot
                break
        for slot in self.fallback_slot_order:
            if slot not in self.required_slots:
                return slot
        for slot in self.slot_names:
            if slot not in self.required_slots:
                return slot
        return self.slot_names[fallback_index % len(self.slot_names)]

    def initialize(self) -> None:
        key_list = list(self.shapes.keys())
        for idx, key in enumerate(key_list):
            slot = self._slot_for_key(key, idx)
            self.key_to_slot[key] = slot
            if slot not in self.required_slots:
                self.required_slots.append(slot)

        for slot in self.required_slots:
            prim_path = self.wrist_camera_prim_path if slot == "wrist" else f"/World/VLACameras/{slot}"
            cam = Camera(
                prim_path=prim_path,
                name=f"smolvla_{slot}",
                frequency=20,
                resolution=self.resolution,
                annotator_device="cpu",
            )
            cam.initialize()
            if slot == "side":
                render_product_path = cam.get_render_product_path()
                bound_camera_path = get_camera_prim_path(render_product_path)
                clip_near, clip_far = cam.get_clipping_range()
                carb.log_info(
                    "Side camera init: "
                    f"render_product={render_product_path}, "
                    f"bound_camera={bound_camera_path}, "
                    f"clipping_range={[float(clip_near), float(clip_far)]}, "
                    f"focal_length={float(cam.get_focal_length()):.4f}, "
                    f"horizontal_aperture={float(cam.get_horizontal_aperture()):.4f}, "
                    f"vertical_aperture={float(cam.get_vertical_aperture()):.4f}"
                )
                cam.set_clipping_range(0.01, 1.0e5)
                clip_near, clip_far = cam.get_clipping_range()
                carb.log_info(
                    "Side camera clipping override: "
                    f"clipping_range={[float(clip_near), float(clip_far)]}"
                )
            self.slot_to_camera[slot] = cam
            if slot == "wrist":
                self._update_wrist_camera_pose()
            else:
                self._update_scene_camera_pose(slot)
        carb.log_info(
            f"SmolVLA image provider initialized with {len(self.required_slots)} camera views "
            f"({', '.join(self.required_slots)}): "
            + ", ".join(f"{k}->{v}" for k, v in self.key_to_slot.items())
            + f", resolution={self.resolution}"
        )

    def _get_prim_world_transform(self, prim_path: str) -> Gf.Matrix4d | None:
        stage = omni.usd.get_context().get_stage()
        prim = stage.GetPrimAtPath(prim_path)
        if not prim or not prim.IsValid():
            return None
        return UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(0.0)

    def _get_prim_world_pos(self, prim_path: str) -> np.ndarray | None:
        mat = self._get_prim_world_transform(prim_path)
        if mat is None:
            return None
        t = mat.ExtractTranslation()
        return np.array([float(t[0]), float(t[1]), float(t[2])], dtype=np.float32)

    def _get_robot_body_points(self) -> list[np.ndarray]:
        points: list[np.ndarray] = []
        for prim_path in (
            self.robot_root_path,
            self.base_prim_path,
            self.shoulder_prim_path,
            self.upper_arm_prim_path,
            self.lower_arm_prim_path,
            self.wrist_prim_path,
        ):
            pos = self._get_prim_world_pos(prim_path)
            if pos is not None:
                points.append(pos)
        return points

    def _get_robot_body_center(self) -> np.ndarray:
        body_points = self._get_robot_body_points()
        if body_points:
            body_stack = np.stack(body_points, axis=0)
            return (0.5 * (body_stack.min(axis=0) + body_stack.max(axis=0))).astype(np.float32)
        return np.array([0.24, -0.04, 0.84], dtype=np.float32)

    def _get_scene_focus(self) -> tuple[np.ndarray, np.ndarray | None, float]:
        body_points = self._get_robot_body_points()
        body_center = self._get_robot_body_center()

        object_pos = self._get_prim_world_pos(self.pick_object_prim_path)
        focus = body_center.copy()
        if object_pos is not None:
            focus = 0.72 * body_center + 0.28 * object_pos
        focus[2] = max(float(focus[2]), 0.84)

        scene_points = list(body_points)
        if object_pos is not None:
            scene_points.append(object_pos)
        if scene_points:
            scene_stack = np.stack(scene_points, axis=0)
            span_xy = float(np.linalg.norm(scene_stack.max(axis=0)[:2] - scene_stack.min(axis=0)[:2]))
        else:
            span_xy = 0.30
        span_xy = max(span_xy, 0.30)
        return focus.astype(np.float32), object_pos, span_xy

    def _get_side_camera_pose(self, object_pos: np.ndarray | None, span_xy: float) -> tuple[np.ndarray, np.ndarray]:
        body_center = self._get_robot_body_center()
        object_target = object_pos if object_pos is not None else np.array([0.34, 0.10, 0.78], dtype=np.float32)
        scene_center = 0.78 * body_center + 0.22 * object_target
        distance_scale = max(0.0, span_xy - 0.30)

        # Match the dataset side view more closely: front-left of the desk, above tabletop, pitched downward.
        eye = np.array(
            [
                float(scene_center[0]) - 0.16 - 0.18 * distance_scale,
                float(scene_center[1]) - 0.34 - 0.10 * distance_scale,
                0.97 + 0.10 * distance_scale,
            ],
            dtype=np.float32,
        )
        target = np.array(
            [
                float(scene_center[0]) + 0.05,
                float(scene_center[1]) + 0.04,
                max(float(scene_center[2]), 0.82) - 0.03,
            ],
            dtype=np.float32,
        )
        self._last_side_camera_debug = {
            "body_center": body_center,
            "object_target": object_target,
            "scene_center": scene_center,
            "eye": eye,
            "target": target,
        }
        return eye, target

    def _update_scene_camera_pose(self, slot: str) -> None:
        if slot not in self.slot_eye_offsets:
            return
        cam = self.slot_to_camera.get(slot)
        prim_path = self.wrist_camera_prim_path if slot == "wrist" else f"/World/VLACameras/{slot}"

        world_up = np.array([0.0, 0.0, 1.0], dtype=np.float32)
        focus, object_pos, span_xy = self._get_scene_focus()
        if slot == "side":
            eye, target = self._get_side_camera_pose(object_pos, span_xy)
        else:
            span_scale = max(0.0, span_xy - 0.30)
            eye = focus + self.slot_eye_offsets[slot]
            eye = eye + np.array([0.30 * span_scale, -0.05 * span_scale, 0.90 * span_scale], dtype=np.float32)
            target = focus + self.slot_target_offsets[slot]
            if object_pos is not None:
                object_target = object_pos + np.array([0.0, 0.0, 0.015], dtype=np.float32)
                target = 0.72 * focus + 0.28 * object_target

        debug_counter = self._scene_debug_counters.get(slot, 0) + 1
        self._scene_debug_counters[slot] = debug_counter
        if slot == "side" and self._last_side_camera_debug and (
            debug_counter <= 3 or debug_counter % self.scene_debug_print_every == 0
        ):
            side_debug = self._last_side_camera_debug
            carb.log_info(
                "Side camera pose "
                f"#{debug_counter}: body_center={np.round(side_debug['body_center'], 3).tolist()}, "
                f"object_target={np.round(side_debug['object_target'], 3).tolist()}, "
                f"scene_center={np.round(side_debug['scene_center'], 3).tolist()}, "
                f"eye={np.round(side_debug['eye'], 3).tolist()}, "
                f"target={np.round(side_debug['target'], 3).tolist()}"
            )

        cam_quat = _look_at_quat_wxyz(eye, target, world_up)
        if slot == "side":
            # Diagnostic A/B: drive the side camera through the viewport helper so the camera prim
            # gets an explicit eye/target update, then read back the resolved world pose.
            set_camera_view(
                eye=eye.tolist(),
                target=target.tolist(),
                camera_prim_path=prim_path,
            )
        elif cam is not None:
            cam.set_world_pose(position=eye, orientation=cam_quat, camera_axes="world")
        else:
            set_camera_view(
                eye=eye.tolist(),
                target=target.tolist(),
                camera_prim_path=prim_path,
            )

        if slot == "side" and cam is not None and (
            debug_counter <= 3 or debug_counter % self.scene_debug_print_every == 0
        ):
            cam_pos_np, cam_quat_np = cam.get_world_pose(camera_axes="world")
            cam_pos = np.array(cam_pos_np, dtype=np.float32).reshape(-1)[:3]
            cam_quat_world = np.array(cam_quat_np, dtype=np.float32).reshape(-1)[:4]
            cam_rot = _quat_wxyz_to_rotmat(cam_quat_world)
            cam_forward = cam_rot[:, 0]
            desired_forward = target - eye
            desired_norm = float(np.linalg.norm(desired_forward))
            if desired_norm > 1e-6:
                desired_forward = desired_forward / desired_norm
            else:
                desired_forward = np.array([1.0, 0.0, 0.0], dtype=np.float32)
            carb.log_info(
                "Side camera actual "
                f"#{debug_counter}: cam_pos={np.round(cam_pos, 3).tolist()}, "
                f"cam_quat={np.round(cam_quat_world, 3).tolist()}, "
                f"cam_forward={np.round(cam_forward, 3).tolist()}, "
                f"desired_forward={np.round(desired_forward, 3).tolist()}"
            )

    def _update_wrist_camera_pose(self) -> None:
        wrist_tf = self._get_prim_world_transform(self.wrist_prim_path)
        if wrist_tf is None:
            set_camera_view(
                eye=self.wrist_fallback_eye.tolist(),
                target=self.look_target.tolist(),
                camera_prim_path=self.wrist_camera_prim_path,
            )
            return
        t = wrist_tf.ExtractTranslation()
        wrist_pos = np.array([float(t[0]), float(t[1]), float(t[2])], dtype=np.float32)
        gripper_pos = self._get_prim_world_pos(self.gripper_prim_path)
        jaw_pos = self._get_prim_world_pos(self.jaw_prim_path)

        world_up = np.array([0.0, 0.0, 1.0], dtype=np.float32)
        object_pos = self._get_prim_world_pos(self.pick_object_prim_path)
        aim_target = object_pos if object_pos is not None else self.look_target
        aim_dir = aim_target - wrist_pos
        aim_norm = float(np.linalg.norm(aim_dir))
        if aim_norm > 1e-6:
            aim_dir = aim_dir / aim_norm
        else:
            aim_dir = np.array([1.0, 0.0, 0.0], dtype=np.float32)

        # Choose wrist-local axis that best points to workspace/object target, then use it as camera forward axis.
        candidate_axes: list[np.ndarray] = []
        for local_axis in (
            Gf.Vec3d(1.0, 0.0, 0.0),
            Gf.Vec3d(-1.0, 0.0, 0.0),
            Gf.Vec3d(0.0, 1.0, 0.0),
            Gf.Vec3d(0.0, -1.0, 0.0),
            Gf.Vec3d(0.0, 0.0, 1.0),
            Gf.Vec3d(0.0, 0.0, -1.0),
        ):
            axis = np.array(wrist_tf.TransformDir(local_axis), dtype=np.float32)
            axis_norm = float(np.linalg.norm(axis))
            if axis_norm > 1e-6:
                candidate_axes.append(axis / axis_norm)

        if candidate_axes:
            forward = max(candidate_axes, key=lambda a: float(np.dot(a, aim_dir)))
            if float(np.dot(forward, aim_dir)) < 0.20:
                forward = aim_dir.copy()
        else:
            forward = aim_dir.copy()

        side_axis = np.cross(forward, world_up)
        side_norm = float(np.linalg.norm(side_axis))
        if side_norm > 1e-6:
            side_axis = side_axis / side_norm
        else:
            side_axis = np.array([0.0, 1.0, 0.0], dtype=np.float32)

        anchor_pos = jaw_pos if jaw_pos is not None else wrist_pos
        eye = anchor_pos - self.wrist_cam_backoff * forward + self.wrist_cam_up * world_up + self.wrist_cam_side * side_axis
        target = wrist_pos + self.wrist_cam_look_ahead * forward - self.wrist_cam_look_down * world_up
        if object_pos is not None:
            target = object_pos + np.array([0.0, 0.0, 0.01], dtype=np.float32)

        wrist_cam = self.slot_to_camera.get("wrist")
        if wrist_cam is not None:
            wrist_quat = _look_at_quat_wxyz(eye, target, world_up)
            wrist_cam.set_world_pose(position=eye, orientation=wrist_quat, camera_axes="world")
        else:
            set_camera_view(
                eye=eye.tolist(),
                target=target.tolist(),
                camera_prim_path=self.wrist_camera_prim_path,
            )
        self._wrist_debug_counter += 1
        if self._wrist_debug_counter % self.wrist_debug_print_every == 0:
            cam_pos = np.zeros(3, dtype=np.float32)
            cam_quat = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
            cam_forward = np.array([1.0, 0.0, 0.0], dtype=np.float32)
            if wrist_cam is not None:
                cam_pos_np, cam_quat_np = wrist_cam.get_world_pose(camera_axes="world")
                cam_pos = np.array(cam_pos_np, dtype=np.float32).reshape(-1)[:3]
                cam_quat = np.array(cam_quat_np, dtype=np.float32).reshape(-1)[:4]
                cam_rot = _quat_wxyz_to_rotmat(cam_quat)
                cam_forward = cam_rot[:, 0]
            carb.log_info(
                "Wrist camera debug: "
                f"wrist={np.array2string(wrist_pos, precision=3)}, "
                f"gripper={np.array2string(gripper_pos if gripper_pos is not None else np.zeros(3), precision=3)}, "
                f"jaw={np.array2string(jaw_pos if jaw_pos is not None else np.zeros(3), precision=3)}, "
                f"aim_dir={np.array2string(aim_dir, precision=3)}, "
                f"forward={np.array2string(forward, precision=3)}, "
                f"eye={np.array2string(eye, precision=3)}, target={np.array2string(target, precision=3)}, "
                f"cam_pos={np.array2string(cam_pos, precision=3)}, "
                f"cam_quat={np.array2string(cam_quat, precision=3)}, "
                f"cam_forward={np.array2string(cam_forward, precision=3)}"
            )

    def _zeros_batch(self, device: torch.device) -> dict[str, torch.Tensor]:
        batch: dict[str, torch.Tensor] = {}
        for key, (c, h, w) in self.shapes.items():
            batch[key] = torch.zeros((1, c, h, w), dtype=torch.float32, device=device)
        return batch

    def get_image_tensors(self, device: torch.device) -> dict[str, torch.Tensor]:
        if len(self.slot_to_camera) < len(self.required_slots):
            return self._zeros_batch(device)

        for slot in self.required_slots:
            if slot == "wrist":
                continue
            self._update_scene_camera_pose(slot)
        if "wrist" in self.required_slots:
            self._update_wrist_camera_pose()

        # Isaac camera annotators can lag behind prim pose updates by a render or two.
        # Refresh after moving the cameras so get_rgba() reflects the latest eye/target.
        if self.refresh_render_fn is not None:
            for _ in range(self.post_pose_render_updates):
                self.refresh_render_fn()

        slot_rgb_chw: dict[str, torch.Tensor] = {}
        slot_rgb_u8: dict[str, np.ndarray] = {}
        for slot, cam in self.slot_to_camera.items():
            rgba = cam.get_rgba()
            if rgba is None or getattr(rgba, "size", 0) == 0:
                if not self.warned_empty_frame:
                    carb.log_warn(f"SmolVLA camera frame is empty for slot={slot}. Using per-slot cache temporarily.")
                    self.warned_empty_frame = True
                if slot in self.cached_slot_rgb_chw:
                    slot_rgb_chw[slot] = self.cached_slot_rgb_chw[slot]
                    if slot in self.latest_slot_rgb_u8:
                        slot_rgb_u8[slot] = self.latest_slot_rgb_u8[slot]
                    continue
                if self.cached_batch is not None:
                    return {k: v.to(device) for k, v in self.cached_batch.items()}
                return self._zeros_batch(device)
            rgb_u8 = np.asarray(rgba[:, :, :3], dtype=np.uint8)
            slot_rgb_u8[slot] = rgb_u8.copy()
            rgb = rgb_u8.astype(np.float32) / 255.0
            chw = torch.from_numpy(rgb).permute(2, 0, 1).contiguous()
            slot_rgb_chw[slot] = chw
            self.cached_slot_rgb_chw[slot] = chw
        self.latest_slot_rgb_u8 = slot_rgb_u8

        batch_cpu: dict[str, torch.Tensor] = {}
        for key, (c, h, w) in self.shapes.items():
            slot = self.key_to_slot[key]
            chw = slot_rgb_chw[slot]
            if chw.shape[1] != h or chw.shape[2] != w:
                chw = F.interpolate(chw.unsqueeze(0), size=(h, w), mode="bilinear", align_corners=False).squeeze(0)
            chw = _adapt_image_channels(chw, c)
            batch_cpu[key] = chw.unsqueeze(0).to(torch.float32)

        self.cached_batch = batch_cpu
        return {k: v.to(device) for k, v in batch_cpu.items()}

    def get_latest_slot_images(self) -> dict[str, np.ndarray]:
        return {k: v.copy() for k, v in self.latest_slot_rgb_u8.items()}


def save_vla_io_snapshot(
    output_dir: Path,
    frame_idx: int,
    infer_idx: int,
    task_id: str,
    task_text: str,
    object_name: str,
    target_zone: str,
    joint_state_6: np.ndarray,
    smolvla_output_6: np.ndarray,
    image_tensors: dict[str, torch.Tensor] | None,
    image_provider: SmolVLAImageProvider | None,
    task_success: bool | None = None,
    success_details: dict[str, Any] | None = None,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = f"frame_{frame_idx:07d}_infer_{infer_idx:06d}"
    saved_images: list[str] = []

    if image_provider is not None:
        slot_images = image_provider.get_latest_slot_images()
        for slot, image_u8 in slot_images.items():
            img_name = f"{stem}_{slot}.png"
            _save_image_u8(output_dir / img_name, image_u8)
            saved_images.append(img_name)

    if image_tensors is not None:
        for key in sorted(image_tensors.keys()):
            img_name = f"{stem}_model_{_safe_name(key)}.png"
            _save_image_u8(output_dir / img_name, _tensor_batch_to_u8_hwc(image_tensors[key]))
            saved_images.append(img_name)

    record = {
        "timestamp_unix": time.time(),
        "frame_idx": int(frame_idx),
        "infer_idx": int(infer_idx),
        "task_id": task_id,
        "task_text": task_text,
        "object_name": object_name,
        "target_zone": target_zone,
        "joint_state_6": [float(v) for v in joint_state_6.tolist()],
        "smolvla_output_6": [float(v) for v in smolvla_output_6.tolist()],
        "task_success": task_success,
        "success_details": _jsonify_record_value(success_details) if success_details is not None else None,
        "saved_images": saved_images,
    }
    with open(output_dir / "vla_io.jsonl", "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def save_scripted_grasp_debug_snapshot(
    output_dir: Path,
    frame_idx: int,
    scripted_mode: str,
    task_id: str,
    task_text: str,
    object_name: str,
    target_zone: str,
    joint_state_6: np.ndarray,
    phase_target_6: np.ndarray,
    applied_target_6: np.ndarray,
    phase_name: str,
    scripted_frame_idx: int,
    scripted_done: bool,
    task_success: bool,
    debug_details: dict[str, Any],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    record = {
        "timestamp_unix": time.time(),
        "frame_idx": int(frame_idx),
        "scripted_mode": scripted_mode,
        "task_id": task_id,
        "task_text": task_text,
        "object_name": object_name,
        "target_zone": target_zone,
        "joint_state_6": joint_state_6,
        "phase_target_6": phase_target_6,
        "applied_target_6": applied_target_6,
        "scripted_phase": phase_name,
        "scripted_frame_idx": int(scripted_frame_idx),
        "scripted_done": bool(scripted_done),
        "task_success": bool(task_success),
        "debug_details": debug_details,
    }
    with open(output_dir / "scripted_grasp_debug.jsonl", "a", encoding="utf-8") as f:
        f.write(json.dumps(_jsonify_record_value(record), ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--robot-model",
        type=str,
        default="so100",
        choices=["so100", "so101"],
        help="robot model profile for URDF, limits and command templates",
    )
    parser.add_argument(
        "--task-id",
        type=str,
        default=DEFAULT_TASK_ID,
        help="structured tabletop task id; use --list-task-ids to inspect the built-in 10-task registry",
    )
    parser.add_argument(
        "--list-task-ids",
        action="store_true",
        help="print the built-in tabletop task registry and exit",
    )
    parser.add_argument(
        "--command",
        type=str,
        default="",
        help="optional initial text command or SmolVLA task prompt; if empty, uses the prompt from --task-id",
    )
    parser.add_argument(
        "--command-file",
        type=str,
        default=str(Path.home() / ".cache" / "isaacsim" / "so101_command.txt"),
        help="text file to update command while simulation is running",
    )
    parser.add_argument(
        "--enable-joint-target-file",
        action="store_true",
        help="enable live joint target control from a file; file values override command/model targets",
    )
    parser.add_argument(
        "--joint-target-file",
        type=str,
        default=str(Path.home() / ".cache" / "isaacsim" / "so101_joint_target.json"),
        help="file containing 6 joint targets; supports JSON, comma-separated, or whitespace-separated numbers",
    )
    parser.add_argument(
        "--joint-target-units",
        type=str,
        default="rad",
        choices=["rad", "deg"],
        help="units for plain-text joint target files; JSON can override with unit/units",
    )
    parser.add_argument(
        "--hf-model-id",
        type=str,
        default="lerobot/act_aloha_sim_transfer_cube_human",
        help="HuggingFace ACT model id used as pretrained motion prior (disable for SO100 VLA direct-control mode)",
    )
    parser.add_argument(
        "--hf-prior-alpha",
        type=float,
        default=0.35,
        help="blend factor of HF prior action in [0,1], higher means stronger HF influence",
    )
    parser.add_argument(
        "--disable-hf-prior",
        action="store_true",
        help="disable loading HuggingFace pretrained ACT prior",
    )
    parser.add_argument(
        "--hf-local-files-only",
        action="store_true",
        help="load HF model from local cache only",
    )
    parser.add_argument(
        "--download-hf-only",
        action="store_true",
        help="only download/load HF ACT and exit",
    )
    parser.add_argument(
        "--text-delta-gain",
        type=float,
        default=0.35,
        help="gain for free-text parsed joint bias in [0,1]",
    )
    parser.add_argument(
        "--disable-free-text",
        action="store_true",
        help="disable free-text parser and keep keyword-only behavior",
    )
    parser.add_argument(
        "--policy-device",
        type=str,
        default="auto",
        choices=["auto", "cpu", "cuda"],
        help="device for local command ACT policy and ACT prior",
    )
    parser.add_argument(
        "--disable-command-act",
        action="store_true",
        help="disable local command ACT and use command templates as base target",
    )
    parser.add_argument(
        "--smolvla-model-id",
        type=str,
        default="lerobot/smolvla_base",
        help="HuggingFace SmolVLA model id used as language-conditioned motion prior (auto-adapts to the checkpoint image keys)",
    )
    parser.add_argument(
        "--enable-smolvla-prior",
        action="store_true",
        help="enable SmolVLA prior blended with command policy output",
    )
    parser.add_argument(
        "--smolvla-prior-alpha",
        type=float,
        default=0.35,
        help="blend factor of SmolVLA prior action in [0,1], higher means stronger SmolVLA influence",
    )
    parser.add_argument(
        "--smolvla-device",
        type=str,
        default="cpu",
        choices=["cpu", "cuda"],
        help="device for SmolVLA model; default cpu",
    )
    parser.add_argument(
        "--smolvla-local-files-only",
        action="store_true",
        help="load SmolVLA model from local cache only",
    )
    parser.add_argument(
        "--download-smolvla-only",
        action="store_true",
        help="only download/load SmolVLA and exit",
    )
    parser.add_argument(
        "--smolvla-infer-interval",
        type=int,
        default=15,
        help="run SmolVLA inference once every N control frames (CPU-friendly). 1 means infer every frame",
    )
    parser.add_argument(
        "--render-every-n-frames",
        type=int,
        default=2,
        help="render viewport once every N simulation frames to reduce GUI load",
    )
    parser.add_argument(
        "--profile-smolvla",
        action="store_true",
        help="log SmolVLA CPU inference timing statistics during runtime",
    )
    parser.add_argument(
        "--save-vla-io",
        action="store_true",
        help="save SmolVLA multi-view input images and output joint command for each selected inference",
    )
    parser.add_argument(
        "--vla-io-dir",
        type=str,
        default=str(Path.home() / ".cache" / "isaacsim" / "vla_io"),
        help="directory for saved SmolVLA input/output records",
    )
    parser.add_argument(
        "--vla-save-every-infer",
        type=int,
        default=1,
        help="save one VLA I/O snapshot every N SmolVLA inferences",
    )
    parser.add_argument(
        "--disable-target-stabilizer",
        action="store_true",
        help="disable output target stabilizer (slew-rate + low-pass)",
    )
    parser.add_argument(
        "--max-joint-step",
        type=float,
        default=0.08,
        help="max per-step joint target delta in rad for stabilizer",
    )
    parser.add_argument(
        "--target-smoothing",
        type=float,
        default=0.35,
        help="low-pass factor in [0,1] for stabilized joint targets",
    )
    parser.add_argument(
        "--disable-smolvla-direct-joint-control",
        action="store_true",
        help="disable direct SmolVLA->joint target control. By default direct control is enabled when SmolVLA prior is loaded",
    )
    parser.add_argument(
        "--smolvla-delta-control",
        action="store_true",
        help="interpret SmolVLA output as delta-joint command instead of absolute target",
    )
    parser.add_argument(
        "--smolvla-delta-scale",
        type=float,
        default=0.10,
        help="scale factor for delta-joint mode",
    )
    parser.add_argument(
        "--max-control-frames",
        type=int,
        default=0,
        help="run at most this many control frames before exiting; 0 means rely on SimulationApp lifecycle only",
    )
    parser.add_argument(
        "--stop-on-task-success",
        action="store_true",
        help="exit once the active tabletop task success predicate becomes true",
    )
    parser.add_argument(
        "--scripted-grasp-debug",
        action="store_true",
        help="bypass learned priors and run a deterministic joint-space grasp sequence for execution-chain debugging",
    )
    parser.add_argument(
        "--scripted-reach-calibration",
        action="store_true",
        help="run a deterministic joint sweep to calibrate SO-101 end-effector reach; implies scripted debug logging",
    )
    parser.add_argument(
        "--scripted-contact-lift",
        action="store_true",
        help="run a deterministic contact-close-lift validation sequence seeded from calibrated SO-101 contact poses",
    )
    parser.add_argument(
        "--scripted-log-every-frames",
        type=int,
        default=10,
        help="write one scripted_grasp_debug.jsonl record every N control frames when scripted debug mode is enabled",
    )
    parser.add_argument(
        "--scripted-reach-hold-frames",
        type=int,
        default=12,
        help="number of control frames to hold each candidate pose during --scripted-reach-calibration",
    )
    parser.add_argument(
        "--stop-on-scripted-done",
        action="store_true",
        help="exit after the deterministic scripted sequence reaches its final phase",
    )
    args, _ = parser.parse_known_args()

    scripted_mode_count = sum(
        bool(flag)
        for flag in (args.scripted_grasp_debug, args.scripted_reach_calibration, args.scripted_contact_lift)
    )
    if scripted_mode_count > 1:
        raise RuntimeError(
            "Choose only one scripted mode: --scripted-grasp-debug, "
            "--scripted-reach-calibration, or --scripted-contact-lift."
        )

    if args.list_task_ids:
        for spec in list_tabletop_tasks():
            print(
                json.dumps(
                    {
                        "task_id": spec.task_id,
                        "task_text": spec.task_text,
                        "object_name": spec.object_name,
                        "target_zone": spec.target_zone,
                        "base_command": spec.base_command,
                    },
                    ensure_ascii=False,
                )
            )
        simulation_app.close()
        return

    task_spec = resolve_tabletop_task_spec(args.task_id)
    initial_command_text = args.command.strip()
    initial_task_id = task_id_from_text(initial_command_text)
    if initial_task_id is not None:
        task_spec = resolve_tabletop_task_spec(initial_task_id)
        initial_command_text = ""

    vla_io_dir = Path(args.vla_io_dir).expanduser()
    vla_save_every_infer = max(1, int(args.vla_save_every_infer))
    scripted_debug_enabled = bool(args.scripted_grasp_debug or args.scripted_reach_calibration or args.scripted_contact_lift)
    if args.save_vla_io or scripted_debug_enabled:
        vla_io_dir.mkdir(parents=True, exist_ok=True)
    if args.save_vla_io:
        carb.log_info(
            f"VLA I/O saving enabled: dir={str(vla_io_dir)}, every_infer={vla_save_every_infer}"
        )
    if scripted_debug_enabled:
        if args.scripted_contact_lift:
            scripted_mode = "contact_lift"
        elif args.scripted_reach_calibration:
            scripted_mode = "reach_calibration"
        else:
            scripted_mode = "grasp_debug"
        carb.log_info(f"Scripted debug enabled: mode={scripted_mode}, dir={str(vla_io_dir)}")

    robot_model = args.robot_model.lower()
    profile = _get_robot_profile(robot_model)

    global JOINT_LOWER, JOINT_UPPER, COMMAND_TEMPLATES, COMMAND_ORDER
    JOINT_LOWER = profile["joint_lower"].astype(np.float32).copy()
    JOINT_UPPER = profile["joint_upper"].astype(np.float32).copy()
    COMMAND_TEMPLATES = build_command_templates(robot_model, JOINT_LOWER, JOINT_UPPER)
    COMMAND_ORDER = list(COMMAND_TEMPLATES.keys())
    carb.log_info(
        f"Robot profile: {profile['label']} (6DoF), joint_lower={JOINT_LOWER.tolist()}, joint_upper={JOINT_UPPER.tolist()}"
    )
    if robot_model == "so100" and not args.disable_hf_prior and "aloha" in args.hf_model_id.lower():
        carb.log_warn(
            "Current HF ACT prior looks non-SO100-specific (contains 'aloha'). "
            "For SO100 VLA joint control, recommend adding --disable-hf-prior."
        )
    if robot_model == "so100" and args.enable_smolvla_prior and "so100" in args.smolvla_model_id.lower():
        carb.log_info(
            f"SmolVLA model id {args.smolvla_model_id!r} looks SO100-specific. "
            "Runtime will adapt cameras to the checkpoint image keys."
        )

    ensure_urdf_importer_enabled()
    source_urdf_path = resolve_robot_urdf_path(robot_model)
    urdf_path = sanitize_urdf_for_isaac(source_urdf_path)
    carb.log_info(f"Using {profile['label']} URDF: {source_urdf_path}")
    carb.log_info(f"Using sanitized URDF for Isaac import: {urdf_path}")

    world = World(stage_units_in_meters=1.0)
    world.scene.add_default_ground_plane()
    set_camera_view(eye=[1.7, 1.1, 1.3], target=[0.35, 0.0, 0.70], camera_prim_path="/OmniverseKit_Persp")

    tabletop_z = add_desk_scene(world)
    add_lights()
    object_prim_paths = add_tabletop_objects(world, tabletop_z)
    target_zone_centers = add_target_zones(world, tabletop_z)
    active_task_spec = task_spec
    active_object_prim_path = object_prim_paths[active_task_spec.object_name]
    smolvla_task_text = initial_command_text if initial_command_text else active_task_spec.task_text
    carb.log_info(
        "Active tabletop task: "
        f"id={active_task_spec.task_id}, object={active_task_spec.object_name}, "
        f"target_zone={active_task_spec.target_zone}, base_command={active_task_spec.base_command}, "
        f"task_text={smolvla_task_text!r}, target_center={target_zone_centers[active_task_spec.target_zone].tolist()}"
    )

    articulation_path = import_robot_to_stage(urdf_path)
    carb.log_info(f"Imported articulation at: {articulation_path}")

    robot_root_path = articulation_path.rsplit("/", 1)[0]
    XFormPrim(
        prim_paths_expr=robot_root_path,
        name=f"{robot_model}_root",
        positions=np.array([[0.16, -0.06, tabletop_z + 0.01]], dtype=np.float32),
        orientations=np.array([quat_wxyz_from_yaw_deg(90.0)], dtype=np.float32),
    )
    robot = world.scene.add(Articulation(prim_paths_expr=articulation_path, name=robot_model))

    world.reset()
    omni.timeline.get_timeline_interface().play()

    if robot.num_dof is None:
        raise RuntimeError(f"{profile['label']} articulation did not initialize. Please check URDF import logs.")
    if robot.num_dof < 6:
        raise RuntimeError(f"Unexpected DOF count for {profile['label']}: {robot.num_dof}")

    carb.log_info(f"{profile['label']} dof names: {robot.dof_names}")

    if args.policy_device == "cpu":
        policy_device = torch.device("cpu")
    elif args.policy_device == "cuda":
        policy_device = torch.device("cuda")
    else:
        policy_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    act_policy = None
    if args.enable_joint_target_file:
        carb.log_info("Live joint target file control enabled. Skipping local command ACT training.")
    elif args.disable_command_act:
        carb.log_info("Local command ACT disabled. Using command templates as base target.")
    else:
        carb.log_info(f"Training ACT command policy on device: {policy_device}")
        act_policy = train_act_command_policy(policy_device)

    hf_prior = None
    if args.enable_joint_target_file:
        carb.log_info("Live joint target file control enabled. Skipping HF ACT prior loading.")
    elif not args.disable_hf_prior:
        try:
            carb.log_info(f"Loading HF pretrained ACT prior: {args.hf_model_id}")
            hf_policy, hf_cfg = load_hf_pretrained_act_policy(
                args.hf_model_id, device=policy_device, local_files_only=args.hf_local_files_only
            )
            hf_prior = HFActPrior(hf_policy, hf_cfg, policy_device)
            carb.log_info(
                f"HF ACT loaded. action_shape={hf_cfg.action_feature.shape}, image_key={hf_prior.image_key}, "
                f"prior_alpha={args.hf_prior_alpha:.2f}"
            )
        except Exception as exc:
            carb.log_warn(f"Failed to load HF ACT prior, fallback to command-only ACT. reason={exc}")

    smolvla_prior = None
    smolvla_cfg = None
    smolvla_device = torch.device(args.smolvla_device)
    smolvla_image_provider: SmolVLAImageProvider | None = None
    if args.enable_joint_target_file and args.enable_smolvla_prior:
        carb.log_info("Live joint target file control enabled. Skipping SmolVLA prior loading.")
    elif args.enable_smolvla_prior or args.download_smolvla_only:
        try:
            carb.log_info(f"Loading SmolVLA prior: {args.smolvla_model_id} on device={smolvla_device}")
            smolvla_load_t0 = time.perf_counter()
            smolvla_policy, smolvla_cfg_loaded = load_hf_pretrained_smolvla_policy(
                args.smolvla_model_id, device=smolvla_device, local_files_only=args.smolvla_local_files_only
            )
            smolvla_cfg = smolvla_cfg_loaded
            smolvla_load_ms = (time.perf_counter() - smolvla_load_t0) * 1000.0
            smolvla_prior = SmolVLAPrior(
                smolvla_policy,
                smolvla_cfg,
                smolvla_device,
                pretrained_path=args.smolvla_model_id,
            )
            carb.log_info(
                f"SmolVLA loaded. action_shape={smolvla_cfg.action_feature.shape}, image_keys={smolvla_prior.image_keys}, "
                f"prior_alpha={args.smolvla_prior_alpha:.2f}, device={smolvla_device}"
            )
            if args.profile_smolvla:
                carb.log_info(f"SmolVLA model load time: {smolvla_load_ms:.1f} ms")
        except Exception as exc:
            carb.log_warn(f"Failed to load SmolVLA prior, fallback to non-SmolVLA control. reason={exc}")

    if args.download_hf_only:
        carb.log_info("Download/load HF-only mode complete. Exiting.")
        simulation_app.close()
        return
    if args.download_smolvla_only:
        carb.log_info("Download/load SmolVLA-only mode complete. Exiting.")
        simulation_app.close()
        return

    if smolvla_prior is not None and smolvla_cfg is not None:
        try:
            smolvla_image_provider = SmolVLAImageProvider(
                smolvla_cfg.image_features,
                robot_root_path=robot_root_path,
                pick_object_prim_path=active_object_prim_path,
                refresh_render_fn=world.render,
            )
            smolvla_image_provider.initialize()
            # Warm up camera frames to avoid first-inference empty image tensors.
            for _ in range(3):
                world.step(render=True)
        except Exception as exc:
            raise RuntimeError(
                "SmolVLA image provider init failed. "
                f"image_keys={list(smolvla_cfg.image_features.keys())}, reason={exc}"
            )

    watcher = CommandFileWatcher(args.command_file)
    initial_command_file_text = initial_command_text if initial_command_text else f"task:{active_task_spec.task_id}"
    watcher.command_file.write_text(initial_command_file_text, encoding="utf-8")
    joint_target_watcher = CommandFileWatcher(args.joint_target_file)
    live_joint_target: np.ndarray | None = (
        COMMAND_TEMPLATES[active_task_spec.base_command].copy() if args.enable_joint_target_file else None
    )
    last_joint_target_file_error = ""
    if args.enable_joint_target_file:
        joint_target_path = Path(args.joint_target_file).expanduser()
        ensure_joint_target_file(joint_target_path, COMMAND_TEMPLATES[active_task_spec.base_command])
        carb.log_info(
            "Live joint target file ready: "
            f"path={str(joint_target_path)}, plain_text_units={args.joint_target_units}. "
            "Edit and save this file to move the first 6 joints."
        )

    cmd_id = COMMAND_ORDER.index(active_task_spec.base_command)
    text_delta_cmd = np.zeros(6, dtype=np.float32)
    if initial_command_text:
        if args.disable_free_text:
            new_id = command_to_id(initial_command_text)
            if new_id is not None:
                cmd_id = new_id
        else:
            cmd_id, text_delta_cmd, parsed = parse_free_text_command(initial_command_text, cmd_id)
            if parsed != "no-intent" or bool(np.linalg.norm(text_delta_cmd) > 1e-4):
                carb.log_info(
                    f"Initial free-text parse: {initial_command_text!r} -> intents={parsed}, delta={np.array2string(text_delta_cmd, precision=3)}"
                )
            else:
                carb.log_info(
                    f"Initial command treated as task prompt only: {initial_command_text!r} -> base={active_task_spec.base_command}"
                )

    current_command_name = COMMAND_ORDER[cmd_id]
    carb.log_info(
        f"Initial control state: task_id={active_task_spec.task_id}, command_text={smolvla_task_text!r}, base_command={current_command_name}"
    )
    carb.log_info(f"Update command by editing file: {args.command_file}")

    scripted_phases: list[tuple[str, int, np.ndarray]] = []
    scripted_frame_idx = 0
    scripted_done_announced = False
    scripted_log_every = max(1, int(args.scripted_log_every_frames))

    def reset_scripted_sequence(reason: str) -> None:
        nonlocal scripted_phases, scripted_frame_idx, scripted_done_announced
        if not scripted_debug_enabled:
            return
        if args.scripted_contact_lift:
            scripted_phases = build_contact_lift_validation_sequence(JOINT_LOWER, JOINT_UPPER)
        elif args.scripted_reach_calibration:
            scripted_phases = build_reach_calibration_sequence(
                JOINT_LOWER, JOINT_UPPER, args.scripted_reach_hold_frames
            )
        else:
            scripted_phases = build_scripted_grasp_sequence(active_task_spec, JOINT_LOWER, JOINT_UPPER)
        scripted_frame_idx = 0
        scripted_done_announced = False
        phase_summary = [(name, frames) for name, frames, _target in scripted_phases]
        carb.log_info(
            "Scripted debug sequence reset: "
            f"mode={scripted_mode}, reason={reason}, task_id={active_task_spec.task_id}, "
            f"phases={phase_summary}, log_every={scripted_log_every}"
        )

    reset_scripted_sequence("initial")

    reset_needed = False
    frame_idx = 0
    smoothed_text_delta = np.zeros(6, dtype=np.float32)
    render_every_n = max(1, int(args.render_every_n_frames))
    smolvla_infer_interval = max(1, int(args.smolvla_infer_interval))
    cached_smolvla_target: np.ndarray | None = None
    smolvla_last_infer_frame = -10**9
    smolvla_force_update = True
    smolvla_infer_calls = 0
    smolvla_infer_total_ms = 0.0
    smolvla_infer_max_ms = 0.0
    smolvla_direct_mode = bool(smolvla_prior is not None and not args.disable_smolvla_direct_joint_control)
    stabilizer_prev_target: np.ndarray | None = None
    stabilizer_max_step = max(1e-4, float(args.max_joint_step))
    stabilizer_beta = float(np.clip(args.target_smoothing, 0.0, 1.0))
    stabilizer_clamp_count = 0
    task_success = False
    task_success_details: dict[str, Any] = {"reason": "not_checked_yet"}
    task_success_announced = False
    carb.log_info(
        f"Performance config: render_every_n={render_every_n}, smolvla_infer_interval={smolvla_infer_interval}"
    )
    if smolvla_direct_mode:
        direct_mode_name = "delta" if args.smolvla_delta_control else "absolute"
        carb.log_info(
            f"SmolVLA direct joint control enabled ({direct_mode_name}). "
            f"delta_scale={float(args.smolvla_delta_scale):.3f}"
        )
    if args.disable_target_stabilizer:
        carb.log_info("Target stabilizer disabled.")
    else:
        carb.log_info(
            f"Target stabilizer enabled: max_joint_step={stabilizer_max_step:.3f} rad, smoothing={stabilizer_beta:.2f}"
        )

    bounded_runtime_warned = False
    max_control_frames = max(0, int(args.max_control_frames))
    keep_alive_without_runner = bool(args.enable_joint_target_file and max_control_frames <= 0)
    if max_control_frames > 0:
        carb.log_info(f"Bounded control-frame mode enabled: max_control_frames={max_control_frames}")
    if keep_alive_without_runner:
        carb.log_info("Live joint target file mode will keep running until the SimulationApp exits.")

    while True:
        if simulation_app.is_exiting():
            break
        if max_control_frames > 0 and frame_idx >= max_control_frames:
            carb.log_info(f"Reached max control frames ({max_control_frames}). Exiting.")
            break
        if not simulation_app.is_running():
            stage = omni.usd.get_context().get_stage()
            if stage is None or (max_control_frames <= 0 and not keep_alive_without_runner):
                break
            if not bounded_runtime_warned:
                carb.log_warn(
                    "SimulationApp.is_running() returned False before control loop completed. "
                    "Continuing because a bounded or live-file control mode is active."
                )
                bounded_runtime_warned = True

        should_render = (frame_idx % render_every_n) == 0
        world.step(render=should_render)

        if world.is_stopped() and not reset_needed:
            reset_needed = True

        if world.is_playing():
            if reset_needed:
                world.reset()
                if act_policy is not None:
                    act_policy.reset()
                reset_needed = False
                frame_idx = 0
                smolvla_force_update = True
                stabilizer_prev_target = None
                task_success_announced = False
                task_success = False
                task_success_details = {"reason": "world_reset"}
                reset_scripted_sequence("world_reset")

            command_text = watcher.read_if_changed()
            if command_text:
                carb.log_info(f"Received command text: {command_text!r}")
                next_task_id = task_id_from_text(command_text)
                if next_task_id is not None:
                    active_task_spec = resolve_tabletop_task_spec(next_task_id)
                    active_object_prim_path = object_prim_paths[active_task_spec.object_name]
                    smolvla_task_text = active_task_spec.task_text
                    cmd_id = COMMAND_ORDER.index(active_task_spec.base_command)
                    current_command_name = COMMAND_ORDER[cmd_id]
                    text_delta_cmd = np.zeros(6, dtype=np.float32)
                    cached_smolvla_target = None
                    smolvla_force_update = True
                    stabilizer_prev_target = None
                    task_success_announced = False
                    task_success = False
                    task_success_details = {"reason": "task_switched"}
                    if smolvla_image_provider is not None:
                        smolvla_image_provider.set_pick_object_prim_path(active_object_prim_path)
                    if act_policy is not None:
                        act_policy.reset()
                    reset_scripted_sequence("task_switch")
                    carb.log_info(
                        "Task updated: "
                        f"task_id={active_task_spec.task_id}, object={active_task_spec.object_name}, "
                        f"target_zone={active_task_spec.target_zone}, base_command={current_command_name}, "
                        f"task_text={smolvla_task_text!r}, target_center={target_zone_centers[active_task_spec.target_zone].tolist()}"
                    )
                else:
                    smolvla_task_text = command_text
                    smolvla_force_update = True
                    if args.disable_free_text:
                        new_id = command_to_id(command_text)
                        if new_id is not None and new_id != cmd_id:
                            cmd_id = new_id
                            current_command_name = COMMAND_ORDER[cmd_id]
                            if act_policy is not None:
                                act_policy.reset()
                            carb.log_info(f"Command updated: {command_text!r} -> {current_command_name}")
                        elif new_id is None:
                            carb.log_warn(
                                f"Unknown command: {command_text!r}. Supported: {', '.join(COMMAND_ORDER)}"
                            )
                    else:
                        prev_id = cmd_id
                        prev_delta = text_delta_cmd.copy()
                        cmd_id, text_delta_cmd, parsed = parse_free_text_command(command_text, cmd_id)
                        delta_changed = bool(np.linalg.norm(text_delta_cmd - prev_delta) > 1e-4)
                        if cmd_id != prev_id or delta_changed:
                            current_command_name = COMMAND_ORDER[cmd_id]
                            if act_policy is not None:
                                act_policy.reset()
                            carb.log_info(
                                f"Command updated: {command_text!r} -> base={current_command_name}, intents={parsed}, "
                                f"delta={np.array2string(text_delta_cmd, precision=3)}"
                            )
                        elif parsed == "no-intent":
                            carb.log_info(
                                f"Free-text parser kept base command {current_command_name}; using text as SmolVLA task prompt: {command_text!r}"
                            )

            joint_pos = robot.get_joint_positions()
            if joint_pos is None:
                continue
            joint_pos = np.array(joint_pos, dtype=np.float32).reshape(-1)
            if joint_pos.shape[0] < 6:
                continue
            joint_pos6 = joint_pos[:6]
            task_success, task_success_details = evaluate_tabletop_task_success(active_task_spec, object_prim_paths, tabletop_z)
            just_reached_task_success = bool(task_success and not task_success_announced)
            if args.enable_joint_target_file:
                joint_target_text = joint_target_watcher.read_if_changed()
                if joint_target_text is not None:
                    try:
                        parsed_target, parsed_units = parse_joint_target_file_text(joint_target_text, args.joint_target_units)
                    except Exception as exc:
                        error_text = str(exc)
                        if error_text != last_joint_target_file_error:
                            carb.log_warn(f"Invalid joint target file {args.joint_target_file!r}: {error_text}")
                            last_joint_target_file_error = error_text
                    else:
                        live_joint_target = parsed_target
                        last_joint_target_file_error = ""
                        carb.log_info(
                            "Joint target file updated: "
                            f"units={parsed_units}, target_rad={np.array2string(live_joint_target, precision=3)}"
                        )

            scripted_phase_name = ""
            scripted_phase_target = np.zeros(6, dtype=np.float32)
            scripted_done = False
            scripted_debug_record: dict[str, Any] | None = None
            if args.enable_joint_target_file and live_joint_target is not None:
                target = live_joint_target.copy()
            elif scripted_debug_enabled:
                scripted_phase_name, scripted_phase_target, scripted_done = get_scripted_phase(
                    scripted_frame_idx, scripted_phases
                )
                target = scripted_phase_target.copy()
                scripted_debug_record = scripted_grasp_debug_details(
                    robot_root_path=robot_root_path,
                    active_object_prim_path=active_object_prim_path,
                    object_size=TABLETOP_OBJECT_SPECS[active_task_spec.object_name].size,
                    phase_name=scripted_phase_name,
                    phase_target=scripted_phase_target,
                    frame_idx=scripted_frame_idx,
                    task_success_details=task_success_details,
                )
                scripted_frame_idx += 1
                if scripted_done and not scripted_done_announced:
                    scripted_done_announced = True
                    carb.log_info(f"Scripted grasp sequence done for task_id={active_task_spec.task_id}")
            elif smolvla_direct_mode and smolvla_prior is not None:
                need_smolvla_infer = (
                    smolvla_force_update
                    or cached_smolvla_target is None
                    or (frame_idx - smolvla_last_infer_frame) >= smolvla_infer_interval
                )
                if need_smolvla_infer:
                    infer_t0 = time.perf_counter()
                    if not should_render and smolvla_image_provider is not None:
                        world.render()
                    image_tensors = (
                        smolvla_image_provider.get_image_tensors(smolvla_device)
                        if smolvla_image_provider is not None
                        else None
                    )
                    cached_smolvla_target = smolvla_prior.predict_target(joint_pos6, smolvla_task_text, image_tensors)
                    infer_ms = (time.perf_counter() - infer_t0) * 1000.0
                    smolvla_infer_calls += 1
                    if args.save_vla_io and smolvla_infer_calls % vla_save_every_infer == 0:
                        save_vla_io_snapshot(
                            output_dir=vla_io_dir,
                            frame_idx=frame_idx,
                            infer_idx=smolvla_infer_calls,
                            task_id=active_task_spec.task_id,
                            task_text=smolvla_task_text,
                            object_name=active_task_spec.object_name,
                            target_zone=active_task_spec.target_zone,
                            joint_state_6=joint_pos6,
                            smolvla_output_6=cached_smolvla_target,
                            image_tensors=image_tensors,
                            image_provider=smolvla_image_provider,
                            task_success=task_success,
                            success_details=task_success_details,
                        )
                    smolvla_infer_total_ms += infer_ms
                    if infer_ms > smolvla_infer_max_ms:
                        smolvla_infer_max_ms = infer_ms
                    if args.profile_smolvla and smolvla_infer_calls % 10 == 0:
                        avg_ms = smolvla_infer_total_ms / smolvla_infer_calls
                        eff_hz = 1000.0 / max(avg_ms, 1e-6)
                        carb.log_info(
                            f"SmolVLA profile: calls={smolvla_infer_calls}, avg={avg_ms:.1f}ms, "
                            f"last={infer_ms:.1f}ms, max={smolvla_infer_max_ms:.1f}ms, eff_hz={eff_hz:.2f}"
                        )
                    smolvla_last_infer_frame = frame_idx
                    smolvla_force_update = False
                if args.smolvla_delta_control:
                    target = joint_pos6 + float(args.smolvla_delta_scale) * cached_smolvla_target
                else:
                    target = cached_smolvla_target.copy()
            else:
                obs_state = torch.from_numpy(joint_pos6).to(policy_device).unsqueeze(0)
                cmd_vec = make_command_onehot(cmd_id, policy_device).unsqueeze(0)

                if act_policy is not None:
                    with torch.no_grad():
                        predicted_next = act_policy.select_action({OBS_STATE: obs_state, OBS_ENV_STATE: cmd_vec})
                    target = predicted_next.squeeze(0).detach().cpu().numpy().astype(np.float32)
                else:
                    target = COMMAND_TEMPLATES[current_command_name].copy()

                if hf_prior is not None:
                    hf_target = hf_prior.predict_target(joint_pos6)
                    alpha = float(np.clip(args.hf_prior_alpha, 0.0, 1.0))
                    target = (1.0 - alpha) * target + alpha * hf_target
                if smolvla_prior is not None:
                    need_smolvla_infer = (
                        smolvla_force_update
                        or cached_smolvla_target is None
                        or (frame_idx - smolvla_last_infer_frame) >= smolvla_infer_interval
                    )
                    if need_smolvla_infer:
                        infer_t0 = time.perf_counter()
                        if not should_render and smolvla_image_provider is not None:
                            world.render()
                        image_tensors = (
                            smolvla_image_provider.get_image_tensors(smolvla_device)
                            if smolvla_image_provider is not None
                            else None
                        )
                        cached_smolvla_target = smolvla_prior.predict_target(joint_pos6, smolvla_task_text, image_tensors)
                        infer_ms = (time.perf_counter() - infer_t0) * 1000.0
                        smolvla_infer_calls += 1
                        if args.save_vla_io and smolvla_infer_calls % vla_save_every_infer == 0:
                            save_vla_io_snapshot(
                                output_dir=vla_io_dir,
                                frame_idx=frame_idx,
                                infer_idx=smolvla_infer_calls,
                                task_id=active_task_spec.task_id,
                                task_text=smolvla_task_text,
                                object_name=active_task_spec.object_name,
                                target_zone=active_task_spec.target_zone,
                                joint_state_6=joint_pos6,
                                smolvla_output_6=cached_smolvla_target,
                                image_tensors=image_tensors,
                                image_provider=smolvla_image_provider,
                                task_success=task_success,
                                success_details=task_success_details,
                            )
                        smolvla_infer_total_ms += infer_ms
                        if infer_ms > smolvla_infer_max_ms:
                            smolvla_infer_max_ms = infer_ms
                        if args.profile_smolvla and smolvla_infer_calls % 10 == 0:
                            avg_ms = smolvla_infer_total_ms / smolvla_infer_calls
                            eff_hz = 1000.0 / max(avg_ms, 1e-6)
                            carb.log_info(
                                f"SmolVLA profile: calls={smolvla_infer_calls}, avg={avg_ms:.1f}ms, "
                                f"last={infer_ms:.1f}ms, max={smolvla_infer_max_ms:.1f}ms, eff_hz={eff_hz:.2f}"
                            )
                        smolvla_last_infer_frame = frame_idx
                        smolvla_force_update = False
                    smolvla_target = cached_smolvla_target
                    smol_alpha = float(np.clip(args.smolvla_prior_alpha, 0.0, 1.0))
                    target = (1.0 - smol_alpha) * target + smol_alpha * smolvla_target

            if (
                not args.enable_joint_target_file
                and not args.disable_free_text
                and not smolvla_direct_mode
                and not scripted_debug_enabled
            ):
                smoothed_text_delta = 0.9 * smoothed_text_delta + 0.1 * text_delta_cmd
                gain = float(np.clip(args.text_delta_gain, 0.0, 1.0))
                target = target + gain * smoothed_text_delta

            target = np.clip(target, JOINT_LOWER, JOINT_UPPER)
            if stabilizer_prev_target is None:
                stabilizer_prev_target = joint_pos6.copy()
            if not args.disable_target_stabilizer:
                desired_delta = target - stabilizer_prev_target
                clipped_delta = np.clip(desired_delta, -stabilizer_max_step, stabilizer_max_step)
                if bool(np.any(np.abs(desired_delta) > (stabilizer_max_step + 1e-6))):
                    stabilizer_clamp_count += 1
                stabilized_target = stabilizer_prev_target + clipped_delta
                target = (1.0 - stabilizer_beta) * stabilizer_prev_target + stabilizer_beta * stabilized_target
                target = np.clip(target, JOINT_LOWER, JOINT_UPPER)
            stabilizer_prev_target = target.copy()

            if scripted_debug_enabled and scripted_debug_record is not None:
                should_log_scripted = (
                    (frame_idx % scripted_log_every) == 0
                    or bool(task_success)
                    or bool(scripted_done)
                )
                if should_log_scripted:
                    save_scripted_grasp_debug_snapshot(
                        output_dir=vla_io_dir,
                        frame_idx=frame_idx,
                        scripted_mode=scripted_mode,
                        task_id=active_task_spec.task_id,
                        task_text=smolvla_task_text,
                        object_name=active_task_spec.object_name,
                        target_zone=active_task_spec.target_zone,
                        joint_state_6=joint_pos6,
                        phase_target_6=scripted_phase_target,
                        applied_target_6=target,
                        phase_name=scripted_phase_name,
                        scripted_frame_idx=scripted_frame_idx,
                        scripted_done=scripted_done,
                        task_success=task_success,
                        debug_details=scripted_debug_record,
                    )

            robot.set_joint_position_targets(np.expand_dims(target, axis=0), joint_indices=np.arange(6, dtype=np.int64))
            frame_idx += 1

            if just_reached_task_success:
                task_success_announced = True
                carb.log_info(
                    "Task success reached: "
                    f"task_id={active_task_spec.task_id}, object={active_task_spec.object_name}, "
                    f"target_zone={active_task_spec.target_zone}, details={json.dumps(task_success_details, ensure_ascii=False)}"
                )
                if args.stop_on_task_success:
                    carb.log_info(
                        f"Stopping after task success because --stop-on-task-success is enabled: {active_task_spec.task_id}"
                    )
                    break

            if scripted_debug_enabled and scripted_done and args.stop_on_scripted_done:
                carb.log_info(
                    f"Stopping after scripted {scripted_mode} sequence because --stop-on-scripted-done is enabled: "
                    f"{active_task_spec.task_id}"
                )
                break

            if frame_idx % 300 == 0:
                carb.log_info(
                    f"Running task_id={active_task_spec.task_id}, success={task_success}, command={current_command_name}, "
                    f"scripted_phase={scripted_phase_name if scripted_debug_enabled else 'off'}, "
                    f"q={np.array2string(target, precision=3)}, stabilizer_clamp_count={stabilizer_clamp_count}"
                )

    simulation_app.close()


if __name__ == "__main__":
    main()
