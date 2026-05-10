# SPDX-FileCopyrightText: Copyright (c) 2020-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from isaacsim import SimulationApp

simulation_app = SimulationApp({"headless": False, "renderer": "RaytracedLighting"})

import math
import os

import carb
import numpy as np
import omni.kit.app
import omni.kit.commands
import omni.timeline
import omni.usd
from isaacsim.core.api import World
from isaacsim.core.api.objects import FixedCuboid
from isaacsim.core.prims import Articulation, XFormPrim
from isaacsim.core.utils.viewports import set_camera_view
from pxr import Gf, Sdf, UsdGeom, UsdLux


def ensure_urdf_importer_enabled() -> None:
    ext_name = "isaacsim.asset.importer.urdf"
    ext_manager = omni.kit.app.get_app().get_extension_manager()
    if not ext_manager.is_extension_enabled(ext_name):
        carb.log_info(f"Enabling extension: {ext_name}")
        ext_manager.set_extension_enabled_immediate(ext_name, True)
        simulation_app.update()


def resolve_so101_urdf_path() -> str:
    env_path = os.getenv("LEROBOT_SO101_URDF") or os.getenv("LEROBOT_URDF")
    if env_path:
        path = os.path.expanduser(env_path)
        if os.path.isfile(path):
            return path
        raise RuntimeError(f"URDF path from environment does not exist: {path}")

    try:
        from robot_descriptions.so_arm101_description import URDF_PATH

        if os.path.isfile(URDF_PATH):
            return URDF_PATH
    except Exception as exc:
        carb.log_warn(f"Import robot_descriptions.so_arm101_description failed: {exc}")

    fallback = os.path.expanduser("~/.cache/robot_descriptions/SO-ARM100/Simulation/SO101/so101_new_calib.urdf")
    if os.path.isfile(fallback):
        return fallback

    raise RuntimeError(
        "Cannot find SO-ARM101 URDF. Install with: "
        "_build/linux-x86_64/release/python.sh -m pip install robot_descriptions"
    )


def sanitize_so101_urdf_for_isaac(urdf_path: str) -> str:
    """
    SO-101 uses numeric joint names (1..6). Isaac's URDF importer normalizes
    those into the same token, which breaks articulation creation.
    """
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


def quat_wxyz_from_yaw_deg(yaw_deg: float) -> np.ndarray:
    half = math.radians(yaw_deg) * 0.5
    return np.array([math.cos(half), 0.0, 0.0, math.sin(half)], dtype=np.float32)


def import_so101_to_stage(urdf_path: str) -> str:
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


def main() -> None:
    ensure_urdf_importer_enabled()
    source_urdf_path = resolve_so101_urdf_path()
    urdf_path = sanitize_so101_urdf_for_isaac(source_urdf_path)
    carb.log_info(f"Using SO101 URDF: {source_urdf_path}")
    carb.log_info(f"Using sanitized URDF for Isaac import: {urdf_path}")

    world = World(stage_units_in_meters=1.0)
    world.scene.add_default_ground_plane()
    set_camera_view(eye=[1.7, 1.1, 1.3], target=[0.35, 0.0, 0.70], camera_prim_path="/OmniverseKit_Persp")

    tabletop_z = add_desk_scene(world)
    add_lights()

    articulation_path = import_so101_to_stage(urdf_path)
    carb.log_info(f"Imported articulation at: {articulation_path}")

    robot_root_path = articulation_path.rsplit("/", 1)[0]
    XFormPrim(
        prim_paths_expr=robot_root_path,
        name="so101_root",
        positions=np.array([[0.03, -0.20, tabletop_z + 0.01]], dtype=np.float32),
        orientations=np.array([quat_wxyz_from_yaw_deg(90.0)], dtype=np.float32),
    )
    so101 = world.scene.add(Articulation(prim_paths_expr=articulation_path, name="so101"))

    world.reset()
    omni.timeline.get_timeline_interface().play()

    if so101.num_dof is None:
        raise RuntimeError("SO101 articulation did not initialize. Please check URDF import logs.")

    carb.log_info(f"SO101 dof names: {so101.dof_names}")

    base_pose = np.zeros((1, so101.num_dof), dtype=np.float32)
    if so101.num_dof >= 6:
        base_pose[0, 1] = -0.55
        base_pose[0, 2] = 1.10
        base_pose[0, 3] = -0.55

    reset_needed = False
    frame_idx = 0

    while simulation_app.is_running():
        world.step(render=True)

        if world.is_stopped() and not reset_needed:
            reset_needed = True

        if world.is_playing():
            if reset_needed:
                world.reset()
                reset_needed = False
                frame_idx = 0

            t = frame_idx / 60.0
            pose = base_pose.copy()
            if so101.num_dof >= 1:
                pose[0, 0] = 0.35 * math.sin(0.7 * t)
            if so101.num_dof >= 2:
                pose[0, 1] += 0.20 * math.sin(0.5 * t)
            if so101.num_dof >= 3:
                pose[0, 2] += 0.25 * math.sin(0.9 * t)
            if so101.num_dof >= 4:
                pose[0, 3] += 0.20 * math.sin(1.2 * t)
            if so101.num_dof >= 5:
                pose[0, 4] = 0.45 * math.sin(0.6 * t)
            if so101.num_dof >= 6:
                pose[0, 5] = 0.65 + 0.35 * math.sin(1.5 * t)

            so101.set_joint_positions(pose)
            frame_idx += 1

    simulation_app.close()


if __name__ == "__main__":
    main()
