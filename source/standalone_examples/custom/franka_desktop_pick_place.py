# SPDX-FileCopyrightText: Copyright (c) 2020-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from isaacsim import SimulationApp

simulation_app = SimulationApp({"headless": False})

import sys

import carb
import numpy as np
import omni.timeline
import os
from isaacsim.core.api import World
from isaacsim.core.api.objects import DynamicCuboid, FixedCuboid
from isaacsim.core.utils.stage import add_reference_to_stage
from isaacsim.core.utils.viewports import set_camera_view
from isaacsim.robot.manipulators import SingleManipulator
from isaacsim.robot.manipulators.examples.franka.controllers.pick_place_controller import PickPlaceController
from isaacsim.robot.manipulators.grippers import ParallelGripper
from isaacsim.storage.native import get_assets_root_path

DEFAULT_ASSETS_ROOT = "https://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/5.1"


def resolve_assets_root() -> str:
    """Resolve Isaac asset root without strict connectivity check."""
    env_root = os.getenv("ISAACSIM_ASSETS_ROOT") or os.getenv("ISAAC_ASSETS_ROOT")
    if env_root:
        return env_root.rstrip("/")
    try:
        root = get_assets_root_path(skip_check=True)
        if isinstance(root, str) and root:
            return root.rstrip("/")
    except Exception:
        pass
    return DEFAULT_ASSETS_ROOT


def add_desk_scene(world: World) -> float:
    """Create a simple desk with colliders and return tabletop height (z)."""
    tabletop_z = 0.75
    top_size = np.array([1.00, 0.80, 0.06])
    top_center = np.array([0.45, 0.00, tabletop_z - top_size[2] / 2.0])

    # Table top
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

    # Table legs
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


def main():
    assets_root_path = resolve_assets_root()
    carb.log_info(f"Using assets root: {assets_root_path}")

    world = World(stage_units_in_meters=1.0)
    world.scene.add_default_ground_plane()
    set_camera_view(eye=[1.8, 1.4, 1.4], target=[0.35, 0.0, 0.65], camera_prim_path="/OmniverseKit_Persp")

    tabletop_z = add_desk_scene(world)

    # Add Franka
    franka_usd = assets_root_path + "/Isaac/Robots/FrankaRobotics/FrankaPanda/franka.usd"
    robot_prim = add_reference_to_stage(usd_path=franka_usd, prim_path="/World/Franka")
    robot_prim.GetVariantSet("Gripper").SetVariantSelection("AlternateFinger")
    robot_prim.GetVariantSet("Mesh").SetVariantSelection("Quality")

    gripper = ParallelGripper(
        end_effector_prim_path="/World/Franka/panda_rightfinger",
        joint_prim_names=["panda_finger_joint1", "panda_finger_joint2"],
        joint_opened_positions=np.array([0.05, 0.05]),
        joint_closed_positions=np.array([0.02, 0.02]),
        action_deltas=np.array([0.01, 0.01]),
    )

    my_franka = world.scene.add(
        SingleManipulator(
            prim_path="/World/Franka",
            name="my_franka",
            end_effector_prim_path="/World/Franka/panda_rightfinger",
            gripper=gripper,
        )
    )
    my_franka.set_world_pose(position=np.array([0.00, -0.18, tabletop_z]))

    # Add pick object on desktop
    cube_size = 0.0515
    cube = world.scene.add(
        DynamicCuboid(
            name="cube",
            prim_path="/World/Cube",
            position=np.array([0.35, 0.18, tabletop_z + cube_size / 2.0 + 0.002]),
            scale=np.array([cube_size, cube_size, cube_size]),
            size=1.0,
            color=np.array([0.10, 0.30, 0.95]),
        )
    )

    my_franka.gripper.set_default_state(my_franka.gripper.joint_opened_positions)
    world.reset()
    omni.timeline.get_timeline_interface().play()

    controller = PickPlaceController(
        name="desktop_pick_place_controller",
        gripper=my_franka.gripper,
        robot_articulation=my_franka,
    )
    articulation_controller = my_franka.get_articulation_controller()

    reset_needed = False
    task_completed = False

    while simulation_app.is_running():
        world.step(render=True)

        if world.is_stopped() and not reset_needed:
            reset_needed = True
            task_completed = False

        if world.is_playing():
            if reset_needed:
                world.reset()
                controller.reset()
                reset_needed = False
                task_completed = False

            actions = controller.forward(
                picking_position=cube.get_local_pose()[0],
                placing_position=np.array([0.15, -0.20, tabletop_z + cube_size / 2.0 + 0.002]),
                current_joint_positions=my_franka.get_joint_positions(),
                end_effector_offset=np.array([0.0, 0.005, 0.0]),
            )
            articulation_controller.apply_action(actions)

            if controller.is_done() and not task_completed:
                print("Desktop pick-and-place done.")
                task_completed = True

    simulation_app.close()


if __name__ == "__main__":
    main()
