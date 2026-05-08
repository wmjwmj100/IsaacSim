import argparse
import hashlib
import json
import math
import os
import re
import shutil
import struct
import time
import zlib
from pathlib import Path

from isaacsim import SimulationApp


def env_flag(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def env_int(name: str, default: int = 0) -> int:
    value = os.getenv(name)
    if not value:
        return default
    return int(value)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Panthera-HT arm on an 80x80 cm Isaac Sim table.")
    parser.add_argument("--headless", action="store_true", default=env_flag("PANTHERA_HEADLESS"))
    parser.add_argument("--max-frames", type=int, default=env_int("PANTHERA_MAX_FRAMES"))
    parser.add_argument("--no-motion", action="store_true", default=env_flag("PANTHERA_NO_MOTION"))
    parser.add_argument("--urdf", default=os.getenv("PANTHERA_URDF", ""))
    parser.add_argument("--ros2-root", default=os.getenv("PANTHERA_ROS2_ROOT", ""))
    parser.add_argument("--mesh-dir", default=os.getenv("PANTHERA_MESH_DIR", ""))
    parser.add_argument("--table-size", type=float, default=float(os.getenv("PANTHERA_TABLE_SIZE", "0.80")))
    parser.add_argument("--table-height", type=float, default=float(os.getenv("PANTHERA_TABLE_HEIGHT", "0.75")))
    parser.add_argument("--use-official-grey", action="store_true", default=env_flag("PANTHERA_USE_OFFICIAL_GREY"))
    parser.add_argument(
        "--disable-layout-cameras",
        "--disable-vla-cameras",
        dest="disable_layout_cameras",
        action="store_true",
        default=env_flag("PANTHERA_DISABLE_LAYOUT_CAMERAS", env_flag("PANTHERA_DISABLE_VLA_CAMERAS")),
    )
    parser.add_argument(
        "--skip-layout-screenshots",
        "--skip-vla-screenshots",
        dest="skip_layout_screenshots",
        action="store_true",
        default=env_flag("PANTHERA_SKIP_LAYOUT_SCREENSHOTS", env_flag("PANTHERA_SKIP_VLA_SCREENSHOTS")),
    )
    parser.add_argument("--disable-realism", action="store_true", default=env_flag("PANTHERA_DISABLE_REALISM"))
    parser.add_argument("--randomize-realism", action="store_true", default=env_flag("PANTHERA_RANDOMIZE_REALISM"))
    parser.add_argument("--realism-seed", type=int, default=env_int("PANTHERA_REALISM_SEED", 17))
    parser.add_argument(
        "--layout-sequence-frames",
        default=os.getenv("PANTHERA_LAYOUT_SEQUENCE_FRAMES", "0,45,90,135,180,225"),
        help="Comma-separated motion frames to save as VLA-style layout camera observations.",
    )
    parser.add_argument(
        "--show-layout-camera-viewports",
        dest="show_layout_camera_viewports",
        action="store_true",
        default=env_flag("PANTHERA_SHOW_LAYOUT_CAMERA_VIEWPORTS", True),
        help="Open live Isaac viewport windows for the wrist and D435i layout cameras in non-headless mode.",
    )
    parser.add_argument(
        "--disable-layout-camera-viewports",
        dest="show_layout_camera_viewports",
        action="store_false",
        default=argparse.SUPPRESS,
        help="Disable live layout camera viewport windows even in GUI mode.",
    )
    if env_flag("PANTHERA_DISABLE_LAYOUT_CAMERA_VIEWPORTS"):
        parser.set_defaults(show_layout_camera_viewports=False)
    return parser.parse_args()


args = parse_args()
simulation_app = SimulationApp({"headless": args.headless, "renderer": "RaytracedLighting"})
_reported_is_running_false = False

import carb
import numpy as np
import omni.kit.app
import omni.kit.commands
import omni.timeline
import omni.usd
from isaacsim.core.api import World
from isaacsim.core.api.objects import DynamicCuboid, FixedCuboid
from isaacsim.core.prims import Articulation, XFormPrim
from isaacsim.core.utils.viewports import create_viewport_for_camera, set_camera_view
from isaacsim.sensors.camera import Camera
from pxr import Gf, Sdf, Usd, UsdGeom, UsdLux, UsdShade


PANTHERA_DESCRIPTION_XACRO_RELATIVE = Path(
    "src/panthera_ht_description_with_finger/urdf/Panthera-HT_description_with_finger.urdf.xacro"
)
PANTHERA_DESCRIPTION_URDF_RELATIVE = Path(
    "src/panthera_ht_description_with_finger/urdf/Panthera-HT_description_with_finger.urdf"
)
PANTHERA_MATERIAL_SPECS = {
    "polished_silver": ((0.46, 0.48, 0.50), 0.70, 0.58),
}
PANTHERA_LINK_MATERIALS = {
    "base_link": "polished_silver",
    "link1": "polished_silver",
    "link2": "polished_silver",
    "link3": "polished_silver",
    "link4": "polished_silver",
    "link5": "polished_silver",
    "link6": "polished_silver",
    "L_finger": "polished_silver",
    "R_finger": "polished_silver",
}
REAL_LAYOUT_ARM_EDGE_SIGN = -1.0
REAL_LAYOUT_ARM_EDGE_INSET = 0.09
REAL_LAYOUT_ARM_CORNER_INSET = 0.09
REAL_LAYOUT_ROBOT_BASE_Z_OFFSET = 0.025
REAL_LAYOUT_ROBOT_YAW_DEG = 90.0
REAL_LAYOUT_D435I_HEIGHT_ABOVE_TABLE = 0.60
REAL_LAYOUT_D435I_DOWNWARD_ANGLE_DEG = 45.0
WRIST_CAMERA_LINK_NAME = "link6"
LAYOUT_CAMERA_CAPTURE_WARMUP_FRAMES = 24
WRIST_RGB_CAMERA_RESOLUTION = (640, 480)
WRIST_RGB_CAMERA_FOCAL_LENGTH_M = 0.0028
WRIST_RGB_CAMERA_HORIZONTAL_FOV_DEG = 85.0
D435I_RGB_CAMERA_RESOLUTION = (640, 480)
D435I_DEPTH_CAMERA_RESOLUTION = (640, 480)
LAYOUT_CAMERA_MOTION_PROOF_FRAME = 180
LAYOUT_CAMERA_MOTION_PROOF_WARMUP_FRAMES = 12
LAYOUT_CAMERA_SEQUENCE_WARMUP_FRAMES = 8
LIVE_LAYOUT_CAMERA_VIEWPORT_NAMES = ("left_wrist_rgb", "right_wrist_rgb", "d435i_rgb")
LIVE_LAYOUT_CAMERA_VIEWPORT_SIZE = (640, 480)
D435I_RGB_INTRINSICS = {
    "fx": 611.9111,
    "fy": 612.2362,
    "cx": 322.5813,
    "cy": 242.8662,
    "distortion": [0.0, 0.0, 0.0, 0.0, 0.0],
    "physical_focal_length_m": 0.00188,
}
D435I_DEPTH_INTRINSICS = {
    "fx": 377.5064,
    "fy": 377.5064,
    "cx": 319.6759,
    "cy": 245.6448,
    "distortion": [0.0, 0.0, 0.0, 0.0, 0.0],
    "physical_focal_length_m": 0.00193,
}
UGREEN_WRIST_CAMERA_REPORT = {
    "source_camera": "UGREEN Camera 1080P",
    "stable_node": "/dev/v4l/by-id/usb-UGREEN_Camera_1080P_UGREEN_Camera_1080P_SN0001-video-index0",
    "sim_profile": "MJPG 640x480@30",
    "intrinsics_status": "approximate_uncalibrated",
    "calibration_note": "V4L2 does not expose fx/fy/cx/cy/distortion; run OpenCV/ROS calibration for 1:1 geometry.",
}
D435I_CAMERA_REPORT = {
    "source_camera": "Intel RealSense D435i",
    "serial": "918512073830",
    "firmware": "5.17.0.10",
    "usb_descriptor_observed": "2.1 / 480M",
    "usb_note": "Use USB 3 for stable simultaneous RGB+depth 30 FPS on real hardware.",
    "rgb_profile": "rgb8 640x480@30",
    "depth_profile": "z16 640x480@30",
}
REALISM_MATERIAL_SPECS = {
    "matte_wall": ((0.70, 0.73, 0.72), 0.86, 0.0),
    "warm_floor": ((0.38, 0.42, 0.40), 0.78, 0.0),
    "laminate_table": ((0.42, 0.29, 0.17), 0.66, 0.0),
    "black_rubber": ((0.015, 0.014, 0.013), 0.88, 0.0),
    "brushed_mount": ((0.26, 0.27, 0.28), 0.66, 0.45),
    "marker_red": ((0.78, 0.08, 0.04), 0.62, 0.0),
    "marker_green": ((0.05, 0.42, 0.14), 0.68, 0.0),
    "marker_blue": ((0.04, 0.18, 0.72), 0.60, 0.0),
    "paper_label": ((0.92, 0.90, 0.84), 0.72, 0.0),
}


def repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def file_sha256(path: Path) -> str | None:
    if not path.is_file():
        return None
    return hashlib.sha256(path.read_bytes()).hexdigest()


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
    with path.open("w", encoding="utf-8") as json_file:
        json.dump(json_ready(payload), json_file, indent=2, sort_keys=True)
        json_file.write("\n")


def relative_to_repo(path: Path) -> str:
    try:
        return path.resolve().relative_to(repo_root()).as_posix()
    except ValueError:
        return path.resolve().as_posix()


def parse_layout_sequence_frames(raw_value: str) -> list[int]:
    raw_value = (raw_value or "").strip()
    if not raw_value:
        return []
    frames = []
    for token in raw_value.split(","):
        token = token.strip()
        if not token:
            continue
        try:
            frame = int(token)
        except ValueError as exc:
            raise RuntimeError(f"Invalid --layout-sequence-frames token: {token!r}") from exc
        if frame < 0:
            raise RuntimeError(f"Layout sequence frame must be non-negative: {frame}")
        if frame not in frames:
            frames.append(frame)
    return frames


def realism_rng() -> np.random.Generator:
    return np.random.default_rng(args.realism_seed)


def jitter_color(
    rng: np.random.Generator, color: tuple[float, float, float], amount: float
) -> tuple[float, float, float]:
    if not args.randomize_realism:
        return color
    noise = rng.uniform(-amount, amount, size=3)
    return tuple(float(np.clip(channel + delta, 0.0, 1.0)) for channel, delta in zip(color, noise))


def jitter_scalar(rng: np.random.Generator, value: float, amount: float, minimum: float, maximum: float) -> float:
    if not args.randomize_realism:
        return value
    return float(np.clip(value + float(rng.uniform(-amount, amount)), minimum, maximum))


def ensure_urdf_importer_enabled() -> None:
    ext_name = "isaacsim.asset.importer.urdf"
    ext_manager = omni.kit.app.get_app().get_extension_manager()
    if not ext_manager.is_extension_enabled(ext_name):
        carb.log_info(f"Enabling extension: {ext_name}")
        ext_manager.set_extension_enabled_immediate(ext_name, True)
        simulation_app.update()


def resolve_panthera_urdf_path(cli_urdf: str, cli_ros2_root: str) -> Path:
    if cli_urdf:
        urdf_path = Path(cli_urdf).expanduser().resolve()
        if urdf_path.is_file():
            return urdf_path
        raise RuntimeError(f"PANTHERA_URDF/--urdf does not exist: {urdf_path}")

    candidate_roots = []
    if cli_ros2_root:
        candidate_roots.append(Path(cli_ros2_root).expanduser())
    candidate_roots.extend(
        [
            repo_root() / "external/Panthera-HT-ROS2",
            repo_root() / "external/Panthera-HT-ROS2/src",
        ]
    )

    for root in candidate_roots:
        root = root.resolve()
        for relative_path in (PANTHERA_DESCRIPTION_XACRO_RELATIVE, PANTHERA_DESCRIPTION_URDF_RELATIVE):
            direct = root / relative_path
            if direct.is_file():
                return direct
        for file_name in (
            "Panthera-HT_description_with_finger.urdf.xacro",
            "Panthera-HT_description_with_finger.urdf",
        ):
            nested_src = root / "panthera_ht_description_with_finger/urdf" / file_name
            if nested_src.is_file():
                return nested_src

    raise RuntimeError(
        "Cannot find Panthera-HT URDF. Clone the official ROS2 repo with:\n"
        "  git clone https://github.com/HighTorque-Robotics/Panthera-HT-ROS2.git external/Panthera-HT-ROS2\n"
        "or set PANTHERA_URDF=/absolute/path/to/Panthera-HT_description_with_finger.urdf.xacro"
    )


def resolve_mesh_dir(urdf_path: Path, cli_mesh_dir: str) -> Path:
    if cli_mesh_dir:
        mesh_dir = Path(cli_mesh_dir).expanduser().resolve()
        if mesh_dir.is_dir():
            return mesh_dir
        raise RuntimeError(f"PANTHERA_MESH_DIR/--mesh-dir does not exist: {mesh_dir}")

    description_root = urdf_path.parents[1]
    mesh_dir = description_root / "meshes"
    if mesh_dir.is_dir():
        return mesh_dir.resolve()

    raise RuntimeError(
        f"Cannot find Panthera mesh directory next to URDF: {mesh_dir}. "
        "Set PANTHERA_MESH_DIR=/absolute/path/to/meshes."
    )


def prepare_panthera_urdf_for_isaac(urdf_path: Path, mesh_dir: Path, robot_name: str) -> Path:
    with urdf_path.open("r", encoding="utf-8") as source_file:
        content = source_file.read()

    patched = content
    if urdf_path.suffix == ".xacro":
        patched = patched.replace(' xmlns:xacro="http://www.ros.org/wiki/xacro"', "")
        patched = re.sub(r"\s*<xacro:property[^>]*/>\n", "", patched)
        patched = patched.replace("${package_name}", "panthera_ht_description_with_finger")
        patched = patched.replace("${default_color}", "0.752941176470588 0.752941176470588 0.752941176470588 1")

    patched = re.sub(r'<robot\s+name="[^"]+"', f'<robot name="{robot_name}"', patched, count=1)

    package_refs = {
        "package://Panthera-HT_description/meshes/": mesh_dir.as_posix() + "/",
        "package://Panthera-HT_description_with_finger/meshes/": mesh_dir.as_posix() + "/",
        "package://panthera_ht_description_with_finger/meshes/": mesh_dir.as_posix() + "/",
    }
    for package_ref, absolute_ref in package_refs.items():
        patched = patched.replace(package_ref, absolute_ref)

    patched = re.sub(
        r'(<joint\s+name="(?:L_finger|R_finger)"[\s\S]*?<limit\s+lower="0"\s+upper="0\.04"\s+)effort="0"\s+velocity="0"',
        r'\1effort="5" velocity="0.2"',
        patched,
    )

    output_dir = repo_root() / "outputs/panthera_ht"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{robot_name}_isaacsim.urdf"
    if not output_path.exists() or output_path.read_text(encoding="utf-8") != patched:
        output_path.write_text(patched, encoding="utf-8")
    return output_path


def real_layout_arm_specs(table_size: float, tabletop_z: float) -> list[dict[str, object]]:
    edge_y = REAL_LAYOUT_ARM_EDGE_SIGN * (table_size / 2.0 - REAL_LAYOUT_ARM_EDGE_INSET)
    corner_x = table_size / 2.0 - REAL_LAYOUT_ARM_CORNER_INSET
    base_z = tabletop_z + REAL_LAYOUT_ROBOT_BASE_Z_OFFSET
    return [
        {
            "label": "front_left",
            "urdf_robot_name": "Panthera_HT_FrontLeft",
            "stage_root_path": "/Panthera_HT_FrontLeft",
            "articulation_name": "panthera_ht_front_left",
            "root_view_name": "panthera_ht_front_left_root",
            "mount_path": "/World/Table/PantheraMountPlate_FrontLeft",
            "mount_name": "panthera_mount_plate_front_left",
            "position": np.array([-corner_x, edge_y, base_z], dtype=np.float32),
            "yaw_deg": REAL_LAYOUT_ROBOT_YAW_DEG,
        },
        {
            "label": "front_right",
            "urdf_robot_name": "Panthera_HT_FrontRight",
            "stage_root_path": "/Panthera_HT_FrontRight",
            "articulation_name": "panthera_ht_front_right",
            "root_view_name": "panthera_ht_front_right_root",
            "mount_path": "/World/Table/PantheraMountPlate_FrontRight",
            "mount_name": "panthera_mount_plate_front_right",
            "position": np.array([corner_x, edge_y, base_z], dtype=np.float32),
            "yaw_deg": REAL_LAYOUT_ROBOT_YAW_DEG,
        },
    ]


def add_table_scene(world: World, table_size: float, table_height: float) -> float:
    tabletop_thickness = 0.06
    tabletop = np.array([table_size, table_size, tabletop_thickness], dtype=np.float32)
    tabletop_center = np.array([0.0, 0.0, table_height - tabletop_thickness / 2.0], dtype=np.float32)

    world.scene.add(
        FixedCuboid(
            prim_path="/World/Table/Top",
            name="panthera_table_top",
            position=tabletop_center,
            scale=tabletop,
            size=1.0,
            color=np.array([0.42, 0.28, 0.16], dtype=np.float32),
        )
    )

    leg_width = 0.055
    leg_height = table_height - tabletop_thickness
    leg_offset = table_size / 2.0 - 0.07
    leg_size = np.array([leg_width, leg_width, leg_height], dtype=np.float32)
    leg_z = leg_height / 2.0
    for index, (x_pos, y_pos) in enumerate(
        [
            (leg_offset, leg_offset),
            (leg_offset, -leg_offset),
            (-leg_offset, leg_offset),
            (-leg_offset, -leg_offset),
        ],
        start=1,
    ):
        world.scene.add(
            FixedCuboid(
                prim_path=f"/World/Table/Leg_{index}",
                name=f"panthera_table_leg_{index}",
                position=np.array([x_pos, y_pos, leg_z], dtype=np.float32),
                scale=leg_size,
                size=1.0,
                color=np.array([0.18, 0.18, 0.18], dtype=np.float32),
            )
        )

    plate_size = np.array([0.18, 0.18, 0.02], dtype=np.float32)
    for arm_spec in real_layout_arm_specs(table_size, table_height):
        mount_position = np.array(arm_spec["position"], dtype=np.float32).copy()
        mount_position[2] = table_height + plate_size[2] / 2.0
        world.scene.add(
            FixedCuboid(
                prim_path=arm_spec["mount_path"],
                name=arm_spec["mount_name"],
                position=mount_position,
                scale=plate_size,
                size=1.0,
                color=np.array([0.08, 0.08, 0.09], dtype=np.float32),
            )
        )

    cube_size = 0.045
    world.scene.add(
        DynamicCuboid(
            prim_path="/World/Table/TargetCube",
            name="panthera_target_cube",
            position=np.array([-0.18, 0.16, table_height + cube_size / 2.0 + 0.003], dtype=np.float32),
            scale=np.array([cube_size, cube_size, cube_size], dtype=np.float32),
            size=1.0,
            color=np.array([0.05, 0.28, 0.95], dtype=np.float32),
        )
    )

    return table_height


def add_lights(rng: np.random.Generator | None = None) -> None:
    rng = rng or realism_rng()
    stage = omni.usd.get_context().get_stage()

    dome = UsdLux.DomeLight.Define(stage, Sdf.Path("/World/DomeLight"))
    dome.CreateIntensityAttr(jitter_scalar(rng, 560.0, 110.0, 420.0, 760.0))
    dome.CreateExposureAttr(0.0)
    dome.CreateColorAttr(Gf.Vec3f(*jitter_color(rng, (0.78, 0.86, 1.0), 0.025)))

    sun = UsdLux.DistantLight.Define(stage, Sdf.Path("/World/SunLight"))
    sun.CreateIntensityAttr(jitter_scalar(rng, 1080.0, 220.0, 760.0, 1450.0))
    sun.CreateAngleAttr(2.5)
    sun.CreateColorAttr(Gf.Vec3f(*jitter_color(rng, (1.0, 0.95, 0.86), 0.025)))
    sun_xform = UsdGeom.Xformable(sun.GetPrim())
    sun_xform.AddRotateXYZOp().Set(Gf.Vec3f(-36.0, -28.0, 0.0))

    window_light = UsdLux.RectLight.Define(stage, Sdf.Path("/World/WindowDaylight"))
    window_light.CreateIntensityAttr(jitter_scalar(rng, 360.0, 90.0, 240.0, 540.0))
    window_light.CreateColorAttr(Gf.Vec3f(*jitter_color(rng, (0.92, 0.97, 1.0), 0.02)))
    window_light.CreateWidthAttr(1.6)
    window_light.CreateHeightAttr(1.0)
    window_xform = UsdGeom.Xformable(window_light.GetPrim())
    window_xform.AddTranslateOp().Set(Gf.Vec3f(-0.85, -1.15, 1.45))
    window_xform.AddRotateXYZOp().Set(Gf.Vec3f(-60.0, 0.0, -35.0))


def create_preview_surface_material(
    stage, material_path: str, color: tuple[float, float, float], roughness: float, metallic: float
) -> UsdShade.Material:
    UsdGeom.Scope.Define(stage, Sdf.Path("/World/Looks"))
    UsdGeom.Scope.Define(stage, Sdf.Path("/World/Looks/PantheraHT"))

    material = UsdShade.Material.Define(stage, Sdf.Path(material_path))
    shader = UsdShade.Shader.Define(stage, Sdf.Path(f"{material_path}/Shader"))
    shader.CreateIdAttr("UsdPreviewSurface")
    shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*color))
    shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(float(roughness))
    shader.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(float(metallic))
    material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
    return material


def create_realism_materials(stage, rng: np.random.Generator) -> dict[str, UsdShade.Material]:
    UsdGeom.Scope.Define(stage, Sdf.Path("/World/Looks/Realism"))
    materials = {}
    for material_name, (color, roughness, metallic) in REALISM_MATERIAL_SPECS.items():
        material_color = jitter_color(rng, color, 0.035)
        material_roughness = jitter_scalar(rng, roughness, 0.05, 0.05, 0.96)
        materials[material_name] = create_preview_surface_material(
            stage,
            f"/World/Looks/Realism/{material_name}",
            material_color,
            material_roughness,
            metallic,
        )
    return materials


def bind_material(prim_path: str, material: UsdShade.Material) -> None:
    stage = omni.usd.get_context().get_stage()
    prim = stage.GetPrimAtPath(prim_path)
    if prim.IsValid():
        UsdShade.MaterialBindingAPI.Apply(prim).Bind(
            material, bindingStrength=UsdShade.Tokens.strongerThanDescendants
        )
    else:
        carb.log_warn(f"Cannot bind realism material; prim not found: {prim_path}")


def create_panthera_materials(stage) -> dict[str, UsdShade.Material]:
    materials = {}
    for material_name, (color, roughness, metallic) in PANTHERA_MATERIAL_SPECS.items():
        materials[material_name] = create_preview_surface_material(
            stage, f"/World/Looks/PantheraHT/{material_name}", color, roughness, metallic
        )
    return materials


def apply_panthera_materials(root_path: str) -> int:
    stage = omni.usd.get_context().get_stage()
    materials = create_panthera_materials(stage)
    bound_count = 0

    for link_name, material_name in PANTHERA_LINK_MATERIALS.items():
        link_path = f"{root_path}/{link_name}"
        link_prim = stage.GetPrimAtPath(link_path)
        if not link_prim.IsValid():
            carb.log_warn(f"Panthera-HT link prim not found for material binding: {link_path}")
            continue
        UsdShade.MaterialBindingAPI.Apply(link_prim).Bind(
            materials[material_name], bindingStrength=UsdShade.Tokens.strongerThanDescendants
        )
        bound_count += 1

    return bound_count


def add_realism_environment(world: World, table_size: float, table_height: float, rng: np.random.Generator) -> None:
    stage = omni.usd.get_context().get_stage()
    materials = create_realism_materials(stage, rng)

    floor_size = np.array([3.4, 3.0, 0.025], dtype=np.float32)
    floor_path = "/World/RealismRoom/Floor"
    world.scene.add(
        FixedCuboid(
            prim_path=floor_path,
            name="panthera_realism_floor",
            position=np.array([0.22, 0.0, -0.0175], dtype=np.float32),
            scale=floor_size,
            size=1.0,
            color=np.array([0.38, 0.42, 0.40], dtype=np.float32),
        )
    )
    bind_material(floor_path, materials["warm_floor"])

    wall_specs = [
        (
            "/World/RealismRoom/BackWall",
            "panthera_realism_back_wall",
            np.array([0.22, 1.28, 0.82], dtype=np.float32),
            np.array([3.4, 0.04, 1.65], dtype=np.float32),
        ),
        (
            "/World/RealismRoom/SideWall",
            "panthera_realism_side_wall",
            np.array([-1.43, 0.0, 0.82], dtype=np.float32),
            np.array([0.04, 3.0, 1.65], dtype=np.float32),
        ),
    ]
    for prim_path, name, position, scale in wall_specs:
        world.scene.add(
            FixedCuboid(
                prim_path=prim_path,
                name=name,
                position=position,
                scale=scale,
                size=1.0,
                color=np.array([0.70, 0.73, 0.72], dtype=np.float32),
            )
        )
        bind_material(prim_path, materials["matte_wall"])

    tabletop_z = table_height
    edge_thickness = 0.025
    edge_drop = 0.035
    edge_specs = [
        (
            "/World/RealismProps/TableFrontEdge",
            "panthera_table_front_edge",
            np.array([0.0, -table_size / 2.0 - edge_thickness / 2.0, tabletop_z - edge_drop], dtype=np.float32),
            np.array([table_size + 0.035, edge_thickness, 0.07], dtype=np.float32),
        ),
        (
            "/World/RealismProps/TableRightEdge",
            "panthera_table_right_edge",
            np.array([table_size / 2.0 + edge_thickness / 2.0, 0.0, tabletop_z - edge_drop], dtype=np.float32),
            np.array([edge_thickness, table_size + 0.035, 0.07], dtype=np.float32),
        ),
    ]
    for prim_path, name, position, scale in edge_specs:
        world.scene.add(
            FixedCuboid(
                prim_path=prim_path,
                name=name,
                position=position,
                scale=scale,
                size=1.0,
                color=np.array([0.42, 0.29, 0.17], dtype=np.float32),
            )
        )
        bind_material(prim_path, materials["laminate_table"])

    cable_specs = [
        (
            "/World/RealismProps/RearCableA",
            "panthera_rear_cable_a",
            np.array([0.10, 0.31, tabletop_z + 0.012], dtype=np.float32),
            np.array([0.42, 0.018, 0.018], dtype=np.float32),
        ),
        (
            "/World/RealismProps/RearCableB",
            "panthera_rear_cable_b",
            np.array([0.40, 0.25, tabletop_z + 0.014], dtype=np.float32),
            np.array([0.018, 0.22, 0.018], dtype=np.float32),
        ),
    ]
    for prim_path, name, position, scale in cable_specs:
        world.scene.add(
            FixedCuboid(
                prim_path=prim_path,
                name=name,
                position=position,
                scale=scale,
                size=1.0,
                color=np.array([0.02, 0.02, 0.02], dtype=np.float32),
            )
        )
        bind_material(prim_path, materials["black_rubber"])

    marker_specs = [
        ("/World/RealismProps/MarkerRed", "panthera_marker_red", "marker_red", [-0.30, -0.24, tabletop_z + 0.012], [0.08, 0.05, 0.024]),
        ("/World/RealismProps/MarkerGreen", "panthera_marker_green", "marker_green", [-0.20, -0.29, tabletop_z + 0.010], [0.06, 0.04, 0.020]),
        ("/World/RealismProps/MarkerBlue", "panthera_marker_blue", "marker_blue", [-0.32, 0.28, tabletop_z + 0.011], [0.05, 0.05, 0.022]),
        ("/World/RealismProps/PaperLabel", "panthera_paper_label", "paper_label", [0.02, -0.32, tabletop_z + 0.003], [0.18, 0.08, 0.006]),
    ]
    for prim_path, name, material_name, position, scale in marker_specs:
        if args.randomize_realism:
            position = [position[0] + float(rng.uniform(-0.025, 0.025)), position[1] + float(rng.uniform(-0.02, 0.02)), position[2]]
        world.scene.add(
            FixedCuboid(
                prim_path=prim_path,
                name=name,
                position=np.array(position, dtype=np.float32),
                scale=np.array(scale, dtype=np.float32),
                size=1.0,
                color=np.array(REALISM_MATERIAL_SPECS[material_name][0], dtype=np.float32),
            )
        )
        bind_material(prim_path, materials[material_name])

    bind_material("/World/Table/Top", materials["laminate_table"])
    for arm_spec in real_layout_arm_specs(table_size, table_height):
        bind_material(arm_spec["mount_path"], materials["brushed_mount"])
    for index in range(1, 5):
        bind_material(f"/World/Table/Leg_{index}", materials["black_rubber"])


def quat_wxyz_from_yaw_deg(yaw_deg: float) -> np.ndarray:
    half = math.radians(yaw_deg) * 0.5
    return np.array([math.cos(half), 0.0, 0.0, math.sin(half)], dtype=np.float32)


def look_at_quat_wxyz(eye: np.ndarray, target: np.ndarray, up_hint: np.ndarray) -> np.ndarray:
    forward = target - eye
    forward_norm = float(np.linalg.norm(forward))
    if forward_norm < 1e-6:
        forward = np.array([1.0, 0.0, 0.0], dtype=np.float64)
    else:
        forward = (forward / forward_norm).astype(np.float64)

    up = up_hint.astype(np.float64).copy()
    up_norm = float(np.linalg.norm(up))
    if up_norm < 1e-6:
        up = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    else:
        up = up / up_norm

    right = np.cross(up, forward)
    right_norm = float(np.linalg.norm(right))
    if right_norm < 1e-6:
        up = np.array([0.0, 1.0, 0.0], dtype=np.float64)
        right = np.cross(up, forward)
        right_norm = float(np.linalg.norm(right))
        if right_norm < 1e-6:
            return np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
    right = right / right_norm
    camera_up = np.cross(forward, right)

    rot = np.array(
        [
            [forward[0], right[0], camera_up[0]],
            [forward[1], right[1], camera_up[1]],
            [forward[2], right[2], camera_up[2]],
        ],
        dtype=np.float64,
    )
    trace = float(np.trace(rot))
    if trace > 0.0:
        scale = math.sqrt(trace + 1.0) * 2.0
        quat = np.array(
            [
                0.25 * scale,
                (rot[2, 1] - rot[1, 2]) / scale,
                (rot[0, 2] - rot[2, 0]) / scale,
                (rot[1, 0] - rot[0, 1]) / scale,
            ],
            dtype=np.float64,
        )
    else:
        diagonal_index = int(np.argmax(np.diag(rot)))
        if diagonal_index == 0:
            scale = math.sqrt(1.0 + rot[0, 0] - rot[1, 1] - rot[2, 2]) * 2.0
            quat = np.array(
                [
                    (rot[2, 1] - rot[1, 2]) / scale,
                    0.25 * scale,
                    (rot[0, 1] + rot[1, 0]) / scale,
                    (rot[0, 2] + rot[2, 0]) / scale,
                ],
                dtype=np.float64,
            )
        elif diagonal_index == 1:
            scale = math.sqrt(1.0 + rot[1, 1] - rot[0, 0] - rot[2, 2]) * 2.0
            quat = np.array(
                [
                    (rot[0, 2] - rot[2, 0]) / scale,
                    (rot[0, 1] + rot[1, 0]) / scale,
                    0.25 * scale,
                    (rot[1, 2] + rot[2, 1]) / scale,
                ],
                dtype=np.float64,
            )
        else:
            scale = math.sqrt(1.0 + rot[2, 2] - rot[0, 0] - rot[1, 1]) * 2.0
            quat = np.array(
                [
                    (rot[1, 0] - rot[0, 1]) / scale,
                    (rot[0, 2] + rot[2, 0]) / scale,
                    (rot[1, 2] + rot[2, 1]) / scale,
                    0.25 * scale,
                ],
                dtype=np.float64,
            )

    quat_norm = float(np.linalg.norm(quat))
    if quat_norm > 1e-6:
        quat = quat / quat_norm
    return quat.astype(np.float32)


def horizontal_aperture_for_fov(focal_length_m: float, horizontal_fov_deg: float) -> float:
    return 2.0 * focal_length_m * math.tan(math.radians(horizontal_fov_deg) * 0.5)


def world_point_from_prim_local_offset(prim_path: str, local_offset: np.ndarray) -> np.ndarray:
    transform = world_transform_for_prim(prim_path)
    point = transform.Transform(
        Gf.Vec3d(float(local_offset[0]), float(local_offset[1]), float(local_offset[2]))
    )
    return np.array([point[0], point[1], point[2]], dtype=np.float32)


def world_transform_for_prim(prim_path: str) -> Gf.Matrix4d:
    stage = omni.usd.get_context().get_stage()
    prim = stage.GetPrimAtPath(prim_path)
    if not prim.IsValid():
        raise RuntimeError(f"Cannot find prim: {prim_path}")
    return UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default())


def rotation_matrix_from_gf_transform(transform: Gf.Matrix4d) -> np.ndarray:
    return np.array(
        [
            [float(transform[0][0]), float(transform[0][1]), float(transform[0][2])],
            [float(transform[1][0]), float(transform[1][1]), float(transform[1][2])],
            [float(transform[2][0]), float(transform[2][1]), float(transform[2][2])],
        ],
        dtype=np.float64,
    )


def real_layout_camera_specs(table_size: float, tabletop_z: float, arm_root_paths: dict[str, str]) -> list[dict[str, object]]:
    table_center = np.array([0.0, 0.0, tabletop_z + 0.03], dtype=np.float32)
    d435i_position = np.array(
        [
            0.0,
            -REAL_LAYOUT_ARM_EDGE_SIGN * table_size / 2.0,
            tabletop_z + REAL_LAYOUT_D435I_HEIGHT_ABOVE_TABLE,
        ],
        dtype=np.float32,
    )
    d435i_horizontal_reach = REAL_LAYOUT_D435I_HEIGHT_ABOVE_TABLE / math.tan(
        math.radians(REAL_LAYOUT_D435I_DOWNWARD_ANGLE_DEG)
    )
    d435i_target = np.array(
        [
            0.0,
            d435i_position[1] + REAL_LAYOUT_ARM_EDGE_SIGN * d435i_horizontal_reach,
            tabletop_z,
        ],
        dtype=np.float32,
    )
    return [
        {
            "name": "left_wrist_rgb",
            "prim_path": f"{arm_root_paths['front_left']}/{WRIST_CAMERA_LINK_NAME}/WristRGBCamera",
            "mode": "wrist",
            "stream": "rgb",
            "source_camera": UGREEN_WRIST_CAMERA_REPORT["source_camera"],
            "intrinsics_status": UGREEN_WRIST_CAMERA_REPORT["intrinsics_status"],
            "resolution": WRIST_RGB_CAMERA_RESOLUTION,
            "translation": np.array([0.22, 0.0, 0.075], dtype=np.float32),
            "target": table_center,
            "up": np.array([0.0, 0.0, 1.0], dtype=np.float32),
            "local_forward": np.array([1.0, 0.0, -0.65], dtype=np.float32),
            "local_up": np.array([0.0, 0.0, 1.0], dtype=np.float32),
            "intrinsics": None,
            "focal_length_m": WRIST_RGB_CAMERA_FOCAL_LENGTH_M,
            "horizontal_aperture_m": horizontal_aperture_for_fov(
                WRIST_RGB_CAMERA_FOCAL_LENGTH_M, WRIST_RGB_CAMERA_HORIZONTAL_FOV_DEG
            ),
        },
        {
            "name": "right_wrist_rgb",
            "prim_path": f"{arm_root_paths['front_right']}/{WRIST_CAMERA_LINK_NAME}/WristRGBCamera",
            "mode": "wrist",
            "stream": "rgb",
            "source_camera": UGREEN_WRIST_CAMERA_REPORT["source_camera"],
            "intrinsics_status": UGREEN_WRIST_CAMERA_REPORT["intrinsics_status"],
            "resolution": WRIST_RGB_CAMERA_RESOLUTION,
            "translation": np.array([0.22, 0.0, 0.075], dtype=np.float32),
            "target": table_center,
            "up": np.array([0.0, 0.0, 1.0], dtype=np.float32),
            "local_forward": np.array([1.0, 0.0, -0.65], dtype=np.float32),
            "local_up": np.array([0.0, 0.0, 1.0], dtype=np.float32),
            "intrinsics": None,
            "focal_length_m": WRIST_RGB_CAMERA_FOCAL_LENGTH_M,
            "horizontal_aperture_m": horizontal_aperture_for_fov(
                WRIST_RGB_CAMERA_FOCAL_LENGTH_M, WRIST_RGB_CAMERA_HORIZONTAL_FOV_DEG
            ),
        },
        {
            "name": "d435i_rgb",
            "prim_path": "/World/RealSenseD435i/RGBCamera",
            "mode": "world",
            "stream": "rgb",
            "source_camera": D435I_CAMERA_REPORT["source_camera"],
            "intrinsics_status": "sdk_profile_640x480_at_30fps",
            "resolution": D435I_RGB_CAMERA_RESOLUTION,
            "position": d435i_position,
            "target": d435i_target,
            "up": np.array([0.0, 0.0, 1.0], dtype=np.float32),
            "intrinsics": D435I_RGB_INTRINSICS,
            "focal_length_m": D435I_RGB_INTRINSICS["physical_focal_length_m"],
            "horizontal_aperture_m": D435I_RGB_INTRINSICS["physical_focal_length_m"]
            * D435I_RGB_CAMERA_RESOLUTION[0]
            / D435I_RGB_INTRINSICS["fx"],
        },
        {
            "name": "d435i_depth",
            "prim_path": "/World/RealSenseD435i/DepthCamera",
            "mode": "world",
            "stream": "depth",
            "source_camera": D435I_CAMERA_REPORT["source_camera"],
            "intrinsics_status": "sdk_profile_640x480_at_30fps",
            "resolution": D435I_DEPTH_CAMERA_RESOLUTION,
            "position": d435i_position,
            "target": d435i_target,
            "up": np.array([0.0, 0.0, 1.0], dtype=np.float32),
            "intrinsics": D435I_DEPTH_INTRINSICS,
            "depth_intrinsics": D435I_DEPTH_INTRINSICS,
            "focal_length_m": D435I_DEPTH_INTRINSICS["physical_focal_length_m"],
            "horizontal_aperture_m": D435I_DEPTH_INTRINSICS["physical_focal_length_m"]
            * D435I_DEPTH_CAMERA_RESOLUTION[0]
            / D435I_DEPTH_INTRINSICS["fx"],
        },
    ]


def create_layout_cameras(table_size: float, tabletop_z: float, arm_root_paths: dict[str, str]) -> dict[str, Camera]:
    cameras = {}
    for spec in real_layout_camera_specs(table_size, tabletop_z, arm_root_paths):
        camera = Camera(
            prim_path=spec["prim_path"],
            name=spec["name"],
            frequency=30,
            resolution=spec["resolution"],
        )
        if spec["mode"] == "world":
            orientation = look_at_quat_wxyz(spec["position"], spec["target"], spec["up"])
            camera.set_world_pose(position=spec["position"], orientation=orientation, camera_axes="world")
            focus_distance = float(np.linalg.norm(spec["position"] - spec["target"]))
        else:
            parent_path = spec["prim_path"].rsplit("/", 1)[0]
            parent_transform = world_transform_for_prim(parent_path)
            parent_rot = rotation_matrix_from_gf_transform(parent_transform)
            position = world_point_from_prim_local_offset(parent_path, spec["translation"])
            target = position + (parent_rot @ spec["local_forward"]).astype(np.float32)
            orientation = look_at_quat_wxyz(position, target, (parent_rot @ spec["local_up"]).astype(np.float32))
            camera.set_world_pose(position=position, orientation=orientation, camera_axes="world")
            camera._panthera_wrist_parent_path = parent_path
            focus_distance = 0.35
        camera.set_focal_length(spec["focal_length_m"])
        camera.set_horizontal_aperture(spec["horizontal_aperture_m"])
        camera.set_focus_distance(focus_distance)
        camera.set_lens_aperture(0.0)
        camera.set_clipping_range(near_distance=0.01, far_distance=10.0)
        if spec["intrinsics"] is not None:
            intrinsics = spec["intrinsics"]
            camera.set_opencv_pinhole_properties(
                cx=intrinsics["cx"],
                cy=intrinsics["cy"],
                fx=intrinsics["fx"],
                fy=intrinsics["fy"],
                pinhole=intrinsics["distortion"],
            )
        if spec.get("depth_intrinsics") is not None:
            camera._panthera_enable_distance_to_image_plane = True
        camera._panthera_layout_spec = json_ready(
            {
                **spec,
                "frequency_hz": 30,
                "focus_distance_m": focus_distance,
                "parent_path": spec["prim_path"].rsplit("/", 1)[0],
                "distance_to_image_plane_enabled": bool(spec.get("depth_intrinsics") is not None),
            }
        )
        print(
            f"[Panthera-HT] Layout camera {spec['name']}: "
            f"prim={spec['prim_path']}, mode={spec['mode']}, resolution={spec['resolution']}"
        )
        cameras[spec["name"]] = camera
    return cameras


def verify_layout_camera_bindings(cameras: dict[str, Camera]) -> None:
    stage = omni.usd.get_context().get_stage()
    for camera_name, camera in cameras.items():
        expected_parent = getattr(camera, "_panthera_wrist_parent_path", "")
        if not expected_parent:
            continue
        camera_prim = stage.GetPrimAtPath(camera.prim_path)
        if not camera_prim.IsValid():
            raise RuntimeError(f"Layout camera {camera_name} prim does not exist: {camera.prim_path}")
        actual_parent = str(camera_prim.GetParent().GetPath())
        if actual_parent != expected_parent or not actual_parent.endswith(f"/{WRIST_CAMERA_LINK_NAME}"):
            raise RuntimeError(
                f"Layout camera {camera_name} is not bound to wrist link {WRIST_CAMERA_LINK_NAME}: "
                f"parent={actual_parent}, expected={expected_parent}"
            )
        print(f"[Panthera-HT] Layout camera {camera_name} bound to wrist parent: {actual_parent}")
    return


def layout_camera_pose_signatures(cameras: dict[str, Camera]) -> dict[str, dict[str, np.ndarray]]:
    signatures = {}
    for camera_name, camera in cameras.items():
        transform = world_transform_for_prim(camera.prim_path)
        origin = transform.Transform(Gf.Vec3d(0.0, 0.0, 0.0))
        forward_point = transform.Transform(Gf.Vec3d(1.0, 0.0, 0.0))
        up_point = transform.Transform(Gf.Vec3d(0.0, 0.0, 1.0))
        position = np.array([origin[0], origin[1], origin[2]], dtype=np.float64)
        forward = np.array(
            [forward_point[0] - origin[0], forward_point[1] - origin[1], forward_point[2] - origin[2]],
            dtype=np.float64,
        )
        up = np.array([up_point[0] - origin[0], up_point[1] - origin[1], up_point[2] - origin[2]], dtype=np.float64)
        for axis in (forward, up):
            axis_norm = float(np.linalg.norm(axis))
            if axis_norm > 1e-9:
                axis /= axis_norm
        signatures[camera_name] = {"position": position, "forward": forward, "up": up}
    return signatures


def angle_between_vectors_deg(first: np.ndarray, second: np.ndarray) -> float:
    first_norm = float(np.linalg.norm(first))
    second_norm = float(np.linalg.norm(second))
    if first_norm < 1e-9 or second_norm < 1e-9:
        return 0.0
    dot = float(np.clip(np.dot(first / first_norm, second / second_norm), -1.0, 1.0))
    return math.degrees(math.acos(dot))


def print_layout_camera_motion_evidence(
    before: dict[str, dict[str, np.ndarray]], after: dict[str, dict[str, np.ndarray]], frame_index: int
) -> None:
    for camera_name in sorted(before):
        if camera_name not in after:
            continue
        position_delta = float(np.linalg.norm(after[camera_name]["position"] - before[camera_name]["position"]))
        forward_delta_deg = angle_between_vectors_deg(before[camera_name]["forward"], after[camera_name]["forward"])
        up_delta_deg = angle_between_vectors_deg(before[camera_name]["up"], after[camera_name]["up"])
        print(
            f"[Panthera-HT] Layout camera motion proof {camera_name}: "
            f"frame={frame_index}, position_delta={position_delta:.5f}m, "
            f"forward_delta={forward_delta_deg:.3f}deg, up_delta={up_delta_deg:.3f}deg"
        )


def enable_layout_depth_streams(cameras: dict[str, Camera]) -> None:
    for camera_name, camera in cameras.items():
        if not getattr(camera, "_panthera_enable_distance_to_image_plane", False):
            continue
        try:
            camera.add_distance_to_image_plane_to_frame()
        except Exception as exc:
            raise RuntimeError(f"Failed to enable depth stream for layout camera {camera_name}") from exc
        print(
            f"[Panthera-HT] Layout camera {camera_name} depth stream enabled; intrinsics recorded: "
            f"fx={D435I_DEPTH_INTRINSICS['fx']}, fy={D435I_DEPTH_INTRINSICS['fy']}, "
            f"cx={D435I_DEPTH_INTRINSICS['cx']}, cy={D435I_DEPTH_INTRINSICS['cy']}"
        )
    return


def create_layout_camera_live_viewports(cameras: dict[str, Camera]) -> list[object]:
    if not cameras or not args.show_layout_camera_viewports:
        return []
    if args.headless:
        print("[Panthera-HT] Live layout camera viewports require non-headless mode; skipping.")
        carb.log_info("Panthera-HT live layout camera viewports skipped because headless=True")
        return []

    viewport_width, viewport_height = LIVE_LAYOUT_CAMERA_VIEWPORT_SIZE
    windows = []
    for viewport_index, camera_name in enumerate(LIVE_LAYOUT_CAMERA_VIEWPORT_NAMES):
        camera = cameras.get(camera_name)
        if camera is None:
            raise RuntimeError(f"Requested live layout camera viewport is missing camera: {camera_name}")
        position_x = 20 + (viewport_width + 40) * (viewport_index % 2)
        position_y = 40 + (viewport_height + 60) * (viewport_index // 2)
        try:
            window = create_viewport_for_camera(
                viewport_name=f"Panthera {camera_name}",
                camera_prim_path=camera.prim_path,
                width=viewport_width,
                height=viewport_height,
                position_x=position_x,
                position_y=position_y,
            )
        except Exception as exc:
            raise RuntimeError(f"Failed to create live viewport for layout camera {camera_name}") from exc
        windows.append(window)
        print(f"[Panthera-HT] Live layout camera viewport ready: {camera_name} -> {camera.prim_path}")
        carb.log_info(f"Panthera-HT live layout camera viewport ready: {camera_name} -> {camera.prim_path}")
    return windows


def realism_capture_postprocess(rgb: np.ndarray, camera_name: str) -> np.ndarray:
    if args.disable_realism:
        return rgb
    image = np.asarray(rgb, dtype=np.float32)
    rng = np.random.default_rng(args.realism_seed + sum(ord(char) for char in camera_name))
    brightness = 1.0
    contrast = 1.0
    noise_sigma = 0.0
    if args.randomize_realism:
        brightness = float(rng.uniform(0.94, 1.08))
        contrast = float(rng.uniform(0.96, 1.06))
        noise_sigma = float(rng.uniform(0.0, 2.2))
    image = (image - 127.5) * contrast + 127.5
    image *= brightness
    if noise_sigma > 0.0:
        image += rng.normal(0.0, noise_sigma, size=image.shape)
    return np.clip(image, 0, 255).astype(np.uint8)


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


def layout_camera_rgb(camera_name: str, camera: Camera) -> np.ndarray:
    rgb = camera.get_rgb()
    if rgb is None or getattr(rgb, "size", 0) == 0:
        raise RuntimeError(f"layout camera {camera_name} returned no RGB data")
    return realism_capture_postprocess(np.asarray(rgb), camera_name)


def save_layout_camera_screenshots(cameras: dict[str, Camera], frame_label: str) -> list[Path]:
    output_dir = repo_root() / "outputs/panthera_ht/layout_cameras"
    saved_paths = []
    for camera_name, camera in cameras.items():
        output_path = output_dir / f"{camera_name}_{frame_label}.png"
        write_png_rgb(output_path, layout_camera_rgb(camera_name, camera))
        saved_paths.append(output_path)
    return saved_paths


def layout_depth_array(camera_name: str, camera: Camera) -> np.ndarray | None:
    if not getattr(camera, "_panthera_enable_distance_to_image_plane", False):
        return None
    depth = camera.get_depth()
    if depth is None or getattr(depth, "size", 0) == 0:
        raise RuntimeError(f"layout camera {camera_name} returned no distance_to_image_plane data")
    depth_array = np.asarray(depth, dtype=np.float32)
    if depth_array.ndim == 3 and depth_array.shape[-1] == 1:
        depth_array = depth_array[:, :, 0]
    if depth_array.ndim != 2:
        raise RuntimeError(f"Expected HxW depth array for {camera_name}, got shape {depth_array.shape}")
    return depth_array


def depth_stats(depth: np.ndarray) -> dict[str, object]:
    finite_mask = np.isfinite(depth)
    finite_count = int(np.count_nonzero(finite_mask))
    stats = {
        "shape": list(depth.shape),
        "dtype": str(depth.dtype),
        "finite_count": finite_count,
        "finite_ratio": float(finite_count / depth.size) if depth.size else 0.0,
    }
    if finite_count:
        finite_depth = depth[finite_mask]
        stats.update(
            {
                "min_m": float(np.min(finite_depth)),
                "max_m": float(np.max(finite_depth)),
                "mean_m": float(np.mean(finite_depth)),
            }
        )
    return stats


def pose_targets_by_name(panthera: Articulation, frame_index: int) -> dict[str, float]:
    pose = pose_for_dofs(panthera.dof_names, frame_index, args.no_motion).reshape(-1)
    return {dof_name: float(pose[dof_index]) for dof_index, dof_name in enumerate(panthera.dof_names)}


def actual_joint_positions_by_name(panthera: Articulation) -> dict[str, float]:
    try:
        positions = np.asarray(panthera.get_joint_positions(), dtype=np.float32).reshape(-1)
    except Exception as exc:
        return {"read_error": str(exc)}
    return {dof_name: float(positions[dof_index]) for dof_index, dof_name in enumerate(panthera.dof_names[: len(positions)])}


def arm_sequence_state(
    arms: list[dict[str, object]],
    pantheras: list[Articulation],
    frame_index: int,
    previous_targets: dict[str, dict[str, float]],
) -> list[dict[str, object]]:
    states = []
    for arm, panthera in zip(arms, pantheras):
        target_by_name = pose_targets_by_name(panthera, frame_index)
        previous_by_name = previous_targets.get(str(arm["label"]), {})
        action_delta = {
            dof_name: float(target_value - previous_by_name.get(dof_name, target_value))
            for dof_name, target_value in target_by_name.items()
        }
        states.append(
            {
                "label": arm["label"],
                "root_path": arm["root_path"],
                "articulation_path": arm["articulation_path"],
                "target_joint_positions": target_by_name,
                "target_delta_from_previous": action_delta,
                "actual_joint_positions": actual_joint_positions_by_name(panthera),
            }
        )
        previous_targets[str(arm["label"])] = target_by_name
    return states


def camera_sequence_specs(cameras: dict[str, Camera]) -> dict[str, dict[str, object]]:
    return {
        camera_name: {
            **getattr(camera, "_panthera_layout_spec", {}),
            "prim_path": camera.prim_path,
            "wrist_parent_path": getattr(camera, "_panthera_wrist_parent_path", None),
        }
        for camera_name, camera in cameras.items()
    }


def save_layout_camera_sequence(
    world: World,
    arms: list[dict[str, object]],
    pantheras: list[Articulation],
    cameras: dict[str, Camera],
    frame_indices: list[int],
    source_urdf_path: Path,
    mesh_dir: Path,
    tabletop_z: float,
) -> Path | None:
    if not frame_indices:
        print("[Panthera-HT] Layout VLA sequence export disabled: no frames requested.")
        return None

    output_dir = repo_root() / "outputs/panthera_ht/layout_cameras/sequence"
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    script_path = Path(__file__).resolve()
    launcher_path = repo_root() / "run_panthera_ht_one_click.sh"
    previous_targets: dict[str, dict[str, float]] = {}
    manifest_frames = []
    started_wall_time = time.time()

    for sequence_index, frame_index in enumerate(frame_indices):
        for panthera in pantheras:
            panthera.set_joint_positions(pose_for_dofs(panthera.dof_names, frame_index, args.no_motion))
        for _ in range(LAYOUT_CAMERA_SEQUENCE_WARMUP_FRAMES):
            world.step(render=True)

        frame_dir = output_dir / f"frame_{frame_index:06d}"
        frame_dir.mkdir(parents=True, exist_ok=True)
        observations = {}
        for camera_name, camera in cameras.items():
            rgb_path = frame_dir / f"{camera_name}.png"
            write_png_rgb(rgb_path, layout_camera_rgb(camera_name, camera))
            observation = {
                "rgb_path": relative_to_repo(rgb_path),
                "stream": getattr(camera, "_panthera_layout_spec", {}).get("stream", "rgb"),
                "prim_path": camera.prim_path,
            }
            depth = layout_depth_array(camera_name, camera)
            if depth is not None:
                depth_path = frame_dir / f"{camera_name}_distance_to_image_plane_m.npy"
                np.save(depth_path, depth)
                observation["distance_to_image_plane_m_path"] = relative_to_repo(depth_path)
                observation["distance_to_image_plane_stats"] = depth_stats(depth)
            observations[camera_name] = observation

        camera_poses = layout_camera_pose_signatures(cameras)
        frame_payload = {
            "schema": "panthera_ht_layout_vla_frame_v1",
            "sequence_index": sequence_index,
            "motion_frame_index": frame_index,
            "capture_wall_time_s": time.time(),
            "motion_enabled": not args.no_motion,
            "arms": arm_sequence_state(arms, pantheras, frame_index, previous_targets),
            "camera_world_poses": camera_poses,
            "observations": observations,
        }
        frame_json_path = frame_dir / "frame.json"
        write_json(frame_json_path, frame_payload)
        manifest_frames.append(
            {
                "sequence_index": sequence_index,
                "motion_frame_index": frame_index,
                "frame_json_path": relative_to_repo(frame_json_path),
                "observations": observations,
            }
        )
        print(f"[Panthera-HT] Saved VLA layout sequence frame {frame_index}: {frame_dir}")

    manifest_path = output_dir / "episode_manifest.json"
    manifest = {
        "schema": "panthera_ht_layout_vla_episode_v1",
        "generated_wall_time_s": started_wall_time,
        "output_dir": relative_to_repo(output_dir),
        "source_hashes": {
            "panthera_ht_table_py": file_sha256(script_path),
            "run_panthera_ht_one_click_sh": file_sha256(launcher_path),
        },
        "command_context": {
            "headless": args.headless,
            "max_frames": args.max_frames,
            "no_motion": args.no_motion,
            "layout_sequence_frames_arg": args.layout_sequence_frames,
            "disable_realism": args.disable_realism,
            "randomize_realism": args.randomize_realism,
            "realism_seed": args.realism_seed,
        },
        "scene": {
            "stage_units_in_meters": 1.0,
            "table_size_m": args.table_size,
            "table_height_m": args.table_height,
            "tabletop_z_m": tabletop_z,
            "arm_edge_sign": REAL_LAYOUT_ARM_EDGE_SIGN,
            "arm_edge_inset_m": REAL_LAYOUT_ARM_EDGE_INSET,
            "arm_corner_inset_m": REAL_LAYOUT_ARM_CORNER_INSET,
            "robot_yaw_deg": REAL_LAYOUT_ROBOT_YAW_DEG,
            "d435i_height_above_table_m": REAL_LAYOUT_D435I_HEIGHT_ABOVE_TABLE,
            "d435i_downward_angle_deg": REAL_LAYOUT_D435I_DOWNWARD_ANGLE_DEG,
        },
        "camera_reports": {
            "d435i": D435I_CAMERA_REPORT,
            "d435i_rgb_intrinsics": D435I_RGB_INTRINSICS,
            "d435i_depth_intrinsics": D435I_DEPTH_INTRINSICS,
            "ugreen_wrist": UGREEN_WRIST_CAMERA_REPORT,
            "ugreen_wrist_fov_deg": WRIST_RGB_CAMERA_HORIZONTAL_FOV_DEG,
            "ugreen_wrist_resolution": WRIST_RGB_CAMERA_RESOLUTION,
        },
        "physical_camera_groups": {
            "left_wrist_ugreen": {"logical_streams": ["left_wrist_rgb"]},
            "right_wrist_ugreen": {"logical_streams": ["right_wrist_rgb"]},
            "d435i": {"logical_streams": ["d435i_rgb", "d435i_depth"]},
        },
        "cameras": camera_sequence_specs(cameras),
        "arms": arms,
        "source_urdf_path": source_urdf_path,
        "mesh_dir": mesh_dir,
        "frames": manifest_frames,
        "vla_readiness_note": (
            "This export is a synchronized visual/depth observation sequence with joint targets and metadata. "
            "It is suitable for VLA input plumbing and QA, but control training still requires task labels, "
            "action semantics, policy schema, calibration validation, and real/sim normalization."
        ),
        "sim_to_real_note": (
            "D435i RGB/depth intrinsics match the supplied 640x480@30 SDK profiles. "
            "UGREEN wrist cameras remain approximate 85deg FOV proxies until real calibration supplies fx/fy/cx/cy/distortion."
        ),
    }
    write_json(manifest_path, manifest)
    print(f"[Panthera-HT] Saved VLA layout sequence manifest: {manifest_path}")
    carb.log_info(f"Saved Panthera-HT VLA layout sequence manifest: {manifest_path}")
    return manifest_path


def import_panthera_to_stage(urdf_path: Path, expected_root_path: str) -> str:
    status, import_config = omni.kit.commands.execute("URDFCreateImportConfig")
    if not status:
        raise RuntimeError("Failed to create URDF import config")

    import_config.merge_fixed_joints = False
    import_config.fix_base = True
    import_config.make_default_prim = False
    import_config.create_physics_scene = False
    import_config.convex_decomp = False
    import_config.import_inertia_tensor = True
    import_config.parse_mimic = False
    import_config.distance_scale = 1.0

    stage = omni.usd.get_context().get_stage()
    expected_path = Sdf.Path(expected_root_path)
    if stage.GetPrimAtPath(expected_path).IsValid():
        raise RuntimeError(f"Panthera-HT import root already exists: {expected_root_path}")

    before_paths = {str(prim.GetPath()) for prim in stage.Traverse()}
    status, articulation_path = omni.kit.commands.execute(
        "URDFParseAndImportFile",
        urdf_path=urdf_path.as_posix(),
        import_config=import_config,
        get_articulation_root=True,
    )
    if not status or not articulation_path:
        raise RuntimeError(f"Failed to import Panthera-HT URDF: {urdf_path}")

    root_path = imported_robot_root_path(articulation_path)
    if root_path != expected_root_path:
        imported_root_path = Sdf.Path(root_path)
        if root_path.count("/") > 1:
            if stage.GetPrimAtPath(expected_path).IsValid():
                raise RuntimeError(f"Cannot relocate nested Panthera-HT import; target exists: {expected_root_path}")
            omni.kit.commands.execute(
                "MovePrimCommand",
                path_from=root_path,
                path_to=expected_root_path,
                keep_world_transform=False,
            )
            articulation_path = expected_root_path + articulation_path[len(root_path):]
        elif root_path not in before_paths:
            if stage.GetPrimAtPath(expected_path).IsValid():
                raise RuntimeError(f"Cannot rename Panthera-HT import; target exists: {expected_root_path}")
            omni.kit.commands.execute(
                "MovePrimCommand",
                path_from=root_path,
                path_to=expected_root_path,
                keep_world_transform=False,
            )
            articulation_path = expected_root_path + articulation_path[len(root_path):]

    root_path = imported_robot_root_path(articulation_path)
    if root_path != expected_root_path:
        raise RuntimeError(
            f"Panthera-HT import root mismatch: expected {expected_root_path}, got {root_path}. "
            "The URDF importer did not isolate this robot instance."
        )
    return articulation_path


def imported_robot_root_path(articulation_path: str) -> str:
    if articulation_path.count("/") <= 1:
        return articulation_path
    return articulation_path.rsplit("/", 1)[0]


def world_position_for_prim(prim_path: str) -> np.ndarray:
    transform = world_transform_for_prim(prim_path)
    origin = transform.Transform(Gf.Vec3d(0.0, 0.0, 0.0))
    return np.array([origin[0], origin[1], origin[2]], dtype=np.float64)


def print_layout_arm_reach_evidence(arms: list[dict[str, object]], frame_label: str) -> None:
    for arm in arms:
        root_path = str(arm["root_path"])
        wrist_path = f"{root_path}/{WRIST_CAMERA_LINK_NAME}"
        root_position = world_position_for_prim(root_path)
        wrist_position = world_position_for_prim(wrist_path)
        delta = wrist_position - root_position
        inward_sign = float(delta[1] * -REAL_LAYOUT_ARM_EDGE_SIGN)
        print(
            f"[Panthera-HT] Layout arm reach proof {arm['label']} {frame_label}: "
            f"root=({root_position[0]:.4f},{root_position[1]:.4f},{root_position[2]:.4f}), "
            f"wrist=({wrist_position[0]:.4f},{wrist_position[1]:.4f},{wrist_position[2]:.4f}), "
            f"delta=({delta[0]:.4f},{delta[1]:.4f},{delta[2]:.4f}), inward_delta={inward_sign:.4f}m"
        )


def pose_for_dofs(dof_names: list[str], frame_index: int, no_motion: bool) -> np.ndarray:
    t = frame_index / 60.0
    targets = {
        "joint1": 0.0,
        "joint2": 1.00,
        "joint3": 0.45,
        "joint4": -0.10,
        "joint5": 0.0,
        "joint6": 0.0,
        "L_finger": 0.018,
        "R_finger": 0.018,
        "L_finger_joint": 0.018,
        "R_finger_joint": 0.018,
    }
    if not no_motion:
        targets["joint1"] = 0.16 * math.sin(0.55 * t)
        targets["joint2"] += 0.08 * math.sin(0.45 * t)
        targets["joint3"] += 0.10 * math.sin(0.70 * t)
        targets["joint4"] += 0.08 * math.sin(0.95 * t)
        targets["joint5"] = 0.18 * math.sin(0.60 * t)
        targets["joint6"] = 0.22 * math.sin(1.10 * t)
        finger_opening = 0.020 + 0.012 * (0.5 + 0.5 * math.sin(1.30 * t))
        targets["L_finger"] = finger_opening
        targets["R_finger"] = finger_opening
        targets["L_finger_joint"] = finger_opening
        targets["R_finger_joint"] = finger_opening

    pose = np.zeros((1, len(dof_names)), dtype=np.float32)
    for dof_index, dof_name in enumerate(dof_names):
        pose[0, dof_index] = targets.get(dof_name, 0.0)
    return pose


def should_keep_running() -> bool:
    global _reported_is_running_false

    if simulation_app.is_exiting():
        return False
    if not args.headless and not simulation_app.is_running():
        if not _reported_is_running_false:
            _reported_is_running_false = True
            print("[Panthera-HT] SimulationApp.is_running() became false; keeping scene alive until explicit close.")
            carb.log_warn("SimulationApp.is_running() is false; keeping Panthera scene alive until explicit close.")
    return True


def import_layout_arms(source_urdf_path: Path, mesh_dir: Path, tabletop_z: float) -> list[dict[str, object]]:
    arms = []
    for arm_spec in real_layout_arm_specs(args.table_size, tabletop_z):
        urdf_path = prepare_panthera_urdf_for_isaac(source_urdf_path, mesh_dir, arm_spec["urdf_robot_name"])
        articulation_path = import_panthera_to_stage(urdf_path, arm_spec["stage_root_path"])
        root_path = imported_robot_root_path(articulation_path)
        for existing_arm in arms:
            if root_path.startswith(existing_arm["root_path"] + "/") or existing_arm["root_path"].startswith(root_path + "/"):
                raise RuntimeError(
                    f"Panthera-HT import root collision: {root_path} conflicts with {existing_arm['root_path']}"
                )
        XFormPrim(
            prim_paths_expr=root_path,
            name=arm_spec["root_view_name"],
            positions=np.array([arm_spec["position"]], dtype=np.float32),
            orientations=np.array([quat_wxyz_from_yaw_deg(arm_spec["yaw_deg"])], dtype=np.float32),
        )
        if not args.use_official_grey:
            material_bind_count = apply_panthera_materials(root_path)
            print(
                f"[Panthera-HT] Applied scene material overrides to {material_bind_count} "
                f"links on {arm_spec['label']}."
            )
            carb.log_info(f"Applied Panthera-HT material overrides to {material_bind_count} links on {root_path}.")
        arms.append(
            {
                "label": arm_spec["label"],
                "root_path": root_path,
                "articulation_path": articulation_path,
                "articulation_name": arm_spec["articulation_name"],
                "position": arm_spec["position"],
                "yaw_deg": arm_spec["yaw_deg"],
            }
        )
        print(
            f"[Panthera-HT] Imported {arm_spec['label']} arm: "
            f"articulation={articulation_path}, root={root_path}, "
            f"position={np.array(arm_spec['position']).tolist()}, yaw_deg={arm_spec['yaw_deg']}"
        )
        carb.log_info(f"Imported Panthera-HT {arm_spec['label']} articulation at: {articulation_path}")
    return arms


def main() -> None:
    ensure_urdf_importer_enabled()
    source_urdf_path = resolve_panthera_urdf_path(args.urdf, args.ros2_root)
    mesh_dir = resolve_mesh_dir(source_urdf_path, args.mesh_dir)
    carb.log_info(f"Using Panthera-HT URDF: {source_urdf_path}")
    carb.log_info(f"Using Panthera-HT mesh directory: {mesh_dir}")

    world = World(stage_units_in_meters=1.0)
    world.scene.add_default_ground_plane()
    set_camera_view(eye=[1.35, 1.05, 1.10], target=[0.0, 0.0, 0.78], camera_prim_path="/OmniverseKit_Persp")
    rng = realism_rng()

    tabletop_z = add_table_scene(world, args.table_size, args.table_height)
    add_lights(rng)
    if args.disable_realism:
        print("[Panthera-HT] Realism environment disabled; using minimal table scene.")
        carb.log_info("Panthera-HT realism environment disabled.")
    else:
        add_realism_environment(world, args.table_size, args.table_height, rng)
        print(
            "[Panthera-HT] Realism environment enabled: room panels, table details, props, "
            f"real-layout camera optics/postprocess. randomize={args.randomize_realism}, seed={args.realism_seed}"
        )
        carb.log_info(
            "Panthera-HT realism environment enabled: "
            f"randomize={args.randomize_realism}, seed={args.realism_seed}"
        )

    arms = import_layout_arms(source_urdf_path, mesh_dir, tabletop_z)
    if args.use_official_grey:
        print("[Panthera-HT] Using official grey URDF material; scene material override is disabled.")
        carb.log_info("Panthera-HT material override disabled; using official grey URDF material.")

    pantheras = [
        world.scene.add(Articulation(prim_paths_expr=arm["articulation_path"], name=arm["articulation_name"]))
        for arm in arms
    ]

    world.reset()
    omni.timeline.get_timeline_interface().play()

    for arm, panthera in zip(arms, pantheras):
        if panthera.num_dof is None:
            raise RuntimeError(f"Panthera-HT {arm['label']} articulation did not initialize. Check URDF importer logs.")
        print(f"[Panthera-HT] {arm['label']} DOF names: {panthera.dof_names}")
        carb.log_info(f"Panthera-HT {arm['label']} dof names: {panthera.dof_names}")

    for panthera in pantheras:
        panthera.set_joint_positions(pose_for_dofs(panthera.dof_names, 0, args.no_motion))
    world.step(render=True)
    print_layout_arm_reach_evidence(arms, "initial")

    arm_root_paths = {arm["label"]: arm["root_path"] for arm in arms}
    layout_cameras = {} if args.disable_layout_cameras else create_layout_cameras(args.table_size, tabletop_z, arm_root_paths)
    for camera in layout_cameras.values():
        camera.initialize()
    verify_layout_camera_bindings(layout_cameras)
    enable_layout_depth_streams(layout_cameras)
    live_layout_viewports = create_layout_camera_live_viewports(layout_cameras)

    if layout_cameras:
        print(f"[Panthera-HT] Layout cameras ready: {', '.join(layout_cameras.keys())}")
        if live_layout_viewports:
            print(
                "[Panthera-HT] Live layout camera viewports updating: "
                f"{', '.join(LIVE_LAYOUT_CAMERA_VIEWPORT_NAMES)}"
            )
        carb.log_info(f"Panthera-HT layout cameras ready: {list(layout_cameras.keys())}")
    print(
        "[Panthera-HT] Running real-layout scene: "
        f"headless={args.headless}, max_frames={args.max_frames}, arms={len(pantheras)}, "
        f"table_size={args.table_size}m. "
        "Close Isaac Sim or press Ctrl+C to stop; use --max-frames N for finite headless runs."
    )
    carb.log_info(
        "Panthera-HT lifetime controls: "
        f"headless={args.headless}, max_frames={args.max_frames}, "
        f"PANTHERA_MAX_FRAMES={os.getenv('PANTHERA_MAX_FRAMES', '')!r}"
    )

    if layout_cameras and not args.skip_layout_screenshots:
        sequence_frames = parse_layout_sequence_frames(args.layout_sequence_frames)
        for _ in range(LAYOUT_CAMERA_CAPTURE_WARMUP_FRAMES):
            world.step(render=True)
        initial_camera_poses = layout_camera_pose_signatures(layout_cameras)
        screenshot_paths = save_layout_camera_screenshots(layout_cameras, "initial")
        for screenshot_path in screenshot_paths:
            print(f"[Panthera-HT] Saved layout camera screenshot: {screenshot_path}")
            carb.log_info(f"Saved Panthera-HT layout camera screenshot: {screenshot_path}")

        if args.no_motion:
            print("[Panthera-HT] Skipping layout camera after-motion proof because --no-motion is enabled.")
        else:
            for panthera in pantheras:
                panthera.set_joint_positions(
                    pose_for_dofs(panthera.dof_names, LAYOUT_CAMERA_MOTION_PROOF_FRAME, args.no_motion)
                )
            for _ in range(LAYOUT_CAMERA_MOTION_PROOF_WARMUP_FRAMES):
                world.step(render=True)
            print_layout_arm_reach_evidence(arms, "after_motion")
            after_motion_camera_poses = layout_camera_pose_signatures(layout_cameras)
            print_layout_camera_motion_evidence(
                initial_camera_poses, after_motion_camera_poses, LAYOUT_CAMERA_MOTION_PROOF_FRAME
            )
            after_motion_paths = save_layout_camera_screenshots(layout_cameras, "after_motion")
            for screenshot_path in after_motion_paths:
                print(f"[Panthera-HT] Saved layout camera screenshot: {screenshot_path}")
                carb.log_info(f"Saved Panthera-HT layout camera screenshot: {screenshot_path}")

        sequence_manifest_path = save_layout_camera_sequence(
            world, arms, pantheras, layout_cameras, sequence_frames, source_urdf_path, mesh_dir, tabletop_z
        )
        if sequence_manifest_path is not None:
            print(f"[Panthera-HT] VLA layout sequence ready: {sequence_manifest_path}")

    reset_needed = False
    frame_index = 0
    exit_reason = "simulation_app requested exit"

    try:
        while should_keep_running():
            world.step(render=True)

            if world.is_stopped() and not reset_needed:
                reset_needed = True

            if world.is_playing():
                if reset_needed:
                    world.reset()
                    reset_needed = False
                    frame_index = 0

                for panthera in pantheras:
                    panthera.set_joint_positions(pose_for_dofs(panthera.dof_names, frame_index, args.no_motion))
                frame_index += 1

                if args.max_frames > 0 and frame_index >= args.max_frames:
                    exit_reason = f"max_frames reached ({args.max_frames})"
                    break

            if args.headless and args.max_frames <= 0 and frame_index % 60 == 0:
                time.sleep(0.001)
    except KeyboardInterrupt:
        exit_reason = "keyboard interrupt"
    finally:
        print(f"[Panthera-HT] Scene exiting: {exit_reason}")
        carb.log_info(f"Panthera-HT scene exiting: {exit_reason}")
        omni.timeline.get_timeline_interface().stop()
        simulation_app.close()


if __name__ == "__main__":
    main()
