from __future__ import annotations

import json
import math
import os
from pathlib import Path
from typing import Any, Sequence

from PIL import Image, ImageDraw


DEFAULT_ARM1_CONFIG = Path(".data/field_execution/site_config.a4_candidate6_redblock_exec_20260626.json")
DEFAULT_ARM2_CONFIG = Path(".data/field_execution/site_config.arm2_topdown_3point.json")


class WorkspaceGuard:
    """Image-space view of a configured robot workspace.

    This is an MCP/VLM guard only. It does not replace the downstream motion
    safety checks in field_transfer; it makes configured XYZ-workspace violations
    invisible or invalid before the model can execute them. It is not an IK
    feasibility proof for a specific gripper orientation or TCP offset.
    """

    def __init__(self, *, arm_target: str, config_path: Path, site_config: dict[str, Any]):
        self.arm_target = arm_target
        self.config_path = config_path
        self.site_config = site_config
        camera = site_config["camera"]
        self.intrinsics = camera["intrinsics"]
        self.world_from_camera = camera["world_from_camera"]["matrix"]
        limits = site_config["robot"]["safety_limits"]
        self.workspace_min_xyz = [float(v) for v in limits["workspace_min_xyz_m"]]
        self.workspace_max_xyz = [float(v) for v in limits["workspace_max_xyz_m"]]
        obj = site_config.get("object") or {}
        table = site_config.get("table") or {}
        self.plane_z_m = float(
            obj.get(
                "localization_plane_z_m",
                table.get("plane_z_m", self.workspace_min_xyz[2]),
            )
        )

    @classmethod
    def from_site_config(cls, *, arm_target: str, config_path: Path, site_config: dict[str, Any]) -> "WorkspaceGuard":
        return cls(arm_target=arm_target, config_path=config_path, site_config=site_config)

    def applies_to_camera(self, camera_id: str) -> bool:
        key = str(camera_id or "").strip().lower().replace("-", "_")
        return key in {"top", "overhead", "camera1"}

    def mask_observation(self, obs: Any) -> Any:
        src = Path(str(obs.image_path)).expanduser().resolve()
        with Image.open(src).convert("RGB") as image:
            width, height = image.size
            polygon = self.reachable_polygon_px(width, height)
            masked = Image.new("RGB", (width, height), (24, 24, 24))
            mask = Image.new("L", (width, height), 0)
            mask_draw = ImageDraw.Draw(mask)
            mask_draw.polygon(polygon, fill=255)
            masked.paste(image, (0, 0), mask)
            draw = ImageDraw.Draw(masked)
            draw.line([*polygon, polygon[0]], fill=(235, 175, 35), width=3)
            label = f"{self.arm_target} XYZ workspace"
            draw.rectangle((8, 8, 8 + 8 * len(label), 30), fill=(24, 24, 24))
            draw.text((12, 12), label, fill=(235, 175, 35))
            dst = src.with_name(f"{src.stem}_reachable{src.suffix or '.png'}")
            masked.save(dst)

        metadata = dict(getattr(obs, "metadata", {}) or {})
        metadata["raw_image_path"] = str(src)
        metadata["workspace_guard"] = self.metadata(width=width, height=height, polygon=polygon)
        metadata["workspace_guard"]["masked_image_path"] = str(dst)
        return obs.model_copy(update={"image_path": str(dst), "metadata": metadata})

    def validate_box_input(
        self,
        box: Sequence[int],
        *,
        width: int,
        height: int,
        coordinate_system: str,
        name: str,
    ) -> dict[str, Any]:
        pixel_box = box_to_pixel_box(box, width, height, coordinate_system, name)
        projection = self.project_box(pixel_box)
        point = projection["world_point_m"]
        violations = [
            f"{axis}={float(value):.4f} outside [{lo:.4f},{hi:.4f}]"
            for axis, value, lo, hi in zip(("x", "y", "z"), point, self.workspace_min_xyz, self.workspace_max_xyz)
            if float(value) < lo or float(value) > hi
        ]
        if violations:
            raise ValueError(
                f"{name} center projects outside {self.arm_target} configured XYZ workspace: "
                + ", ".join(violations)
                + ". Redraw this box inside the unmasked XYZ workspace region."
            )
        return {
            "box": pixel_box,
            "pixel_xy": projection["pixel_xy"],
            "world_point_m": point,
            "reachable": True,
        }

    def metadata(self, *, width: int, height: int, polygon: list[tuple[float, float]] | None = None) -> dict[str, Any]:
        return {
            "enabled": True,
            "arm_target": self.arm_target,
            "config_path": str(self.config_path),
            "plane_z_m": self.plane_z_m,
            "workspace_min_xyz_m": self.workspace_min_xyz,
            "workspace_max_xyz_m": self.workspace_max_xyz,
            "image_size": [int(width), int(height)],
            "xyz_workspace_polygon_px": [
                [round(float(x), 3), round(float(y), 3)]
                for x, y in (polygon if polygon is not None else self.reachable_polygon_px(width, height))
            ],
            # Keep the old key for compatibility with existing records and tools.
            "reachable_polygon_px": [
                [round(float(x), 3), round(float(y), 3)]
                for x, y in (polygon if polygon is not None else self.reachable_polygon_px(width, height))
            ],
            "semantics": (
                "Pixels outside xyz_workspace_polygon_px are masked for VLM grounding and rejected for "
                "source/target boxes. This is a configured XYZ workspace guard, not an IK feasibility "
                "map for the current gripper orientation, TCP offset, or full waypoint sequence."
            ),
        }

    def reachable_polygon_px(self, width: int, height: int) -> list[tuple[float, float]]:
        lo = self.workspace_min_xyz
        hi = self.workspace_max_xyz
        z = self.plane_z_m
        corners_world = [
            [lo[0], lo[1], z],
            [lo[0], hi[1], z],
            [hi[0], hi[1], z],
            [hi[0], lo[1], z],
        ]
        polygon = []
        for point in corners_world:
            u, v = self.world_to_pixel(point)
            polygon.append((min(max(u, 0.0), float(width - 1)), min(max(v, 0.0), float(height - 1))))
        return polygon

    def project_box(self, box_xyxy: Sequence[int]) -> dict[str, Any]:
        x1, y1, x2, y2 = [float(v) for v in box_xyxy]
        u = (x1 + x2) / 2.0
        v = (y1 + y2) / 2.0
        fx = float(self.intrinsics["fx"])
        fy = float(self.intrinsics["fy"])
        cx = float(self.intrinsics["cx"])
        cy = float(self.intrinsics["cy"])
        ray_camera = _normalize([(u - cx) / fx, (v - cy) / fy, 1.0])
        origin = _transform_point(self.world_from_camera, [0.0, 0.0, 0.0])
        ray_tip = _transform_point(self.world_from_camera, ray_camera)
        ray_world = _normalize([ray_tip[i] - origin[i] for i in range(3)])
        denom = ray_world[2]
        if abs(denom) < 1e-9:
            raise ValueError(f"camera ray for box {list(box_xyxy)} is parallel to the localization plane")
        t = (self.plane_z_m - origin[2]) / denom
        point = [origin[i] + t * ray_world[i] for i in range(3)]
        return {
            "pixel_xy": [u, v],
            "world_point_m": point,
            "plane_z_m": self.plane_z_m,
        }

    def world_to_pixel(self, point_xyz: Sequence[float]) -> tuple[float, float]:
        matrix = self.world_from_camera
        rotation = [row[:3] for row in matrix[:3]]
        translation = [float(matrix[i][3]) for i in range(3)]
        q = [float(point_xyz[i]) - translation[i] for i in range(3)]
        # world_from_camera is rigid, so camera_from_world uses R^T.
        camera_point = [sum(float(rotation[row][col]) * q[row] for row in range(3)) for col in range(3)]
        z = camera_point[2]
        if abs(z) < 1e-9:
            raise ValueError(f"world point projects with near-zero camera z: {point_xyz}")
        u = float(self.intrinsics["fx"]) * camera_point[0] / z + float(self.intrinsics["cx"])
        v = float(self.intrinsics["fy"]) * camera_point[1] / z + float(self.intrinsics["cy"])
        return u, v


def load_workspace_guard(cfg: Any) -> WorkspaceGuard | None:
    enabled = os.environ.get("ISAACSIM_VLA_WORKSPACE_GUARD", "1").strip().lower()
    if enabled in {"0", "false", "no", "off"}:
        return None
    arm_target = str(getattr(cfg, "arm_target", "") or getattr(cfg, "arm_id", "") or "").strip().lower()
    if arm_target not in {"arm1", "arm2"}:
        return None
    config_env = os.environ.get("ROBOCLAW_MCP_VLA_ALGORITHM_CONFIG", "").strip()
    if config_env:
        config_path = Path(config_env).expanduser().resolve()
    else:
        default_rel = DEFAULT_ARM2_CONFIG if arm_target == "arm2" else DEFAULT_ARM1_CONFIG
        config_path = (Path(getattr(cfg, "repo_root", Path.cwd())) / default_rel).resolve()
    if not config_path.is_file():
        return None
    site_config = json.loads(config_path.read_text(encoding="utf-8"))
    return WorkspaceGuard.from_site_config(arm_target=arm_target, config_path=config_path, site_config=site_config)


def box_to_pixel_box(
    box: Sequence[int],
    width: int,
    height: int,
    coordinate_system: str,
    name: str,
) -> list[int]:
    if len(box) != 4:
        raise ValueError(f"{name} must be [x1, y1, x2, y2]")
    x1, y1, x2, y2 = [int(v) for v in box]
    normalized = str(coordinate_system or "").strip().lower().replace("-", "_")
    if normalized in {"pixel", "pixels", "px"}:
        pixel_box = [x1, y1, x2, y2]
    elif normalized in {"normalized_1000", "normalised_1000", "doubao_1000", "ark_1000", "bbox_1000"}:
        if any(v < 0 or v > 999 for v in (x1, y1, x2, y2)):
            raise ValueError(f"{name}={list(box)} uses normalized_1000 coordinates outside [0, 999]")
        pixel_box = [
            int(x1 * width / 1000),
            int(y1 * height / 1000),
            int(x2 * width / 1000),
            int(y2 * height / 1000),
        ]
    else:
        raise ValueError("coordinate_system must be 'normalized_1000' or 'pixel'")
    _validate_bounds(pixel_box, width, height, name)
    return pixel_box


def _validate_bounds(box: Sequence[int], width: int, height: int, name: str) -> None:
    x1, y1, x2, y2 = [int(v) for v in box]
    if x1 < 0 or y1 < 0 or x2 > width or y2 > height or x2 <= x1 or y2 <= y1:
        raise ValueError(f"{name}={list(box)} is outside image bounds {width}x{height}")


def _normalize(values: Sequence[float]) -> list[float]:
    norm = math.sqrt(sum(float(v) * float(v) for v in values))
    if norm <= 0.0:
        raise ValueError("cannot normalize zero-length vector")
    return [float(v) / norm for v in values]


def _transform_point(matrix_4x4: Sequence[Sequence[float]], point_xyz: Sequence[float]) -> list[float]:
    x, y, z = [float(v) for v in point_xyz]
    vec = [x, y, z, 1.0]
    return [sum(float(matrix_4x4[r][c]) * vec[c] for c in range(4)) for r in range(3)]
