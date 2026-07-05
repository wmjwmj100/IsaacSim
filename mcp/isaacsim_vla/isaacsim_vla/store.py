from __future__ import annotations

import json
import shutil
import time
import uuid
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw

from .models import BoxLayer, BoxOverlay, CameraObservation

SOURCE_BOX_COLOR = (30, 190, 70)
TARGET_BOX_COLOR = (30, 90, 230)


def now_id(prefix: str) -> str:
    return f"{prefix}_{int(time.time() * 1000)}_{uuid.uuid4().hex[:8]}"


class ArtifactStore:
    def __init__(self, root: Path):
        self.root = root
        self.images_dir = root / "images"
        self.overlays_dir = root / "overlays"
        self.layers_dir = root / "layers"
        self.records_dir = root / "records"
        for path in (self.images_dir, self.overlays_dir, self.layers_dir, self.records_dir):
            path.mkdir(parents=True, exist_ok=True)

    def record_path(self, kind: str, item_id: str) -> Path:
        return self.records_dir / f"{kind}_{item_id}.json"

    def save_json(self, kind: str, item_id: str, payload: dict[str, Any]) -> Path:
        path = self.record_path(kind, item_id)
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        return path

    def load_json(self, kind: str, item_id: str) -> dict[str, Any]:
        path = self.record_path(kind, item_id)
        if not path.is_file():
            raise KeyError(f"{kind} record not found: {item_id}")
        return json.loads(path.read_text(encoding="utf-8"))

    def import_observation(
        self,
        src_path: str | Path,
        *,
        camera_id: str,
        observation_id: str | None = None,
    ) -> CameraObservation:
        src = Path(src_path).expanduser().resolve()
        if not src.is_file():
            raise FileNotFoundError(f"image file does not exist: {src}")
        observation_id = observation_id or now_id("obs")
        suffix = src.suffix.lower() if src.suffix else ".png"
        dst = self.images_dir / f"{observation_id}_{_safe_component(camera_id)}{suffix}"
        if src != dst:
            shutil.copy2(src, dst)
        with Image.open(dst) as img:
            width, height = img.size
        obs = CameraObservation(
            observation_id=observation_id,
            camera_id=camera_id,
            image_path=str(dst),
            width=width,
            height=height,
            timestamp=time.time(),
            metadata={"source_path": str(src)},
        )
        self.save_json("observation", observation_id, obs.model_dump())
        self.save_json("camera_latest", _safe_component(camera_id), {"observation_id": observation_id})
        return obs

    def get_observation(self, observation_id: str) -> CameraObservation:
        return CameraObservation.model_validate(self.load_json("observation", observation_id))

    def get_latest_observation(self, camera_id: str) -> CameraObservation:
        latest = self.load_json("camera_latest", _safe_component(camera_id))
        observation_id = latest.get("observation_id")
        if not isinstance(observation_id, str) or not observation_id:
            raise KeyError(f"latest observation is missing for camera_id={camera_id!r}")
        return self.get_observation(observation_id)

    def save_box_layer(
        self,
        *,
        camera_id: str,
        red_box: list[int],
        green_box: list[int],
        coordinate_system: str = "pixel",
        observation_id: str | None = None,
        label_red: str = "",
        label_green: str = "",
        metadata: dict[str, Any] | None = None,
    ) -> BoxLayer:
        obs = self.get_observation(observation_id) if observation_id else self.get_latest_observation(camera_id)
        if obs.camera_id != camera_id:
            raise ValueError(
                f"observation_id={obs.observation_id!r} belongs to camera_id={obs.camera_id!r}, not {camera_id!r}"
            )
        normalized_coordinate_system = _normalize_coordinate_system(coordinate_system)
        red_box_input = [int(v) for v in red_box]
        green_box_input = [int(v) for v in green_box]
        red_box_pixels = _box_to_pixel_box(
            red_box_input,
            obs.width,
            obs.height,
            normalized_coordinate_system,
            "source_box",
        )
        green_box_pixels = _box_to_pixel_box(
            green_box_input,
            obs.width,
            obs.height,
            normalized_coordinate_system,
            "target_box",
        )
        layer_metadata = {
            **(metadata or {}),
            "coordinate_system": normalized_coordinate_system,
            "red_box_input": red_box_input,
            "green_box_input": green_box_input,
            "source_box_input": red_box_input,
            "target_box_input": green_box_input,
            "box_color_convention": "source_box_green_target_box_blue",
            "validated": False,
            "validation": None,
        }
        warnings = _box_warnings(red_box_pixels, green_box_pixels, width=obs.width, height=obs.height)
        if warnings:
            layer_metadata["warnings"] = warnings
        layer_id = now_id("layer")
        preview = self._render_boxes(
            obs=obs,
            red_box=red_box_pixels,
            green_box=green_box_pixels,
            label_red=label_red,
            label_green=label_green,
            box_layer_id=layer_id,
            metadata=layer_metadata,
        )
        layer = BoxLayer(
            box_layer_id=layer_id,
            camera_id=camera_id,
            reference_observation_id=obs.observation_id,
            reference_image_path=obs.image_path,
            preview_overlay_id=preview.box_overlay_id,
            preview_overlay_path=preview.overlay_path,
            red_box=red_box_pixels,
            green_box=green_box_pixels,
            source_box=red_box_pixels,
            target_box=green_box_pixels,
            width=preview.width,
            height=preview.height,
            red_label=label_red,
            green_label=label_green,
            created_at=time.time(),
            metadata=layer_metadata,
        )
        self.save_json("box_layer", layer_id, layer.model_dump())
        self.save_json("overlay", preview.box_overlay_id, preview.model_dump())
        return layer

    def save_overlay(
        self,
        *,
        camera_id: str,
        red_box: list[int],
        green_box: list[int],
        coordinate_system: str = "pixel",
        observation_id: str | None = None,
        label_red: str = "",
        label_green: str = "",
        metadata: dict[str, Any] | None = None,
    ) -> BoxOverlay:
        layer = self.save_box_layer(
            camera_id=camera_id,
            red_box=red_box,
            green_box=green_box,
            coordinate_system=coordinate_system,
            observation_id=observation_id,
            label_red=label_red,
            label_green=label_green,
            metadata=metadata,
        )
        return self.get_overlay(layer.preview_overlay_id)

    def get_box_layer(self, layer_id: str) -> BoxLayer:
        return BoxLayer.model_validate(self.load_json("box_layer", layer_id))

    def confirm_box_layer(
        self,
        layer_id: str,
        *,
        source_confirmation: str,
        target_confirmation: str,
        preview_observation_id: str | None = None,
    ) -> BoxLayer:
        layer = self.get_box_layer(layer_id)
        metadata = dict(layer.metadata)
        metadata["validated"] = True
        metadata["validation"] = {
            "source_confirmation": source_confirmation,
            "target_confirmation": target_confirmation,
            "preview_observation_id": preview_observation_id,
            "validated_at": time.time(),
        }
        updated = layer.model_copy(update={"metadata": metadata})
        self.save_json("box_layer", layer_id, updated.model_dump())
        return updated

    def render_layer_on_observation(
        self,
        *,
        box_layer_id: str,
        observation_id: str | None = None,
        camera_id: str | None = None,
    ) -> BoxOverlay:
        layer = self.get_box_layer(box_layer_id)
        obs = self.get_observation(observation_id) if observation_id else self.get_latest_observation(camera_id or layer.camera_id)
        if obs.camera_id != layer.camera_id:
            raise ValueError(
                f"box_layer_id={box_layer_id!r} belongs to camera_id={layer.camera_id!r}, not {obs.camera_id!r}"
            )
        if obs.width != layer.width or obs.height != layer.height:
            raise ValueError(
                f"box_layer_id={box_layer_id!r} was defined for {layer.width}x{layer.height}, "
                f"but observation_id={obs.observation_id!r} is {obs.width}x{obs.height}"
            )
        overlay = self._render_boxes(
            obs=obs,
            red_box=layer.red_box,
            green_box=layer.green_box,
            label_red=layer.red_label,
            label_green=layer.green_label,
            box_layer_id=layer.box_layer_id,
            metadata=layer.metadata,
        )
        self.save_json("overlay", overlay.box_overlay_id, overlay.model_dump())
        return overlay

    def get_overlay(self, overlay_id: str) -> BoxOverlay:
        return BoxOverlay.model_validate(self.load_json("overlay", overlay_id))

    def resolve_overlay(
        self,
        *,
        box_layer_id: str | None = None,
        box_overlay_id: str | None = None,
        observation_id: str | None = None,
        camera_id: str | None = None,
    ) -> BoxOverlay:
        if box_layer_id:
            return self.render_layer_on_observation(
                box_layer_id=box_layer_id,
                observation_id=observation_id,
                camera_id=camera_id,
            )
        if box_overlay_id:
            return self.get_overlay(box_overlay_id)
        raise ValueError("either box_layer_id or box_overlay_id is required")

    def _render_boxes(
        self,
        *,
        obs: CameraObservation,
        red_box: list[int],
        green_box: list[int],
        label_red: str,
        label_green: str,
        box_layer_id: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> BoxOverlay:
        overlay_id = now_id("box")
        overlay_path = self.overlays_dir / f"{overlay_id}.png"
        with Image.open(obs.image_path).convert("RGB") as img:
            width, height = img.size
            _validate_bounds(red_box, width, height, "source_box")
            _validate_bounds(green_box, width, height, "target_box")
            draw = ImageDraw.Draw(img)
            _draw_box(draw, red_box, color=SOURCE_BOX_COLOR, label=label_red)
            _draw_box(draw, green_box, color=TARGET_BOX_COLOR, label=label_green)
            img.save(overlay_path)
        return BoxOverlay(
            box_overlay_id=overlay_id,
            box_layer_id=box_layer_id,
            camera_id=obs.camera_id,
            observation_id=obs.observation_id,
            original_image_path=obs.image_path,
            overlay_path=str(overlay_path),
            red_box=red_box,
            green_box=green_box,
            source_box=red_box,
            target_box=green_box,
            width=width,
            height=height,
            created_at=time.time(),
            metadata=metadata or {},
        )


def _normalize_coordinate_system(value: str) -> str:
    normalized = str(value or "").strip().lower().replace("-", "_")
    aliases = {
        "pixel": "pixel",
        "pixels": "pixel",
        "px": "pixel",
        "normalized_1000": "normalized_1000",
        "normalised_1000": "normalized_1000",
        "doubao_1000": "normalized_1000",
        "ark_1000": "normalized_1000",
        "bbox_1000": "normalized_1000",
    }
    if normalized not in aliases:
        raise ValueError(
            "coordinate_system must be 'normalized_1000' for Doubao/Ark "
            "grounding <bbox> coordinates or 'pixel' for absolute image pixels"
        )
    return aliases[normalized]


def _box_to_pixel_box(
    box: list[int],
    width: int,
    height: int,
    coordinate_system: str,
    name: str,
) -> list[int]:
    x1, y1, x2, y2 = [int(v) for v in box]
    if coordinate_system == "pixel":
        pixel_box = [x1, y1, x2, y2]
    elif coordinate_system == "normalized_1000":
        if any(v < 0 or v > 999 for v in (x1, y1, x2, y2)):
            raise ValueError(
                f"{name}={box} uses normalized_1000 coordinates but contains "
                "values outside the official [0, 999] range"
            )
        pixel_box = [
            int(x1 * width / 1000),
            int(y1 * height / 1000),
            int(x2 * width / 1000),
            int(y2 * height / 1000),
        ]
    else:
        raise ValueError(f"unsupported coordinate_system={coordinate_system!r}")
    _validate_bounds(pixel_box, width, height, name)
    return pixel_box


def _validate_bounds(box: list[int], width: int, height: int, name: str) -> None:
    x1, y1, x2, y2 = [int(v) for v in box]
    if x1 < 0 or y1 < 0 or x2 > width or y2 > height or x2 <= x1 or y2 <= y1:
        raise ValueError(f"{name}={box} is outside image bounds {width}x{height}")


def _box_warnings(red_box: list[int], green_box: list[int], *, width: int, height: int) -> list[str]:
    warnings: list[str] = []
    for name, box in (("source_box", red_box), ("target_box", green_box)):
        x1, y1, x2, y2 = [int(v) for v in box]
        area_ratio = ((x2 - x1) * (y2 - y1)) / max(1, width * height)
        if area_ratio < 0.002:
            warnings.append(f"{name} is very small relative to the image")
        if area_ratio > 0.35:
            warnings.append(f"{name} is very large relative to the image")
        margin = min(x1, y1, width - x2, height - y2)
        if margin < min(width, height) * 0.02:
            warnings.append(f"{name} is close to an image edge")
    sx1, sy1, sx2, sy2 = [int(v) for v in red_box]
    tx1, ty1, tx2, ty2 = [int(v) for v in green_box]
    overlap_w = max(0, min(sx2, tx2) - max(sx1, tx1))
    overlap_h = max(0, min(sy2, ty2) - max(sy1, ty1))
    if overlap_w * overlap_h:
        warnings.append("source_box and target_box overlap")
    return warnings


def _safe_component(value: str) -> str:
    safe = "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in value.strip())
    return safe or "camera"


def _draw_box(draw: ImageDraw.ImageDraw, box: list[int], *, color: tuple[int, int, int], label: str) -> None:
    x1, y1, x2, y2 = [int(v) for v in box]
    thickness = max(3, int(min(max(x2 - x1, y2 - y1), 300) / 50))
    for offset in range(thickness):
        draw.rectangle((x1 - offset, y1 - offset, x2 + offset, y2 + offset), outline=color)
    text = label.strip()[:32]
    if text:
        pad = 4
        text_box = draw.textbbox((x1, y1), text)
        tw = text_box[2] - text_box[0]
        th = text_box[3] - text_box[1]
        label_y = max(0, y1 - th - pad * 2)
        draw.rectangle((x1, label_y, x1 + tw + pad * 2, label_y + th + pad * 2), fill=color)
        draw.text((x1 + pad, label_y + pad), text, fill=(255, 255, 255))
