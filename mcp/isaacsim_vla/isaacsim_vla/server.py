from __future__ import annotations

import json
import logging
import sys
from typing import Any

from .backends import build_backend
from .config import load_config
from .store import ArtifactStore

logger = logging.getLogger("isaacsim_vla.mcp")


def _build_server() -> Any:
    try:
        from mcp.server.fastmcp import FastMCP
    except ImportError as exc:
        raise ImportError("Install this package with its MCP dependencies first: pip install -e .") from exc

    cfg = load_config()
    cfg.workdir.mkdir(parents=True, exist_ok=True)
    store = ArtifactStore(cfg.workdir)
    backend = build_backend(cfg, store)

    mcp = FastMCP(
        "isaacsim-vla",
        instructions=(
            "Robotics visual-prompt tools for a VLM -> box layer -> VLA workflow. "
            "Use the top/overhead camera once for VLM grounding, call draw_boxes "
            "once to create a reusable red/green box_layer_id, then reuse that "
            "layer for VLA execution. Do not draw boxes on wrist cameras; wrist "
            "camera observations are for local gripper checks only."
        ),
    )

    @mcp.tool(structured_output=False)
    def look_camera(camera_id: str = "top", box_layer_id: str | None = None) -> list[Any]:
        """Capture or fetch one observation from a robotics camera stream.

        Args:
            camera_id: Logical camera stream. Use "top" or "overhead" for global planning;
                use "wrist", "left_wrist", or "right_wrist" for local gripper checks.
            box_layer_id: Optional reusable top-camera box layer returned by draw_boxes.
                Pass this only with the same top/overhead camera stream when you want
                the current top frame returned with the fixed red/green boxes rendered.
                Do not pass a box layer for wrist cameras.

        Returns:
            MCP content blocks: JSON metadata plus the captured image. If box_layer_id
            is provided, the image block is the current camera frame with that layer
            rendered on top.
        """
        result = backend.look_camera(camera_id)
        image_path = result.get("image_path")
        if not image_path:
            raise RuntimeError("backend look_camera response must include image_path")
        obs = store.import_observation(
            image_path,
            camera_id=camera_id,
            observation_id=result.get("observation_id"),
        )
        payload = obs.model_dump()
        payload["backend"] = backend.name
        payload["backend_result"] = {k: v for k, v in result.items() if k != "image_path"}
        if box_layer_id:
            overlay = store.render_layer_on_observation(
                box_layer_id=box_layer_id,
                observation_id=obs.observation_id,
            )
            payload["image_path"] = overlay.overlay_path
            payload["raw_image_path"] = obs.image_path
            payload["box_layer_id"] = box_layer_id
            payload["box_overlay_id"] = overlay.box_overlay_id
            payload["red_box"] = overlay.red_box
            payload["green_box"] = overlay.green_box
            payload["metadata"] = {
                **payload.get("metadata", {}),
                "rendered_box_layer": True,
                "raw_observation_id": obs.observation_id,
            }
            return _json_with_image(payload, overlay.overlay_path)
        return _json_with_image(payload, obs.image_path)

    @mcp.tool(structured_output=False)
    def draw_boxes(
        camera_id: str,
        red_box: list[int],
        green_box: list[int],
        coordinate_system: str,
        observation_id: str | None = None,
        red_label: str = "source",
        green_label: str = "target",
    ) -> list[Any]:
        """Create a reusable red/green box layer for one camera stream.

        Args:
            camera_id: Logical camera stream to attach the layer to. Use "top" or
                "overhead". Do not draw boxes on wrist cameras because wrist images
                move with the gripper.
            red_box: Source object box [x1, y1, x2, y2]. Use Doubao/Ark
                grounding coordinates when coordinate_system="normalized_1000";
                use absolute image pixels when coordinate_system="pixel".
            green_box: Target region box [x1, y1, x2, y2], in the same coordinate
                system as red_box.
            coordinate_system: Required. Use "normalized_1000" for Doubao/Ark
                grounding <bbox>x1 y1 x2 y2</bbox> output, whose values are
                normalized to a 1000x1000 coordinate frame in [0, 999]. Use
                "pixel" only when the boxes are already absolute pixel coordinates
                for the current image.
            observation_id: Optional exact frame returned by look_camera. If omitted,
                the latest observation from camera_id is used. Pass this when the VLM
                grounded boxes on a specific top frame and you need strict frame binding.
            red_label: Optional label rendered on the red box.
            green_label: Optional label rendered on the green box.

        Returns:
            MCP content blocks: JSON metadata plus a preview image. The returned
            box_layer_id is the stable handle to pass to look_camera and vla_execute.
            box_overlay_id is included only as a compatibility preview id.
        """
        camera_key = camera_id.strip().lower().replace("-", "_")
        if "wrist" in camera_key:
            raise ValueError("draw_boxes creates reusable layers only for top/overhead cameras; do not draw boxes on wrist cameras")
        layer = store.save_box_layer(
            camera_id=camera_id,
            red_box=[int(v) for v in red_box],
            green_box=[int(v) for v in green_box],
            coordinate_system=coordinate_system,
            observation_id=observation_id,
            label_red=red_label,
            label_green=green_label,
        )
        payload = layer.model_dump()
        payload["box_overlay_id"] = layer.preview_overlay_id
        payload["overlay_path"] = layer.preview_overlay_path
        return _json_with_image(payload, layer.preview_overlay_path)

    @mcp.tool(structured_output=False)
    def vla_execute(
        instruction: str,
        box_layer_id: str | None = None,
        box_overlay_id: str | None = None,
        atomic_action: str = "pick_and_place",
        dry_run: bool = False,
    ) -> str:
        """Execute a red/green-box visual prompt with a VLA robotics backend.

        Args:
            instruction: Simplified instruction, e.g. "move the object in the red box to the green box".
            box_layer_id: Stable layer id returned by draw_boxes. Prefer this. When
                provided, the tool fetches the current layer camera frame, renders
                the same boxes onto that fresh frame, and sends that image to VLA.
            box_overlay_id: Legacy static overlay id. Use only for backward
                compatibility; it binds execution to the old rendered frame.
            atomic_action: Atomic action type. Start with "pick_and_place".
            dry_run: If true, return the resolved overlay payload without commanding the backend.

        Returns:
            JSON execution result from the backend.
        """
        if box_layer_id:
            layer = store.get_box_layer(box_layer_id)
            result = backend.look_camera(layer.camera_id)
            image_path = result.get("image_path")
            if not image_path:
                raise RuntimeError("backend look_camera response must include image_path")
            obs = store.import_observation(
                image_path,
                camera_id=layer.camera_id,
                observation_id=result.get("observation_id"),
            )
            overlay = store.render_layer_on_observation(
                box_layer_id=box_layer_id,
                observation_id=obs.observation_id,
            )
        else:
            overlay = store.resolve_overlay(box_overlay_id=box_overlay_id)
        if dry_run:
            return _json(
                {
                    "success": True,
                    "dry_run": True,
                    "backend": backend.name,
                    "instruction": instruction,
                    "atomic_action": atomic_action,
                    "box_layer_id": overlay.box_layer_id,
                    "box_overlay_id": overlay.box_overlay_id,
                    "overlay": overlay.model_dump(),
                }
            )
        result = backend.vla_execute(instruction, overlay, atomic_action)
        result.setdefault("backend", backend.name)
        result.setdefault("box_layer_id", overlay.box_layer_id)
        result.setdefault("box_overlay_id", overlay.box_overlay_id)
        result.setdefault("instruction", instruction)
        return _json(result)

    return mcp


def _json(payload: dict[str, Any]) -> str:
    return json.dumps(payload, ensure_ascii=False, indent=2)


def _json_with_image(payload: dict[str, Any], image_path: str) -> list[Any]:
    from mcp.server.fastmcp import Image

    return [_json(payload), Image(path=image_path)]


def main(argv: list[str] | None = None) -> int:
    argv = argv or sys.argv[1:]
    verbose = "--verbose" in argv or "-v" in argv
    logging.basicConfig(
        level=logging.INFO if verbose else logging.WARNING,
        stream=sys.stderr,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    try:
        server = _build_server()
    except ImportError as exc:
        sys.stderr.write(f"isaacsim-vla MCP server cannot start: {exc}\n")
        return 2
    except Exception as exc:
        logger.exception("failed to build isaacsim-vla MCP server")
        sys.stderr.write(f"isaacsim-vla MCP server configuration error: {exc}\n")
        return 1

    try:
        server.run()
    except KeyboardInterrupt:
        return 0
    except Exception as exc:
        logger.exception("isaacsim-vla MCP server crashed")
        sys.stderr.write(f"isaacsim-vla MCP server error: {exc}\n")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
