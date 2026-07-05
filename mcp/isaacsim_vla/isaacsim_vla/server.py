from __future__ import annotations

import json
import logging
import os
import sys
from typing import Any, Literal

from .audit import append_event, exception_fields
from .backends import build_backend
from .config import load_config
from .store import ArtifactStore
from .workspace_guard import load_workspace_guard

logger = logging.getLogger("isaacsim_vla.mcp")

DEFAULT_VLA_INSTRUCTION = "Pick up the object inside the green source box and place it at the location marked by the blue target box."
CameraId = Literal["top", "overhead", "wrist", "left_wrist", "right_wrist"]
TopCameraId = Literal["top", "overhead"]
CoordinateSystem = Literal["normalized_1000", "pixel"]


def _build_server() -> Any:
    try:
        from mcp.server.fastmcp import FastMCP
    except ImportError as exc:
        raise ImportError("Install this package with its MCP dependencies first: pip install -e .") from exc

    cfg = load_config()
    cfg.workdir.mkdir(parents=True, exist_ok=True)
    store = ArtifactStore(cfg.workdir)
    backend = build_backend(cfg, store)
    workspace_guard = load_workspace_guard(cfg)
    arm_identity = _arm_identity(cfg)
    audit_log_path = os.environ.get("ISAACSIM_VLA_AUDIT_LOG") or str(cfg.workdir / "events.jsonl")
    append_event(
        audit_log_path,
        "mcp_server_start",
        backend=backend.name,
        workdir=cfg.workdir,
        bridge_dir=cfg.bridge_dir,
    )

    mcp = FastMCP(
        f"isaacsim-vla-{cfg.arm_target or cfg.arm_id or 'unbound'}",
        instructions=(
            f"This MCP instance is bound to {arm_identity['arm_id'] or 'an unspecified arm'} "
            f"({arm_identity['camera_side'] or 'unknown camera side'}, {arm_identity['physical_side'] or 'unknown physical side'}). "
            "Only use this instance for tasks routed to that arm. "
            "Robotics visual-prompt tools for a VLM -> box layer -> pick-and-place workflow. "
            "Use the top/overhead camera once for VLM grounding and keep its observation_id. "
            "Call draw_boxes with that exact observation_id, source_box, target_box, and labels. "
            "Then inspect the returned preview image, or call look_camera with the box_layer_id, and call "
            "confirm_box_layer only if the green source box tightly encloses the object to move and the "
            "blue target box covers the requested placement region. Only after this preview validation, reuse that "
            "layer with transfer_between_boxes for execution. Do not draw boxes on wrist cameras; wrist "
            "camera observations are for local gripper checks only. The source/object "
            "box is rendered green and the target/place box is rendered blue. "
            "The execution goal is always to move the object inside source_box to target_box; "
            "the instruction text is optional context and should not be used as a second grounding source. "
            "Darkened or masked image regions are outside this arm's configured XYZ workspace; never draw "
            "source_box or target_box there. Unmasked regions are not guaranteed IK-solvable for every "
            "gripper orientation, TCP offset, or waypoint sequence; if transfer_between_boxes reports an IK "
            "failure, redraw the target in a different unmasked region instead of retrying the same pose. "
            "For glue tasks in the RoboClaw top view, glue means the independent upright bottle with a "
            "blue pointed cap and pale/yellow label, usually left of the robot arm; never use the robot "
            "gripper, black/blue cylindrical gripper part, or colored blocks as the glue source. "
            "For normal planning validation, call transfer_between_boxes only after confirm_box_layer, with box_layer_id and leave dry_run unset; "
            "the real-robot bridge defaults to a non-motion algorithmic dry-run. "
            "Set dry_run=true only when you want to inspect the resolved overlay without planning. "
            f"For vla_execute, use this instruction unless the user explicitly overrides it: {DEFAULT_VLA_INSTRUCTION}"
        ),
    )

    @mcp.tool(structured_output=False)
    def look_camera(camera_id: CameraId = "top", box_layer_id: str | None = None) -> list[Any]:
        """Capture or fetch one observation from a robotics camera stream.

        Args:
            camera_id: Logical camera stream. Use "top" or "overhead" for global planning;
                use "wrist", "left_wrist", or "right_wrist" for local gripper checks.
            box_layer_id: Optional reusable top-camera box layer returned by draw_boxes.
                Pass this only with the same top/overhead camera stream when you want
                the current top frame returned with the fixed green/blue boxes rendered.
                Do not pass a box layer for wrist cameras.

        Returns:
            MCP content blocks: JSON metadata plus the captured image. If box_layer_id
            is provided, the image block is the current camera frame with that layer
            rendered on top.
        """
        append_event(
            audit_log_path,
            "mcp_tool_call_start",
            tool="look_camera",
            camera_id=camera_id,
            box_layer_id=box_layer_id,
        )
        try:
            result = backend.look_camera(camera_id)
            image_path = result.get("image_path")
            if not image_path:
                raise RuntimeError("backend look_camera response must include image_path")
            obs = store.import_observation(
                image_path,
                camera_id=camera_id,
                observation_id=result.get("observation_id"),
            )
            if workspace_guard is not None and workspace_guard.applies_to_camera(camera_id):
                obs = workspace_guard.mask_observation(obs)
                store.save_json("observation", obs.observation_id, obs.model_dump())
            payload = obs.model_dump()
            payload["arm_identity"] = arm_identity
            payload["backend"] = backend.name
            if isinstance(obs.metadata, dict) and obs.metadata.get("workspace_guard"):
                payload["workspace_guard"] = obs.metadata["workspace_guard"]
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
                payload["source_box"] = overlay.source_box or overlay.red_box
                payload["target_box"] = overlay.target_box or overlay.green_box
                payload["metadata"] = {
                    **payload.get("metadata", {}),
                    "rendered_box_layer": True,
                    "raw_observation_id": obs.observation_id,
                }
                append_event(
                    audit_log_path,
                    "mcp_tool_call_end",
                    tool="look_camera",
                    success=True,
                    camera_id=camera_id,
                    observation_id=obs.observation_id,
                    image_path=payload["image_path"],
                    raw_image_path=obs.image_path,
                    box_layer_id=box_layer_id,
                    box_overlay_id=overlay.box_overlay_id,
                    source_box=payload["source_box"],
                    target_box=payload["target_box"],
                    workspace_guard=payload.get("workspace_guard"),
                )
                return _json_with_image(payload, overlay.overlay_path)
            append_event(
                audit_log_path,
                "mcp_tool_call_end",
                tool="look_camera",
                success=True,
                camera_id=camera_id,
                observation_id=obs.observation_id,
                image_path=obs.image_path,
                workspace_guard=payload.get("workspace_guard"),
            )
            return _json_with_image(payload, obs.image_path)
        except Exception as exc:
            append_event(
                audit_log_path,
                "mcp_tool_call_end",
                tool="look_camera",
                success=False,
                camera_id=camera_id,
                box_layer_id=box_layer_id,
                **exception_fields(exc),
            )
            raise

    @mcp.tool(structured_output=False)
    def draw_boxes(
        camera_id: TopCameraId,
        coordinate_system: CoordinateSystem,
        source_box: list[int] | None = None,
        target_box: list[int] | None = None,
        red_box: list[int] | None = None,
        green_box: list[int] | None = None,
        observation_id: str = "",
        source_label: str = "",
        target_label: str = "",
        red_label: str | None = None,
        green_label: str | None = None,
    ) -> list[Any]:
        """Create a reusable green/blue box layer for one camera stream.

        Args:
            camera_id: Logical camera stream to attach the layer to. Use "top" or
                "overhead". Do not draw boxes on wrist cameras because wrist images
                move with the gripper.
            coordinate_system: Required. Use "normalized_1000" for Doubao/Ark
                grounding <bbox>x1 y1 x2 y2</bbox> output, whose values are
                normalized to a 1000x1000 coordinate frame in [0, 999]. Use
                "pixel" only when the boxes are already absolute pixel coordinates
                for the current image.
            source_box: Source object box [x1, y1, x2, y2]. It is rendered green.
            target_box: Target/place box [x1, y1, x2, y2]. It is rendered blue.
            red_box: Legacy alias for source_box.
            green_box: Legacy alias for target_box.
            observation_id: Required exact frame returned by look_camera. Bind boxes
                to the frame the VLM grounded.
            source_label: Required label rendered on the green source box.
            target_label: Required label rendered on the blue target box.
            red_label: Legacy alias for source_label.
            green_label: Legacy alias for target_label.

        Returns:
            MCP content blocks: JSON metadata plus a preview image. The returned
            box_layer_id is the stable handle to pass to look_camera and vla_execute.
            box_overlay_id is included only as a compatibility preview id.
        """
        append_event(
            audit_log_path,
            "mcp_tool_call_start",
            tool="draw_boxes",
            camera_id=camera_id,
            coordinate_system=coordinate_system,
            source_box=source_box,
            target_box=target_box,
            red_box=red_box,
            green_box=green_box,
            observation_id=observation_id,
            source_label=source_label if red_label is None else red_label,
            target_label=target_label if green_label is None else green_label,
        )
        try:
            camera_key = camera_id.strip().lower().replace("-", "_")
            if "wrist" in camera_key:
                raise ValueError("draw_boxes creates reusable layers only for top/overhead cameras; do not draw boxes on wrist cameras")
            if camera_key not in {"top", "overhead"}:
                raise ValueError("draw_boxes camera_id must be 'top' or 'overhead'")
            if not observation_id.strip():
                raise ValueError("draw_boxes requires observation_id from look_camera so boxes bind to the exact grounded frame")
            resolved_source_box = source_box if source_box is not None else red_box
            resolved_target_box = target_box if target_box is not None else green_box
            if resolved_source_box is None or resolved_target_box is None:
                raise ValueError("draw_boxes requires source_box and target_box; legacy red_box and green_box are also accepted")
            resolved_source_label = source_label if red_label is None else red_label
            resolved_target_label = target_label if green_label is None else green_label
            if not str(resolved_source_label or "").strip() or not str(resolved_target_label or "").strip():
                raise ValueError("draw_boxes requires source_label and target_label so the preview can be audited")
            layer_metadata: dict[str, Any] = {}
            if workspace_guard is not None and workspace_guard.applies_to_camera(camera_id):
                obs = store.get_observation(observation_id)
                source_check = workspace_guard.validate_box_input(
                    resolved_source_box,
                    width=obs.width,
                    height=obs.height,
                    coordinate_system=coordinate_system,
                    name="source_box",
                )
                target_check = workspace_guard.validate_box_input(
                    resolved_target_box,
                    width=obs.width,
                    height=obs.height,
                    coordinate_system=coordinate_system,
                    name="target_box",
                )
                layer_metadata["workspace_guard"] = {
                    **workspace_guard.metadata(width=obs.width, height=obs.height),
                    "source_projection": source_check,
                    "target_projection": target_check,
                }
            layer = store.save_box_layer(
                camera_id=camera_id,
                red_box=[int(v) for v in resolved_source_box],
                green_box=[int(v) for v in resolved_target_box],
                coordinate_system=coordinate_system,
                observation_id=observation_id,
                label_red=resolved_source_label,
                label_green=resolved_target_label,
                metadata=layer_metadata,
            )
            payload = layer.model_dump()
            payload["arm_identity"] = arm_identity
            payload["source_box"] = layer.red_box
            payload["target_box"] = layer.green_box
            payload["box_color_convention"] = "source_box_green_target_box_blue"
            payload["box_overlay_id"] = layer.preview_overlay_id
            payload["overlay_path"] = layer.preview_overlay_path
            payload["validated"] = bool(layer.metadata.get("validated", False))
            payload["warnings"] = layer.metadata.get("warnings", [])
            append_event(
                audit_log_path,
                "mcp_tool_call_end",
                tool="draw_boxes",
                success=True,
                camera_id=camera_id,
                coordinate_system=coordinate_system,
                observation_id=observation_id,
                box_layer_id=layer.box_layer_id,
                box_overlay_id=layer.preview_overlay_id,
                source_box=layer.red_box,
                target_box=layer.green_box,
                source_box_input=resolved_source_box,
                target_box_input=resolved_target_box,
                overlay_path=layer.preview_overlay_path,
                validated=payload["validated"],
                warnings=payload["warnings"],
                workspace_guard=payload.get("metadata", {}).get("workspace_guard") if isinstance(payload.get("metadata"), dict) else None,
            )
            return _json_with_image(payload, layer.preview_overlay_path)
        except Exception as exc:
            append_event(
                audit_log_path,
                "mcp_tool_call_end",
                tool="draw_boxes",
                success=False,
                camera_id=camera_id,
                coordinate_system=coordinate_system,
                observation_id=observation_id,
                **exception_fields(exc),
            )
            raise

    @mcp.tool(structured_output=False)
    def confirm_box_layer(
        box_layer_id: str,
        source_confirmation: str,
        target_confirmation: str,
        preview_observation_id: str | None = None,
    ) -> str:
        """Confirm a VLM-grounded box layer after inspecting the rendered preview.

        Args:
            box_layer_id: Layer returned by draw_boxes.
            source_confirmation: Short text confirming the green source box
                tightly encloses the object to move, e.g. "green box encloses
                the glue bottle, not the gripper".
            target_confirmation: Short text confirming the blue target box covers
                the requested placement region.
            preview_observation_id: Optional observation id of the preview frame
                inspected with look_camera(box_layer_id=...).

        Returns:
            JSON payload containing validated=true and the layer metadata.
        """
        append_event(
            audit_log_path,
            "mcp_tool_call_start",
            tool="confirm_box_layer",
            box_layer_id=box_layer_id,
            source_confirmation=source_confirmation,
            target_confirmation=target_confirmation,
            preview_observation_id=preview_observation_id,
        )
        try:
            if len(source_confirmation.strip()) < 12 or len(target_confirmation.strip()) < 12:
                raise ValueError("confirm_box_layer requires non-empty source and target confirmation text")
            layer = store.confirm_box_layer(
                box_layer_id,
                source_confirmation=source_confirmation,
                target_confirmation=target_confirmation,
                preview_observation_id=preview_observation_id,
            )
            payload = layer.model_dump()
            payload["arm_identity"] = arm_identity
            payload["validated"] = True
            payload["warnings"] = layer.metadata.get("warnings", [])
            append_event(
                audit_log_path,
                "mcp_tool_call_end",
                tool="confirm_box_layer",
                success=True,
                box_layer_id=box_layer_id,
                source_box=layer.source_box or layer.red_box,
                target_box=layer.target_box or layer.green_box,
                preview_overlay_path=layer.preview_overlay_path,
                warnings=payload["warnings"],
            )
            return _json(payload)
        except Exception as exc:
            append_event(
                audit_log_path,
                "mcp_tool_call_end",
                tool="confirm_box_layer",
                success=False,
                box_layer_id=box_layer_id,
                **exception_fields(exc),
            )
            raise

    @mcp.tool(structured_output=False)
    def vla_execute(
        instruction: str = DEFAULT_VLA_INSTRUCTION,
        box_layer_id: str | None = None,
        box_overlay_id: str | None = None,
        atomic_action: str = "pick_and_place",
        dry_run: bool = False,
    ) -> str:
        """Execute the stored source-to-target box transfer.

        Args:
            instruction: Optional natural-language context. The actionable goal is
                always to move the object inside source_box to target_box. Use the
                default instruction unless the user explicitly asks for different
                wording.
            box_layer_id: Stable layer id returned by draw_boxes. Prefer this. When
                provided, the tool fetches the current layer camera frame, renders
                the same source/target boxes onto that fresh frame, and sends the
                resolved source_box/target_box to the execution backend.
            box_overlay_id: Legacy static overlay id. Use only for backward
                compatibility; it binds execution to the old rendered frame.
            atomic_action: Atomic action type. Use "pick_and_place" for moving the
                object from the green source box to the blue target box.
            dry_run: MCP-layer overlay inspection only. If true, the tool returns
                the resolved overlay payload and does not call the planning or
                execution backend. Leave this false/unset for the default
                real-robot bridge dry-run, which plans with the deterministic
                algorithm but does not move hardware.

        Returns:
            JSON execution result from the backend.
        """
        append_event(
            audit_log_path,
            "mcp_tool_call_start",
            tool="vla_execute",
            instruction=instruction,
            box_layer_id=box_layer_id,
            box_overlay_id=box_overlay_id,
            atomic_action=atomic_action,
            dry_run=dry_run,
        )
        try:
            if box_layer_id:
                layer = store.get_box_layer(box_layer_id)
                _require_validated_layer(layer)
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
                append_event(
                    audit_log_path,
                    "mcp_tool_call_end",
                    tool="vla_execute",
                    success=True,
                    mcp_layer_dry_run=True,
                    backend=backend.name,
                    box_layer_id=overlay.box_layer_id,
                    box_overlay_id=overlay.box_overlay_id,
                    source_box=overlay.source_box or overlay.red_box,
                    target_box=overlay.target_box or overlay.green_box,
                    overlay_path=overlay.overlay_path,
                )
                return _json(
                    {
                        "success": True,
                        "dry_run": True,
                        "backend": backend.name,
                        "arm_identity": arm_identity,
                        "instruction": instruction,
                        "atomic_action": atomic_action,
                        "box_layer_id": overlay.box_layer_id,
                        "box_overlay_id": overlay.box_overlay_id,
                        "overlay": overlay.model_dump(),
                    }
                )
            result = backend.vla_execute(instruction, overlay, atomic_action)
            metadata = result.get("metadata")
            if not isinstance(metadata, dict):
                metadata = {}
                result["metadata"] = metadata
            metadata.setdefault("arm_identity", arm_identity)
            result.setdefault("backend", backend.name)
            result.setdefault("box_layer_id", overlay.box_layer_id)
            result.setdefault("box_overlay_id", overlay.box_overlay_id)
            result.setdefault("instruction", instruction)
            append_event(
                audit_log_path,
                "mcp_tool_call_end",
                tool="vla_execute",
                success=bool(result.get("success", False)),
                backend=backend.name,
                box_layer_id=overlay.box_layer_id,
                box_overlay_id=overlay.box_overlay_id,
                source_box=overlay.source_box or overlay.red_box,
                target_box=overlay.target_box or overlay.green_box,
                overlay_path=overlay.overlay_path,
                result_status=result.get("status"),
                applied=result.get("applied"),
                result_id=result.get("result_id"),
                result_path=(result.get("metadata") or {}).get("result_path") if isinstance(result.get("metadata"), dict) else None,
            )
            return _json(result)
        except Exception as exc:
            append_event(
                audit_log_path,
                "mcp_tool_call_end",
                tool="vla_execute",
                success=False,
                box_layer_id=box_layer_id,
                box_overlay_id=box_overlay_id,
                **exception_fields(exc),
            )
            raise

    @mcp.tool(structured_output=False)
    def transfer_between_boxes(
        box_layer_id: str,
        instruction: str = DEFAULT_VLA_INSTRUCTION,
        atomic_action: str = "pick_and_place",
        dry_run: bool = False,
    ) -> str:
        """Move the object from the green source box to the blue target box.

        Args:
            box_layer_id: Stable layer id returned by draw_boxes. This is
                required so the backend uses the exact source_box and target_box
                that the VLM/grounding step already created.
            instruction: Optional natural-language context. It does not define
                the executable target; execution uses the stored source_box and
                target_box.
            atomic_action: Keep as "pick_and_place" for this tool.
            dry_run: MCP-layer overlay inspection only. Leave false/unset for
                the default real-robot bridge dry-run, which plans with the
                deterministic algorithm but does not move hardware.

        Returns:
            JSON execution result from the configured backend.
        """
        append_event(
            audit_log_path,
            "mcp_tool_call_start",
            tool="transfer_between_boxes",
            box_layer_id=box_layer_id,
            instruction=instruction,
            atomic_action=atomic_action,
            dry_run=dry_run,
        )
        try:
            _require_validated_layer(store.get_box_layer(box_layer_id))
            result = vla_execute(
                instruction=instruction,
                box_layer_id=box_layer_id,
                atomic_action=atomic_action,
                dry_run=dry_run,
            )
            append_event(
                audit_log_path,
                "mcp_tool_call_end",
                tool="transfer_between_boxes",
                success=True,
                box_layer_id=box_layer_id,
            )
            return result
        except Exception as exc:
            append_event(
                audit_log_path,
                "mcp_tool_call_end",
                tool="transfer_between_boxes",
                success=False,
                box_layer_id=box_layer_id,
                **exception_fields(exc),
            )
            raise

    return mcp


def _require_validated_layer(layer: Any) -> None:
    if not isinstance(layer.metadata, dict) or layer.metadata.get("validated") is not True:
        raise PermissionError(
            "box_layer_id is not preview-validated. Inspect the rendered preview, then call confirm_box_layer before transfer_between_boxes."
        )


def _arm_identity(cfg: Any) -> dict[str, str]:
    return {
        "arm_id": str(getattr(cfg, "arm_id", "") or ""),
        "arm_target": str(getattr(cfg, "arm_target", "") or ""),
        "label": str(getattr(cfg, "arm_label", "") or ""),
        "physical_side": str(getattr(cfg, "arm_side", "") or ""),
        "camera_side": str(getattr(cfg, "arm_camera_side", "") or ""),
    }


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
