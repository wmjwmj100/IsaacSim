# IsaacSim VLA MCP

This MCP server exposes the robotics tools for the VLM -> visual prompt -> pick/place execution loop:

- `look_camera`: capture or fetch a camera observation (`top`, `wrist`, etc.).
- `draw_boxes`: create one reusable green/blue `box_layer_id` on the fixed top camera view.
- `confirm_box_layer`: mark a layer as preview-validated after VLM inspection.
- `transfer_between_boxes`: execute a stored source-to-target box transfer.
- `vla_execute`: legacy alias for the same execution path.

The vision model is intentionally outside this MCP. The agent should call a VLM/grounding model once on the initial top camera image to produce `source_box` for the object and `target_box` for the placement region, then call `draw_boxes` with the exact `observation_id` returned by `look_camera`. The source/object box is rendered green and the target/place box is rendered blue. The agent must inspect the rendered preview and call `confirm_box_layer` before execution calls may reuse the returned `box_layer_id`; the MCP server fetches the current top frame and renders the same layer onto it automatically. Wrist camera images are never boxed because that camera moves with the gripper.

This server is backend-neutral: use `mock` for protocol tests, `file` to bridge a running Isaac Sim process through JSON request/response files, or `http` for a future simulator/robot controller service.

## Install

```bash
cd /home/capper/RoboClaw/external/IsaacSim/mcp/isaacsim_vla
python3 -m venv .venv
.venv/bin/pip install -e .
```

## Run directly

```bash
ISAACSIM_VLA_BACKEND=mock .venv/bin/isaacsim-vla-mcp
```

MCP uses stdio, so the command will wait for an MCP client. This direct mock
startup is only for protocol tests. In the RoboClaw dual-arm workspace, do not
use an unbound single MCP server for real camera or robot tasks; use the
generated `arm1-vla` and `arm2-vla` entries from `.data/hermes/config.yaml`.

## Register with Hermes

For RoboClaw dual-arm sessions, prefer the generated Hermes profile:

```bash
cd /home/capper/RoboClaw
./scripts/agent/configure.sh
./scripts/agent/start.sh mcp list
```

The generated config registers two bound MCP servers:

- `arm1-vla`: `ROBOCLAW_ARM=arm1`, `ROBOCLAW_ARM_TARGET=arm1`,
  `ROBOCLAW_ARM_SIDE=left`, `ROBOCLAW_ARM_CAMERA_SIDE=top-camera-left`.
- `arm2-vla`: `ROBOCLAW_ARM=arm2`, `ROBOCLAW_ARM_TARGET=arm2`,
  `ROBOCLAW_ARM_SIDE=right`, `ROBOCLAW_ARM_CAMERA_SIDE=top-camera-right`.

If you must register manually for RoboClaw, register both bound servers with
separate bridge/work directories:

```bash
hermes mcp add arm1-vla \
  --command /home/capper/RoboClaw/external/IsaacSim/mcp/isaacsim_vla/.venv/bin/isaacsim-vla-mcp \
  --env ISAACSIM_VLA_BACKEND=file \
  --env ROBOCLAW_ARM=arm1 \
  --env ROBOCLAW_ARM_TARGET=arm1 \
  --env ROBOCLAW_ARM_LABEL="RoboClaw arm1 top-camera left-side ROS/MoveIt arm" \
  --env ROBOCLAW_ARM_SIDE=left \
  --env ROBOCLAW_ARM_CAMERA_SIDE=top-camera-left \
  --env ISAACSIM_VLA_BRIDGE_DIR=/home/capper/RoboClaw/.data/isaacsim_vla_bridge/arm1 \
  --env ISAACSIM_VLA_WORKDIR=/home/capper/RoboClaw/.data/isaacsim_vla_mcp/arm1

hermes mcp add arm2-vla \
  --command /home/capper/RoboClaw/external/IsaacSim/mcp/isaacsim_vla/.venv/bin/isaacsim-vla-mcp \
  --env ISAACSIM_VLA_BACKEND=file \
  --env ROBOCLAW_ARM=arm2 \
  --env ROBOCLAW_ARM_TARGET=arm2 \
  --env ROBOCLAW_ARM_LABEL="RoboClaw arm2 top-camera right-side dual-CAN low-level arm" \
  --env ROBOCLAW_ARM_SIDE=right \
  --env ROBOCLAW_ARM_CAMERA_SIDE=top-camera-right \
  --env ISAACSIM_VLA_BRIDGE_DIR=/home/capper/RoboClaw/.data/isaacsim_vla_bridge/arm2 \
  --env ISAACSIM_VLA_WORKDIR=/home/capper/RoboClaw/.data/isaacsim_vla_mcp/arm2
```

Generic protocol-only registration:

```bash
hermes mcp add isaacsim-vla \
  --command /home/capper/RoboClaw/external/IsaacSim/mcp/isaacsim_vla/.venv/bin/isaacsim-vla-mcp \
  --env ISAACSIM_VLA_BACKEND=mock \
  --env ISAACSIM_VLA_WORKDIR=/home/capper/RoboClaw/.data/isaacsim_vla_mcp
```

For a non-RoboClaw file bridge backend:

```bash
hermes mcp add isaacsim-vla \
  --command /home/capper/RoboClaw/external/IsaacSim/mcp/isaacsim_vla/.venv/bin/isaacsim-vla-mcp \
  --env ISAACSIM_VLA_BACKEND=file \
  --env ISAACSIM_VLA_BRIDGE_DIR=/home/capper/RoboClaw/.data/isaacsim_vla_bridge \
  --env ISAACSIM_VLA_WORKDIR=/home/capper/RoboClaw/.data/isaacsim_vla_mcp
```

## File Bridge Protocol

When `ISAACSIM_VLA_BACKEND=file`, each tool call writes:

```text
$ISAACSIM_VLA_BRIDGE_DIR/requests/<request_id>.json
```

and waits for:

```text
$ISAACSIM_VLA_BRIDGE_DIR/responses/<request_id>.json
```

The request file contains:

```json
{
  "request_id": "...",
  "operation": "look_camera",
  "payload": {"camera_id": "top"}
}
```

The response should be:

```json
{
  "success": true,
  "result": {"image_path": "/abs/path/to/image.png"}
}
```

For `vla_execute`, the backend receives a freshly rendered overlay record, including `overlay_path`, `box_layer_id`, `source_box`, `target_box`, `camera_id`, and `observation_id`. Legacy `red_box`/`green_box` fields are still present for compatibility, but they now mean source/target, not display colors.

The file bridge is the simulator/robot adapter boundary. The MCP server owns artifact bookkeeping, reusable box layers, and rendering boxes onto the current top image; the adapter only needs to implement:

- `look_camera`: return an image from a logical camera stream.
- `transfer_between_boxes`: run the configured controller from the resolved green/blue overlay after the layer has been preview-confirmed. The default real-robot bridge maps `source_box` and `target_box` into the deterministic field-transfer pick/place algorithm. Set `ROBOCLAW_MCP_VLA_EXECUTION_BACKEND=vla` only when intentionally routing back through the data613 SmolVLA wrapper.

This is why `draw_boxes` takes `camera_id`, not a public `image_id`. `camera_id` names the live top stream. `observation_id` is the required strict frame binding returned by `look_camera`; pass it when the VLM grounded boxes on that exact initial top frame.

### `look_camera` Contract

Request payload:

```json
{"camera_id": "top", "box_layer_id": "optional_layer_id"}
```

Response result:

```json
{
  "image_path": "/abs/path/to/rgb.png",
  "observation_id": "optional_backend_frame_id",
  "camera_id": "top",
  "width": 640,
  "height": 480,
  "timestamp": 1780992000.0,
  "metadata": {
    "camera_role": "global"
  }
}
```

Only `image_path` is required from the backend today. If the MCP caller passes `box_layer_id` with `camera_id="top"`/`"overhead"`, the MCP response image is a freshly rendered current top frame with that reusable layer overlaid, and the raw unboxed image path is preserved as `raw_image_path`. Do not pass `box_layer_id` for wrist cameras.

The other fields are preserved as backend metadata so a real robot adapter can add camera intrinsics/extrinsics, depth paths, robot state ids, or hardware timestamps without changing the MCP tools.

### `draw_boxes` Contract

`draw_boxes` creates a reusable layer, not a one-shot image annotation. Call it once after the VLM has grounded the task on the initial top image, and then inspect the returned preview before confirming the layer.

Boxes are explicit about coordinate units. Use:

- `coordinate_system: "normalized_1000"` for Doubao/Ark grounding output like `<bbox>x1 y1 x2 y2</bbox>`. These values are normalized to a 1000x1000 coordinate frame in the official `[0, 999]` range, and the MCP converts them to the current image's pixels.
- `coordinate_system: "pixel"` only when `source_box` and `target_box` are already absolute pixel coordinates for the referenced observation.

Request payload:

```json
{
  "camera_id": "top",
  "observation_id": "required_initial_top_observation_id",
  "coordinate_system": "normalized_1000",
  "source_box": [609, 510, 734, 635],
  "target_box": [656, 281, 859, 416],
  "source_label": "glue",
  "target_label": "between blocks"
}
```

Response metadata:

```json
{
  "box_layer_id": "layer_...",
  "camera_id": "top",
  "reference_observation_id": "obs_...",
  "preview_overlay_id": "box_...",
  "preview_overlay_path": "/abs/path/to/preview.png",
  "source_box": [389, 244, 469, 304],
  "target_box": [419, 134, 549, 199],
  "metadata": {
    "coordinate_system": "normalized_1000",
    "source_box_input": [609, 510, 734, 635],
    "target_box_input": [656, 281, 859, 416],
    "box_color_convention": "source_box_green_target_box_blue"
  }
}
```

The layer stores pixel coordinates after conversion because the renderer, VLA overlay, simulator bridge, and real robot adapter operate on actual camera image dimensions. The original model coordinates are preserved in metadata for debugging.

The returned image block is only a preview of the layer on the reference frame. Inspect it and call `confirm_box_layer` before using `box_layer_id` for later execution. `box_overlay_id` may also appear for backward compatibility, but it is a static rendered image id and should not be used for the normal closed-loop workflow.

### `transfer_between_boxes` / `vla_execute` Contract

Request payload:

```json
{
  "instruction": "Pick up the object inside the green source box and place it at the location marked by the blue target box.",
  "atomic_action": "pick_and_place",
  "overlay": {
    "box_overlay_id": "box_...",
    "box_layer_id": "layer_...",
    "camera_id": "top",
    "observation_id": "obs_...",
    "overlay_path": "/abs/path/to/green_blue_overlay.png",
    "source_box": [120, 180, 220, 280],
    "target_box": [430, 160, 560, 300]
  }
}
```

Response result:

```json
{
  "success": true,
  "backend": "isaacsim-file-bridge",
  "result_id": "sim_exec_...",
  "status": "completed",
  "instruction": "Pick up the object inside the green source box and place it at the location marked by the blue target box.",
  "atomic_action": "pick_and_place",
  "box_layer_id": "layer_...",
  "box_overlay_id": "box_...",
  "applied": true,
  "metadata": {
    "frame_json_path": "/abs/path/to/frame.json",
    "result_json_path": "/abs/path/to/result.json"
  }
}
```

The current tool is synchronous because Hermes needs a direct action result. A real robot adapter can still execute internally through a job/controller layer and return after completion; the `result_id`, `status`, and `metadata` fields are already shaped so later `get_status`/`cancel` tools can be added without renaming the core tools.

## Real-Robot Box-Transfer Bridge

Start one real-robot bridge per target arm. In the top-camera workspace view,
`left` resolves to `arm1-vla` and `right` resolves to `arm2-vla`:

```bash
cd /home/capper/RoboClaw
./scripts/agent/start_bridge.sh --arm left
./scripts/agent/start_bridge.sh --arm right
```

By default, `transfer_between_boxes` and the legacy `vla_execute` alias run
deterministic field-transfer pick/place in dry-run mode. Each bridge watches its
own arm-specific directory, for example:

```text
/home/capper/RoboClaw/.data/isaacsim_vla_bridge/arm1/requests
/home/capper/RoboClaw/.data/isaacsim_vla_bridge/arm2/requests
```

and writes responses to the matching arm-specific `responses` directory.

Use the generated Hermes config for the MCP server env. If you run the lower
level MCP server manually, include the same arm identity metadata and bridge dir:

```bash
ROBOCLAW_ARM=arm1 \
ROBOCLAW_ARM_TARGET=arm1 \
ROBOCLAW_ARM_LABEL="RoboClaw arm1 top-camera left-side ROS/MoveIt arm" \
ROBOCLAW_ARM_SIDE=left \
ROBOCLAW_ARM_CAMERA_SIDE=top-camera-left \
ISAACSIM_VLA_BACKEND=file \
ISAACSIM_VLA_BRIDGE_DIR=/home/capper/RoboClaw/.data/isaacsim_vla_bridge/arm1 \
ISAACSIM_VLA_WORKDIR=/home/capper/RoboClaw/.data/isaacsim_vla_mcp/arm1 \
/home/capper/RoboClaw/external/IsaacSim/mcp/isaacsim_vla/.venv/bin/isaacsim-vla-mcp
```

Direct lower-level bridge startup is allowed only with an explicit arm target:

```bash
ROBOCLAW_ARM_TARGET=arm1 ./scripts/vla/run_real_robot_mcp_bridge.sh
ROBOCLAW_ARM_TARGET=arm2 ./scripts/vla/run_real_robot_mcp_bridge.sh
```

In this mode, execution writes a per-run `top_overlay.json` containing the stored
`source_box` and `target_box`, then runs the algorithmic pick/place wrapper with
`--bbox-json`. It validates the live top camera file and does not send robot
motion unless the bridge is explicitly started with
`ROBOCLAW_MCP_VLA_ALGORITHM_MODE=execute` and
`ROBOCLAW_MCP_VLA_YES_I_CHECKED_WORKSPACE=1`.

To route back through the data613 SmolVLA wrapper instead:

```bash
ROBOCLAW_MCP_VLA_EXECUTION_BACKEND=vla ./scripts/agent/start_bridge.sh --arm left
ROBOCLAW_MCP_VLA_EXECUTION_BACKEND=vla ./scripts/agent/start_bridge.sh --arm right
```

## Environment

- `ISAACSIM_VLA_BACKEND`: `mock`, `file`, or `http`. Default: `mock`.
- `ISAACSIM_VLA_WORKDIR`: persistent output directory. Default: `./.data/isaacsim_vla_mcp`.
- `ISAACSIM_VLA_SAMPLE_TOP`: optional image for mock top camera.
- `ISAACSIM_VLA_SAMPLE_WRIST`: optional image for mock wrist camera.
- `ISAACSIM_VLA_BRIDGE_DIR`: required for file backend.
- `ISAACSIM_VLA_HTTP_URL`: required for http backend.
- `ISAACSIM_VLA_TIMEOUT_SEC`: backend wait timeout. Default: `120`.
- `ROBOCLAW_MCP_VLA_EXECUTION_BACKEND`: real-robot bridge controller, `algorithm`
  or `vla`. Default: `algorithm`.
- `ROBOCLAW_MCP_VLA_ALGORITHM_MODE`: `dry-run` or `execute`. Default: `dry-run`.
- `ROBOCLAW_MCP_VLA_ALGORITHM_SCRIPT`: optional override for the pick/place
  wrapper. Defaults by arm target.

## Typical Agent Flow

1. `look_camera(camera_id="top")`
2. External VLM grounding returns `source_box` for the object and `target_box` for the placement region.
3. `draw_boxes(camera_id="top", observation_id="<from look_camera>", source_box=[...], target_box=[...], source_label="...", target_label="...")` returns `box_layer_id`
4. Inspect the preview image, or call `look_camera(camera_id="top", box_layer_id="...")` for a current top debug image with the same layer rendered.
5. If the preview is correct, call `confirm_box_layer(box_layer_id="...", source_confirmation="...", target_confirmation="...")`.
6. `transfer_between_boxes(instruction="Pick up the object inside the green source box and place it at the location marked by the blue target box.", box_layer_id="...")`
7. For local progress checks, use `look_camera(camera_id="wrist")` without any box layer.

Do not call `draw_boxes` again unless the top camera moved, the scene was reset, or the original VLM grounding was wrong.
