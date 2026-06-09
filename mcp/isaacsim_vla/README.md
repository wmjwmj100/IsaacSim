# IsaacSim VLA MCP

This MCP server exposes the robotics tools for the VLM -> visual prompt -> VLA execution loop:

- `look_camera`: capture or fetch a camera observation (`top`, `wrist`, etc.).
- `draw_boxes`: create one reusable red/green `box_layer_id` on the fixed top camera view.
- `vla_execute`: execute a simplified instruction against a stored box layer.

The vision model is intentionally outside this MCP. The agent should call a VLM/grounding model once on the initial top camera image to produce `red_box` and `green_box`, then call `draw_boxes` once. Later VLA calls reuse the returned `box_layer_id`; the MCP server fetches the current top frame and renders the same layer onto it automatically. Wrist camera images are never boxed because that camera moves with the gripper.

This server is backend-neutral: use `mock` for protocol tests, `file` to bridge a running Isaac Sim process through JSON request/response files, or `http` for a future simulator/robot controller service.

## Install

```bash
cd /work/IsaacSim/mcp/isaacsim_vla
python3 -m venv .venv
.venv/bin/pip install -e .
```

## Run directly

```bash
ISAACSIM_VLA_BACKEND=mock .venv/bin/isaacsim-vla-mcp
```

MCP uses stdio, so the command will wait for an MCP client.

## Register with Hermes

Manual registration:

```bash
hermes mcp add isaacsim-vla \
  --command /work/IsaacSim/mcp/isaacsim_vla/.venv/bin/isaacsim-vla-mcp \
  --env ISAACSIM_VLA_BACKEND=mock \
  --env ISAACSIM_VLA_WORKDIR=/work/IsaacSim/outputs/isaacsim_vla_mcp
```

For the file bridge backend:

```bash
hermes mcp add isaacsim-vla \
  --command /work/IsaacSim/mcp/isaacsim_vla/.venv/bin/isaacsim-vla-mcp \
  --env ISAACSIM_VLA_BACKEND=file \
  --env ISAACSIM_VLA_BRIDGE_DIR=/work/IsaacSim/outputs/isaacsim_vla_bridge \
  --env ISAACSIM_VLA_WORKDIR=/work/IsaacSim/outputs/isaacsim_vla_mcp
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

For `vla_execute`, the backend receives a freshly rendered overlay record, including `overlay_path`, `box_layer_id`, `red_box`, `green_box`, `camera_id`, and `observation_id`.

The file bridge is the simulator/robot adapter boundary. The MCP server owns artifact bookkeeping, reusable box layers, and rendering boxes onto the current top image; the adapter only needs to implement:

- `look_camera`: return an image from a logical camera stream.
- `vla_execute`: run the VLA policy/controller from the resolved red/green overlay.

This is why `draw_boxes` takes `camera_id`, not a public `image_id`. `camera_id` names the live top stream. `observation_id` is only an optional strict frame binding returned by `look_camera`; pass it when the VLM grounded boxes on that exact initial top frame.

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

`draw_boxes` creates a reusable layer, not a one-shot image annotation. Call it once after the VLM has grounded the task on the initial top image.

Boxes are explicit about coordinate units. Use:

- `coordinate_system: "normalized_1000"` for Doubao/Ark grounding output like `<bbox>x1 y1 x2 y2</bbox>`. These values are normalized to a 1000x1000 coordinate frame in the official `[0, 999]` range, and the MCP converts them to the current image's pixels.
- `coordinate_system: "pixel"` only when `red_box` and `green_box` are already absolute pixel coordinates for the referenced observation.

Request payload:

```json
{
  "camera_id": "top",
  "observation_id": "optional_initial_top_observation_id",
  "coordinate_system": "normalized_1000",
  "red_box": [609, 510, 734, 635],
  "green_box": [656, 281, 859, 416]
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
  "red_box": [389, 244, 469, 304],
  "green_box": [419, 134, 549, 199],
  "metadata": {
    "coordinate_system": "normalized_1000",
    "red_box_input": [609, 510, 734, 635],
    "green_box_input": [656, 281, 859, 416]
  }
}
```

The layer stores pixel coordinates after conversion because the renderer, VLA overlay, simulator bridge, and real robot adapter operate on actual camera image dimensions. The original model coordinates are preserved in metadata for debugging.

The returned image block is only a preview of the layer on the reference frame. Use `box_layer_id` for later execution. `box_overlay_id` may also appear for backward compatibility, but it is a static rendered image id and should not be used for the normal closed-loop workflow.

### `vla_execute` Contract

Request payload:

```json
{
  "instruction": "move the object in the red box to the green box",
  "atomic_action": "pick_and_place",
  "overlay": {
    "box_overlay_id": "box_...",
    "box_layer_id": "layer_...",
    "camera_id": "top",
    "observation_id": "obs_...",
    "overlay_path": "/abs/path/to/red_green_overlay.png",
    "red_box": [120, 180, 220, 280],
    "green_box": [430, 160, 560, 300]
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
  "instruction": "move the object in the red box to the green box",
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

## Isaac Sim Bridge

Start the virtual environment bridge:

```bash
cd /work/IsaacSim
./run_roboclaw_mcp_vla_bridge.sh --headless
```

This launches `source/standalone_examples/custom/roboclaw_smolvla_rollout.py` in bridge mode. It watches:

```text
/work/IsaacSim/outputs/isaacsim_vla_bridge/requests
```

and writes responses to:

```text
/work/IsaacSim/outputs/isaacsim_vla_bridge/responses
```

Then register or run the MCP server with the same bridge directory:

```bash
ISAACSIM_VLA_BACKEND=file \
ISAACSIM_VLA_BRIDGE_DIR=/work/IsaacSim/outputs/isaacsim_vla_bridge \
ISAACSIM_VLA_WORKDIR=/work/IsaacSim/outputs/isaacsim_vla_mcp \
/work/IsaacSim/mcp/isaacsim_vla/.venv/bin/isaacsim-vla-mcp
```

In bridge mode, Isaac Sim does not run autonomous VLA inference on an interval. It waits for the agent sequence: initial `look_camera(top)` -> external VLM grounding -> one `draw_boxes(top)` -> repeated `vla_execute(box_layer_id=...)`.

## Environment

- `ISAACSIM_VLA_BACKEND`: `mock`, `file`, or `http`. Default: `mock`.
- `ISAACSIM_VLA_WORKDIR`: persistent output directory. Default: `/work/IsaacSim/outputs/isaacsim_vla_mcp`.
- `ISAACSIM_VLA_SAMPLE_TOP`: optional image for mock top camera.
- `ISAACSIM_VLA_SAMPLE_WRIST`: optional image for mock wrist camera.
- `ISAACSIM_VLA_BRIDGE_DIR`: required for file backend.
- `ISAACSIM_VLA_HTTP_URL`: required for http backend.
- `ISAACSIM_VLA_TIMEOUT_SEC`: backend wait timeout. Default: `120`.

## Typical Agent Flow

1. `look_camera(camera_id="top")`
2. External VLM grounding returns `red_box` and `green_box`
3. `draw_boxes(camera_id="top", red_box=[...], green_box=[...])` returns `box_layer_id`
4. `vla_execute(instruction="move the object in the red box to the green box", box_layer_id="...")`
5. For local progress checks, use `look_camera(camera_id="wrist")` without any box layer.
6. For a current top debug image with the same layer rendered, use `look_camera(camera_id="top", box_layer_id="...")`.

If the VLM grounded boxes on a specific frame, pass that frame explicitly:

```text
draw_boxes(camera_id="top", observation_id="<from look_camera>", red_box=[...], green_box=[...])
```

Do not call `draw_boxes` again unless the top camera moved, the scene was reset, or the original VLM grounding was wrong.
