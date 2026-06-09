from __future__ import annotations

import argparse
import json
import time
import uuid
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description="Tiny file-bridge worker example for isaacsim-vla MCP.")
    parser.add_argument("--bridge-dir", required=True)
    parser.add_argument("--top-image", required=True)
    parser.add_argument("--wrist-image", default="")
    args = parser.parse_args()

    bridge_dir = Path(args.bridge_dir).expanduser().resolve()
    requests_dir = bridge_dir / "requests"
    responses_dir = bridge_dir / "responses"
    requests_dir.mkdir(parents=True, exist_ok=True)
    responses_dir.mkdir(parents=True, exist_ok=True)

    print(f"Watching {requests_dir}", flush=True)
    handled: set[str] = set()
    while True:
        for request_path in sorted(requests_dir.glob("*.json")):
            if request_path.name in handled:
                continue
            handled.add(request_path.name)
            try:
                request = json.loads(request_path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                handled.discard(request_path.name)
                continue
            response = handle_request(request, top_image=args.top_image, wrist_image=args.wrist_image)
            response_path = responses_dir / request_path.name
            write_json_atomic(response_path, response)
        time.sleep(0.05)


def handle_request(request: dict, *, top_image: str, wrist_image: str) -> dict:
    operation = request.get("operation")
    payload = request.get("payload") or {}
    try:
        if operation == "look_camera":
            camera_id = str(payload.get("camera_id") or "top")
            image_path = wrist_image if "wrist" in camera_id and wrist_image else top_image
            return {"success": True, "result": {"image_path": image_path}}
        if operation == "vla_execute":
            return {
                "success": True,
                "result": {
                    "success": True,
                    "backend": "file-example",
                    "result_id": request.get("request_id"),
                    "instruction": payload.get("instruction", ""),
                    "atomic_action": payload.get("atomic_action"),
                    "box_layer_id": (payload.get("overlay") or {}).get("box_layer_id", ""),
                    "box_overlay_id": (payload.get("overlay") or {}).get("box_overlay_id", ""),
                    "metadata": {"dry_run": True, "request_id": request.get("request_id")},
                },
            }
        return {"success": False, "error": f"unsupported operation: {operation}"}
    except Exception as exc:
        return {"success": False, "error": str(exc)}


def write_json_atomic(path: Path, payload: dict) -> None:
    tmp_path = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    tmp_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp_path.replace(path)


if __name__ == "__main__":
    raise SystemExit(main())
