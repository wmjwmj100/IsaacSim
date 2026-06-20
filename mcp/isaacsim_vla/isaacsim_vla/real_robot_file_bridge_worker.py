from __future__ import annotations

import argparse
import json
import subprocess
import time
import uuid
from pathlib import Path
from typing import Any

from PIL import Image, ImageStat


DEFAULT_TASK = "Pick up the object inside the green box and place it at the location marked by the blue box."
DEFAULT_POLL_INTERVAL = 0.05
DEFAULT_INFER_TIMEOUT_SEC = 900.0
DEFAULT_DRY_RUN_STEPS = 1
DEFAULT_CAMERA_FILE_MAX_AGE = 5.0
DEFAULT_CAMERA_DARK_FRAME_MEAN_THRESHOLD = 0.08


def _discover_repo_root() -> Path:
    here = Path(__file__).resolve()
    for candidate in (here.parent, *here.parents):
        if (candidate / "scripts" / "vla" / "infer_data613.sh").is_file():
            return candidate
    return Path.cwd().resolve()


REPO_ROOT = _discover_repo_root()
DEFAULT_TOP_IMAGE = REPO_ROOT / ".data" / "camera" / "latest_rgb.png"
DEFAULT_WRIST_IMAGE = REPO_ROOT / ".data" / "camera" / "latest_ugreen_rgb.png"
DEFAULT_POLICY_PATH = REPO_ROOT / "model" / "pretrained_model"
DEFAULT_INFER_SCRIPT = REPO_ROOT / "scripts" / "vla" / "infer_data613.sh"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Real-robot file bridge for isaacsim-vla MCP dry-run acceptance.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--bridge-dir", required=True, help="Shared MCP request/response directory.")
    parser.add_argument("--top-image", default=str(DEFAULT_TOP_IMAGE), help="Live top camera preview PNG.")
    parser.add_argument("--wrist-image", default=str(DEFAULT_WRIST_IMAGE), help="Live wrist/side camera PNG.")
    parser.add_argument(
        "--policy-path",
        default=str(DEFAULT_POLICY_PATH),
        help="SmolVLA checkpoint path used for the dry-run acceptance invocation.",
    )
    parser.add_argument(
        "--infer-script",
        default=str(DEFAULT_INFER_SCRIPT),
        help="Wrapper used to launch the data613 dry-run inference path.",
    )
    parser.add_argument("--task", default=DEFAULT_TASK, help="Task text passed to the policy wrapper.")
    parser.add_argument("--steps", type=int, default=DEFAULT_DRY_RUN_STEPS, help="Dry-run prediction steps.")
    parser.add_argument(
        "--camera-file-max-age",
        type=float,
        default=DEFAULT_CAMERA_FILE_MAX_AGE,
        help="Maximum accepted age in seconds for live camera files during dry-run acceptance.",
    )
    parser.add_argument(
        "--camera-dark-frame-mean-threshold",
        type=float,
        default=DEFAULT_CAMERA_DARK_FRAME_MEAN_THRESHOLD,
        help="Reject live camera files darker than this normalized mean intensity threshold.",
    )
    parser.add_argument(
        "--infer-timeout-sec",
        type=float,
        default=DEFAULT_INFER_TIMEOUT_SEC,
        help="Timeout for the dry-run policy invocation.",
    )
    parser.add_argument(
        "--poll-interval",
        type=float,
        default=DEFAULT_POLL_INTERVAL,
        help="Seconds between request directory polls.",
    )
    parser.add_argument(
        "--clear-pending",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Remove stale request/response JSON before starting the worker.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    bridge_dir = Path(args.bridge_dir).expanduser().resolve()
    requests_dir = bridge_dir / "requests"
    responses_dir = bridge_dir / "responses"
    runs_dir = bridge_dir / "runs"
    requests_dir.mkdir(parents=True, exist_ok=True)
    responses_dir.mkdir(parents=True, exist_ok=True)
    runs_dir.mkdir(parents=True, exist_ok=True)
    if bool(args.clear_pending):
        for directory in (requests_dir, responses_dir):
            for path in directory.glob("*.json"):
                try:
                    path.unlink()
                except OSError as exc:
                    print(f"[real-bridge][WARN] failed to remove {path}: {exc}", flush=True)
    print(f"[real-bridge] watching {requests_dir}", flush=True)
    print(f"[real-bridge] top image: {Path(args.top_image).expanduser().resolve()}", flush=True)
    print(f"[real-bridge] wrist image: {Path(args.wrist_image).expanduser().resolve()}", flush=True)
    print(f"[real-bridge] policy: {Path(args.policy_path).expanduser().resolve()}", flush=True)
    print(f"[real-bridge] infer: {Path(args.infer_script).expanduser().resolve()}", flush=True)

    handled: set[str] = set()
    poll_interval = max(0.01, float(args.poll_interval))
    while True:
        for request_path in sorted(requests_dir.glob("*.json")):
            if request_path.name in handled:
                continue
            response_path = responses_dir / request_path.name
            try:
                request = json.loads(request_path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                continue
            response = handle_request(
                request,
                bridge_dir=bridge_dir,
                runs_dir=runs_dir,
                top_image=Path(args.top_image).expanduser().resolve(),
                wrist_image=Path(args.wrist_image).expanduser().resolve(),
                policy_path=Path(args.policy_path).expanduser().resolve(),
                infer_script=Path(args.infer_script).expanduser().resolve(),
                task=str(args.task),
                steps=int(args.steps),
                camera_file_max_age=float(args.camera_file_max_age),
                camera_dark_frame_mean_threshold=float(args.camera_dark_frame_mean_threshold),
                infer_timeout_sec=float(args.infer_timeout_sec),
            )
            write_json_atomic(response_path, response)
            handled.add(request_path.name)
        time.sleep(poll_interval)


def handle_request(
    request: dict[str, Any],
    *,
    bridge_dir: Path,
    runs_dir: Path,
    top_image: Path,
    wrist_image: Path,
    policy_path: Path,
    infer_script: Path,
    task: str,
    steps: int,
    camera_file_max_age: float,
    camera_dark_frame_mean_threshold: float,
    infer_timeout_sec: float,
) -> dict[str, Any]:
    operation = str(request.get("operation") or "")
    payload = request.get("payload") or {}
    if not isinstance(payload, dict):
        payload = {}
    request_id = str(request.get("request_id") or f"req_{uuid.uuid4().hex[:8]}")
    try:
        if operation == "look_camera":
            return {
                "success": True,
                "result": _handle_look_camera(
                    payload,
                    top_image=top_image,
                    wrist_image=wrist_image,
                    camera_file_max_age=camera_file_max_age,
                    camera_dark_frame_mean_threshold=camera_dark_frame_mean_threshold,
                ),
            }
        if operation == "vla_execute":
            result = _handle_vla_execute(
                request_id,
                payload,
                bridge_dir=bridge_dir,
                runs_dir=runs_dir,
                top_image=top_image,
                wrist_image=wrist_image,
                policy_path=policy_path,
                infer_script=infer_script,
                task=task,
                steps=steps,
                camera_file_max_age=camera_file_max_age,
                camera_dark_frame_mean_threshold=camera_dark_frame_mean_threshold,
                infer_timeout_sec=infer_timeout_sec,
            )
            if result.get("success", False):
                return {"success": True, "result": result}
            return {
                "success": False,
                "error": result.get("error") or "dry-run inference failed",
                "result": result,
            }
        return {"success": False, "error": f"unsupported operation: {operation}"}
    except Exception as exc:
        return {"success": False, "error": str(exc)}


def _handle_look_camera(
    payload: dict[str, Any],
    *,
    top_image: Path,
    wrist_image: Path,
    camera_file_max_age: float,
    camera_dark_frame_mean_threshold: float,
) -> dict[str, Any]:
    camera_id = str(payload.get("camera_id") or "top")
    image_path = _resolve_camera_path(camera_id, top_image=top_image, wrist_image=wrist_image)
    _assert_live_image(
        image_path,
        camera_file_max_age=float(payload.get("camera_file_max_age") or camera_file_max_age),
        dark_frame_mean_threshold=float(payload.get("camera_dark_frame_mean_threshold") or camera_dark_frame_mean_threshold),
    )
    return {
        "image_path": str(image_path),
        "camera_id": camera_id,
        "timestamp": time.time(),
        "metadata": {
            "camera_role": "global" if _is_top_camera(camera_id) else "wrist",
            "bridge_mode": "real-robot-file",
        },
    }


def _handle_vla_execute(
    request_id: str,
    payload: dict[str, Any],
    *,
    bridge_dir: Path,
    runs_dir: Path,
    top_image: Path,
    wrist_image: Path,
    policy_path: Path,
    infer_script: Path,
    task: str,
    steps: int,
    camera_file_max_age: float,
    camera_dark_frame_mean_threshold: float,
    infer_timeout_sec: float,
) -> dict[str, Any]:
    overlay = payload.get("overlay")
    if not isinstance(overlay, dict):
        raise ValueError("vla_execute payload must include an overlay object")
    instruction = str(payload.get("instruction") or task)
    atomic_action = payload.get("atomic_action")
    resolved_top_image = _resolve_overlay_source_image(overlay, fallback=top_image)
    if not resolved_top_image.is_file():
        raise FileNotFoundError(f"top image does not exist: {resolved_top_image}")
    if not wrist_image.is_file():
        raise FileNotFoundError(f"wrist image does not exist: {wrist_image}")
    if not infer_script.is_file():
        raise FileNotFoundError(f"infer script does not exist: {infer_script}")
    if not policy_path.exists():
        raise FileNotFoundError(f"policy path does not exist: {policy_path}")

    run_dir = runs_dir / request_id
    run_dir.mkdir(parents=True, exist_ok=True)
    overlay_json_path = run_dir / "top_overlay.json"
    overlay_payload = _build_overlay_json_payload(
        overlay,
        top_image=resolved_top_image,
        overlay_path=Path(str(overlay.get("overlay_path") or "")) if overlay.get("overlay_path") else None,
    )
    overlay_json_path.write_text(json.dumps(overlay_payload, ensure_ascii=False, indent=2), encoding="utf-8")
    _assert_live_image(
        resolved_top_image,
        camera_file_max_age=camera_file_max_age,
        dark_frame_mean_threshold=camera_dark_frame_mean_threshold,
    )
    _assert_live_image(
        wrist_image,
        camera_file_max_age=camera_file_max_age,
        dark_frame_mean_threshold=camera_dark_frame_mean_threshold,
    )

    stdout_path = run_dir / "infer.stdout.log"
    stderr_path = run_dir / "infer.stderr.log"
    result_path = run_dir / "infer_result.json"
    command = [
        str(infer_script),
        "--dry-run",
        "--steps",
        str(max(1, int(steps))),
        "--top-overlay",
        str(Path(str(overlay.get("overlay_path") or resolved_top_image)).expanduser().resolve()),
        "--top-overlay-json",
        str(overlay_json_path),
        "--wrist-image",
        str(wrist_image),
        "--policy-path",
        str(policy_path),
        "--task",
        instruction,
        "--camera-file-max-age",
        str(max(0.0, float(camera_file_max_age))),
        "--camera-dark-frame-mean-threshold",
        str(max(0.0, float(camera_dark_frame_mean_threshold))),
    ]

    proc = subprocess.run(
        command,
        cwd=str(REPO_ROOT),
        text=True,
        capture_output=True,
        timeout=infer_timeout_sec,
        check=False,
    )
    stdout_path.write_text(proc.stdout or "", encoding="utf-8")
    stderr_path.write_text(proc.stderr or "", encoding="utf-8")

    result: dict[str, Any] = {
        "success": proc.returncode == 0,
        "backend": "real-robot-file-bridge",
        "result_id": request_id,
        "status": "completed" if proc.returncode == 0 else "failed",
        "instruction": instruction,
        "atomic_action": atomic_action,
        "box_layer_id": str(overlay.get("box_layer_id") or ""),
        "box_overlay_id": str(overlay.get("box_overlay_id") or ""),
        "applied": False,
        "metadata": {
            "dry_run": True,
            "request_id": request_id,
            "bridge_dir": str(bridge_dir),
            "run_dir": str(run_dir),
            "top_image_path": str(resolved_top_image),
            "wrist_image_path": str(wrist_image),
            "overlay_json_path": str(overlay_json_path),
            "stdout_path": str(stdout_path),
            "stderr_path": str(stderr_path),
            "command": command,
            "returncode": proc.returncode,
        },
    }
    if proc.stdout:
        result["metadata"]["stdout_tail"] = _tail_text(proc.stdout)
    if proc.stderr:
        result["metadata"]["stderr_tail"] = _tail_text(proc.stderr)
    if proc.returncode != 0:
        result["error"] = f"dry-run inference failed with return code {proc.returncode}"
    result["metadata"]["result_path"] = str(result_path)
    result_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return result


def _build_overlay_json_payload(
    overlay: dict[str, Any],
    *,
    top_image: Path,
    overlay_path: Path | None,
) -> dict[str, Any]:
    source_box = _resolve_box(overlay, "source_box", "red_box")
    target_box = _resolve_box(overlay, "target_box", "green_box")
    metadata = overlay.get("metadata") if isinstance(overlay.get("metadata"), dict) else {}
    payload = {
        "image_path": str(top_image),
        "overlay_path": str(overlay_path or overlay.get("overlay_path") or ""),
        "source_box": source_box,
        "target_box": target_box,
        "red_box": source_box,
        "green_box": target_box,
        "source_label": str(overlay.get("source_label") or overlay.get("red_label") or ""),
        "target_label": str(overlay.get("target_label") or overlay.get("green_label") or ""),
        "box_color_convention": "source_box_green_target_box_blue",
        "camera_id": str(overlay.get("camera_id") or "top"),
        "observation_id": str(overlay.get("observation_id") or ""),
        "width": overlay.get("width"),
        "height": overlay.get("height"),
        "coordinate_system": str(metadata.get("coordinate_system") or "pixel"),
        "original_image_path": str(overlay.get("original_image_path") or top_image),
    }
    if metadata:
        payload["metadata"] = metadata
    return payload


def _resolve_overlay_source_image(overlay: dict[str, Any], *, fallback: Path) -> Path:
    for key in ("original_image_path", "raw_image_path", "image_path"):
        value = overlay.get(key)
        if isinstance(value, str) and value.strip():
            return Path(value).expanduser().resolve()
    return fallback.expanduser().resolve()


def _resolve_box(overlay: dict[str, Any], preferred_key: str, fallback_key: str) -> list[int]:
    box = overlay.get(preferred_key)
    if box is None:
        box = overlay.get(fallback_key)
    if not isinstance(box, list) or len(box) != 4:
        raise ValueError(f"overlay is missing {preferred_key}/{fallback_key}")
    return [int(value) for value in box]


def _resolve_camera_path(camera_id: str, *, top_image: Path, wrist_image: Path) -> Path:
    if _is_top_camera(camera_id):
        return top_image.expanduser().resolve()
    if _is_wrist_camera(camera_id):
        return wrist_image.expanduser().resolve()
    return top_image.expanduser().resolve()


def _assert_live_image(path: Path, *, camera_file_max_age: float, dark_frame_mean_threshold: float) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"camera image does not exist: {path}")
    if camera_file_max_age > 0.0:
        age_s = max(0.0, time.time() - path.stat().st_mtime)
        if age_s > camera_file_max_age:
            raise TimeoutError(f"camera image is stale ({age_s:.2f}s old): {path}")
    if dark_frame_mean_threshold > 0.0:
        with Image.open(path).convert("RGB") as image:
            stat = ImageStat.Stat(image)
            mean_intensity = float(sum(stat.mean) / len(stat.mean) / 255.0)
        if mean_intensity < dark_frame_mean_threshold:
            raise RuntimeError(
                f"camera image is too dark (mean={mean_intensity:.4f} < {dark_frame_mean_threshold:.4f}): {path}"
            )


def _is_top_camera(camera_id: str) -> bool:
    key = str(camera_id or "").strip().lower().replace("-", "_")
    return key in {"top", "overhead", "camera1"}


def _is_wrist_camera(camera_id: str) -> bool:
    key = str(camera_id or "").strip().lower().replace("-", "_")
    return "wrist" in key or key == "camera2"


def _tail_text(text: str, limit: int = 4000) -> str:
    if len(text) <= limit:
        return text
    return text[-limit:]


def write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    tmp_path = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    tmp_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp_path.replace(path)


if __name__ == "__main__":
    raise SystemExit(main())
