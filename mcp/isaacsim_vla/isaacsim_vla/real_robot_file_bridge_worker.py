from __future__ import annotations

import argparse
import contextlib
import fcntl
import json
import os
import subprocess
import time
import uuid
from pathlib import Path
from typing import Any

from PIL import Image, ImageStat

try:
    from isaacsim_vla.audit import append_event, exception_fields
except ModuleNotFoundError:  # Direct script execution sets sys.path to this package dir.
    from audit import append_event, exception_fields


DEFAULT_TASK = "Pick up the object inside the green box and place it at the location marked by the blue box."
DEFAULT_POLL_INTERVAL = 0.05
DEFAULT_INFER_TIMEOUT_SEC = 900.0
DEFAULT_DRY_RUN_STEPS = 1
DEFAULT_CAMERA_FILE_MAX_AGE = 5.0
DEFAULT_CAMERA_DARK_FRAME_MEAN_THRESHOLD = 0.08
DEFAULT_EXECUTION_BACKEND = "algorithm"
DEFAULT_ALGORITHM_MODE = "dry-run"
PROCESSING_SUFFIX_PREFIX = ".processing."


def _discover_repo_root() -> Path:
    here = Path(__file__).resolve()
    for candidate in (here.parent, *here.parents):
        if (candidate / "scripts" / "vla" / "infer_data613.sh").is_file():
            return candidate
    return Path.cwd().resolve()


REPO_ROOT = _discover_repo_root()
DEFAULT_TOP_IMAGE = REPO_ROOT / ".data" / "camera" / "latest_rgb.png"
DEFAULT_WRIST_IMAGE = REPO_ROOT / ".data" / "camera" / "latest_ugreen_rgb.png"
DEFAULT_POLICY_PATH = REPO_ROOT / "checkpoint" / "2026-06-14" / "data613" / "pretrained_model"
DEFAULT_INFER_SCRIPT = REPO_ROOT / "scripts" / "vla" / "infer_data613.sh"
DEFAULT_ARM2_ALGORITHM_SCRIPT = REPO_ROOT / "scripts" / "arm2" / "pick_place.sh"
DEFAULT_FIELD_ALGORITHM_SCRIPT = REPO_ROOT / "scripts" / "field" / "pick_place.sh"
DEFAULT_ARM1_ALGORITHM_CONFIG = REPO_ROOT / ".data" / "field_execution" / "site_config.a4_candidate6_redblock_exec_20260626.json"
DEFAULT_ARM2_ALGORITHM_CONFIG = REPO_ROOT / ".data" / "field_execution" / "site_config.arm2_topdown_3point.json"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Real-robot file bridge for isaacsim-vla MCP box-transfer execution.",
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
        "--execution-backend",
        choices=("algorithm", "vla"),
        default=os.environ.get("ROBOCLAW_MCP_VLA_EXECUTION_BACKEND", DEFAULT_EXECUTION_BACKEND),
        help=(
            "Controller used for vla_execute. algorithm maps source_box/target_box to "
            "field-transfer pick_place; vla runs the data613 SmolVLA wrapper."
        ),
    )
    parser.add_argument(
        "--arm-target",
        choices=("arm1", "arm2"),
        default=os.environ.get("ROBOCLAW_ARM_TARGET", os.environ.get("ROBOCLAW_ARM", "")).strip() or None,
        required=not bool(os.environ.get("ROBOCLAW_ARM_TARGET", os.environ.get("ROBOCLAW_ARM", "")).strip()),
        help="Robot arm used by the algorithmic pick/place backend. Required in dual-arm workspaces.",
    )
    parser.add_argument(
        "--algorithm-script",
        default=os.environ.get("ROBOCLAW_MCP_VLA_ALGORITHM_SCRIPT", ""),
        help="Override the algorithmic pick/place wrapper. Defaults are selected only after explicit --arm-target.",
    )
    parser.add_argument(
        "--algorithm-config",
        default=os.environ.get("ROBOCLAW_MCP_VLA_ALGORITHM_CONFIG", ""),
        help="Override the field-transfer site config used by the algorithmic pick/place backend.",
    )
    parser.add_argument(
        "--algorithm-mode",
        choices=("dry-run", "execute"),
        default=os.environ.get("ROBOCLAW_MCP_VLA_ALGORITHM_MODE", DEFAULT_ALGORITHM_MODE),
        help="Run algorithmic pick/place as planning dry-run or real execution.",
    )
    parser.add_argument(
        "--yes-i-checked-workspace",
        action="store_true",
        default=os.environ.get("ROBOCLAW_MCP_VLA_YES_I_CHECKED_WORKSPACE", "").lower() in {"1", "true", "yes", "on"},
        help="Required with --algorithm-mode execute before real robot motion is allowed.",
    )
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
    parser.add_argument(
        "--audit-log",
        default=os.environ.get("ROBOCLAW_MCP_VLA_AUDIT_LOG", ""),
        help="JSONL audit log path. Defaults to <bridge-dir>/events.jsonl.",
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
    audit_log_path = Path(args.audit_log).expanduser().resolve() if args.audit_log else bridge_dir / "events.jsonl"
    if bool(args.clear_pending):
        cleared_paths = _clear_pending_requests(requests_dir, responses_dir)
    else:
        cleared_paths = []
    print(f"[real-bridge] watching {requests_dir}", flush=True)
    print(f"[real-bridge] top image: {Path(args.top_image).expanduser().resolve()}", flush=True)
    print(f"[real-bridge] wrist image: {Path(args.wrist_image).expanduser().resolve()}", flush=True)
    print(f"[real-bridge] policy: {Path(args.policy_path).expanduser().resolve()}", flush=True)
    print(f"[real-bridge] infer: {Path(args.infer_script).expanduser().resolve()}", flush=True)
    print(f"[real-bridge] execution backend: {args.execution_backend}", flush=True)
    print(f"[real-bridge] arm target: {args.arm_target}", flush=True)
    if args.execution_backend == "algorithm":
        print(f"[real-bridge] algorithm mode: {args.algorithm_mode}", flush=True)
        if args.algorithm_script:
            print(f"[real-bridge] algorithm script: {Path(args.algorithm_script).expanduser().resolve()}", flush=True)
        if args.algorithm_config:
            print(f"[real-bridge] algorithm config: {Path(args.algorithm_config).expanduser().resolve()}", flush=True)
    print(f"[real-bridge] audit log: {audit_log_path}", flush=True)
    lock_path = bridge_dir / "worker.lock"
    with _exclusive_worker_lock(lock_path, audit_log_path):
        append_event(
            audit_log_path,
            "bridge_worker_start",
            bridge_dir=bridge_dir,
            requests_dir=requests_dir,
            responses_dir=responses_dir,
            runs_dir=runs_dir,
            top_image=Path(args.top_image).expanduser().resolve(),
            wrist_image=Path(args.wrist_image).expanduser().resolve(),
            execution_backend=args.execution_backend,
            arm_target=args.arm_target,
            algorithm_mode=args.algorithm_mode,
            algorithm_script=Path(args.algorithm_script).expanduser().resolve() if args.algorithm_script else "",
            algorithm_config=Path(args.algorithm_config).expanduser().resolve() if args.algorithm_config else "",
            yes_i_checked_workspace=bool(args.yes_i_checked_workspace),
            cleared_pending=cleared_paths,
            lock_path=lock_path,
        )

        poll_interval = max(0.01, float(args.poll_interval))
        while True:
            for request_path in sorted(requests_dir.glob("*.json")):
                response_path = responses_dir / request_path.name
                if _discard_answered_request(request_path, response_path, audit_log_path):
                    continue
                claim_path = _claim_request(request_path, audit_log_path)
                if claim_path is None:
                    continue
                request: dict[str, Any] | None = None
                try:
                    request = json.loads(claim_path.read_text(encoding="utf-8"))
                except json.JSONDecodeError:
                    append_event(
                        audit_log_path,
                        "bridge_request_claim_invalid_json",
                        request_file=request_path,
                        claim_file=claim_path,
                    )
                    _release_claim(claim_path)
                    continue
                try:
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
                        execution_backend=str(args.execution_backend),
                        arm_target=str(args.arm_target),
                        algorithm_script=Path(args.algorithm_script).expanduser().resolve() if args.algorithm_script else None,
                        algorithm_config=Path(args.algorithm_config).expanduser().resolve() if args.algorithm_config else None,
                        algorithm_mode=str(args.algorithm_mode),
                        yes_i_checked_workspace=bool(args.yes_i_checked_workspace),
                        audit_log_path=audit_log_path,
                    )
                    write_json_atomic(response_path, response)
                    append_event(
                        audit_log_path,
                        "bridge_response_written",
                        request_file=request_path,
                        claim_file=claim_path,
                        response_file=response_path,
                        request_id=request.get("request_id"),
                        operation=request.get("operation"),
                        success=bool(response.get("success", False)),
                        error=response.get("error"),
                    )
                finally:
                    _release_claim(claim_path)
            time.sleep(poll_interval)


@contextlib.contextmanager
def _exclusive_worker_lock(lock_path: Path, audit_log_path: Path | str | None):
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    handle = lock_path.open("a+", encoding="utf-8")
    try:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            handle.seek(0)
            owner = handle.read().strip()
            message = f"another real-robot bridge worker already owns {lock_path}"
            if owner:
                message = f"{message}: {owner}"
            print(f"[real-bridge][ERROR] {message}", flush=True)
            append_event(audit_log_path, "bridge_worker_lock_busy", lock_path=lock_path, owner=owner)
            raise SystemExit(75)
        handle.seek(0)
        handle.truncate()
        handle.write(json.dumps({"pid": os.getpid(), "started_at": time.time()}, sort_keys=True))
        handle.flush()
        append_event(audit_log_path, "bridge_worker_lock_acquired", lock_path=lock_path)
        yield
    finally:
        with contextlib.suppress(Exception):
            append_event(audit_log_path, "bridge_worker_lock_released", lock_path=lock_path)
        with contextlib.suppress(Exception):
            handle.seek(0)
            handle.truncate()
            handle.flush()
        with contextlib.suppress(Exception):
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        handle.close()


def _claim_request(request_path: Path, audit_log_path: Path | str | None) -> Path | None:
    claim_path = request_path.with_name(f"{request_path.name}{PROCESSING_SUFFIX_PREFIX}{os.getpid()}")
    try:
        request_path.rename(claim_path)
    except FileNotFoundError:
        return None
    except OSError as exc:
        append_event(audit_log_path, "bridge_request_claim_failed", request_file=request_path, error=str(exc))
        return None
    append_event(audit_log_path, "bridge_request_claimed", request_file=request_path, claim_file=claim_path)
    return claim_path


def _release_claim(claim_path: Path) -> None:
    with contextlib.suppress(FileNotFoundError):
        claim_path.unlink()


def _discard_answered_request(request_path: Path, response_path: Path, audit_log_path: Path | str | None) -> bool:
    if not response_path.is_file():
        return False
    try:
        request_path.unlink()
    except FileNotFoundError:
        return True
    except OSError as exc:
        append_event(
            audit_log_path,
            "bridge_answered_request_discard_failed",
            request_file=request_path,
            response_file=response_path,
            error=str(exc),
        )
        return False
    append_event(
        audit_log_path,
        "bridge_answered_request_discarded",
        request_file=request_path,
        response_file=response_path,
    )
    return True


def _clear_pending_requests(requests_dir: Path, responses_dir: Path) -> list[str]:
    cleared_paths: list[str] = []
    for path in requests_dir.glob("*.json"):
        try:
            path.unlink()
            cleared_paths.append(str(path))
        except OSError as exc:
            print(f"[real-bridge][WARN] failed to remove {path}: {exc}", flush=True)
    for path in responses_dir.glob("*.json"):
        try:
            path.unlink()
            cleared_paths.append(str(path))
        except OSError as exc:
            print(f"[real-bridge][WARN] failed to remove {path}: {exc}", flush=True)
    return cleared_paths


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
    execution_backend: str = DEFAULT_EXECUTION_BACKEND,
    arm_target: str = "",
    algorithm_script: Path | None = None,
    algorithm_config: Path | None = None,
    algorithm_mode: str = DEFAULT_ALGORITHM_MODE,
    yes_i_checked_workspace: bool = False,
    audit_log_path: Path | str | None = None,
) -> dict[str, Any]:
    operation = str(request.get("operation") or "")
    payload = request.get("payload") or {}
    if not isinstance(payload, dict):
        payload = {}
    request_id = str(request.get("request_id") or f"req_{uuid.uuid4().hex[:8]}")
    append_event(
        audit_log_path,
        "bridge_request_start",
        request_id=request_id,
        operation=operation,
        payload_summary=_payload_summary(payload),
        execution_backend=execution_backend,
        arm_target=arm_target,
        algorithm_mode=algorithm_mode,
    )
    try:
        if operation == "look_camera":
            response = {
                "success": True,
                "result": _handle_look_camera(
                    payload,
                    top_image=top_image,
                    wrist_image=wrist_image,
                    camera_file_max_age=camera_file_max_age,
                    camera_dark_frame_mean_threshold=camera_dark_frame_mean_threshold,
                ),
            }
            append_event(
                audit_log_path,
                "bridge_request_end",
                request_id=request_id,
                operation=operation,
                success=True,
                result_summary=_payload_summary(response.get("result") or {}),
            )
            return response
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
                execution_backend=execution_backend,
                arm_target=arm_target,
                algorithm_script=algorithm_script,
                algorithm_config=algorithm_config,
                algorithm_mode=algorithm_mode,
                yes_i_checked_workspace=yes_i_checked_workspace,
                audit_log_path=audit_log_path,
            )
            if result.get("success", False):
                response = {"success": True, "result": result}
                append_event(
                    audit_log_path,
                    "bridge_request_end",
                    request_id=request_id,
                    operation=operation,
                    success=True,
                    result_summary=_payload_summary(result),
                )
                return response
            response = {
                "success": False,
                "error": result.get("error") or "dry-run inference failed",
                "result": result,
            }
            append_event(
                audit_log_path,
                "bridge_request_end",
                request_id=request_id,
                operation=operation,
                success=False,
                error=response["error"],
                result_summary=_payload_summary(result),
            )
            return response
        response = {"success": False, "error": f"unsupported operation: {operation}"}
        append_event(
            audit_log_path,
            "bridge_request_end",
            request_id=request_id,
            operation=operation,
            success=False,
            error=response["error"],
        )
        return response
    except Exception as exc:
        append_event(
            audit_log_path,
            "bridge_request_end",
            request_id=request_id,
            operation=operation,
            success=False,
            **exception_fields(exc),
        )
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
    execution_backend: str,
    arm_target: str,
    algorithm_script: Path | None,
    algorithm_config: Path | None,
    algorithm_mode: str,
    yes_i_checked_workspace: bool,
    audit_log_path: Path | str | None = None,
) -> dict[str, Any]:
    overlay = payload.get("overlay")
    if not isinstance(overlay, dict):
        raise ValueError("vla_execute payload must include an overlay object")
    instruction = str(payload.get("instruction") or task)
    atomic_action = payload.get("atomic_action")
    resolved_top_image = _resolve_overlay_source_image(overlay, fallback=top_image)
    if not resolved_top_image.is_file():
        raise FileNotFoundError(f"top image does not exist: {resolved_top_image}")

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
    normalized_backend = str(execution_backend or DEFAULT_EXECUTION_BACKEND).strip().lower()
    if normalized_backend == "algorithm":
        return _run_algorithm_pick_place(
            request_id,
            overlay,
            run_dir=run_dir,
            bridge_dir=bridge_dir,
            overlay_json_path=overlay_json_path,
            top_image=resolved_top_image,
            instruction=instruction,
            atomic_action=atomic_action,
            arm_target=arm_target,
            algorithm_script=algorithm_script,
            algorithm_config=algorithm_config,
            algorithm_mode=algorithm_mode,
            yes_i_checked_workspace=yes_i_checked_workspace,
            infer_timeout_sec=infer_timeout_sec,
            audit_log_path=audit_log_path,
        )
    if normalized_backend != "vla":
        raise ValueError(f"unsupported execution backend: {execution_backend!r}")

    if not wrist_image.is_file():
        raise FileNotFoundError(f"wrist image does not exist: {wrist_image}")
    if not infer_script.is_file():
        raise FileNotFoundError(f"infer script does not exist: {infer_script}")
    if not policy_path.exists():
        raise FileNotFoundError(f"policy path does not exist: {policy_path}")
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
    append_event(
        audit_log_path,
        "bridge_vla_subprocess_start",
        request_id=request_id,
        command=command,
        run_dir=run_dir,
        overlay_json_path=overlay_json_path,
        box_layer_id=str(overlay.get("box_layer_id") or ""),
        box_overlay_id=str(overlay.get("box_overlay_id") or ""),
    )

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
            "execution_backend": "vla",
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
        result["error"] = _subprocess_error(
            "dry-run inference",
            proc.returncode,
            stdout=proc.stdout,
            stderr=proc.stderr,
        )
    result["metadata"]["result_path"] = str(result_path)
    result_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    append_event(
        audit_log_path,
        "bridge_vla_subprocess_end",
        request_id=request_id,
        success=bool(result["success"]),
        returncode=proc.returncode,
        stdout_path=stdout_path,
        stderr_path=stderr_path,
        result_path=result_path,
    )
    return result


def _run_algorithm_pick_place(
    request_id: str,
    overlay: dict[str, Any],
    *,
    run_dir: Path,
    bridge_dir: Path,
    overlay_json_path: Path,
    top_image: Path,
    instruction: str,
    atomic_action: Any,
    arm_target: str,
    algorithm_script: Path | None,
    algorithm_config: Path | None,
    algorithm_mode: str,
    yes_i_checked_workspace: bool,
    infer_timeout_sec: float,
    audit_log_path: Path | str | None = None,
) -> dict[str, Any]:
    normalized_arm = str(arm_target or "").strip().lower()
    if not normalized_arm:
        raise ValueError("algorithm backend requires an explicit arm target: arm1 or arm2")
    if normalized_arm not in {"arm1", "arm2"}:
        raise ValueError(f"unsupported arm target for algorithm backend: {arm_target!r}")
    normalized_mode = str(algorithm_mode or DEFAULT_ALGORITHM_MODE).strip().lower()
    if normalized_mode not in {"dry-run", "execute"}:
        raise ValueError(f"unsupported algorithm mode: {algorithm_mode!r}")
    if normalized_mode == "execute" and not yes_i_checked_workspace:
        raise PermissionError("--algorithm-mode execute requires --yes-i-checked-workspace")

    script = algorithm_script or _default_algorithm_script(normalized_arm)
    if not script.is_file():
        raise FileNotFoundError(f"algorithm script does not exist: {script}")
    config_path = algorithm_config or _default_algorithm_config(normalized_arm)
    if not config_path.is_file():
        raise FileNotFoundError(f"algorithm config does not exist: {config_path}")

    source_box = _resolve_box(overlay, "source_box", "red_box")
    target_box = _resolve_box(overlay, "target_box", "green_box")
    command = [
        str(script),
        "--config",
        str(config_path),
        "--bbox-json",
        str(overlay_json_path),
        "--arm",
        normalized_arm,
        "--allow-uncalibrated",
        "--allow-placeholder-adapter",
        "--plan-ik",
    ]
    if normalized_arm == "arm1":
        command.extend([
            "--approach-style",
            "top-down",
            "--tcp-offset",
            "0.15580,0,0",
            "--grasp-z-offset",
            "0.003",
            "--final-grasp-drop",
            "0.005",
            "--place-z-offset",
            "0.015",
            "--approach-height",
            "0.06",
            "--lift-height",
            "0.06",
            "--transit-height",
            "0.06",
            "--top-down-step",
            "0.025",
            "--top-down-roll-candidates",
            "0,1.570796,3.141593,4.712389",
            "--ik-timeout",
            "0.2",
            "--ik-attempts",
            "2",
            "--ik-subprocess-timeout",
            "8",
            "--move-time",
            "12",
            "--gripper-time",
            "5",
            "--no-retreat",
        ])
    stdout_path = run_dir / "algorithm.stdout.log"
    stderr_path = run_dir / "algorithm.stderr.log"
    result_path = run_dir / "algorithm_result.json"
    preflight_stdout_path = run_dir / "algorithm_preflight.stdout.log"
    preflight_stderr_path = run_dir / "algorithm_preflight.stderr.log"
    preflight_command: list[str] | None = None
    preflight_proc: subprocess.CompletedProcess[str] | None = None

    if normalized_mode == "execute":
        preflight_command = list(command)
        append_event(
            audit_log_path,
            "bridge_algorithm_preflight_start",
            request_id=request_id,
            algorithm_mode=normalized_mode,
            arm_target=normalized_arm,
            source_box=source_box,
            target_box=target_box,
            command=preflight_command,
            run_dir=run_dir,
            overlay_json_path=overlay_json_path,
            box_layer_id=str(overlay.get("box_layer_id") or ""),
            box_overlay_id=str(overlay.get("box_overlay_id") or ""),
        )
        preflight_proc = subprocess.run(
            preflight_command,
            cwd=str(REPO_ROOT),
            text=True,
            capture_output=True,
            timeout=infer_timeout_sec,
            check=False,
        )
        preflight_stdout_path.write_text(preflight_proc.stdout or "", encoding="utf-8")
        preflight_stderr_path.write_text(preflight_proc.stderr or "", encoding="utf-8")
        append_event(
            audit_log_path,
            "bridge_algorithm_preflight_end",
            request_id=request_id,
            success=preflight_proc.returncode == 0,
            returncode=preflight_proc.returncode,
            stdout_path=preflight_stdout_path,
            stderr_path=preflight_stderr_path,
        )
        if preflight_proc.returncode != 0:
            result: dict[str, Any] = {
                "success": False,
                "backend": "real-robot-file-bridge",
                "result_id": request_id,
                "status": "failed",
                "failure_stage": "preflight_ik",
                "instruction": instruction,
                "atomic_action": atomic_action,
                "box_layer_id": str(overlay.get("box_layer_id") or ""),
                "box_overlay_id": str(overlay.get("box_overlay_id") or ""),
                "applied": False,
                "metadata": {
                    "dry_run": False,
                    "execution_backend": "algorithm",
                    "algorithm_mode": normalized_mode,
                    "request_id": request_id,
                    "bridge_dir": str(bridge_dir),
                    "run_dir": str(run_dir),
                    "top_image_path": str(top_image),
                    "overlay_json_path": str(overlay_json_path),
                    "stdout_path": str(stdout_path),
                    "stderr_path": str(stderr_path),
                    "preflight_stdout_path": str(preflight_stdout_path),
                    "preflight_stderr_path": str(preflight_stderr_path),
                    "source_box": source_box,
                    "target_box": target_box,
                    "algorithm_script": str(script),
                    "algorithm_config": str(config_path),
                    "arm_target": normalized_arm,
                    "arm_side": os.environ.get("ROBOCLAW_ARM_SIDE", ""),
                    "arm_camera_side": os.environ.get("ROBOCLAW_ARM_CAMERA_SIDE", ""),
                    "arm_label": os.environ.get("ROBOCLAW_ARM_LABEL", ""),
                    "command": command,
                    "preflight_command": preflight_command,
                    "preflight_returncode": preflight_proc.returncode,
                    "returncode": preflight_proc.returncode,
                },
                "error": (
                    "algorithmic pick/place preflight failed before real motion; "
                    "the selected boxes are inside the configured XYZ workspace but the full "
                    "IK waypoint sequence is not solvable for the current arm profile. "
                    + _subprocess_error(
                        "algorithmic pick/place preflight",
                        preflight_proc.returncode,
                        stdout=preflight_proc.stdout,
                        stderr=preflight_proc.stderr,
                    )
                ),
            }
            if preflight_proc.stdout:
                result["metadata"]["preflight_stdout_tail"] = _tail_text(preflight_proc.stdout)
            if preflight_proc.stderr:
                result["metadata"]["preflight_stderr_tail"] = _tail_text(preflight_proc.stderr)
            result["metadata"]["result_path"] = str(result_path)
            result_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
            append_event(
                audit_log_path,
                "bridge_algorithm_subprocess_end",
                request_id=request_id,
                success=False,
                applied=False,
                returncode=preflight_proc.returncode,
                stdout_path=stdout_path,
                stderr_path=stderr_path,
                result_path=result_path,
                failure_stage="preflight_ik",
            )
            return result
        command.extend(["--execute", "--yes-i-checked-workspace"])

    append_event(
        audit_log_path,
        "bridge_algorithm_subprocess_start",
        request_id=request_id,
        algorithm_mode=normalized_mode,
        arm_target=normalized_arm,
        source_box=source_box,
        target_box=target_box,
        command=command,
        run_dir=run_dir,
        overlay_json_path=overlay_json_path,
        box_layer_id=str(overlay.get("box_layer_id") or ""),
        box_overlay_id=str(overlay.get("box_overlay_id") or ""),
    )
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
        "applied": normalized_mode == "execute" and proc.returncode == 0,
        "metadata": {
            "dry_run": normalized_mode == "dry-run",
            "execution_backend": "algorithm",
            "algorithm_mode": normalized_mode,
            "request_id": request_id,
            "bridge_dir": str(bridge_dir),
            "run_dir": str(run_dir),
            "top_image_path": str(top_image),
            "overlay_json_path": str(overlay_json_path),
            "stdout_path": str(stdout_path),
            "stderr_path": str(stderr_path),
            "source_box": source_box,
            "target_box": target_box,
            "algorithm_script": str(script),
            "algorithm_config": str(config_path),
            "arm_target": normalized_arm,
            "arm_side": os.environ.get("ROBOCLAW_ARM_SIDE", ""),
            "arm_camera_side": os.environ.get("ROBOCLAW_ARM_CAMERA_SIDE", ""),
            "arm_label": os.environ.get("ROBOCLAW_ARM_LABEL", ""),
            "command": command,
            "returncode": proc.returncode,
        },
    }
    if preflight_command is not None and preflight_proc is not None:
        result["metadata"]["preflight_command"] = preflight_command
        result["metadata"]["preflight_returncode"] = preflight_proc.returncode
        result["metadata"]["preflight_stdout_path"] = str(preflight_stdout_path)
        result["metadata"]["preflight_stderr_path"] = str(preflight_stderr_path)
    if proc.stdout:
        result["metadata"]["stdout_tail"] = _tail_text(proc.stdout)
    if proc.stderr:
        result["metadata"]["stderr_tail"] = _tail_text(proc.stderr)
    if proc.returncode != 0:
        result["error"] = _subprocess_error(
            "algorithmic pick/place",
            proc.returncode,
            stdout=proc.stdout,
            stderr=proc.stderr,
        )
    result["metadata"]["result_path"] = str(result_path)
    result_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    append_event(
        audit_log_path,
        "bridge_algorithm_subprocess_end",
        request_id=request_id,
        success=bool(result["success"]),
        applied=bool(result["applied"]),
        returncode=proc.returncode,
        stdout_path=stdout_path,
        stderr_path=stderr_path,
        result_path=result_path,
    )
    return result


def _default_algorithm_script(arm_target: str) -> Path:
    if arm_target == "arm2":
        return DEFAULT_ARM2_ALGORITHM_SCRIPT
    return DEFAULT_FIELD_ALGORITHM_SCRIPT


def _default_algorithm_config(arm_target: str) -> Path:
    if arm_target == "arm2":
        return DEFAULT_ARM2_ALGORITHM_CONFIG
    return DEFAULT_ARM1_ALGORITHM_CONFIG


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


def _subprocess_error(label: str, returncode: int, *, stdout: str = "", stderr: str = "") -> str:
    detail = _tail_text((stderr or stdout or "").strip(), limit=1200)
    message = f"{label} failed with return code {returncode}"
    if detail:
        message = f"{message}: {detail}"
    return message


def _payload_summary(payload: dict[str, Any]) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    for key in (
        "camera_id",
        "instruction",
        "atomic_action",
        "box_layer_id",
        "box_overlay_id",
        "image_path",
        "overlay_path",
        "source_box",
        "target_box",
        "red_box",
        "green_box",
        "success",
        "status",
        "applied",
        "result_id",
    ):
        if key in payload:
            summary[key] = payload[key]
    overlay = payload.get("overlay")
    if isinstance(overlay, dict):
        summary["overlay"] = _payload_summary(overlay)
    metadata = payload.get("metadata")
    if isinstance(metadata, dict):
        for key in (
            "dry_run",
            "execution_backend",
            "algorithm_mode",
            "request_id",
            "run_dir",
            "result_path",
            "returncode",
        ):
            if key in metadata:
                summary[f"metadata.{key}"] = metadata[key]
    return summary


def write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    tmp_path = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    tmp_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp_path.replace(path)


if __name__ == "__main__":
    raise SystemExit(main())
