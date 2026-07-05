from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ServerConfig:
    backend: str
    workdir: Path
    bridge_dir: Path | None
    http_url: str
    timeout_sec: float
    sample_top: Path | None
    sample_wrist: Path | None
    live_top_image: Path
    live_wrist_image: Path
    camera_file_max_age: float
    camera_dark_frame_mean_threshold: float
    repo_root: Path
    arm_id: str
    arm_target: str
    arm_label: str
    arm_side: str
    arm_camera_side: str


def _path_env(name: str) -> Path | None:
    value = os.getenv(name, "").strip()
    if not value:
        return None
    return Path(value).expanduser().resolve()


def _discover_repo_root() -> Path:
    here = Path(__file__).resolve()
    for candidate in (here.parent, *here.parents):
        if (candidate / "scripts" / "vla" / "infer_data613.sh").is_file():
            return candidate
    return here.parents[4]


def load_config() -> ServerConfig:
    default_repo_root = _discover_repo_root()
    repo_root = Path(os.getenv("ISAACSIM_REPO_ROOT", str(default_repo_root))).expanduser().resolve()
    workdir = _path_env("ISAACSIM_VLA_WORKDIR") or (repo_root / ".data" / "isaacsim_vla_mcp")
    backend = os.getenv("ISAACSIM_VLA_BACKEND", "mock").strip().lower() or "mock"
    timeout_raw = os.getenv("ISAACSIM_VLA_TIMEOUT_SEC", "120").strip()
    try:
        timeout_sec = float(timeout_raw)
    except ValueError:
        timeout_sec = 120.0
    camera_file_max_age_raw = os.getenv("ROBOCLAW_MCP_VLA_CAMERA_FILE_MAX_AGE", "5").strip()
    try:
        camera_file_max_age = float(camera_file_max_age_raw)
    except ValueError:
        camera_file_max_age = 5.0
    camera_dark_raw = os.getenv("ROBOCLAW_MCP_VLA_CAMERA_DARK_FRAME_MEAN_THRESHOLD", "0.08").strip()
    try:
        camera_dark_frame_mean_threshold = float(camera_dark_raw)
    except ValueError:
        camera_dark_frame_mean_threshold = 0.08
    return ServerConfig(
        backend=backend,
        workdir=workdir,
        bridge_dir=_path_env("ISAACSIM_VLA_BRIDGE_DIR"),
        http_url=os.getenv("ISAACSIM_VLA_HTTP_URL", "").strip().rstrip("/"),
        timeout_sec=timeout_sec,
        sample_top=_path_env("ISAACSIM_VLA_SAMPLE_TOP"),
        sample_wrist=_path_env("ISAACSIM_VLA_SAMPLE_WRIST"),
        live_top_image=_path_env("ROBOCLAW_MCP_VLA_TOP_IMAGE") or (repo_root / ".data" / "camera" / "latest_rgb.png"),
        live_wrist_image=_path_env("ROBOCLAW_MCP_VLA_WRIST_IMAGE") or (repo_root / ".data" / "camera" / "latest_ugreen_rgb.png"),
        camera_file_max_age=camera_file_max_age,
        camera_dark_frame_mean_threshold=camera_dark_frame_mean_threshold,
        repo_root=repo_root,
        arm_id=os.getenv("ROBOCLAW_ARM", "").strip(),
        arm_target=os.getenv("ROBOCLAW_ARM_TARGET", os.getenv("ROBOCLAW_ARM", "")).strip(),
        arm_label=os.getenv("ROBOCLAW_ARM_LABEL", "").strip(),
        arm_side=os.getenv("ROBOCLAW_ARM_SIDE", "").strip(),
        arm_camera_side=os.getenv("ROBOCLAW_ARM_CAMERA_SIDE", "").strip(),
    )
