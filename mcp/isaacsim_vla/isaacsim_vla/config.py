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
    repo_root: Path


def _path_env(name: str) -> Path | None:
    value = os.getenv(name, "").strip()
    if not value:
        return None
    return Path(value).expanduser().resolve()


def load_config() -> ServerConfig:
    repo_root = Path(os.getenv("ISAACSIM_REPO_ROOT", "/work/IsaacSim")).expanduser().resolve()
    workdir = _path_env("ISAACSIM_VLA_WORKDIR") or (repo_root / "outputs" / "isaacsim_vla_mcp")
    backend = os.getenv("ISAACSIM_VLA_BACKEND", "mock").strip().lower() or "mock"
    timeout_raw = os.getenv("ISAACSIM_VLA_TIMEOUT_SEC", "120").strip()
    try:
        timeout_sec = float(timeout_raw)
    except ValueError:
        timeout_sec = 120.0
    return ServerConfig(
        backend=backend,
        workdir=workdir,
        bridge_dir=_path_env("ISAACSIM_VLA_BRIDGE_DIR"),
        http_url=os.getenv("ISAACSIM_VLA_HTTP_URL", "").strip().rstrip("/"),
        timeout_sec=timeout_sec,
        sample_top=_path_env("ISAACSIM_VLA_SAMPLE_TOP"),
        sample_wrist=_path_env("ISAACSIM_VLA_SAMPLE_WRIST"),
        repo_root=repo_root,
    )
