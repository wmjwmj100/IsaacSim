from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Protocol

import requests
from PIL import Image

from .config import ServerConfig
from .models import BoxOverlay, ExecutionResult
from .store import ArtifactStore, now_id


class VLABackend(Protocol):
    name: str

    def look_camera(self, camera_id: str) -> dict[str, Any]:
        ...

    def vla_execute(self, instruction: str, overlay: BoxOverlay, atomic_action: str | None) -> dict[str, Any]:
        ...


class MockBackend:
    name = "mock"

    def __init__(self, cfg: ServerConfig, store: ArtifactStore):
        self.cfg = cfg
        self.store = store

    def look_camera(self, camera_id: str) -> dict[str, Any]:
        sample = self._sample_for_camera(camera_id)
        if sample and sample.is_file():
            return {"image_path": str(sample)}
        generated = self._generate_sample(camera_id)
        return {"image_path": str(generated)}

    def vla_execute(self, instruction: str, overlay: BoxOverlay, atomic_action: str | None) -> dict[str, Any]:
        return ExecutionResult(
            success=True,
            instruction=instruction,
            box_overlay_id=overlay.box_overlay_id,
            box_layer_id=overlay.box_layer_id,
            atomic_action=atomic_action,
            backend=self.name,
            result_id=now_id("exec"),
            metadata={
                "mode": "dry_run",
                "overlay_path": overlay.overlay_path,
                "red_box": overlay.red_box,
                "green_box": overlay.green_box,
            },
        ).model_dump()

    def _sample_for_camera(self, camera_id: str) -> Path | None:
        camera_key = camera_id.lower()
        if camera_key in {"top", "overhead"}:
            return self.cfg.sample_top
        if "wrist" in camera_key:
            return self.cfg.sample_wrist
        return self.cfg.sample_top or self.cfg.sample_wrist

    def _generate_sample(self, camera_id: str) -> Path:
        path = self.store.images_dir / f"{now_id('mock')}_{camera_id}.png"
        width, height = 640, 480
        img = Image.new("RGB", (width, height), (232, 235, 238))
        pixels = img.load()
        for y in range(height):
            for x in range(width):
                if (x // 40 + y // 40) % 2 == 0:
                    pixels[x, y] = (222, 226, 230)
        img.save(path)
        return path


class FileBridgeBackend:
    name = "file"

    def __init__(self, cfg: ServerConfig, store: ArtifactStore):
        if cfg.bridge_dir is None:
            raise RuntimeError("ISAACSIM_VLA_BRIDGE_DIR is required for file backend")
        self.cfg = cfg
        self.store = store
        self.requests_dir = cfg.bridge_dir / "requests"
        self.responses_dir = cfg.bridge_dir / "responses"
        self.requests_dir.mkdir(parents=True, exist_ok=True)
        self.responses_dir.mkdir(parents=True, exist_ok=True)

    def look_camera(self, camera_id: str) -> dict[str, Any]:
        return self._roundtrip("look_camera", {"camera_id": camera_id})

    def vla_execute(self, instruction: str, overlay: BoxOverlay, atomic_action: str | None) -> dict[str, Any]:
        return self._roundtrip(
            "vla_execute",
            {"instruction": instruction, "atomic_action": atomic_action, "overlay": overlay.model_dump()},
        )

    def _roundtrip(self, operation: str, payload: dict[str, Any]) -> dict[str, Any]:
        request_id = now_id("req")
        request_path = self.requests_dir / f"{request_id}.json"
        response_path = self.responses_dir / f"{request_id}.json"
        request = {
            "request_id": request_id,
            "operation": operation,
            "payload": payload,
            "created_at": time.time(),
        }
        _write_json_atomic(request_path, request)
        deadline = time.monotonic() + self.cfg.timeout_sec
        while time.monotonic() < deadline:
            if response_path.is_file():
                try:
                    response = json.loads(response_path.read_text(encoding="utf-8"))
                except json.JSONDecodeError:
                    time.sleep(0.05)
                    continue
                if response.get("success") is False:
                    raise RuntimeError(str(response.get("error") or "file bridge backend returned failure"))
                result = response.get("result")
                if not isinstance(result, dict):
                    raise RuntimeError(f"file bridge response missing result object: {response_path}")
                return result
            time.sleep(0.05)
        raise TimeoutError(f"timed out waiting for {response_path}")


class HttpBackend:
    name = "http"

    def __init__(self, cfg: ServerConfig, store: ArtifactStore):
        if not cfg.http_url:
            raise RuntimeError("ISAACSIM_VLA_HTTP_URL is required for http backend")
        self.cfg = cfg
        self.store = store

    def look_camera(self, camera_id: str) -> dict[str, Any]:
        return self._post("look_camera", {"camera_id": camera_id})

    def vla_execute(self, instruction: str, overlay: BoxOverlay, atomic_action: str | None) -> dict[str, Any]:
        return self._post("vla_execute", {"instruction": instruction, "atomic_action": atomic_action, "overlay": overlay.model_dump()})

    def _post(self, operation: str, payload: dict[str, Any]) -> dict[str, Any]:
        response = requests.post(
            f"{self.cfg.http_url}/{operation}",
            json=payload,
            timeout=self.cfg.timeout_sec,
        )
        response.raise_for_status()
        data = response.json()
        if isinstance(data, dict) and "success" in data:
            if data.get("success") is False:
                raise RuntimeError(str(data.get("error") or f"http backend {operation} failed"))
            result = data.get("result")
            if isinstance(result, dict):
                return result
        if not isinstance(data, dict):
            raise RuntimeError(f"http backend returned non-object JSON for {operation}")
        return data


def build_backend(cfg: ServerConfig, store: ArtifactStore) -> VLABackend:
    if cfg.backend == "mock":
        return MockBackend(cfg, store)
    if cfg.backend == "file":
        return FileBridgeBackend(cfg, store)
    if cfg.backend == "http":
        return HttpBackend(cfg, store)
    raise RuntimeError(f"unsupported ISAACSIM_VLA_BACKEND={cfg.backend!r}")


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    tmp_path = path.with_name(f".{path.name}.{now_id('tmp')}.tmp")
    tmp_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp_path.replace(path)
