from __future__ import annotations

import json
import os
import time
import traceback
from pathlib import Path
from typing import Any


def append_event(log_path: str | Path | None, event: str, **fields: Any) -> None:
    """Append one structured audit event as JSONL.

    Audit logging must never break robot control or MCP responses, so all file
    errors are intentionally swallowed after best-effort serialization.
    """
    if not log_path:
        return
    try:
        path = Path(log_path).expanduser().resolve()
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "ts": time.time(),
            "ts_iso": time.strftime("%Y-%m-%dT%H:%M:%S%z", time.localtime()),
            "pid": os.getpid(),
            "event": event,
            **_jsonable(fields),
        }
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n")
    except Exception:
        return


def exception_fields(exc: BaseException) -> dict[str, Any]:
    return {
        "error_type": type(exc).__name__,
        "error": str(exc),
        "traceback_tail": traceback.format_exc(limit=6),
    }


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)
