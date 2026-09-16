"""Persistent learned capabilities for LLM request features."""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Literal

_CACHE_ENV = "CHUNKHOUND_LLM_CAPABILITY_CACHE"
_CACHE_FILENAME = "llm-capabilities.json"
_FORMAT_VERSION = 1
_TTL_SECONDS = 30 * 24 * 60 * 60
CapabilityState = Literal["unknown", "accepted", "rejected"]


def _default_cache_path() -> Path:
    """Return the user-scoped capability cache file path."""
    override = os.environ.get(_CACHE_ENV)
    if override:
        return Path(override).expanduser()

    if sys.platform == "win32":
        local_appdata = os.environ.get("LOCALAPPDATA")
        cache_dir = (
            Path(local_appdata) / "ChunkHound"
            if local_appdata
            else Path.home() / "AppData" / "Local" / "ChunkHound"
        )
    else:
        cache_root = os.environ.get("XDG_CACHE_HOME")
        cache_dir = (
            Path(cache_root) / "chunkhound"
            if cache_root
            else Path.home() / ".cache" / "chunkhound"
        )
    return cache_dir / _CACHE_FILENAME


class LLMCapabilityStore:
    """Store accepted or rejected provider/model capabilities for 30 days."""

    def __init__(self) -> None:
        self._path = _default_cache_path()

    def get(self, provider: str, model: str) -> CapabilityState:
        entry = self._read().get(f"{provider}:{model}")
        if not isinstance(entry, dict):
            return "unknown"

        accepted = entry.get("accepted")
        timestamp = entry.get("ts")
        if (
            entry.get("format_version") != _FORMAT_VERSION
            or not isinstance(accepted, bool)
            or isinstance(timestamp, bool)
            or not isinstance(timestamp, (int, float))
            or time.time() - timestamp >= _TTL_SECONDS
        ):
            return "unknown"
        return "accepted" if accepted else "rejected"

    def set(self, provider: str, model: str, state: CapabilityState) -> None:
        if state not in {"accepted", "rejected"}:
            raise ValueError("Capability state must be 'accepted' or 'rejected'")

        payload = self._read()
        payload[f"{provider}:{model}"] = {
            "accepted": state == "accepted",
            "ts": time.time(),
            "format_version": _FORMAT_VERSION,
        }
        self._path.parent.mkdir(parents=True, exist_ok=True)
        temporary_path = self._path.with_name(
            f"{self._path.name}.{os.getpid()}.{time.time_ns()}.tmp"
        )
        temporary_path.write_text(json.dumps(payload) + "\n", encoding="utf-8")
        os.replace(temporary_path, self._path)

    def _read(self) -> dict[str, Any]:
        try:
            data = json.loads(self._path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return {}
        return data if isinstance(data, dict) else {}
