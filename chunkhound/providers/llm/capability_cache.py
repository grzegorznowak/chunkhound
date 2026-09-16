"""Persistent learned capabilities for LLM request features."""

from __future__ import annotations

import io
import json
import logging
import os
import sys
import time
from collections.abc import Generator
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Literal

if os.name == "nt":
    import msvcrt
else:
    import fcntl

_CACHE_ENV = "CHUNKHOUND_LLM_CAPABILITY_CACHE"
_CACHE_FILENAME = "llm-capabilities.json"
_FORMAT_VERSION = 1
_TTL_SECONDS = 30 * 24 * 60 * 60
CapabilityState = Literal["unknown", "accepted", "rejected"]


def _acquire_cache_lock(handle: io.BufferedRandom) -> None:
    handle.seek(0, os.SEEK_END)
    if handle.tell() == 0:
        handle.write(b"0")
        handle.flush()
    handle.seek(0)
    if os.name == "nt":
        msvcrt.locking(  # type: ignore[attr-defined]
            handle.fileno(),
            msvcrt.LK_LOCK,  # type: ignore[attr-defined]
            1,
        )
        return
    fcntl.flock(handle.fileno(), fcntl.LOCK_EX)


def _release_cache_lock(handle: io.BufferedRandom) -> None:
    handle.seek(0)
    if os.name == "nt":
        msvcrt.locking(  # type: ignore[attr-defined]
            handle.fileno(),
            msvcrt.LK_UNLCK,  # type: ignore[attr-defined]
            1,
        )
        return
    fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


@contextmanager
def _locked_cache_file(cache_path: Path) -> Generator[None, None, None]:
    lock_path = cache_path.with_name(f"{cache_path.name}.lock")
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("a+b") as handle:
        _acquire_cache_lock(handle)
        try:
            yield
        finally:
            _release_cache_lock(handle)


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
        format_version = entry.get("format_version")
        if (
            isinstance(format_version, bool)
            or not isinstance(format_version, int)
            or format_version != _FORMAT_VERSION
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

        try:
            with _locked_cache_file(self._path):
                payload = self._read()
                payload[f"{provider}:{model}"] = {
                    "accepted": state == "accepted",
                    "ts": time.time(),
                    "format_version": _FORMAT_VERSION,
                }
                temporary_path = self._path.with_name(
                    f"{self._path.name}.{os.getpid()}.{time.time_ns()}.tmp"
                )
                try:
                    temporary_path.write_text(
                        json.dumps(payload) + "\n", encoding="utf-8"
                    )
                    os.replace(temporary_path, self._path)
                finally:
                    temporary_path.unlink(missing_ok=True)
        except OSError:
            logging.warning(
                "Failed to persist LLM capability cache at %s",
                self._path,
                exc_info=True,
            )

    def _read(self) -> dict[str, Any]:
        try:
            data = json.loads(self._path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return {}
        return data if isinstance(data, dict) else {}
