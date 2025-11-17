"""MCP-related environment helpers.

Centralizes small pieces of environment logic so different layers
can share consistent behavior without duplicating code.
"""

from __future__ import annotations

import os


def get_idle_disconnect_ms() -> int:
    """Return the idle-disconnect interval in milliseconds.

    Logic:
    - If CHUNKHOUND_MCP__RW_IDLE_DISCONNECT_MS is set, use its integer value.
    - Otherwise, when CHUNKHOUND_MCP_MODE=1 (MCP stdio/HTTP), default to 1500ms
      to create RO windows for followers.
    - Else default to 0 (disabled).
    """
    try:
        raw = os.getenv("CHUNKHOUND_MCP__RW_IDLE_DISCONNECT_MS")
        if raw is None and os.getenv("CHUNKHOUND_MCP_MODE") == "1":
            return 1500
        return int(raw or 0)
    except Exception:
        return 1500 if os.getenv("CHUNKHOUND_MCP_MODE") == "1" else 0


def get_rw_gate_ms() -> tuple[int, int]:
    """Return writer duty-cycle gating (on_ms, off_ms).

    Logic:
    - If CHUNKHOUND_MCP__RW_GATE_ON_MS/OFF_MS are set, use them (ints; clamp >=0).
    - Otherwise, when CHUNKHOUND_MCP_MODE=1, default to (2000, 1000) to
      guarantee at least 1s off every ~3s under sustained write load.
    - Else return (0, 0) meaning disabled.
    """
    try:
        on_raw = os.getenv("CHUNKHOUND_MCP__RW_GATE_ON_MS")
        off_raw = os.getenv("CHUNKHOUND_MCP__RW_GATE_OFF_MS")
        if on_raw is not None or off_raw is not None:
            on_ms = int(on_raw or 0)
            off_ms = int(off_raw or 0)
            if on_ms < 0:
                on_ms = 0
            if off_ms < 0:
                off_ms = 0
            return on_ms, off_ms
    except Exception:
        pass
    if os.getenv("CHUNKHOUND_MCP_MODE") == "1":
        return 2000, 1000
    return 0, 0


def thin_rw_enabled() -> bool:
    """Return True when thin-RW mode is enabled.

    Thin-RW concentrates on minimizing the writer’s RW hold time by
    aggressively releasing connections and applying duty-cycle gating.

    Enabled when CHUNKHOUND_MCP__THIN_RW is truthy.
    """
    try:
        raw = os.getenv("CHUNKHOUND_MCP__THIN_RW", "")
        return str(raw).strip().lower() in ("1", "true", "yes", "on")
    except Exception:
        return False


def rw_disabled() -> bool:
    """Return True when all RW operations should be disabled in MCP.

    Controlled by CHUNKHOUND_MCP__DISABLE_RW. When enabled, providers
    should avoid opening RW connections and writers should not perform
    schema/index changes or store data.
    """
    try:
        raw = os.getenv("CHUNKHOUND_MCP__DISABLE_RW", "")
        return str(raw).strip().lower() in ("1", "true", "yes", "on")
    except Exception:
        return False


def indexing_writes_enabled() -> bool:
    """Return True if indexing writes are permitted.

    Global RW disable takes precedence. Otherwise, can be disabled with
    CHUNKHOUND_MCP__DISABLE_INDEXING_WRITES.
    """
    if rw_disabled():
        return False
    try:
        raw = os.getenv("CHUNKHOUND_MCP__DISABLE_INDEXING_WRITES", "")
        return not (str(raw).strip().lower() in ("1", "true", "yes", "on"))
    except Exception:
        return True
