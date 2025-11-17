"""
E2E test for RW connection duty cycle and thin connections in MCP mode.

Guarantee: In MCP mode with the realtime watcher running, the writer opens
short RO windows automatically without flags. Specifically, within any
~3s interval there should be at least ~1s where the underlying DuckDB
connection is closed (db_connected=False).
"""

from __future__ import annotations

import asyncio
import json
import os
import time
from pathlib import Path

import pytest

from tests.utils import (
    SubprocessJsonRpcClient,
    create_subprocess_exec_safe,
    get_safe_subprocess_env,
)
from tests.utils.windows_compat import windows_safe_tempdir


def _has_duckdb() -> bool:
    try:
        import duckdb  # noqa: F401
        return True
    except Exception:
        return False


async def _start_stdio(tmp: Path, env: dict[str, str]) -> SubprocessJsonRpcClient:
    proc = await create_subprocess_exec_safe(
        "uv",
        "run",
        "chunkhound",
        "mcp",
        str(tmp),
        cwd=str(tmp),
        env=env,
        stdin=asyncio.subprocess.PIPE,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    client = SubprocessJsonRpcClient(proc)
    await client.start()
    await client.send_request(
        "initialize",
        {"protocolVersion": "2024-11-05", "capabilities": {}, "clientInfo": {"name": "rw-gate", "version": "1.0"}},
        timeout=10.0,
    )
    await client.send_notification("notifications/initialized")
    return client


async def _provider_status(client: SubprocessJsonRpcClient) -> dict:
    return await client.send_request(
        "tools/call", {"name": "provider_status", "arguments": {}}, timeout=5.0
    )


@pytest.mark.skipif(not _has_duckdb(), reason="duckdb not installed; skipping")
def test_rw_connection_has_regular_off_windows_in_mcp():
    with windows_safe_tempdir() as tmp:
        # Seed content so initial scan has work, then watcher idles
        (tmp / "t.py").write_text("def t():\n    return 1\n")
        cfg = {
            "database": {"path": str(tmp / ".chunkhound" / "test.db"), "provider": "duckdb"},
            "indexing": {"include": ["*.py"]},
        }
        (tmp / ".chunkhound").mkdir(exist_ok=True)
        (tmp / ".chunkhound.json").write_text(json.dumps(cfg))

        env_rw = get_safe_subprocess_env(os.environ)
        env_rw["CHUNKHOUND_MCP_MODE"] = "1"
        # Do NOT set skip-indexing; watcher must run to exercise RW behavior

        async def _run():
            c_rw = await _start_stdio(tmp, env_rw)
            try:
                # Wait briefly for initialization + any initial scan to finish
                await asyncio.sleep(1.2)

                # Sample provider_status for ~3.5s at 0.1s cadence and calculate off time
                off_time = 0.0
                cadence = 0.1
                samples = int(3.5 / cadence)
                for _ in range(samples):
                    st = await _provider_status(c_rw)
                    prov = st.get("provider", {})
                    if not prov.get("db_connected", False):
                        off_time += cadence
                    await asyncio.sleep(cadence)

                assert (
                    off_time >= 1.0
                ), f"Expected at least 1.0s off in ~3.5s window; saw {off_time:.2f}s"
            finally:
                await c_rw.close()

        asyncio.run(_run())

