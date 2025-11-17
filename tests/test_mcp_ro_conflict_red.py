"""
Red test capturing RO follower failure due to writer holding the DB.

Behavior: Start a writer MCP stdio server on a directory with enough files
to keep the initial scan busy. While the writer is scanning (and thus holding
the DuckDB connection), start a follower (RO) stdio server and attempt a
regex search. Expect the follower call to fail with a writer-active error
within its per-call RO backoff budget.

This is intentionally a RED test to reproduce the observed failure reliably
before implementing improvements to RW/RO gating.
"""

from __future__ import annotations

import asyncio
import json
import os
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
        {"protocolVersion": "2024-11-05", "capabilities": {}, "clientInfo": {"name": "ro-conflict", "version": "1.0"}},
        timeout=10.0,
    )
    await client.send_notification("notifications/initialized")
    return client


async def _provider_status(client: SubprocessJsonRpcClient) -> dict:
    return await client.send_request(
        "tools/call", {"name": "provider_status", "arguments": {}}, timeout=5.0
    )


async def _wait_writer_connected(client: SubprocessJsonRpcClient, timeout: float = 3.0) -> bool:
    """Poll provider_status until writer has an open DB connection or timeout."""
    import time

    t0 = time.time()
    while time.time() - t0 < timeout:
        st = await _provider_status(client)
        prov = st.get("provider", {})
        if prov.get("role") == "RW" and prov.get("db_connected") is True:
            return True
        await asyncio.sleep(0.1)
    return False


@pytest.mark.skipif(not _has_duckdb(), reason="duckdb not installed; skipping")
def test_ro_follower_conflicts_during_writer_initial_scan():
    with windows_safe_tempdir() as tmp:
        # Seed many files to keep writer busy during initial scan
        (tmp / ".chunkhound").mkdir(exist_ok=True)
        cfg = {
            "database": {"path": str(tmp / ".chunkhound" / "test.db"), "provider": "duckdb"},
            "indexing": {"include": ["**/*.py"]},
        }
        (tmp / ".chunkhound.json").write_text(json.dumps(cfg))

        # Generate a moderate number of files with some content
        src = tmp / "src"
        src.mkdir(exist_ok=True)
        for i in range(0, 200):
            (src / f"f_{i:04d}.py").write_text(
                "\n".join([f"def f_{i}_{j}(): return {i}+{j}" for j in range(200)])
            )

        env_rw = get_safe_subprocess_env(os.environ)
        env_rw["CHUNKHOUND_MCP_MODE"] = "1"
        # Disable idle-disconnect to maximize the hold window during scan
        env_rw["CHUNKHOUND_MCP__RW_IDLE_DISCONNECT_MS"] = "0"

        async def _run():
            c_rw = await _start_stdio(tmp, env_rw)
            try:
                # Wait until writer is actively connected (initial scan window)
                ok = await _wait_writer_connected(c_rw, timeout=5.0)
                assert ok, "Writer did not open DB connection during initial scan window"

                # Start follower (RO)
                env_ro = get_safe_subprocess_env(os.environ)
                env_ro["CHUNKHOUND_MCP_MODE"] = "1"
                c_ro = await _start_stdio(tmp, env_ro)
                try:
                    # Attempt a regex search; for true red (TDD), we expect it to succeed.
                    # Current behavior is known to fail under contention, so this assertion
                    # will make the test RED until fixes land.
                    res = await c_ro.send_request(
                        "tools/call",
                        {
                            "name": "search_regex",
                            "arguments": {"pattern": "def f_", "page_size": 5},
                        },
                        timeout=10.0,
                    )
                    # RED expectation: follower should succeed; will currently fail with writer-active
                    assert "error" not in res, f"Unexpected error from follower: {res}"
                    assert "results" in res, f"Expected results in response, got: {res}"
                finally:
                    await c_ro.close()
            finally:
                await c_rw.close()

        asyncio.run(_run())
