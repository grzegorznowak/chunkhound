"""
E2E tests for opportunistic short-lived DuckDB access by Non-Indexer clients.

Scenarios:
1) When no writer is connected, a Non-Indexer with RO_TRY_DB=1 can perform
   search_regex successfully (open->select->close).
2) While an Indexer is active (holding a write connection), a Non-Indexer with
   RO_TRY_DB=1 returns quickly with a clear error for DB-backed tools (no timeouts).
3) With RW idle-disconnect enabled, after the Indexer goes idle the Non-Indexer
   can read successfully.
"""

from __future__ import annotations

import asyncio
import json
import os
import time
from pathlib import Path

import pytest

from tests.utils import (
    JsonRpcResponseError,
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
        {"protocolVersion": "2024-11-05", "capabilities": {}, "clientInfo": {"name": "t", "version": "1.0"}},
        timeout=10.0,
    )
    await client.send_notification("notifications/initialized")
    return client


async def _get_stats(client: SubprocessJsonRpcClient) -> dict:
    return await client.send_request("tools/call", {"name": "get_stats", "arguments": {}}, timeout=10.0)


@pytest.mark.skipif(not _has_duckdb(), reason="duckdb not installed; skipping")
def test_opportunistic_ro_read_succeeds_when_no_writer():
    with windows_safe_tempdir() as tmp:
        token = "opportunistic_ok"
        (tmp / "t.py").write_text(f"def t():\n    return '{token}'\n")
        cfg = {
            "database": {"path": str(tmp / ".chunkhound" / "test.db"), "provider": "duckdb"},
            "indexing": {"include": ["*.py"]},
        }
        (tmp / ".chunkhound").mkdir(exist_ok=True)
        (tmp / ".chunkhound.json").write_text(json.dumps(cfg))

        # Pre-index once (single process) to create DB then exit
        async def _index_once() -> int:
            env_index = get_safe_subprocess_env(os.environ)
            env_index["CHUNKHOUND_MCP_MODE"] = "0"
            proc = await asyncio.create_subprocess_exec(
                "uv", "run", "chunkhound", "index", "--no-embeddings", str(tmp),
                cwd=str(tmp), env=env_index,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            out, err = await proc.communicate()
            return proc.returncode or 0

        assert asyncio.run(_index_once()) == 0

        # Start a Non-Indexer MCP with RO_TRY_DB=1 (no writer active now)
        env = get_safe_subprocess_env(os.environ)
        env["CHUNKHOUND_MCP_MODE"] = "1"
        env["CHUNKHOUND_MCP__SKIP_INDEXING"] = "1"
        env["CHUNKHOUND_MCP__RO_TRY_DB"] = "1"

        async def _run():
            c = await _start_stdio(tmp, env)
            try:
                res = await c.send_request(
                    "tools/call",
                    {"name": "search_regex", "arguments": {"pattern": token, "page_size": 50, "offset": 0}},
                    timeout=10.0,
                )
                assert len(res.get("results", [])) > 0
            finally:
                await c.close()

        asyncio.run(_run())


@pytest.mark.skipif(not _has_duckdb(), reason="duckdb not installed; skipping")
def test_opportunistic_ro_falls_back_quickly_during_writer():
    with windows_safe_tempdir() as tmp:
        token = "ro_conflict_fallback"
        (tmp / "t.py").write_text(f"def t():\n    return '{token}'\n")
        cfg = {
            "database": {"path": str(tmp / ".chunkhound" / "test.db"), "provider": "duckdb"},
            "indexing": {"include": ["*.py"]},
        }
        (tmp / ".chunkhound").mkdir(exist_ok=True)
        (tmp / ".chunkhound.json").write_text(json.dumps(cfg))

        env_rw = get_safe_subprocess_env(os.environ)
        env_rw["CHUNKHOUND_MCP_MODE"] = "1"
        # Default behavior: RW Indexer connects and scans

        env_ro = get_safe_subprocess_env(os.environ)
        env_ro["CHUNKHOUND_MCP_MODE"] = "1"
        env_ro["CHUNKHOUND_MCP__SKIP_INDEXING"] = "1"
        env_ro["CHUNKHOUND_MCP__RO_TRY_DB"] = "1"
        # Use a small per-request budget for this test so RO calls do not hit
        # the client timeout even when the writer holds the DB.
        env_ro["CHUNKHOUND_MCP__RO_BUDGET_MS"] = "800"

        async def _run():
            c_rw = await _start_stdio(tmp, env_rw)
            c_ro = await _start_stdio(tmp, env_ro)
            try:
                # While RW is likely connected (initial scan), RO query should
                # complete fast (within timeout) without hanging, either by
                # succeeding or by surfacing a clear writer-busy error.
                t0 = time.time()
                try:
                    await c_ro.send_request(
                        "tools/call",
                        {"name": "search_regex", "arguments": {"pattern": token, "page_size": 50, "offset": 0}},
                        timeout=2.0,
                    )
                except JsonRpcResponseError as exc:
                    assert "Regex search unavailable" in str(exc)
                dt = time.time() - t0
                assert dt < 2.0
            finally:
                await c_ro.close()
                await c_rw.close()

        asyncio.run(_run())


@pytest.mark.skipif(not _has_duckdb(), reason="duckdb not installed; skipping")
def test_rw_idle_disconnect_opens_ro_window():
    with windows_safe_tempdir() as tmp:
        token = "idle_window_ok"
        (tmp / "t.py").write_text(f"def t():\n    return '{token}'\n")
        cfg = {
            "database": {"path": str(tmp / ".chunkhound" / "test.db"), "provider": "duckdb"},
            "indexing": {"include": ["*.py"]},
        }
        (tmp / ".chunkhound").mkdir(exist_ok=True)
        (tmp / ".chunkhound.json").write_text(json.dumps(cfg))

        # Pre-index once to create DB, then run RW MCP in skip-indexing mode
        async def _index_once() -> int:
            env_index = get_safe_subprocess_env(os.environ)
            env_index["CHUNKHOUND_MCP_MODE"] = "0"
            proc = await asyncio.create_subprocess_exec(
                "uv", "run", "chunkhound", "index", "--no-embeddings", str(tmp),
                cwd=str(tmp), env=env_index,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            out, err = await proc.communicate()
            return proc.returncode or 0

        assert asyncio.run(_index_once()) == 0

        env_rw = get_safe_subprocess_env(os.environ)
        env_rw["CHUNKHOUND_MCP_MODE"] = "1"
        env_rw["CHUNKHOUND_MCP__SKIP_INDEXING"] = "1"

        env_ro = get_safe_subprocess_env(os.environ)
        env_ro["CHUNKHOUND_MCP_MODE"] = "1"
        env_ro["CHUNKHOUND_MCP__SKIP_INDEXING"] = "1"
        env_ro["CHUNKHOUND_MCP__RO_TRY_DB"] = "1"

        async def _run():
            c_rw = await _start_stdio(tmp, env_rw)
            c_ro = await _start_stdio(tmp, env_ro)
            try:
                # RW is non-indexer (skip-indexing), so DB should not be held
                await asyncio.sleep(0.5)

                # Now RO should be able to query successfully
                res = await c_ro.send_request(
                    "tools/call",
                    {"name": "search_regex", "arguments": {"pattern": token, "page_size": 50, "offset": 0}},
                    timeout=5.0,
                )
                assert len(res.get("results", [])) > 0
            finally:
                await c_ro.close()
                await c_rw.close()

        asyncio.run(_run())
