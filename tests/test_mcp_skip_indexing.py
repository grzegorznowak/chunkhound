"""
Tests for MCP read-only startup that skips indexing the worktree.

All code and comments in English.

These tests are red-first for the skip-indexing feature: they expect the
server to honor an environment flag/CLI flag that bypasses RealtimeIndexing
and the initial directory scan, while still serving tools.
"""

from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path

import pytest

from tests.utils import SubprocessJsonRpcClient, create_subprocess_exec_safe, get_safe_subprocess_env
from tests.utils.windows_compat import windows_safe_tempdir


def test_mcp_stdio_skips_indexing_when_flag_set():
    try:
        import duckdb  # noqa: F401
    except Exception:
        pytest.skip("duckdb not installed; skipping stdio skip-indexing test")
    """MCP stdio should start without indexing when skip flag is set."""
    with windows_safe_tempdir() as temp_path:
        # Minimal config
        config_path = temp_path / ".chunkhound.json"
        db_path = temp_path / ".chunkhound" / "test.db"
        db_path.parent.mkdir(exist_ok=True)
        config = {
            "database": {"path": str(db_path), "provider": "duckdb"},
            "indexing": {"include": ["*.py"]},
        }
        config_path.write_text(json.dumps(config))

        # Env for MCP + skip indexing
        env = get_safe_subprocess_env(os.environ)
        env["CHUNKHOUND_MCP_MODE"] = "1"
        env["CHUNKHOUND_MCP__SKIP_INDEXING"] = "1"

        # Start stdio server in the temp dir
        async def _run():
            proc = await create_subprocess_exec_safe(
                "uv",
                "run",
                "chunkhound",
                "mcp",
                str(temp_path),
                cwd=str(temp_path),
                env=env,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )

            client = SubprocessJsonRpcClient(proc)
            await client.start()

            try:
                # Initialize handshake
                init_result = await client.send_request(
                    "initialize",
                    {
                        "protocolVersion": "2024-11-05",
                        "capabilities": {},
                        "clientInfo": {"name": "test", "version": "1.0"},
                    },
                    timeout=10.0,
                )
                assert "serverInfo" in init_result

                await client.send_notification("notifications/initialized")

                # Call get_stats and assert no scan is running
                stats = await client.send_request(
                    "tools/call",
                    {"name": "get_stats", "arguments": {}},
                    timeout=10.0,
                )

                # initial_scan should either be absent or explicitly indicate no scanning
                scan = stats.get("initial_scan")
                if scan is not None:
                    assert scan.get("is_scanning") is False
                    # When skipping indexing, we expect no timestamps yet
                    assert scan.get("started_at") in (None, "", 0)
                    assert scan.get("completed_at") in (None, "", 0)

            finally:
                await client.close()

        asyncio.run(_run())


def test_mcp_http_skips_indexing_when_flag_set():
    """MCP HTTP should start and report no scanning when skip flag is set.

    This test is lenient: if FastMCP is unavailable, skip automatically.
    """
    try:
        import fastmcp  # noqa: F401
    except Exception:
        pytest.skip("FastMCP not installed; skipping HTTP skip-indexing test")

    with windows_safe_tempdir() as temp_path:
        # Minimal config
        config_path = temp_path / ".chunkhound.json"
        db_path = temp_path / ".chunkhound" / "test.db"
        db_path.parent.mkdir(exist_ok=True)
        config = {
            "database": {"path": str(db_path), "provider": "duckdb"},
            "indexing": {"include": ["*.py"]},
        }
        config_path.write_text(json.dumps(config))

        env = get_safe_subprocess_env(os.environ)
        env["CHUNKHOUND_MCP_MODE"] = "1"
        env["CHUNKHOUND_MCP__SKIP_INDEXING"] = "1"

        # Start HTTP server
        async def _run_http():
            # Use flag form (--http) rather than treating 'http' as a path argument
            proc = await create_subprocess_exec_safe(
                "uv",
                "run",
                "chunkhound",
                "mcp",
                "--http",
                "--port",
                "5199",
                cwd=str(temp_path),
                env=env,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )

            try:
                # Give the server a moment to boot
                await asyncio.sleep(1.0)

                # We cannot call tools directly here without the HTTP client, but we can
                # assert the process is alive and did not crash due to indexing.
                assert proc.returncode is None
            finally:
                proc.terminate()
                try:
                    await asyncio.wait_for(proc.wait(), timeout=3.0)
                except asyncio.TimeoutError:
                    proc.kill()
                    await proc.wait()

        asyncio.run(_run_http())
