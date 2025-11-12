"""
E2E test: one RW MCP + multiple RO MCP readers using read-only DuckDB connections.

Requires CHUNKHOUND_MCP__ALLOW_RO_DB=1 for the RO processes.
"""

from __future__ import annotations

import asyncio
import json
import os
import time
from pathlib import Path

import pytest

from tests.utils import SubprocessJsonRpcClient, create_subprocess_exec_safe, get_safe_subprocess_env
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


@pytest.mark.skipif(not _has_duckdb() or os.getenv("CH_RUN_RO_DB_E2E") not in ("1","true","yes","on"),
                    reason="duckdb not installed or CH_RUN_RO_DB_E2E not set; skipping")
def test_mcp_multiple_ro_readers_can_query_db():
    with windows_safe_tempdir() as tmp:
        # Seed files
        (tmp / "a.py").write_text("def a(): return 1\n")
        (tmp / "b.py").write_text("def b(): return 2\n")

        # Config
        cfg = {
            "database": {"path": str(tmp / ".chunkhound" / "test.db"), "provider": "duckdb"},
            "indexing": {"include": ["*.py"]},
        }
        (tmp / ".chunkhound").mkdir(exist_ok=True)
        (tmp / ".chunkhound.json").write_text(json.dumps(cfg))

        base_env = get_safe_subprocess_env(os.environ)
        base_env["CHUNKHOUND_MCP_MODE"] = "1"

        async def _run():
            # Leader (RW)
            c_rw = await _start_stdio(tmp, base_env)
            # Two RO readers with RO DB allowed
            ro_env = dict(base_env)
            ro_env["CHUNKHOUND_MCP__ALLOW_RO_DB"] = "1"
            ro_env["CHUNKHOUND_MCP__SKIP_INDEXING"] = "1"
            c_ro1 = await _start_stdio(tmp, ro_env)
            c_ro2 = await _start_stdio(tmp, ro_env)
            try:
                # RW stats should be non-zero after initial scan completes
                t0 = time.time()
                while time.time() - t0 < 10.0:
                    s = await _get_stats(c_rw)
                    if s.get("total_files", 0) >= 2:
                        break
                    await asyncio.sleep(0.2)

                # RO readers should be able to read non-zero stats
                s1 = await _get_stats(c_ro1)
                s2 = await _get_stats(c_ro2)
                assert s1.get("total_files", 0) >= 2
                assert s2.get("total_files", 0) >= 2

                # Create a new file; RW watcher will index it
                token = "multi_ro_token"
                (tmp / "c.py").write_text(f"def c():\n    return '{token}'\n")

                # Allow some time for RW to index and checkpoint
                await asyncio.sleep(2.0)

                # RO readers should find the token via regex
                for client in (c_ro1, c_ro2):
                    res = await client.send_request(
                        "tools/call",
                        {"name": "search_regex", "arguments": {"pattern": token, "page_size": 10, "offset": 0}},
                        timeout=10.0,
                    )
                    assert len(res.get("results", [])) > 0
            finally:
                await c_ro2.close()
                await c_ro1.close()
                await c_rw.close()

        asyncio.run(_run())
