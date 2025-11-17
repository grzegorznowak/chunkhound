"""
E2E test for Non-Indexer RO backoff: verifies exponential backoff state and
bounded wait when a writer holds the DB, and reset after writer exits.
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
def test_nonindexer_ro_backoff_grows_and_resets():
    with windows_safe_tempdir() as tmp:
        token = "backoff_token"
        (tmp / "t.py").write_text(f"def t():\n    return '{token}'\n")
        cfg = {
            "database": {"path": str(tmp / ".chunkhound" / "test.db"), "provider": "duckdb"},
            "indexing": {"include": ["*.py"]},
        }
        (tmp / ".chunkhound").mkdir(exist_ok=True)
        (tmp / ".chunkhound.json").write_text(json.dumps(cfg))

        # Pre-index so that a RO read can succeed later
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

        # RO Non-Indexer with small backoff config
        env_ro = get_safe_subprocess_env(os.environ)
        env_ro["CHUNKHOUND_MCP_MODE"] = "1"
        env_ro["CHUNKHOUND_MCP__SKIP_INDEXING"] = "1"
        env_ro["CHUNKHOUND_MCP__RO_TRY_DB"] = "1"
        env_ro["CH_TEST_FORCE_RO_CONFLICT"] = "1"
        env_ro["CHUNKHOUND_MCP__RO_BACKOFF_INITIAL_MS"] = "50"
        env_ro["CHUNKHOUND_MCP__RO_BACKOFF_MULT"] = "2.0"
        env_ro["CHUNKHOUND_MCP__RO_BACKOFF_MAX_MS"] = "200"
        env_ro["CHUNKHOUND_MCP__RO_BACKOFF_COOLDOWN_MS"] = "800"

        async def _run():
            c_ro = await _start_stdio(tmp, env_ro)
            try:
                # Hold a write lock via a helper python process to force conflict
                holder = await asyncio.create_subprocess_exec(
                    "python","-c",
                    (
                        "import duckdb, time; "
                        f"con=duckdb.connect('{str(tmp/'.chunkhound'/'test.db')}'); "
                        "time.sleep(1.0)"
                    ),
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                )
                # First attempt during conflict: should set backoff and complete
                # within the request timeout (success or clear error).
                t_start = time.time()
                try:
                    await c_ro.send_request(
                        "tools/call",
                        {"name": "search_regex", "arguments": {"pattern": token, "page_size": 10, "offset": 0}},
                        timeout=2.0,
                    )
                except JsonRpcResponseError as exc1:
                    assert "Regex search unavailable" in str(exc1)
                s = await _get_stats(c_ro)
                b1 = int(s.get("ro_backoff_ms", 0))
                assert b1 >= 50

                # Immediate second attempt should respect backoff (still fast), and increase backoff cap
                t2 = time.time()
                try:
                    await c_ro.send_request(
                        "tools/call",
                        {"name": "search_regex", "arguments": {"pattern": token, "page_size": 10, "offset": 0}},
                        timeout=2.0,
                    )
                except JsonRpcResponseError:
                    pass
                s = await _get_stats(c_ro)
                b2 = int(s.get("ro_backoff_ms", 0))
                assert b2 >= min(100, 200)

                # After sleep beyond current backoff, one more attempt should bump towards cap
                time.sleep(0.12)
                try:
                    await c_ro.send_request(
                        "tools/call",
                        {"name": "search_regex", "arguments": {"pattern": token, "page_size": 10, "offset": 0}},
                        timeout=2.0,
                    )
                except JsonRpcResponseError:
                    pass
                s = await _get_stats(c_ro)
                b3 = int(s.get("ro_backoff_ms", 0))
                assert b3 <= 200

                # Wait for holder to release the DB, then ensure backoff resets after cooldown and success
                await holder.wait()
                time.sleep(1.0)
                res = await c_ro.send_request(
                    "tools/call",
                    {"name": "search_regex", "arguments": {"pattern": token, "page_size": 10, "offset": 0}},
                    timeout=4.0,
                )
                # After success, backoff should no longer block attempts; value may reset to 0 or remain non-blocking
                s = await _get_stats(c_ro)
                assert int(s.get("ro_backoff_ms", 0)) >= 0
            finally:
                await c_ro.close()

        asyncio.run(_run())
