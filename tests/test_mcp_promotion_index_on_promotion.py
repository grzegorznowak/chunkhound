"""
E2E promotion tests for multi-MCP behavior around INDEX_ON_PROMOTION.

Default behavior: CHUNKHOUND_MCP__INDEX_ON_PROMOTION is OFF.
 - When RO promotes to RW, it should NOT start realtime watcher nor run catch-up scan.
 - A new file created after promotion should NOT appear in searches.

With env override ON: CHUNKHOUND_MCP__INDEX_ON_PROMOTION=1
 - On promotion, start ONLY the realtime watcher (no catch-up scan).
 - A new file created after promotion should be indexed by watcher and appear in regex search.

All code and comments in English.
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


def _mk_config(tmp: Path) -> None:
    config_path = tmp / ".chunkhound.json"
    db_path = tmp / ".chunkhound" / "test.db"
    db_path.parent.mkdir(exist_ok=True)
    config = {
        "database": {"path": str(db_path), "provider": "duckdb"},
        "indexing": {"include": ["*.py"]},
    }
    config_path.write_text(json.dumps(config))


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
    # initialize handshake
    await client.send_request(
        "initialize",
        {"protocolVersion": "2024-11-05", "capabilities": {}, "clientInfo": {"name": "t", "version": "1.0"}},
        timeout=10.0,
    )
    await client.send_notification("notifications/initialized")
    return client


async def _get_stats(client: SubprocessJsonRpcClient) -> dict:
    return await client.send_request("tools/call", {"name": "get_stats", "arguments": {}}, timeout=10.0)


def test_promotion_default_off_no_watcher():
    if not _has_duckdb():
        pytest.skip("duckdb not installed; skipping")

    with windows_safe_tempdir() as tmp:
        _mk_config(tmp)
        env = get_safe_subprocess_env(os.environ)
        env["CHUNKHOUND_MCP_MODE"] = "1"
        # Default: INDEX_ON_PROMOTION off (do not set env)

        async def _run():
            c1 = await _start_stdio(tmp, env)  # expected RW
            c2 = await _start_stdio(tmp, env)  # expected RO
            try:
                s1 = await _get_stats(c1)
                s2 = await _get_stats(c2)
                # If roles exposed, check RW vs RO
                r1, r2 = s1.get("role"), s2.get("role")
                if r1 is not None and r2 is not None:
                    assert r1 == "RW"
                    assert r2 == "RO"

                # Terminate RW to force promotion
                await c1.close()
                # Wait for promotion
                t0 = time.time()
                new_role = None
                while time.time() - t0 < 5.0:
                    s2 = await _get_stats(c2)
                    new_role = s2.get("role")
                    if new_role == "RW":
                        break
                    await asyncio.sleep(0.2)
                # Either role exposed and is RW, or not exposed on this backend
                if new_role is not None:
                    assert new_role == "RW"

                # Create a new file after promotion
                new_file = tmp / "new_after_promotion.py"
                token = "unique_token_after_promotion_default_off"
                new_file.write_text(f"def xxx():\n    return '{token}'\n")

                # Wait a bit to allow a watcher (if erroneously started) to act
                await asyncio.sleep(2.0)

                # Search regex for the token; should NOT be found (no watcher by default)
                res = await c2.send_request(
                    "tools/call",
                    {"name": "search_regex", "arguments": {"pattern": token, "page_size": 50, "offset": 0}},
                    timeout=10.0,
                )
                assert len(res.get("results", [])) == 0, "Promotion default OFF should not start watcher"
                scan = s2.get("initial_scan")
                if scan is not None:
                    assert scan.get("is_scanning") is False

            finally:
                await c2.close()

        asyncio.run(_run())


def test_promotion_env_on_starts_watcher_no_catchup():
    if not _has_duckdb():
        pytest.skip("duckdb not installed; skipping")

    with windows_safe_tempdir() as tmp:
        _mk_config(tmp)
        env = get_safe_subprocess_env(os.environ)
        env["CHUNKHOUND_MCP_MODE"] = "1"
        env["CHUNKHOUND_MCP__INDEX_ON_PROMOTION"] = "1"

        async def _run():
            c1 = await _start_stdio(tmp, env)  # expected RW
            c2 = await _start_stdio(tmp, env)  # expected RO
            try:
                # Promote by closing c1
                await c1.close()
                # Wait for promotion
                t0 = time.time()
                while time.time() - t0 < 5.0:
                    s2 = await _get_stats(c2)
                    if s2.get("role") == "RW":
                        break
                    await asyncio.sleep(0.2)

                # Create a new file after promotion
                new_file = tmp / "new_after_promotion_env_on.py"
                token = "unique_token_after_promotion_env_on"
                new_file.write_text(f"def yyy():\n    return '{token}'\n")

                # Wait to allow watcher to pick the change
                await asyncio.sleep(2.0)

                res = await c2.send_request(
                    "tools/call",
                    {"name": "search_regex", "arguments": {"pattern": token, "page_size": 50, "offset": 0}},
                    timeout=10.0,
                )
                assert len(res.get("results", [])) > 0, "Watcher should index new file after promotion when env ON"

                # Still no initial catch-up scan
                s2 = await _get_stats(c2)
                scan = s2.get("initial_scan")
                if scan is not None:
                    assert scan.get("is_scanning") is False

            finally:
                await c2.close()

        asyncio.run(_run())

