"""
E2E test for RW→RO degradation: ensure watcher stops when demoted.

Uses a test-only hook CH_TEST_DEMOTE_AFTER_MS to simulate demotion in the
MCP role monitor without requiring cross-process lock stealing.
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


def test_demote_rw_to_ro_stops_watcher():
    if not _has_duckdb():
        pytest.skip("duckdb not installed; skipping")

    with windows_safe_tempdir() as tmp:
        # Minimal config
        cfg = {
            "database": {"path": str(tmp / ".chunkhound" / "test.db"), "provider": "duckdb"},
            "indexing": {"include": ["*.py"]},
        }
        (tmp / ".chunkhound").mkdir(exist_ok=True)
        (tmp / ".chunkhound.json").write_text(json.dumps(cfg))

        env = get_safe_subprocess_env(os.environ)
        env["CHUNKHOUND_MCP_MODE"] = "1"
        env["CHUNKHOUND_MCP__INDEX_ON_PROMOTION"] = "1"  # ensure watcher can run when RW

        async def _run():
            c = await _start_stdio(tmp, env)
            try:
                # Confirm role (RW or unknown)
                s = await _get_stats(c)
                if s.get("role") == "RO":
                    pytest.skip("unexpected RO on single process; environment constrained")

                # Ensure initial scan (if any) has completed so it won't pick up
                # a file created after demotion; we want to verify watcher stop.
                t_scan = time.time()
                while time.time() - t_scan < 5.0:
                    s = await _get_stats(c)
                    scan = s.get("initial_scan") or {}
                    if not scan.get("is_scanning", False):
                        break
                    await asyncio.sleep(0.2)

                # File-based demotion control: write <db>.rw.lock.ctrl to request demotion after 500ms
                db_path = Path(json.loads((tmp / ".chunkhound.json").read_text())["database"]["path"])  # type: ignore
                ctrl_path = Path(str(db_path) + ".rw.lock.ctrl")
                ctrl_path.write_text("demote\ndemote_after_ms=500\n", encoding="utf-8")

                t0 = time.time()
                new_role = None
                while time.time() - t0 < 3.0:
                    s = await _get_stats(c)
                    new_role = s.get("role")
                    if new_role == "RO":
                        break
                    await asyncio.sleep(0.2)

                assert new_role == "RO", "Server should demote to RO via test hook"

                # Give the MCP role monitor a brief moment to stop the realtime
                # watcher after the RO transition is observed. This avoids a race
                # where a filesystem event for the new file could still be picked
                # up by the watcher before it fully stops.
                await asyncio.sleep(0.75)

                # Create new file; since demoted, watcher should be stopped and not index it
                token = "demotion_should_stop_watcher"
                (tmp / "x.py").write_text(f"def z():\n    return '{token}'\n")
                await asyncio.sleep(1.5)
                res = await c.send_request(
                    "tools/call",
                    {"name": "search_regex", "arguments": {"pattern": token, "page_size": 50, "offset": 0}},
                    timeout=10.0,
                )
                assert len(res.get("results", [])) == 0, "Watcher should be stopped after demotion"

            finally:
                await c.close()

        asyncio.run(_run())
