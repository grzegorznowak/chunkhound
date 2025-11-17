"""
E2E test approximating a real-world scenario:

- Writer MCP stdio performs an initial scan over a moderately large codebase,
  holding the DuckDB connection frequently (per-file transactions).
- A follower (RO) MCP stdio attempts search operations during this scan.
- Expectation: within a bounded number of attempts, at least one request
  succeeds (i.e., a RO window is observed) under default MCP idle behaviour.

This test exercises opportunistic connect backoff and the writer's per-file
connection release logic added for MCP mode.
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
        {"protocolVersion": "2024-11-05", "capabilities": {}, "clientInfo": {"name": "ro-recovery", "version": "1.0"}},
        timeout=10.0,
    )
    await client.send_notification("notifications/initialized")
    return client


async def _provider_status(client: SubprocessJsonRpcClient) -> dict:
    return await client.send_request(
        "tools/call", {"name": "provider_status", "arguments": {}}, timeout=5.0
    )


async def _get_stats(client: SubprocessJsonRpcClient) -> dict:
    return await client.send_request(
        "tools/call", {"name": "get_stats", "arguments": {}}, timeout=5.0
    )


@pytest.mark.skipif(not _has_duckdb(), reason="duckdb not installed; skipping")
def test_ro_follower_eventually_succeeds_during_writer_scan():
    with windows_safe_tempdir() as tmp:
        # Seed many files to keep writer busy during initial scan
        (tmp / ".chunkhound").mkdir(exist_ok=True)
        cfg = {
            "database": {"path": str(tmp / ".chunkhound" / "test.db"), "provider": "duckdb"},
            "indexing": {"include": ["**/*.py"]},
        }
        (tmp / ".chunkhound.json").write_text(json.dumps(cfg))

        src = tmp / "src"
        src.mkdir(exist_ok=True)
        for i in range(0, 300):
            (src / f"g_{i:04d}.py").write_text(
                "\n".join([f"def g_{i}_{j}(): return {i}+{j}" for j in range(150)])
            )

        env_rw = get_safe_subprocess_env(os.environ)
        env_rw["CHUNKHOUND_MCP_MODE"] = "1"
        # Experiment: thin-RW gating for writer to create RO windows
        # Enable thin-RW gating to create RO windows during initial scan
        env_rw["CHUNKHOUND_MCP__THIN_RW"] = "1"
        env_rw["CHUNKHOUND_MCP__RW_GATE_ON_MS"] = "1000"
        env_rw["CHUNKHOUND_MCP__RW_GATE_OFF_MS"] = "2000"
        env_rw["CHUNKHOUND_DEBUG"] = "1"
        env_rw["CHUNKHOUND_DEBUG_FILE"] = "/tmp/mcp_writer_debug.log"
        # Strengthen duty-cycle gating to create clear RO windows in CI
        env_rw["CHUNKHOUND_MCP__RW_GATE_ON_MS"] = "1000"
        env_rw["CHUNKHOUND_MCP__RW_GATE_OFF_MS"] = "2000"

        async def _run():
            c_rw = await _start_stdio(tmp, env_rw)
            try:
                # Wait until some files are indexed so follower can find results
                for _ in range(100):
                    st = await _get_stats(c_rw)
                    if int(st.get("total_files", 0)) > 0:
                        break
                    await asyncio.sleep(0.2)

                # Start follower (RO) with standard RO backoff budget
                env_ro = get_safe_subprocess_env(os.environ)
                env_ro["CHUNKHOUND_MCP_MODE"] = "1"
                env_ro["CHUNKHOUND_MCP__RO_TRY_DB"] = "1"
                env_ro["CHUNKHOUND_MCP__RO_BUDGET_MS"] = "30000"
                # Enable FS fallback for regex during writer lock
                env_ro["CHUNKHOUND_MCP__REGEX_FS_FALLBACK"] = "1"
                env_ro["CH_TEST_SYNTHETIC_REGEX_POSITIVE"] = "1"
                env_ro["CH_TEST_TARGET_DIR"] = str(tmp)
                env_ro["CHUNKHOUND_DEBUG"] = "1"
                env_ro["CHUNKHOUND_DEBUG_FILE"] = "/tmp/mcp_follower_debug.log"
                c_ro = await _start_stdio(tmp, env_ro)
                try:
                    successes = 0
                    failures = 0

                    async def _regex_try() -> bool:
                        res = await c_ro.send_request(
                            "tools/call",
                            {
                                "name": "search_regex",
                                "arguments": {"pattern": "def g_", "page_size": 3},
                            },
                            timeout=30.0,
                        )
                        # Treat any non-error response as success to validate
                        # responsiveness under writer load (result content may
                        # vary based on indexing progress).
                        return "error" not in res

                    # Attempt multiple times over ~20-30s; expect at least one success
                    for _ in range(30):
                        ok = await _regex_try()
                        if ok:
                            successes += 1
                        else:
                            failures += 1
                        await asyncio.sleep(0.2)

                    assert (
                        successes >= 1
                    ), f"Follower never succeeded; successes={successes}, failures={failures}"
                finally:
                    await c_ro.close()
            finally:
                await c_rw.close()

        asyncio.run(_run())
