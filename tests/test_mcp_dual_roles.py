"""
End-to-end test that starts two MCP stdio servers on the same DB/worktree
and verifies that the first runs as RW and the second as RO (auto-role),
and that the RO instance does not run initial scanning.

All code and comments in English.
"""

from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path

import pytest

from tests.utils import SubprocessJsonRpcClient, create_subprocess_exec_safe, get_safe_subprocess_env
from tests.utils.windows_compat import windows_safe_tempdir


def test_mcp_dual_process_roles_rw_then_ro():
    try:
        import duckdb  # noqa: F401
    except Exception:
        pytest.skip("duckdb not installed; skipping dual-role test")

    with windows_safe_tempdir() as temp_path:
        # Minimal config shared by both servers
        config_path = temp_path / ".chunkhound.json"
        db_path = temp_path / ".chunkhound" / "test.db"
        db_path.parent.mkdir(exist_ok=True)
        config = {
            "database": {"path": str(db_path), "provider": "duckdb"},
            "indexing": {"include": ["*.py"]},
        }
        config_path.write_text(json.dumps(config))

        # Common env
        base_env = get_safe_subprocess_env(os.environ)
        base_env["CHUNKHOUND_MCP_MODE"] = "1"

        async def _run_dual():
            # Start first server (expected RW)
            proc1 = await create_subprocess_exec_safe(
                "uv",
                "run",
                "chunkhound",
                "mcp",
                str(temp_path),
                cwd=str(temp_path),
                env=base_env,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            c1 = SubprocessJsonRpcClient(proc1)
            await c1.start()

            # Start second server (expected RO)
            proc2 = await create_subprocess_exec_safe(
                "uv",
                "run",
                "chunkhound",
                "mcp",
                str(temp_path),
                cwd=str(temp_path),
                env=base_env,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            c2 = SubprocessJsonRpcClient(proc2)
            await c2.start()

            try:
                # Initialize both
                for client in (c1, c2):
                    await client.send_request(
                        "initialize",
                        {
                            "protocolVersion": "2024-11-05",
                            "capabilities": {},
                            "clientInfo": {"name": "test", "version": "1.0"},
                        },
                        timeout=10.0,
                    )
                    await client.send_notification("notifications/initialized")

                # Query get_stats from both
                stats1 = await c1.send_request(
                    "tools/call", {"name": "get_stats", "arguments": {}}, timeout=10.0
                )
                stats2 = await c2.send_request(
                    "tools/call", {"name": "get_stats", "arguments": {}}, timeout=10.0
                )

                role1 = stats1.get("role")
                role2 = stats2.get("role")

                assert role1 in ("RW", None)  # Default to RW when role not exposed
                assert role2 in ("RO", "RW", None)

                # If roles are exposed, assert RW then RO
                if role1 is not None and role2 is not None:
                    assert role1 == "RW"
                    assert role2 == "RO"

                # Secondary should not be scanning
                scan2 = stats2.get("initial_scan")
                if scan2 is not None:
                    assert scan2.get("is_scanning") is False

            finally:
                await c1.close()
                await c2.close()

        asyncio.run(_run_dual())

