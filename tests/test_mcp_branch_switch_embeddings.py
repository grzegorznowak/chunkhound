"""
Integration test: live re-indexing on a simulated branch switch.

Goals
- Start MCP stdio in a temp worktree (RW role).
- Create a file (v1) and ensure regex search finds its token.
- Overwrite the file with different content (v2) to simulate a branch switch.
- While re-indexing happens, issue another MCP call (get_stats) and ensure it
  succeeds quickly (server remains responsive during updates).
- Verify regex search drops the old token and finds the new token.
- If embeddings are available, verify semantic search finds the new token.

All code and comments in English.
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
    # initialize handshake
    await client.send_request(
        "initialize",
        {
            "protocolVersion": "2024-11-05",
            "capabilities": {},
            "clientInfo": {"name": "branch-switch-test", "version": "1.0"},
        },
        timeout=10.0,
    )
    await client.send_notification("notifications/initialized")
    return client


async def _tools_call(client: SubprocessJsonRpcClient, name: str, arguments: dict, timeout: float = 10.0) -> dict:
    return await client.send_request("tools/call", {"name": name, "arguments": arguments}, timeout=timeout)


def _mk_config(tmp: Path) -> None:
    config_path = tmp / ".chunkhound.json"
    db_path = tmp / ".chunkhound" / "test.db"
    db_path.parent.mkdir(exist_ok=True)
    config = {
        "database": {"path": str(db_path), "provider": "duckdb"},
        "indexing": {"include": ["*.py"]},
    }
    config_path.write_text(json.dumps(config))


def _embeddings_env_present() -> bool:
    # Heuristic: if any common embedding provider hints exist, we can attempt semantic assertions
    return any(
        os.getenv(k)
        for k in (
            "OPENAI_API_KEY",
            "OPENAI_BASE_URL",
            "OLLAMA_HOST",
            "OLLAMA_URL",
        )
    )


def test_mcp_branch_switch_reindexes_and_remains_responsive():
    if not _has_duckdb():
        pytest.skip("duckdb not installed; skipping branch-switch test")

    with windows_safe_tempdir() as tmp:
        tmp = Path(tmp)
        _mk_config(tmp)

        env = get_safe_subprocess_env(os.environ)
        env["CHUNKHOUND_MCP_MODE"] = "1"
        # Ensure default behavior: realtime watcher active (RW) during runtime
        # No special promotion flags needed for single process RW.

        async def _run() -> None:
            client = await _start_stdio(tmp, env)
            try:
                # Wait briefly for any initial scan to complete to avoid race with v1 write
                t0 = time.time()
                while time.time() - t0 < 5.0:
                    stats = await _tools_call(client, "get_stats", {})
                    scan = stats.get("initial_scan") or {}
                    if not scan.get("is_scanning", False):
                        break
                    await asyncio.sleep(0.2)

                # Create v1 of the file with token A
                token_a = "branch_token_A_uniq"
                fpath = tmp / "app.py"
                fpath.write_text(
                    f"def a():\n    # {token_a}\n    return 'A'\n",
                    encoding="utf-8",
                )

                # Poll until regex finds token A
                found_a = False
                deadline = time.time() + 6.0
                while time.time() < deadline:
                    res = await _tools_call(
                        client,
                        "search_regex",
                        {"pattern": token_a, "page_size": 50, "offset": 0},
                    )
                    if (res.get("results") or []):
                        found_a = True
                        break
                    await asyncio.sleep(0.3)
                assert found_a, "regex should find token A after v1 indexing"

                # Optionally probe semantic search (if embeddings configured)
                semantic_enabled = _embeddings_env_present()
                if semantic_enabled:
                    try:
                        _ = await _tools_call(client, "search_semantic", {"query": token_a, "top_k": 5})
                        # Not asserting here; we mainly confirm the tool path works under config
                        pass
                    except Exception:
                        # If semantic provider is misconfigured, continue without failing
                        semantic_enabled = False

                # Overwrite file with v2 (token B), simulating a branch switch
                token_b = "branch_token_B_uniq"
                fpath.write_text(
                    f"class B:\n    def b(self):\n        # {token_b}\n        return 'B'\n",
                    encoding="utf-8",
                )

                # While indexing reacts to the change, issue another tool call (get_stats)
                # to verify responsiveness (should not hang or error)
                stats_resp = await _tools_call(client, "get_stats", {})
                assert isinstance(stats_resp, dict), "get_stats should succeed during re-indexing"

                # Poll for token B to appear via regex and token A to disappear
                saw_b = False
                gone_a = False
                deadline = time.time() + 8.0
                while time.time() < deadline and not (saw_b and gone_a):
                    res_b = await _tools_call(
                        client,
                        "search_regex",
                        {"pattern": token_b, "page_size": 50, "offset": 0},
                    )
                    res_a = await _tools_call(
                        client,
                        "search_regex",
                        {"pattern": token_a, "page_size": 50, "offset": 0},
                    )
                    saw_b = bool(res_b.get("results"))
                    gone_a = len(res_a.get("results", [])) == 0
                    if saw_b and gone_a:
                        break
                    await asyncio.sleep(0.3)

                assert saw_b, "regex should find token B after branch switch"
                assert gone_a, "regex should no longer find token A after branch switch"

                # If embeddings are available, attempt semantic search for token B
                if semantic_enabled:
                    sem_ok = False
                    deadline = time.time() + 10.0
                    while time.time() < deadline and not sem_ok:
                        try:
                            sem = await _tools_call(
                                client,
                                "search_semantic",
                                {"query": token_b, "top_k": 5},
                                timeout=12.0,
                            )
                            # Heuristic: presence of results or a content field indicates success
                            if sem and (sem.get("results") or sem.get("content")):
                                sem_ok = True
                                break
                        except Exception:
                            # Retry within window
                            pass
                        await asyncio.sleep(0.5)
                    # Do not hard-fail if semantic provider is slow/missing; best-effort
                    # However, if embeddings are configured, we expect results eventually
                    if _embeddings_env_present():
                        assert sem_ok, (
                            "semantic search should succeed for token B when embeddings are configured"
                        )
            finally:
                await client.close()

        asyncio.run(_run())

