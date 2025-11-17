import asyncio
import json
import os
from pathlib import Path

import pytest

from tests.utils import SubprocessJsonRpcClient, create_subprocess_exec_safe, get_safe_subprocess_env
from tests.utils.windows_compat import windows_safe_tempdir


@pytest.mark.asyncio
async def test_code_research_obeys_ro_backoff_and_recovers():
    """E2E for code_research backoff using a deterministic forced-conflict hook.

    - Start Non-Indexer MCP stdio with CH_TEST_FORCE_RO_CONFLICT=1 and CALLS=2
    - First two code_research calls must return fast fallback and grow backoff
    - Third call should succeed (stubbed codex provider) after forced conflicts expire
    """

    async def run_index(temp_dir: Path, cfg_path: Path, db_path: Path) -> None:
        cmd = [
            "uv",
            "run",
            "chunkhound",
            "index",
            str(temp_dir),
            "--no-embeddings",
            "--config",
            str(cfg_path),
            "--db",
            str(db_path),
        ]
        proc = await create_subprocess_exec_safe(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env=get_safe_subprocess_env(os.environ.copy()),
        )
        stdout, stderr = await proc.communicate()
        assert (
            proc.returncode == 0
        ), f"indexing failed\nstdout: {stdout.decode()}\nstderr: {stderr.decode()}"

    with windows_safe_tempdir() as temp:
        temp_dir = temp
        src_dir = temp_dir / "src"
        src_dir.mkdir(parents=True, exist_ok=True)
        (src_dir / "app.py").write_text("def alpha():\n    return 1\n", encoding="utf-8")

        # Minimal config with database path (duckdb)
        cfg_path = temp_dir / ".chunkhound.json"
        db_path = temp_dir / ".chunkhound" / "db"
        db_path.parent.mkdir(parents=True, exist_ok=True)
        cfg = {
            "database": {"path": str(db_path), "provider": "duckdb"},
            "indexing": {"include": ["*.py"]},
        }
        cfg_path.write_text(json.dumps(cfg), encoding="utf-8")

        # 1) Index the tiny repo (no embeddings)
        await run_index(temp_dir, cfg_path, db_path)

        # 2) Start MCP stdio server with forced conflicts for code_research
        mark_file = temp_dir / "codex_called.txt"
        env = get_safe_subprocess_env(
            {
                **os.environ,
                "PYTHONPATH": f"{Path('tests/helpers').resolve()}:{os.environ.get('PYTHONPATH','')}",
                "CH_TEST_PATCH_CODEX": "1",
                "CH_TEST_CODEX_MARK_FILE": str(mark_file),
                "CH_TEST_FORCE_RO_CONFLICT": "1",
                "CH_TEST_FORCE_RO_CONFLICT_CALLS": "2",
                "CHUNKHOUND_MCP_MODE": "1",
                "CHUNKHOUND_MCP__SKIP_INDEXING": "1",
                "CHUNKHOUND_MCP__RO_TRY_DB": "1",
                # Small backoff tunables for fast test
                "CHUNKHOUND_MCP__RO_BACKOFF_INITIAL_MS": "50",
                "CHUNKHOUND_MCP__RO_BACKOFF_MULT": "2.0",
                "CHUNKHOUND_MCP__RO_BACKOFF_MAX_MS": "200",
                "CHUNKHOUND_MCP__RO_BACKOFF_COOLDOWN_MS": "800",
                "CHUNKHOUND_MCP__RO_BUDGET_MS": "400",
                "CHUNKHOUND_DEBUG": "1",
                "CHUNKHOUND_DEBUG_FILE": str(temp_dir / "mcp_debug.log"),
            }
        )

        mcp_cmd = [
            "uv",
            "run",
            "chunkhound",
            "mcp",
            str(temp_dir),
            "--stdio",
            "--llm-utility-provider",
            "codex-cli",
            "--llm-utility-model",
            "codex",
            "--llm-synthesis-provider",
            "codex-cli",
            "--llm-synthesis-model",
            "codex",
            "--config",
            str(cfg_path),
        ]
        proc = await create_subprocess_exec_safe(
            *mcp_cmd,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env=env,
            cwd=str(temp_dir),
        )

        client = SubprocessJsonRpcClient(proc)
        await client.start()
        try:
            # Handshake
            await client.send_request(
                "initialize",
                {
                    "protocolVersion": "2024-11-05",
                    "capabilities": {},
                    "clientInfo": {"name": "e2e", "version": "1.0"},
                },
                timeout=10.0,
            )
            await client.send_notification("notifications/initialized")

            async def code_research_call() -> str:
                call = await client.send_request(
                    "tools/call",
                    {
                        "name": "code_research",
                        "arguments": {"query": "alpha", "depth": "shallow"},
                    },
                    timeout=5.0,
                )
                # Normalize MCP ToolResult shape
                result = call.get("result") or {}
                contents = result.get("content") or call.get("content") or []
                if isinstance(contents, dict):
                    contents = contents.get("content", [])
                if not contents and isinstance(result, dict) and "text" in result:
                    contents = [{"text": result["text"]}]
                return "\n".join(
                    [c.get("text", "") for c in contents if isinstance(c, dict)]
                )

            async def get_stats() -> dict:
                return await client.send_request(
                    "tools/call", {"name": "get_stats", "arguments": {}}, timeout=5.0
                )

            # First forced-conflict
            t1 = await code_research_call()
            assert "Research unavailable" in t1
            s1 = await get_stats()
            ro1 = int(s1.get("ro_backoff_ms", 0))
            assert ro1 >= 50

            # Second forced-conflict (grows backoff)
            t2 = await code_research_call()
            assert "Research unavailable" in t2
            s2 = await get_stats()
            ro2 = int(s2.get("ro_backoff_ms", 0))
            assert ro2 >= min(100, 200)

            # Third call should succeed (forced-conflict exhausted) via codex stub
            t3 = await code_research_call()
            assert "SYNTH_OK" in t3
            assert (temp_dir / "codex_called.txt").exists()

        finally:
            await client.close()
