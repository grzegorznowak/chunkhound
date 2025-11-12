"""
Regression reproducer: executor used after shutdown raises RuntimeError.

This test documents the current behavior where the database executor
is shut down (via provider.disconnect()), and a subsequent service call
that tries to schedule work on the same executor results in:

    RuntimeError: cannot schedule new futures after shutdown

The test ensures we can reliably reproduce the issue locally before
we implement a self-healing fix (e.g., lazy re-create executor).
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from tests.utils.windows_compat import windows_safe_tempdir
from chunkhound.database_factory import create_services


def _minimal_config(db_path: Path) -> dict:
    return {
        "database": {"path": str(db_path), "provider": "duckdb"},
        "indexing": {"include": ["*.py"]},
    }


def test_executor_auto_recovers_after_disconnect_regex():
    with windows_safe_tempdir() as tmp:
        tmp_path = Path(tmp)
        db_path = tmp_path / ".chunkhound" / "test.db"
        db_path.parent.mkdir(exist_ok=True)

        # Create minimal config and services
        cfg = _minimal_config(db_path)
        services = create_services(db_path, cfg, embedding_manager=None)
        provider = services.provider

        # Establish connection and perform a quick no-op search to warm the executor
        provider.connect()
        services.search_service.search_regex(pattern="__warm__", page_size=1, offset=0)

        # Disconnect the provider which shuts down the underlying executor
        provider.disconnect(skip_checkpoint=True)

        # After fix: executor lazily re-initializes and the call succeeds
        out, _ = services.search_service.search_regex(pattern="again", page_size=1, offset=0)
        assert isinstance(out, list)


@pytest.mark.asyncio
async def test_executor_auto_recovers_after_disconnect_semantic():
    from tests.fixtures.fake_providers import FakeEmbeddingProvider
    from chunkhound.embeddings import EmbeddingManager

    with windows_safe_tempdir() as tmp:
        tmp_path = Path(tmp)
        db_path = tmp_path / ".chunkhound" / "test.db"
        db_path.parent.mkdir(exist_ok=True)

        cfg = _minimal_config(db_path)
        # Register fake embedding provider so semantic path is enabled
        em = EmbeddingManager()
        em.register_provider(FakeEmbeddingProvider(), set_default=True)
        services = create_services(db_path, cfg, embedding_manager=em)
        provider = services.provider
        provider.connect()
        # Warm up
        await services.search_service.search_semantic("warm", page_size=1, offset=0)
        # Disconnect (shutdown executor)
        provider.disconnect(skip_checkpoint=True)
        # After fix: next semantic search re-initializes executor and succeeds
        results, _ = await services.search_service.search_semantic("again", page_size=1, offset=0)
        assert isinstance(results, list)


@pytest.mark.asyncio
async def test_executor_auto_recovers_after_disconnect_get_stats_tool():
    # Directly invoke tool implementation to ensure it can reconnect after disconnect
    from chunkhound.mcp_server.tools import get_stats_impl

    with windows_safe_tempdir() as tmp:
        tmp_path = Path(tmp)
        db_path = tmp_path / ".chunkhound" / "test.db"
        db_path.parent.mkdir(exist_ok=True)

        cfg = _minimal_config(db_path)
        services = create_services(db_path, cfg, embedding_manager=None)
        provider = services.provider
        provider.connect()
        # Warm: stats with connected provider
        s1 = await get_stats_impl(services, scan_progress=None)
        assert isinstance(s1, dict)
        # Disconnect (shutdown executor)
        provider.disconnect(skip_checkpoint=True)
        # After fix: tool should reconnect and return stats without raising RuntimeError
        s2 = await get_stats_impl(services, scan_progress=None)
        assert isinstance(s2, dict)


@pytest.mark.asyncio
async def test_executor_auto_recovers_after_disconnect_code_research_tool():
    # Use a minimal embedding stub with reranking support and a trivial LLM manager
    from chunkhound.mcp_server.tools import deep_research_impl
    from chunkhound.embeddings import EmbeddingManager

    class _DummyEmbeddingProvider:
        name = "dummy"
        model = "dummy"
        dims = 1
        distance = "cosine"
        batch_size = 8
        def supports_reranking(self) -> bool:  # deep_research requirement
            return True
        async def embed(self, texts):  # pragma: no cover
            return [[0.0] for _ in texts]

    class _DummyLLMManager:
        def is_configured(self) -> bool:
            return True

    with windows_safe_tempdir() as tmp:
        tmp_path = Path(tmp)
        db_path = tmp_path / ".chunkhound" / "test.db"
        db_path.parent.mkdir(exist_ok=True)

        cfg = _minimal_config(db_path)
        em = EmbeddingManager()
        em.register_provider(_DummyEmbeddingProvider(), set_default=True)
        services = create_services(db_path, cfg, embedding_manager=em)
        provider = services.provider
        provider.connect()
        # Warm: ensure executor thread is used at least once
        services.search_service.search_regex(pattern="warm", page_size=1, offset=0)
        # Disconnect (shutdown executor)
        provider.disconnect(skip_checkpoint=True)
        # After fix: deep_research_impl should reconnect during setup and not raise RuntimeError
        res = await deep_research_impl(
            services=services,
            embedding_manager=em,
            llm_manager=_DummyLLMManager(),
            query="probe",
            depth="shallow",
            progress=None,
        )
        assert isinstance(res, dict)
