"""
Red-first test for semantic search RO backoff behaviour.

Goal:
- Simulate a RO follower that initially hits DuckDB-style lock conflicts when
  attempting to connect for semantic search.
- Verify that search_semantic_impl eventually succeeds after transient
  conflicts are resolved (once implementation adds wait+retry logic).

Current behaviour (expected RED):
- _opportunistic_connect_for_semantic calls provider.connect() once.
- On simulated lock conflict, it raises "Semantic search unavailable: writer active"
  and search_semantic_impl never calls the search service.
"""

from __future__ import annotations

import os
from types import SimpleNamespace
from typing import Any

import pytest

from chunkhound.embeddings import EmbeddingManager
from chunkhound.mcp_server.tools import search_semantic_impl


class DummyEmbeddingProvider:
    """Minimal embedding provider stub for EmbeddingManager."""

    def __init__(self) -> None:
        self._name = "dummy"
        self._model = "dummy-model"

    @property
    def name(self) -> str:
        return self._name

    @property
    def model(self) -> str:
        return self._model

    @property
    def dims(self) -> int:
        return 1

    @property
    def distance(self) -> str:
        return "cosine"

    @property
    def batch_size(self) -> int:
        return 1

    async def embed(self, texts: list[str]) -> list[list[float]]:
        return [[0.0] for _ in texts]


class DummyProvider:
    """Simulate a RO follower provider that hits transient lock conflicts."""

    def __init__(self) -> None:
        self.is_connected = False
        self.db_connected = False
        self._role = "RO"
        self.connect_calls = 0

    def get_role(self) -> str:
        return self._role

    def connect(self) -> None:
        self.connect_calls += 1
        # First two attempts simulate a writer-held lock
        if self.connect_calls <= 2:
            # Message chosen to match _is_duckdb_lock_conflict checks
            raise RuntimeError("Could not set lock on file: writer active")
        # Third attempt succeeds
        self.is_connected = True
        self.db_connected = True


class DummySearchService:
    """Stub search service that returns a single semantic result."""

    async def search_semantic(self, **kwargs: Any) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        return (
            [{"file_path": "x.py", "line": 1, "symbol": "dummy"}],
            {
                "offset": kwargs.get("offset", 0),
                "page_size": kwargs.get("page_size", 10),
                "has_more": False,
                "next_offset": None,
                "total": 1,
            },
        )


@pytest.mark.asyncio
async def test_semantic_search_eventually_succeeds_after_lock_conflicts(monkeypatch: pytest.MonkeyPatch) -> None:
    """RED-FIRST: semantic search should wait/retry on transient writer conflicts instead of failing immediately.

    This test encodes the desired behaviour:
    - In MCP RO follower mode with RO_TRY_DB enabled, transient lock conflicts
      during provider.connect() should not surface "Semantic search unavailable: writer active"
      to the caller.
    - Instead, search_semantic_impl should eventually reach the search service
      once a slot becomes available (after implementation adds wait+retry).

    Current implementation raises immediately, so this test is expected to FAIL
    until the retry/backoff logic is implemented.
    """
    # Configure environment for MCP RO follower with opportunistic DB tries
    monkeypatch.setenv("CHUNKHOUND_MCP_MODE", "1")
    monkeypatch.setenv("CHUNKHOUND_MCP__RO_TRY_DB", "1")

    # Set up stub services
    provider = DummyProvider()
    services = SimpleNamespace(provider=provider, search_service=DummySearchService())

    # Minimal embedding manager with a single stub provider
    embedding_manager = EmbeddingManager()
    embedding_manager.register_provider(DummyEmbeddingProvider(), set_default=True)

    # When implementation is complete, this call should:
    # - internally handle the first two lock conflicts via backoff/retry
    # - eventually succeed and return the stubbed semantic result
    result = await search_semantic_impl(
        services=services,
        embedding_manager=embedding_manager,
        query="dummy",
        page_size=10,
        offset=0,
        provider=None,
        model=None,
        threshold=None,
        path_filter=None,
    )

    assert len(result["results"]) == 1
    # We expect multiple connect attempts due to transient conflicts
    assert provider.connect_calls >= 3

