"""
Red-first TDD tests for provider-level auto-role integration with LockManager.

All code and comments in English, per request.

Goals:
 - Provider can enable auto-role (RW/RO) using a lock file.
 - First provider becomes RW, subsequent providers become RO.
 - RO provider auto-promotes to RW after the RW provider disconnects (watcher).
 - RO provider rejects mutating operations (e.g., insert_embeddings_batch).

Assumptions for the provider API (to be implemented):
    from chunkhound.providers.database.duckdb_provider import DuckDBProvider

    provider = DuckDBProvider(db_path, base_directory, embedding_manager=None, config=None)
    provider.enable_auto_role(lock_path, heartbeat_ms=..., lease_ttl_ms=...,
                              election_min_ms=..., election_max_ms=...)
    provider.connect()
    provider.start_role_watcher()  # idempotent
    provider.get_role() -> "RW" | "RO"
    provider.close()

Mutating call expected to be rejected in RO:
    provider.insert_embeddings_batch([{"chunk_id": 1, "provider": "x", "model": "y",
                                      "dims": 2, "embedding": [0.0, 0.0]}])

These tests will be red until the provider integrates with LockManager and
enforces read-only mode. The module is skipped entirely if duckdb is not
available, to keep CI stable on minimal environments.
"""

from __future__ import annotations

import os
import time
from pathlib import Path

import pytest

# Skip entire module if duckdb is not installed
pytest.importorskip("duckdb")

from chunkhound.providers.database.duckdb_provider import DuckDBProvider  # type: ignore


def _mk_provider(db_path: Path, base_dir: Path) -> DuckDBProvider:
    return DuckDBProvider(db_path=db_path, base_directory=base_dir, embedding_manager=None, config=None)


def test_provider_auto_role_rw_ro(tmp_path: Path):
    """First provider becomes RW; second provider becomes RO when auto-role is enabled."""
    db_path = tmp_path / "auto_role.duckdb"
    lock_path = tmp_path / "auto_role.rw.lock"

    p1 = _mk_provider(db_path, tmp_path)
    p1.enable_auto_role(lock_path=str(lock_path), heartbeat_ms=200, lease_ttl_ms=1000)
    p1.connect()
    assert p1.get_role() == "RW"

    p2 = _mk_provider(db_path, tmp_path)
    p2.enable_auto_role(lock_path=str(lock_path), heartbeat_ms=200, lease_ttl_ms=1000)
    p2.connect()
    assert p2.get_role() == "RO"

    p2.close()
    p1.close()


def test_provider_ro_promotes_after_rw_close(tmp_path: Path):
    """RO provider should auto-promote to RW after RW provider closes."""
    db_path = tmp_path / "auto_role.duckdb"
    lock_path = tmp_path / "auto_role.rw.lock"

    p1 = _mk_provider(db_path, tmp_path)
    p1.enable_auto_role(lock_path=str(lock_path), heartbeat_ms=200, lease_ttl_ms=1000, election_min_ms=150, election_max_ms=400)
    p1.connect()
    assert p1.get_role() == "RW"

    p2 = _mk_provider(db_path, tmp_path)
    p2.enable_auto_role(lock_path=str(lock_path), heartbeat_ms=200, lease_ttl_ms=1000, election_min_ms=150, election_max_ms=400)
    p2.connect()
    assert p2.get_role() == "RO"
    p2.start_role_watcher()

    # Close RW; RO should auto-promote within a short window
    p1.close()

    deadline = time.time() + 3.0
    while time.time() < deadline and p2.get_role() != "RW":
        time.sleep(0.05)

    assert p2.get_role() == "RW", "RO provider should auto-promote after leader closes"
    p2.close()


def test_provider_enforces_read_only_on_mutations(tmp_path: Path):
    """Mutating operations must be rejected when provider is in RO role."""
    db_path = tmp_path / "auto_role.duckdb"
    lock_path = tmp_path / "auto_role.rw.lock"

    p1 = _mk_provider(db_path, tmp_path)
    p1.enable_auto_role(lock_path=str(lock_path))
    p1.connect()
    assert p1.get_role() == "RW"

    p2 = _mk_provider(db_path, tmp_path)
    p2.enable_auto_role(lock_path=str(lock_path))
    p2.connect()
    assert p2.get_role() == "RO"

    # Attempt a mutating call; should raise a clear read-only error before touching DB.
    with pytest.raises(RuntimeError, match=r"read[- ]only|readonly|RO"):
        p2.insert_embeddings_batch([
            {
                "chunk_id": 1,
                "provider": "test",
                "model": "dummy",
                "dims": 2,
                "embedding": [0.0, 0.0],
            }
        ])

    p2.close()
    p1.close()

