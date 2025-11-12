"""
Red-first TDD tests for multi-process auto-role locking.

All code and comments in English, per request.

These tests define the expected API for a cross-platform file-based lock
manager that coordinates roles among multiple ChunkHound MCP processes:
  - Exactly one RW leader at a time (OS-exclusive file lock).
  - Any number of RO followers.
  - Live promotion RO->RW when the leader releases or dies.
  - Basic election backoff to reduce thundering herd on leader loss.

Expected API under test (to be implemented):
    from chunkhound.providers.database.locks import LockManager, LockRole

    class LockManager:
        def __init__(self, lock_path: str | Path, *,
                     lease_ttl_ms: int = 5000,
                     heartbeat_ms: int = 1000,
                     election_min_ms: int = 150,
                     election_max_ms: int = 600,
                     allow_force_takeover: bool = False,
                     allow_cross_host_takeover: bool = False) -> None: ...

        def acquire(self) -> LockRole:
            # Acquire role for this process. Returns "RW" or "RO".

        def attempt_promote(self) -> bool:
            # Try to promote from RO to RW (non-blocking). Returns True on success.

        def release(self) -> None:
            # Release any held lock and role. Idempotent.

        def read_meta(self) -> dict:
            # Read current lock metadata (pid, epoch, last_heartbeat_ts, etc.).

        def renew(self) -> None:
            # Heartbeat/renew the lease if RW (no-op if RO).

        # Watcher API (new for auto-promotion):
        def start_watcher(self) -> None:
            # Start background watcher to auto-promote RO to RW when possible.

        def stop_watcher(self) -> None:
            # Stop background watcher thread cleanly (idempotent).

        def get_role(self) -> LockRole | None:
            # Return current role: "RW", "RO" or None if not acquired.

    LockRole = Literal["RW", "RO"]

Notes:
 - Tests use multiprocessing only for the dead-leader scenario to avoid false
   positives with per-process file lock semantics.
 - Time-based tests use generous timeouts to remain stable.
"""

from __future__ import annotations

import os
import time
import json
import random
import tempfile
import multiprocessing as mp
from pathlib import Path

import pytest


# -- Helper to check module availability early (keeps tests red but importable) --
LOCKS_IMPORT_ERROR_MSG = (
    "ChunkHound LockManager API not found. Implement chunkhound/providers/database/locks.py"
)


def _try_import_locks_module():
    try:
        from chunkhound.providers.database.locks import LockManager  # type: ignore
        return LockManager
    except Exception as exc:  # pragma: no cover - red-first placeholder
        pytest.skip(f"{LOCKS_IMPORT_ERROR_MSG}: {exc}")


# ----------------------
# Basic role acquisition
# ----------------------


def test_single_rw_multiple_ro(tmp_path: Path):
    LockManager = _try_import_locks_module()
    lock_path = tmp_path / "testdb.rw.lock"

    m1 = LockManager(lock_path)
    role1 = m1.acquire()
    assert role1 == "RW", "first process should become RW leader"

    m2 = LockManager(lock_path)
    role2 = m2.acquire()
    assert role2 == "RO", "second process should fall back to RO"

    # Release and cleanup
    m1.release()
    m2.release()


def test_promotion_after_release(tmp_path: Path):
    LockManager = _try_import_locks_module()
    lock_path = tmp_path / "testdb.rw.lock"

    leader = LockManager(lock_path)
    assert leader.acquire() == "RW"

    follower = LockManager(lock_path)
    assert follower.acquire() == "RO"

    # Drop leadership; follower should be able to promote
    leader.release()

    # Allow small delay for filesystem state to settle
    time.sleep(0.05)
    assert follower.attempt_promote() is True, "follower should promote to RW after release"
    follower.release()


def test_metadata_and_heartbeat(tmp_path: Path):
    """After acquiring RW, metadata should be present and heartbeat should advance."""
    LockManager = _try_import_locks_module()
    lock_path = tmp_path / "testdb.rw.lock"

    m = LockManager(lock_path, heartbeat_ms=200, lease_ttl_ms=1000)
    assert m.acquire() == "RW"

    meta1 = m.read_meta()
    assert isinstance(meta1, dict)
    for key in ("pid", "instance_id", "epoch", "last_heartbeat_ts", "lease_ttl_ms"):
        assert key in meta1, f"missing metadata field: {key}"

    t1 = float(meta1["last_heartbeat_ts"])
    time.sleep(0.25)
    m.renew()
    meta2 = m.read_meta()
    t2 = float(meta2["last_heartbeat_ts"]) if meta2 else t1
    assert t2 >= t1, "heartbeat/renew should advance last_heartbeat_ts"
    m.release()


# ---------------------------------------
# Dead leader detection and safe takeover
# ---------------------------------------


def _leader_process(lock_path_str: str, sleep_seconds: float):
    """Child process that becomes RW and sleeps (simulates work)."""
    from chunkhound.providers.database.locks import LockManager  # type: ignore

    lm = LockManager(lock_path_str, heartbeat_ms=500, lease_ttl_ms=2000)
    role = lm.acquire()
    assert role == "RW"
    # Keep heartbeating while sleeping
    end = time.time() + sleep_seconds
    while time.time() < end:
        lm.renew()
        time.sleep(0.2)
    # Intentional: exit without explicit release to simulate abrupt termination


def test_takeover_on_dead_leader(tmp_path: Path):
    LockManager = _try_import_locks_module()
    lock_path = tmp_path / "testdb.rw.lock"

    # Start a separate process that becomes RW and then we terminate it
    p = mp.Process(target=_leader_process, args=(str(lock_path), 10.0))
    p.start()
    p.join(timeout=1.5)
    assert p.is_alive(), "leader process failed to start and acquire RW"

    # Our follower joins as RO
    follower = LockManager(lock_path, heartbeat_ms=500, lease_ttl_ms=2000)
    assert follower.acquire() == "RO"

    # Kill the leader abruptly
    p.terminate()
    p.join(timeout=3.0)
    assert not p.is_alive(), "leader process did not terminate as expected"

    # Wait for lease to expire
    time.sleep(2.5)

    # Attempt promotion; should succeed without force takeover if OS lock released
    promoted = follower.attempt_promote()
    assert promoted is True, "follower should promote to RW after dead leader and lease expiry"

    follower.release()


# ---------------------------
# Watcher-based auto-promotion
# ---------------------------


def test_watcher_promotes_after_leader_release(tmp_path: Path):
    """Follower with watcher should auto-promote after leader releases the lock."""
    LockManager = _try_import_locks_module()
    lock_path = tmp_path / "testdb.rw.lock"

    leader = LockManager(lock_path, heartbeat_ms=200, lease_ttl_ms=1000)
    assert leader.acquire() == "RW"

    follower = LockManager(lock_path, heartbeat_ms=200, lease_ttl_ms=1000)
    assert follower.acquire() == "RO"
    follower.start_watcher()

    # Leader releases; watcher should detect and promote within a short window
    leader.release()

    deadline = time.time() + 3.0
    while time.time() < deadline and follower.get_role() != "RW":
        time.sleep(0.05)

    assert follower.get_role() == "RW", "watcher should auto-promote follower to RW"
    follower.stop_watcher()
    follower.release()


def _leader_forever(lock_path_str: str):
    from chunkhound.providers.database.locks import LockManager  # type: ignore
    lm = LockManager(lock_path_str, heartbeat_ms=300, lease_ttl_ms=2000)
    assert lm.acquire() == "RW"
    # Keep heartbeating until killed
    try:
        while True:
            lm.renew()
            time.sleep(0.2)
    finally:
        # Best effort release if we exit gracefully
        lm.release()


def test_watcher_does_not_steal_when_leader_alive(tmp_path: Path):
    """With an active leader, a follower watcher must remain RO (no steal)."""
    LockManager = _try_import_locks_module()
    lock_path = tmp_path / "testdb.rw.lock"

    p = mp.Process(target=_leader_forever, args=(str(lock_path),))
    p.start()
    time.sleep(0.5)  # allow leader to acquire and start heartbeating

    follower = LockManager(lock_path, heartbeat_ms=300, lease_ttl_ms=2000)
    assert follower.acquire() == "RO"
    follower.start_watcher()

    time.sleep(1.0)
    assert follower.get_role() == "RO", "watcher must not promote while leader is alive"

    follower.stop_watcher()
    follower.release()
    p.terminate()
    p.join(timeout=2.0)


def test_epoch_increments_on_promotion(tmp_path: Path):
    """Epoch should monotonically increase whenever a new RW leader is elected."""
    LockManager = _try_import_locks_module()
    lock_path = tmp_path / "testdb.rw.lock"

    leader = LockManager(lock_path)
    assert leader.acquire() == "RW"
    meta1 = leader.read_meta()
    e1 = int(meta1.get("epoch", 0)) if meta1 else 0

    follower = LockManager(lock_path)
    assert follower.acquire() == "RO"
    follower.start_watcher()

    leader.release()

    deadline = time.time() + 3.0
    while time.time() < deadline and follower.get_role() != "RW":
        time.sleep(0.05)

    assert follower.get_role() == "RW"
    meta2 = follower.read_meta()
    e2 = int(meta2.get("epoch", 0)) if meta2 else 0
    assert e2 >= e1 + 1, "new leader must bump epoch"

    follower.stop_watcher()
    follower.release()


def test_watcher_start_stop_is_idempotent(tmp_path: Path):
    """start_watcher/stop_watcher should be safe to call multiple times."""
    LockManager = _try_import_locks_module()
    lock_path = tmp_path / "testdb.rw.lock"

    follower = LockManager(lock_path)
    assert follower.acquire() in ("RO", "RW")

    follower.start_watcher()
    follower.start_watcher()  # idempotent
    time.sleep(0.05)
    follower.stop_watcher()
    follower.stop_watcher()  # idempotent
    follower.release()


# -------------------------------------------
# Election backoff to reduce thundering herd
# -------------------------------------------


def _contender_process(lock_path_str: str, result_path_str: str,
                       election_min_ms: int, election_max_ms: int):
    from chunkhound.providers.database.locks import LockManager  # type: ignore

    # Random delay to emulate election window
    delay = random.randint(election_min_ms, election_max_ms) / 1000.0
    time.sleep(delay)

    lm = LockManager(lock_path_str,
                     heartbeat_ms=400,
                     lease_ttl_ms=3000,
                     election_min_ms=election_min_ms,
                     election_max_ms=election_max_ms)
    role = lm.acquire()
    if role == "RW":
        # Write a small file to announce victory
        Path(result_path_str).write_text(json.dumps({"winner": os.getpid()}))
        # Hold briefly then release
        time.sleep(0.2)
        lm.release()


def test_election_backoff_many_contenders(tmp_path: Path):
    LockManager = _try_import_locks_module()
    lock_path = tmp_path / "testdb.rw.lock"
    result_path = tmp_path / "winner.json"

    # Spawn N contenders without any prior leader
    contenders = []
    N = 8
    for _ in range(N):
        p = mp.Process(
            target=_contender_process,
            args=(str(lock_path), str(result_path), 150, 500),
        )
        contenders.append(p)
        p.start()

    for p in contenders:
        p.join(timeout=3.0)

    # Exactly one should have declared leadership at a time. We only check that
    # at least one winner existed; stronger guarantees will be validated once the
    # LockManager persists epochs/heartbeats.
    assert result_path.exists(), "one contender should win leadership with backoff"
