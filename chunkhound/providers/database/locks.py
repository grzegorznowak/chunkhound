from __future__ import annotations

"""
Cross-platform file-based lock manager for auto-role (RW/RO) coordination.

All code and comments in English (per request).

This implementation is designed to satisfy tests in
tests/test_auto_role_locking.py.
"""

import errno
import json
import os
import random
import threading
import time
import uuid
from pathlib import Path
from typing import Literal, Optional

_IS_WINDOWS = os.name == "nt"
if not _IS_WINDOWS:
    import fcntl  # type: ignore
else:  # pragma: no cover - windows specific
    import msvcrt  # type: ignore


LockRole = Literal["RW", "RO"]


class LockManager:
    def __init__(
        self,
        lock_path: str | Path,
        *,
        lease_ttl_ms: int = 5000,
        heartbeat_ms: int = 1000,
        election_min_ms: int = 150,
        election_max_ms: int = 600,
        allow_force_takeover: bool = False,
        allow_cross_host_takeover: bool = False,
    ) -> None:
        self._lock_path = Path(lock_path)
        self._lease_ttl_ms = int(lease_ttl_ms)
        self._heartbeat_ms = int(heartbeat_ms)
        self._election_min_ms = int(election_min_ms)
        self._election_max_ms = int(election_max_ms)
        self._allow_force_takeover = bool(allow_force_takeover)
        self._allow_cross_host_takeover = bool(allow_cross_host_takeover)

        self._fh: Optional[object] = None
        self._role: Optional[LockRole] = None
        self._instance_id: str = str(uuid.uuid4())
        self._watcher_thread: Optional[threading.Thread] = None
        self._watcher_stop: Optional[threading.Event] = None
        self._acquired_at: float | None = None

    # Public API
    def acquire(self) -> LockRole:
        self._lock_path.parent.mkdir(parents=True, exist_ok=True)
        fh = open(self._lock_path, "a+b", buffering=0)
        fh.close()
        self._fh = open(self._lock_path, "r+b", buffering=0)
        if self._try_lock_exclusive(self._fh):
            self._role = "RW"
            self._acquired_at = time.time()
            self._write_initial_meta(epoch=self._next_epoch())
            return "RW"
        # follower
        try:
            self._fh.close()
        except Exception:
            pass
        self._fh = None
        self._role = "RO"
        return "RO"

    def attempt_promote(self) -> bool:
        if self._role == "RW":
            return True
        temp_fh = open(self._lock_path, "r+b", buffering=0)
        if not self._try_lock_exclusive(temp_fh):
            temp_fh.close()
            return False
        self._fh = temp_fh
        self._role = "RW"
        self._acquired_at = time.time()
        self._write_initial_meta(epoch=self._next_epoch())
        return True

    def release(self) -> None:
        self.stop_watcher()
        if self._fh is not None and self._role == "RW":
            self._unlock_exclusive(self._fh)
            try:
                self._fh.close()
            except Exception:
                pass
        self._fh = None
        self._role = None

    def renew(self) -> None:
        if self._role != "RW" or self._fh is None:
            return
        meta = self.read_meta() or {}
        meta["last_heartbeat_ts"] = time.time()
        meta["lease_ttl_ms"] = self._lease_ttl_ms
        self._write_meta(meta)

    def read_meta(self) -> dict:
        try:
            # Prefer the existing handle when we have one. This avoids
            # platform-specific sharing quirks (especially on Windows) where
            # opening a second handle on a locked file can fail or see stale
            # contents even though the current holder has written metadata.
            if self._fh is not None:
                fh = self._fh
                try:
                    fh.seek(0)
                    data_bytes = fh.read()
                except Exception:
                    data_bytes = b""
            else:
                with open(self._lock_path, "rb") as rfh:
                    data_bytes = rfh.read()

            if not data_bytes:
                return {}

            data = data_bytes.decode("utf-8", errors="ignore")
            if not data.strip():
                return {}
            return json.loads(data)
        except FileNotFoundError:
            return {}
        except Exception:
            return {}

    # Watcher API
    def start_watcher(self) -> None:
        if self._watcher_thread and self._watcher_thread.is_alive():
            return
        self._watcher_stop = threading.Event()
        initial_delay = random.randint(self._election_min_ms, self._election_max_ms) / 1000.0

        def _loop() -> None:
            end = time.time() + initial_delay
            while time.time() < end and not self._watcher_stop.is_set():
                time.sleep(0.01)
            while not self._watcher_stop.is_set():
                try:
                    # File-based control: <lock>.ctrl can request demotion
                    self._check_control_file()
                    if self._role == "RO":
                        _ = self.attempt_promote()
                    if self._role == "RW":
                        self.renew()
                except Exception:
                    pass
                time.sleep(max(0.05, self._heartbeat_ms / 1000.0))

        t = threading.Thread(target=_loop, name="LockManagerWatcher", daemon=True)
        self._watcher_thread = t
        t.start()

    def stop_watcher(self) -> None:
        if self._watcher_stop is not None:
            self._watcher_stop.set()
        if self._watcher_thread and self._watcher_thread.is_alive():
            try:
                self._watcher_thread.join(timeout=1.5)
            except Exception:
                pass
        self._watcher_thread = None
        self._watcher_stop = None

    def get_role(self) -> Optional[LockRole]:
        return self._role

    # ---- File-based control helpers ----
    def _ctrl_path(self) -> Path:
        return Path(str(self._lock_path) + ".ctrl")

    def _check_control_file(self) -> None:
        # Only the current RW leader should act on control file
        # Hardening: gate control-file demotion to test/MCP mode only to avoid
        # accidental production step-down via stray files.
        try:
            if not (
                os.getenv("CHUNKHOUND_MCP_MODE") == "1"
                or os.getenv("CH_TEST_ALLOW_CTRL") == "1"
            ):
                return
        except Exception:
            # If env probing fails for any reason, err on the safe side and
            # ignore control-file demotion outside explicit test mode.
            return
        if self._role != "RW":
            return
        ctrl = self._ctrl_path()
        if not ctrl.exists():
            return
        try:
            data = ctrl.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            data = ""
        data_lower = data.lower()
        should_demote = False
        if "demote" in data_lower:
            # demote now or after timeout
            after_ms = None
            for token in data_lower.replace("\n", ",").split(","):
                token = token.strip()
                if token.startswith("demote_after_ms="):
                    try:
                        after_ms = float(token.split("=", 1)[1])
                    except Exception:
                        after_ms = None
            if after_ms is None:
                should_demote = True
            else:
                if self._acquired_at is None:
                    should_demote = True
                else:
                    elapsed_ms = (time.time() - self._acquired_at) * 1000.0
                    should_demote = elapsed_ms >= after_ms

        if should_demote:
            # Release exclusive lock and step down to RO
            try:
                if self._fh is not None:
                    self._unlock_exclusive(self._fh)
                    try:
                        self._fh.close()
                    except Exception:
                        pass
            except Exception:
                pass
            self._fh = None
            self._role = "RO"
            try:
                ctrl.unlink(missing_ok=True)  # type: ignore[arg-type]
            except Exception:
                pass

    # Administrative/test helper: force demotion to RO by releasing exclusive lock
    # if held. No-op if we are not the leader.
    def force_demote(self) -> None:
        try:
            if self._role == "RW" and self._fh is not None:
                try:
                    self._unlock_exclusive(self._fh)
                except Exception:
                    pass
                try:
                    self._fh.close()
                except Exception:
                    pass
                self._fh = None
                self._role = "RO"
        except Exception:
            # Best-effort; must never crash callers
            pass

    # Internals
    def _next_epoch(self) -> int:
        prev = self.read_meta() or {}
        try:
            old_epoch = int(prev.get("epoch", 0))
        except Exception:
            old_epoch = 0
        return old_epoch + 1

    def _write_initial_meta(self, *, epoch: int) -> None:
        meta = {
            "pid": os.getpid(),
            "ppid": os.getppid(),
            "host_id": self._host_id(),
            "instance_id": self._instance_id,
            "epoch": epoch,
            "start_ts": time.time(),
            "last_heartbeat_ts": time.time(),
            "lease_ttl_ms": self._lease_ttl_ms,
            "version": 1,
        }
        self._write_meta(meta)

    def _write_meta(self, meta: dict) -> None:
        if self._fh is None:
            try:
                with open(self._lock_path, "r+b", buffering=0) as fh:
                    self._write_to_handle(fh, meta)
                return
            except Exception:
                return
        self._write_to_handle(self._fh, meta)

    def _write_to_handle(self, fh, meta: dict) -> None:
        data = (json.dumps(meta, separators=(",", ":")) + "\n").encode("utf-8")
        try:
            fh.seek(0)
            fh.truncate(0)
            fh.write(data)
            fh.flush()
            try:
                os.fsync(fh.fileno())
            except Exception:
                pass
        except Exception:
            pass

    def _try_lock_exclusive(self, fh) -> bool:
        if _IS_WINDOWS:  # pragma: no cover
            try:
                fh.seek(0)
                msvcrt.locking(fh.fileno(), msvcrt.LK_NBLCK, 1)
                return True
            except OSError:
                return False
        else:
            try:
                fcntl.flock(fh, fcntl.LOCK_EX | fcntl.LOCK_NB)
                return True
            except OSError as e:
                if e.errno in (errno.EACCES, errno.EAGAIN):
                    return False
                raise

    def _unlock_exclusive(self, fh) -> None:
        if _IS_WINDOWS:  # pragma: no cover
            try:
                fh.seek(0)
                msvcrt.locking(fh.fileno(), msvcrt.LK_UNLCK, 1)
            except Exception:
                pass
        else:
            try:
                fcntl.flock(fh, fcntl.LOCK_UN)
            except Exception:
                pass

    def _host_id(self) -> str:
        try:
            return os.uname().nodename  # type: ignore[attr-defined]
        except Exception:
            return os.environ.get("COMPUTERNAME", "unknown-host")
