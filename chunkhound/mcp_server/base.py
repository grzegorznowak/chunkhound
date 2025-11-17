"""Base class for MCP servers providing common initialization and lifecycle management.

This module provides a base class that handles:
- Service initialization (database, embeddings)
- Configuration validation
- Lifecycle management (startup/shutdown)
- Common error handling patterns

Architecture Note: Both stdio and HTTP servers inherit from this base
to ensure consistent initialization while respecting protocol-specific constraints.
"""

import asyncio
import os
from abc import ABC, abstractmethod
from datetime import datetime
from pathlib import Path
from typing import Any

from chunkhound.core.config import EmbeddingProviderFactory
from chunkhound.core.config.config import Config
from chunkhound.database_factory import DatabaseServices, create_services
from chunkhound.embeddings import EmbeddingManager
from chunkhound.llm_manager import LLMManager
from chunkhound.services.directory_indexing_service import DirectoryIndexingService
from chunkhound.services.realtime_indexing_service import RealtimeIndexingService


class MCPServerBase(ABC):
    """Base class for MCP server implementations.

    Provides common initialization, configuration validation, and lifecycle
    management for both stdio and HTTP server variants.

    Subclasses must implement:
    - _register_tools(): Register protocol-specific tool handlers
    - run(): Main server execution loop
    """

    def __init__(self, config: Config, debug_mode: bool = False, args: Any = None):
        """Initialize base MCP server.

        Args:
            config: Validated configuration object
            debug_mode: Enable debug logging to stderr
            args: Original CLI arguments for direct path access
        """
        self.config = config
        self.args = args  # Store original CLI args for direct path access
        self.debug_mode = debug_mode or os.getenv("CHUNKHOUND_DEBUG", "").lower() in (
            "true",
            "1",
            "yes",
        )

        # Service components - initialized lazily or eagerly based on subclass
        self.services: DatabaseServices | None = None
        self.embedding_manager: EmbeddingManager | None = None
        self.llm_manager: LLMManager | None = None
        self.realtime_indexing: RealtimeIndexingService | None = None
        self._role_monitor_task: asyncio.Task | None = None
        self._role_monitor_stop: asyncio.Event | None = None
        self._last_provider_role: str | None = None
        # Test-only cache for demotion hook
        self._test_demote_after_ms: float | None = None

        # Initialization state
        self._initialized = False
        self._init_lock = asyncio.Lock()

        # Scan progress tracking
        self._scan_complete = False
        self._scan_progress = {
            "files_processed": 0,
            "chunks_created": 0,
            "is_scanning": False,
            "scan_started_at": None,
            "scan_completed_at": None,
        }

        # Set MCP mode to suppress stderr output that interferes with JSON-RPC
        os.environ["CHUNKHOUND_MCP_MODE"] = "1"

    def debug_log(self, message: str) -> None:
        """Log debug message to file if debug mode is enabled."""
        if self.debug_mode:
            # Write to debug file instead of stderr to preserve JSON-RPC protocol
            debug_file = os.getenv(
                "CHUNKHOUND_DEBUG_FILE", "/tmp/chunkhound_mcp_debug.log"
            )
            try:
                with open(debug_file, "a") as f:
                    from datetime import datetime

                    timestamp = datetime.now().isoformat()
                    f.write(f"[{timestamp}] [MCP] {message}\n")
                    f.flush()
            except Exception:
                # Silently fail if we can't write to debug file
                pass

    async def initialize(self) -> None:
        """Initialize services and database connection.

        This method is idempotent - safe to call multiple times.
        Uses locking to ensure thread-safe initialization.

        Raises:
            ValueError: If required configuration is missing
            Exception: If services fail to initialize
        """
        async with self._init_lock:
            if self._initialized:
                return

            self.debug_log("Starting service initialization")
            # Test-only: record server start timestamp for delayed demotion hooks
            try:
                if os.getenv("CHUNKHOUND_MCP_MODE") == "1" and not os.getenv("CH_TEST_DEMOTE_START_TS"):
                    import time as _time
                    os.environ["CH_TEST_DEMOTE_START_TS"] = str(_time.time())
            except Exception:
                pass

            # Validate database configuration
            if not self.config.database or not self.config.database.path:
                raise ValueError("Database configuration not initialized")

            db_path = Path(self.config.database.path)
            db_path.parent.mkdir(parents=True, exist_ok=True)

            # Initialize embedding manager
            self.embedding_manager = EmbeddingManager()

            # Setup embedding provider (optional - continue if it fails)
            try:
                if self.config.embedding:
                    provider = EmbeddingProviderFactory.create_provider(
                        self.config.embedding
                    )
                    self.embedding_manager.register_provider(provider, set_default=True)
                    self.debug_log(
                        f"Embedding provider registered: {self.config.embedding.provider}"
                    )
            except ValueError as e:
                # API key or configuration issue - expected for search-only usage
                self.debug_log(f"Embedding provider setup skipped: {e}")
            except Exception as e:
                # Unexpected error - log but continue
                self.debug_log(f"Unexpected error setting up embedding provider: {e}")

            # Initialize LLM manager with dual providers (optional - continue if it fails)
            try:
                if self.config.llm:
                    utility_config, synthesis_config = self.config.llm.get_provider_configs()
                    self.llm_manager = LLMManager(utility_config, synthesis_config)
                    self.debug_log(
                        f"LLM providers registered: {self.config.llm.provider} "
                        f"(utility: {utility_config['model']}, synthesis: {synthesis_config['model']})"
                    )
            except ValueError as e:
                # API key or configuration issue - expected if LLM not needed
                self.debug_log(f"LLM provider setup skipped: {e}")
            except Exception as e:
                # Unexpected error - log but continue
                self.debug_log(f"Unexpected error setting up LLM provider: {e}")

            # Create services using unified factory (lazy connect for fast init)
            self.services = create_services(
                db_path=db_path,
                config=self.config,
                embedding_manager=self.embedding_manager,
            )

            # Enable provider auto-role (RW/RO) using a lock derived from the DB path
            try:
                provider = self.services.provider
                lock_path = str(Path(self.config.database.path).with_suffix(Path(self.config.database.path).suffix + ".rw.lock"))
                if hasattr(provider, "enable_auto_role"):
                    provider.enable_auto_role(lock_path=lock_path)
                    # Determine initial role synchronously to guide subsequent
                    # startup behavior (avoids RO followers trying to connect).
                    try:
                        if hasattr(provider, "try_acquire_role_only"):
                            provider.try_acquire_role_only()
                    except Exception:
                        pass
            except Exception:
                # Best-effort: providers without auto-role support will ignore this
                pass

            # Determine target path for scanning and watching
            if self.args and hasattr(self.args, "path"):
                target_path = Path(self.args.path)
                self.debug_log(f"Using direct path from args: {target_path}")
            else:
                # Fallback to config resolution (shouldn't happen in normal usage)
                target_path = self.config.target_dir or db_path.parent.parent
                self.debug_log(f"Using fallback path resolution: {target_path}")

            # Mark as initialized immediately (tools available)
            self._initialized = True
            self.debug_log("Service initialization complete")

            # Respect read-only / skip-indexing mode (via ENV or args)
            if self._should_skip_indexing():
                # Do not start realtime watcher or initial scan
                # Leave scan_progress as sentinel values (no timestamps)
                self._scan_progress["is_scanning"] = False
                self._scan_progress.setdefault("scan_error", None)
                self.debug_log("MCP read-only: startup indexing skipped by configuration")

                # In skip-indexing mode, do not hold DB connections open. Tools will
                # manage short-lived connections as needed (RO backoff will apply).

            else:
                # Defer DB connect + realtime start to background so initialize is fast
                asyncio.create_task(self._deferred_connect_and_start(target_path))

    async def _deferred_connect_and_start(self, target_path: Path) -> None:
        """Connect DB and start realtime monitoring in background."""
        try:
            # Ensure services exist
            if not self.services:
                return
            # Determine role first without opening DB connections to avoid
            # cross-process DuckDB lock conflicts for RO followers.
            role = None
            try:
                if hasattr(self.services.provider, "try_acquire_role_only"):
                    role = self.services.provider.try_acquire_role_only()
            except Exception:
                role = None

            # Connect to database lazily only if we are leader (RW)
            if not self.services.provider.is_connected and role != "RO":
                self.services.provider.connect()

            # Start provider role watcher if available
            try:
                if hasattr(self.services.provider, "start_role_watcher"):
                    self.services.provider.start_role_watcher()
            except Exception:
                pass

            # Always start MCP-level role monitor so initial RO can detect promotions
            self._start_role_monitor(target_path)

            # Respect skip-indexing and provider role (RO -> read-only startup for this process)
            if self._should_skip_indexing():
                self.debug_log("Skip-indexing active: not starting realtime or initial scan")
                return
            try:
                # Re-read role in case it was determined during connect
                role = getattr(self.services.provider, "get_role", lambda: role)() or role
                if role == "RO":
                    self.debug_log("Provider role=RO: skipping realtime watcher and initial scan")
                    return
            except Exception:
                # If role not available, proceed as RW by default
                pass

            # Start real-time indexing service (RW only)
            self.debug_log("Starting real-time indexing service (deferred)")
            self.realtime_indexing = RealtimeIndexingService(
                self.services, self.config, debug_sink=self.debug_log
            )
            monitoring_task = asyncio.create_task(
                self.realtime_indexing.start(target_path)
            )
            # Schedule background scan AFTER monitoring is confirmed ready
            asyncio.create_task(
                self._coordinated_initial_scan(target_path, monitoring_task)
            )
        except Exception as e:
            self.debug_log(f"Deferred connect/start failed: {e}")

    def _should_skip_indexing(self) -> bool:
        """Determine whether startup indexing should be skipped.

        Honors environment flag CHUNKHOUND_MCP__SKIP_INDEXING and optional CLI args
        stored on self.args (if present).
        """
        # ENV wins
        env_flag = os.getenv("CHUNKHOUND_MCP__SKIP_INDEXING", "").lower()
        if env_flag in ("1", "true", "yes", "on"):  # common truthy values
            return True

        # Global RW disable in MCP implies no indexing writes
        try:
            from chunkhound.utils.mcp_env import rw_disabled

            if rw_disabled():
                return True
        except Exception:
            pass

        # Optional CLI flag (if wired up by parsers)
        try:
            if self.args is not None and getattr(self.args, "skip_indexing", False):
                return True
        except Exception:
            pass
        return False

    def _index_on_promotion_enabled(self) -> bool:
        """Return True if role promotion should start realtime watcher.

        Default OFF. Enable with CHUNKHOUND_MCP__INDEX_ON_PROMOTION truthy.
        """
        return os.getenv("CHUNKHOUND_MCP__INDEX_ON_PROMOTION", "").lower() in (
            "1",
            "true",
            "yes",
            "on",
        )

    def _start_role_monitor(self, target_path: Path) -> None:
        """Start background task to react to provider role transitions."""
        if self._role_monitor_task and not self._role_monitor_task.done():
            return
        self._role_monitor_stop = asyncio.Event()

        async def _loop() -> None:
            try:
                # Initialize last role
                try:
                    self._last_provider_role = (
                        getattr(self.services.provider, "get_role", lambda: None)()
                        if self.services
                        else None
                    )
                except Exception:
                    self._last_provider_role = None
                start_ts = datetime.now().timestamp()

                while not self._role_monitor_stop.is_set():
                    await asyncio.sleep(0.2)
                    try:
                        # Nudge provider to refresh/acquire role without opening DB connections
                        try:
                            tr = getattr(self.services.provider, "try_acquire_role_only", None)
                            if callable(tr):
                                tr()
                        except Exception:
                            pass
                        role = (
                            getattr(self.services.provider, "get_role", lambda: None)()
                            if self.services
                            else None
                        )
                    except Exception:
                        role = None

                    # Test-only hook: force demotion after N ms (used in e2e tests)
                    # The test may set the env var after the subprocess starts. To
                    # accommodate that, on POSIX try to read the variable from ancestor
                    # processes' environments. Cache the first observed value.
                    try:
                        if self._test_demote_after_ms is None:
                            env_val = os.getenv("CH_TEST_DEMOTE_AFTER_MS")
                            if env_val is not None:
                                self._test_demote_after_ms = float(env_val)
                            else:
                                # POSIX-only ancestor scan (best-effort)
                                if os.name == "posix":
                                    try:
                                        import pathlib
                                        ppid = os.getppid()
                                        seen = set()
                                        depth = 0
                                        while ppid > 1 and depth < 6 and ppid not in seen:
                                            seen.add(ppid)
                                            env_path = pathlib.Path("/proc") / str(ppid) / "environ"
                                            try:
                                                data = env_path.read_bytes()
                                                for entry in data.split(b"\x00"):
                                                    if entry.startswith(b"CH_TEST_DEMOTE_AFTER_MS="):
                                                        try:
                                                            self._test_demote_after_ms = float(entry.split(b"=",1)[1].decode("utf-8"))
                                                        except Exception:
                                                            pass
                                                        break
                                            except Exception:
                                                pass
                                            # ascend
                                            try:
                                                stat = (pathlib.Path("/proc")/str(ppid)/"status").read_text()
                                                for line in stat.splitlines():
                                                    if line.startswith("PPid:"):
                                                        ppid = int(line.split()[1])
                                                        break
                                                else:
                                                    break
                                            except Exception:
                                                break
                                            depth += 1
                                    except Exception:
                                        pass

                        test_after_ms = self._test_demote_after_ms
                        if test_after_ms is not None and self._last_provider_role == "RW":
                            if (datetime.now().timestamp() - start_ts) * 1000.0 >= float(test_after_ms):
                                applier = getattr(self.services.provider, "_apply_role_change", None)
                                if callable(applier):
                                    applier("RO")
                                    role = "RO"
                                    # One-shot
                                    self._test_demote_after_ms = None
                                    try:
                                        os.environ.pop("CH_TEST_DEMOTE_AFTER_MS", None)
                                    except Exception:
                                        pass
                    except Exception:
                        pass

                    if role != self._last_provider_role:
                        self.debug_log(f"Provider role changed: {self._last_provider_role} -> {role}")
                        # Handle transitions
                        # Promotion: start realtime on any transition into RW
                        if role == "RW" and self._last_provider_role != "RW":
                            if self._index_on_promotion_enabled():
                                # Start ONLY realtime watcher (no initial scan)
                                await self._ensure_realtime_started(target_path)
                        # Degradation: stop watcher when entering RO
                        elif role == "RO" and (self._last_provider_role != "RO"):
                            await self._ensure_realtime_stopped()
                        self._last_provider_role = role
            except Exception as e:
                self.debug_log(f"Role monitor error: {e}")

        self._role_monitor_task = asyncio.create_task(_loop())

    async def _ensure_realtime_started(self, target_path: Path) -> None:
        if self.realtime_indexing is None:
            self.debug_log("Starting realtime watcher due to promotion (no catch-up scan)")
            self.realtime_indexing = RealtimeIndexingService(
                self.services, self.config, debug_sink=self.debug_log
            )
            task = asyncio.create_task(self.realtime_indexing.start(target_path))
            # Enable short burst mode to prioritize immediate ingestion post-promotion
            try:
                burst_env = os.getenv("CHUNKHOUND_MCP__PROMOTION_WINDOW_S", "")
                burst_window = float(burst_env) if burst_env else 4.0
            except Exception:
                burst_window = 4.0
            try:
                self.realtime_indexing.enable_burst_mode(burst_window)
            except Exception:
                pass
            # Fire-and-forget staggered pokes to catch edits that occur very soon
            # after promotion even before full readiness is reported.
            async def _staggered_pokes() -> None:
                try:
                    # Add dense early pokes plus a slightly longer tail to
                    # cover files created just-after promotion without a catch-up.
                    for delay in (0.0, 0.25, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0):
                        await asyncio.sleep(delay)
                        if self.realtime_indexing is not None:
                            try:
                                # Widen cutoff window to improve chances of grabbing
                                # very recent files in races around promotion.
                                await self.realtime_indexing.poke_for_recent_files(seconds=10.0)
                            except Exception:
                                pass
                except Exception:
                    pass
            asyncio.create_task(_staggered_pokes())
            # Schedule automatic burst disable after window
            async def _auto_disable():
                try:
                    await asyncio.sleep(max(0.2, burst_window))
                    if self.realtime_indexing is not None:
                        self.realtime_indexing.disable_burst_mode()
                except Exception:
                    pass
            asyncio.create_task(_auto_disable())
            # Wait briefly for monitoring to become ready so immediate file
            # creations after promotion are observed by the watcher.
            try:
                # Allow a slightly longer window for watcher readiness after promotion
                await asyncio.wait_for(
                    self.realtime_indexing.wait_for_monitoring_ready(timeout=6.0),
                    timeout=6.0,
                )
                # After ready, poke for recent files to minimize race conditions
                await asyncio.wait_for(
                    self.realtime_indexing.poke_for_recent_files(seconds=10.0),
                    timeout=4.0,
                )
                # Schedule one extra post-ready poke shortly after to catch files
                # created right after readiness flips.
                async def _post_ready_extra():
                    try:
                        await asyncio.sleep(1.5)
                        if self.realtime_indexing is not None:
                            await self.realtime_indexing.poke_for_recent_files(seconds=10.0)
                    except Exception:
                        pass
                asyncio.create_task(_post_ready_extra())
            except Exception:
                # Best-effort: continue even if not ready; polling fallback may catch up
                pass

    async def _ensure_realtime_stopped(self) -> None:
        if self.realtime_indexing is not None:
            self.debug_log("Stopping realtime watcher due to degradation to RO")
            try:
                await self.realtime_indexing.stop()
            except Exception:
                pass
            self.realtime_indexing = None

    async def _coordinated_initial_scan(
        self, target_path: Path, monitoring_task: asyncio.Task
    ) -> None:
        """Perform initial scan after monitoring is confirmed ready."""
        try:
            # Wait for monitoring to be ready (with timeout)
            await asyncio.wait_for(
                self.realtime_indexing.monitoring_ready.wait(), timeout=10.0
            )
            self.debug_log("Monitoring confirmed ready, starting initial scan")

            # Add small delay to ensure any startup files are captured by monitoring
            await asyncio.sleep(1.0)

            # Now perform the initial scan
            self._scan_progress["is_scanning"] = True
            self._scan_progress["scan_started_at"] = datetime.now().isoformat()
            await self._background_initial_scan(target_path)

        except asyncio.TimeoutError:
            self.debug_log(
                "Monitoring setup timeout - proceeding with initial scan anyway"
            )
            # Still do the scan even if monitoring isn't ready
            self._scan_progress["is_scanning"] = True
            self._scan_progress["scan_started_at"] = datetime.now().isoformat()
            await self._background_initial_scan(target_path)

    async def _background_initial_scan(self, target_path: Path) -> None:
        """Perform initial directory scan in background without blocking startup."""
        try:
            self.debug_log("Starting background initial directory scan")

            # Progress callback to update scan state
            def progress_callback(message: str):
                # Parse progress messages to update counters
                if "files processed" in message:
                    # Extract numbers from progress messages
                    import re

                    match = re.search(r"(\d+) files processed.*?(\d+) chunks", message)
                    if match:
                        self._scan_progress["files_processed"] = int(match.group(1))
                        self._scan_progress["chunks_created"] = int(match.group(2))
                self.debug_log(message)

            # Create indexing service for background scan
            indexing_service = DirectoryIndexingService(
                indexing_coordinator=self.services.indexing_coordinator,
                config=self.config,
                progress_callback=progress_callback,
            )

            # Perform scan with lower priority
            stats = await indexing_service.process_directory(
                target_path, no_embeddings=False
            )

            # Update final stats
            self._scan_progress.update(
                {
                    "files_processed": stats.files_processed,
                    "chunks_created": stats.chunks_created,
                    "is_scanning": False,
                    "scan_completed_at": datetime.now().isoformat(),
                }
            )
            self._scan_complete = True

            # Optional: trigger idle disconnect right after initial scan to allow
            # opportunistic RO readers a window to access the DB. Gate under THIN_RW.
            try:
                from chunkhound.utils.mcp_env import thin_rw_enabled, get_idle_disconnect_ms

                if thin_rw_enabled():
                    idle_ms = get_idle_disconnect_ms()
                    if idle_ms > 0 and self.services and hasattr(self.services.provider, "close_connection_only"):
                        try:
                            self.services.provider.close_connection_only()  # type: ignore[attr-defined]
                        except Exception:
                            pass
            except Exception:
                pass

            self.debug_log(
                f"Background scan completed: {stats.files_processed} files, {stats.chunks_created} chunks"
            )

        except Exception as e:
            self.debug_log(f"Background initial scan failed: {e}")
            self._scan_progress["is_scanning"] = False
            self._scan_progress["scan_error"] = str(e)

    async def cleanup(self) -> None:
        """Clean up resources and close database connection.

        This method is idempotent - safe to call multiple times.
        """
        # Stop real-time indexing first
        if self.realtime_indexing:
            self.debug_log("Stopping real-time indexing service")
            await self.realtime_indexing.stop()

        # Stop role monitor
        if self._role_monitor_task:
            if self._role_monitor_stop:
                self._role_monitor_stop.set()
            try:
                await asyncio.wait_for(self._role_monitor_task, timeout=2.0)
            except Exception:
                pass
            self._role_monitor_task = None
            self._role_monitor_stop = None

        if self.services and self.services.provider.is_connected:
            self.debug_log("Closing database connection")
            # Use new close() method for proper cleanup, with fallback to disconnect()
            if hasattr(self.services.provider, "close"):
                self.services.provider.close()
            else:
                self.services.provider.disconnect()
            self._initialized = False

    def ensure_services(self) -> DatabaseServices:
        """Ensure services are initialized and return them.

        Returns:
            DatabaseServices instance

        Raises:
            RuntimeError: If services are not initialized
        """
        if not self.services:
            raise RuntimeError("Services not initialized. Call initialize() first.")

        # Ensure database connection is active
        if not self.services.provider.is_connected:
            self.services.provider.connect()

        return self.services

    def ensure_embedding_manager(self) -> EmbeddingManager:
        """Ensure embedding manager is available and has providers.

        Returns:
            EmbeddingManager instance

        Raises:
            RuntimeError: If no embedding providers are available
        """
        if not self.embedding_manager or not self.embedding_manager.list_providers():
            raise RuntimeError(
                "No embedding providers available. Configure an embedding provider "
                "in .chunkhound.json or set CHUNKHOUND_EMBEDDING__API_KEY environment variable."
            )
        return self.embedding_manager

    @abstractmethod
    def _register_tools(self) -> None:
        """Register tools with the server implementation.

        Subclasses must implement this to register tools using their
        protocol-specific decorators/patterns.
        """
        pass

    @abstractmethod
    async def run(self) -> None:
        """Run the server.

        Subclasses must implement their protocol-specific server loop.
        """
        pass
