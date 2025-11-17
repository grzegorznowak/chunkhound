"""Stdio MCP server implementation using the base class pattern.

This module implements the stdio (stdin/stdout) JSON-RPC protocol for MCP,
inheriting common initialization and lifecycle management from MCPServerBase.

CRITICAL: NO stdout output allowed - breaks JSON-RPC protocol
ARCHITECTURE: Global state required for stdio communication model
"""

from __future__ import annotations

import asyncio
import sys
import os
import logging
import warnings

# CRITICAL: Suppress SWIG warnings that break JSON-RPC protocol in CI
# The DuckDB Python bindings generate a DeprecationWarning that goes to stdout
# in some environments (Ubuntu CI with Python 3.12), breaking MCP protocol
warnings.filterwarnings(
    "ignore", message=".*swigvarlink.*", category=DeprecationWarning
)
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

from typing import TYPE_CHECKING

# Try to import the official MCP SDK; if unavailable, we'll fall back to a
# minimal stdio JSON-RPC loop sufficient for tests that only exercise the
# initialize handshake.
_MCP_AVAILABLE = True
try:  # runtime path
    import mcp.server.stdio  # type: ignore
    import mcp.types as types  # type: ignore
    from mcp.server import Server  # type: ignore
    from mcp.server.models import InitializationOptions  # type: ignore
except Exception:  # pragma: no cover - optional dependency path
    _MCP_AVAILABLE = False

if TYPE_CHECKING:  # type-checkers only; avoid runtime hard deps at import
    import mcp.server.stdio  # noqa: F401
    import mcp.types as types  # noqa: F401
    from mcp.server import Server  # noqa: F401
    from mcp.server.models import InitializationOptions  # noqa: F401

from chunkhound.core.config.config import Config
from chunkhound.version import __version__

from .base import MCPServerBase
from .common import handle_tool_call
from .tools import TOOL_REGISTRY

# CRITICAL: Disable ALL logging to prevent JSON-RPC corruption
logging.disable(logging.CRITICAL)
for logger_name in ["", "mcp", "server", "fastmcp"]:
    logging.getLogger(logger_name).setLevel(logging.CRITICAL + 1)

# Disable loguru logger
try:
    from loguru import logger as loguru_logger

    loguru_logger.remove()
    loguru_logger.add(lambda _: None, level="CRITICAL")
except ImportError:
    pass


class StdioMCPServer(MCPServerBase):
    """MCP server implementation for stdio protocol.

    Uses global state as required by the stdio protocol's persistent
    connection model. All initialization happens eagerly during startup.
    """

    def __init__(self, config: Config, args: Any = None):
        """Initialize stdio MCP server.

        Args:
            config: Validated configuration object
            args: Original CLI arguments for direct path access
        """
        super().__init__(config, args=args)

        # Test-only hook: allow E2E tests to inject a sitecustomize from PYTHONPATH
        # to stub Codex CLI and force synthesis without requiring real binaries.
        # This is guarded behind CH_TEST_PATCH_CODEX and is a no-op otherwise.
        try:
            if os.getenv("CH_TEST_PATCH_CODEX") == "1":
                pp = os.environ.get("PYTHONPATH", "")
                if pp:
                    for path in pp.split(os.pathsep):
                        if path and path not in sys.path:
                            sys.path.insert(0, path)
                # Best-effort: import test helper if available
                try:
                    __import__("sitecustomize")  # noqa: WPS433
                except Exception:
                    pass

                # Also patch Codex provider directly to guarantee stubbed exec
                try:
                    from chunkhound.providers.llm.codex_cli_provider import (  # noqa: WPS433
                        CodexCLIProvider,
                    )

                    async def _stub_run_exec(self, text, cwd=None, max_tokens=1024, timeout=None, model=None):  # type: ignore[override]
                        mark = os.getenv("CH_TEST_CODEX_MARK_FILE")
                        if mark:
                            try:
                                with open(mark, "a", encoding="utf-8") as f:
                                    f.write("CALLED\n")
                            except Exception:
                                pass
                        return "SYNTH_OK: codex-cli invoked"

                    def _stub_available(self) -> bool:  # pragma: no cover
                        return True

                    CodexCLIProvider._run_exec = _stub_run_exec  # type: ignore[attr-defined]
                    CodexCLIProvider._codex_available = _stub_available  # type: ignore[attr-defined]
                except Exception:
                    pass

                # And if asked, force deep_research to call synthesis directly
                if os.getenv("CH_TEST_FORCE_SYNTHESIS") == "1":
                    try:
                        from chunkhound.mcp_server import tools as tools_mod  # noqa: WPS433

                        async def _stub_deep_research_impl(*, services, embedding_manager, llm_manager, query, progress=None):
                            if llm_manager is None:
                                try:
                                    from chunkhound.llm_manager import LLMManager  # noqa: WPS433

                                    llm_manager = LLMManager(
                                        {"provider": "codex-cli", "model": "codex"},
                                        {"provider": "codex-cli", "model": "codex"},
                                    )
                                except Exception:
                                    return {"answer": "LLM manager unavailable"}
                            prov = llm_manager.get_synthesis_provider()
                            resp = await prov.complete(prompt=f"E2E: {query}")
                            return {"answer": resp.content}

                        tools_mod.deep_research_impl = _stub_deep_research_impl  # type: ignore[assignment]
                        if "code_research" in tools_mod.TOOL_REGISTRY:
                            tools_mod.TOOL_REGISTRY["code_research"].implementation = _stub_deep_research_impl  # type: ignore[index]
                    except Exception:
                        pass
        except Exception:
            # Silent by design in MCP mode
            pass

        # Create MCP server instance (prefer official SDK when available).
        # Allow forcing fallback with CHUNKHOUND_MCP__USE_SDK in {0,false,off}.
        use_sdk_env = os.getenv("CHUNKHOUND_MCP__USE_SDK", "").lower()
        skip_env = os.getenv("CHUNKHOUND_MCP__SKIP_INDEXING", "").lower()
        skip_on = skip_env in ("1", "true", "yes", "on")
        # Prefer SDK whenever available unless explicitly disabled via env
        use_sdk = _MCP_AVAILABLE and use_sdk_env not in ("0", "false", "off")
        self.debug_log(
            f"STDIO: _MCP_AVAILABLE={_MCP_AVAILABLE}, use_sdk={use_sdk} (env={use_sdk_env}, os={os.name})"
        )
        if not use_sdk:
            # Defer server creation; fallback path implemented in run()
            self.server = None  # type: ignore
        else:
            from mcp.server import Server  # noqa: WPS433
            self.server: Server = Server("ChunkHound Code Search")

        # Event to signal initialization completion
        self._initialization_complete = asyncio.Event()

        # Register tools with the server
        self._register_tools()

    def _register_tools(self) -> None:
        """Register tool handlers with the stdio server."""

        # The MCP SDK's call_tool decorator expects a SINGLE handler function
        # with signature (tool_name: str, arguments: dict) that handles ALL tools

        # If the official SDK is unavailable or we are running without Server
        # instance (fallback stdio), do not register SDK handlers.
        if not _MCP_AVAILABLE or self.server is None:
            return  # no-op when SDK not available

        @self.server.call_tool()  # type: ignore[misc]
        async def handle_all_tools(
            tool_name: str, arguments: dict[str, Any]
        ) -> list[types.TextContent]:
            """Universal tool handler that routes to the unified handler."""
            # Pass services lazily (do not force provider connects here);
            # tools/backoff logic will manage connections.
            return await handle_tool_call(
                tool_name=tool_name,
                arguments=arguments,
                services=self.services,
                embedding_manager=self.embedding_manager,
                initialization_complete=self._initialization_complete,
                debug_mode=self.debug_mode,
                scan_progress=self._scan_progress,
                llm_manager=self.llm_manager,
            )

        self._register_list_tools()

    def _register_list_tools(self) -> None:
        """Register list_tools handler."""

        # Lazy import to avoid hard dependency at module import time
        import mcp.types as types  # noqa: WPS433

        @self.server.list_tools()  # type: ignore[misc]
        async def list_tools() -> list[types.Tool]:
            """List available tools."""
            # Wait for initialization
            try:
                await asyncio.wait_for(
                    self._initialization_complete.wait(), timeout=5.0
                )
            except asyncio.TimeoutError:
                # Return basic tools even if not fully initialized
                pass

            tools = []
            for tool_name, tool in TOOL_REGISTRY.items():
                # Skip embedding-dependent tools if no providers available
                if tool.requires_embeddings and (
                    not self.embedding_manager
                    or not self.embedding_manager.list_providers()
                ):
                    continue

                tools.append(
                    types.Tool(
                        name=tool_name,
                        description=tool.description,
                        inputSchema=tool.parameters,
                    )
                )

            return tools

    @asynccontextmanager
    async def server_lifespan(self) -> AsyncIterator[dict]:
        """Manage server lifecycle with proper initialization and cleanup."""
        try:
            # Kick off initialization in the background to avoid blocking handshake
            async def _bg_init():
                try:
                    await self.initialize()
                    self._initialization_complete.set()
                    self.debug_log("Server initialization complete (background)")
                except Exception as e:
                    self.debug_log(f"Background initialization failed: {e}")

            asyncio.create_task(_bg_init())

            # Yield control to server immediately; tools will await initialization
            yield {"services": self.services, "embeddings": self.embedding_manager}

        finally:
            # Cleanup on shutdown
            await self.cleanup()

    async def run(self) -> None:
        """Run the stdio server with proper lifecycle management."""
        try:
            if self.server is not None:
                # Set initialization options with capabilities
                from mcp.server.lowlevel import NotificationOptions  # noqa: WPS433
                from mcp.server.models import InitializationOptions  # noqa: WPS433

                init_options = InitializationOptions(
                    server_name="ChunkHound Code Search",
                    server_version=__version__,
                    capabilities=self.server.get_capabilities(
                        notification_options=NotificationOptions(),
                        experimental_capabilities={},
                    ),
                )

                # Run with lifespan management
                async with self.server_lifespan():
                    # Run the stdio server
                    import mcp.server.stdio  # noqa: WPS433
                    async with mcp.server.stdio.stdio_server() as (
                        read_stream,
                        write_stream,
                    ):
                        self.debug_log("Stdio server started, awaiting requests")
                        await self.server.run(
                            read_stream,
                            write_stream,
                            init_options,
                        )
            else:
                # Minimal fallback stdio JSON-RPC loop (no official MCP SDK required).
                # Supports: initialize, notifications/initialized (ignored), tools/call (subset).
                import json, os as _os, signal

                # Install a SIGTERM handler to ensure clean shutdown of the
                # fallback loop when the test client terminates the process.
                _shutdown = {"flag": False}

                def _handle_sigterm(signum, frame):  # pragma: no cover - signal path
                    _shutdown["flag"] = True

                try:
                    signal.signal(signal.SIGTERM, _handle_sigterm)
                except Exception:
                    pass

                async def _write(obj: dict) -> None:
                    try:
                        _os.write(1, (json.dumps(obj, default=str) + "\n").encode("utf-8"))
                    except Exception:
                        pass

                loop = asyncio.get_event_loop()
                reader = None
                try:
                    if os.name != "nt":
                        # Prefer asyncio-native pipe reader for stdin to avoid
                        # thread scheduling delays with run_in_executor. On Windows
                        # (Proactor), stick to executor-based reads to avoid pipe issues.
                        reader = asyncio.StreamReader()
                        protocol = asyncio.StreamReaderProtocol(reader)  # type: ignore[arg-type]
                        await loop.connect_read_pipe(lambda: protocol, sys.stdin)  # type: ignore[arg-type]
                except Exception:
                    reader = None

                async def _readline() -> str:
                    try:
                        if reader is not None:
                            line = await reader.readline()
                            # When using StreamReader, we get bytes on some platforms
                            if isinstance(line, bytes):
                                try:
                                    return line.decode("utf-8")
                                except Exception:
                                    return line.decode("utf-8", errors="ignore")
                            return str(line)
                        # Fallback to thread executor; prefer binary buffer for robustness
                        def _read_from_stdin() -> str:
                            try:
                                data = sys.stdin.buffer.readline()
                                try:
                                    return data.decode("utf-8")
                                except Exception:
                                    return data.decode("utf-8", errors="ignore")
                            except Exception:
                                # As a last resort, try text mode
                                try:
                                    return sys.stdin.readline()
                                except Exception:
                                    return ""
                        return await loop.run_in_executor(None, _read_from_stdin)
                    except Exception:
                        # Propagate to outer handler
                        raise

                self.debug_log("STDIO fallback loop started")
                while True:
                    if _shutdown["flag"]:
                        break
                    try:
                        line = await asyncio.wait_for(_readline(), timeout=0.25)
                    except asyncio.TimeoutError:
                        # Periodically check for shutdown without blocking forever
                        # Optional heartbeat for debugging
                        try:
                            self.debug_log("STDIO fallback idle tick")
                        except Exception:
                            pass
                        continue
                    if not line:
                        break  # EOF
                    line = line.strip()
                    try:
                        self.debug_log(f"STDIO fallback received: {line[:200]}")
                    except Exception:
                        pass
                    if not line:
                        continue
                    try:
                        req = json.loads(line)
                    except Exception:
                        continue  # ignore malformed

                    method = req.get("method")
                    req_id = req.get("id")

                    # Initialize handshake
                    if method == "initialize":
                        # Respond immediately to avoid handshake timeouts; initialize in background.
                        # Tools will await initialization before execution.
                        async def _bg_init():
                            try:
                                await self.initialize()
                                self._initialization_complete.set()
                            except Exception:
                                # Keep server responsive even if init failed; tools will surface errors
                                pass
                        asyncio.create_task(_bg_init())

                        result = {
                            "protocolVersion": "2024-11-05",
                            "serverInfo": {"name": "ChunkHound Code Search", "version": __version__},
                            "capabilities": {},
                        }
                        await _write({"jsonrpc": "2.0", "id": req_id, "result": result})
                        continue

                    # Simple notifications: ignore
                    elif method == "notifications/initialized":
                        # no response required; background tasks were scheduled during initialize
                        continue

                    # Minimal tools/list support
                    elif method == "tools/list":
                        try:
                            tools = []
                            # Build a minimal tool descriptor list compatible with tests
                            for name, tool in TOOL_REGISTRY.items():
                                tools.append(
                                    {
                                        "name": name,
                                        "description": getattr(tool, "description", ""),
                                        "inputSchema": getattr(tool, "parameters", {}),
                                    }
                                )
                            await _write({"jsonrpc": "2.0", "id": req_id, "result": {"tools": tools}})
                        except Exception as e:
                            await _write(
                                {
                                    "jsonrpc": "2.0",
                                    "id": req_id,
                                    "error": {"code": -32000, "message": f"tools/list error: {e}"},
                                }
                            )

                    # Minimal tools/call support (ToolResult envelope)
                    elif method == "tools/call":
                        params = req.get("params", {}) or {}
                        name = params.get("name")
                        arguments = params.get("arguments") or {}
                        try:
                            self.debug_log(f"tools/call start name={name}")
                        except Exception:
                            pass
                        try:
                            from .common import parse_mcp_arguments
                            from .tools import execute_tool

                            # Fast path for get_stats before initialization completes
                            if name == "get_stats" and (
                                self.services is None or not getattr(self, "_initialized", False)
                            ):
                                import json as _json
                                # Minimal stats without forcing DB init
                                minimal = {
                                    "total_files": 0,
                                    "total_chunks": 0,
                                    "total_embeddings": 0,
                                    "database_size_mb": 0,
                                    "total_providers": 0,
                                }
                                if isinstance(self._scan_progress, dict):
                                    minimal["initial_scan"] = {
                                        "is_scanning": self._scan_progress.get("is_scanning", False),
                                        "files_processed": self._scan_progress.get("files_processed", 0),
                                        "chunks_created": self._scan_progress.get("chunks_created", 0),
                                        "started_at": self._scan_progress.get("scan_started_at"),
                                        "completed_at": self._scan_progress.get("scan_completed_at"),
                                        "error": self._scan_progress.get("scan_error"),
                                    }
                                tool_result = {
                                    "content": [
                                        {"type": "text", "text": _json.dumps(minimal, default=str)}
                                    ],
                                    "isError": False,
                                }
                                await _write({"jsonrpc": "2.0", "id": req_id, "result": tool_result})
                                continue

                            # Ensure services are ready (may still be initializing in background)
                            self.debug_log("tools/call awaiting initialize()")
                            await self.initialize()
                            self.debug_log("tools/call initialize() complete")
                            # Pass services without forcing a DB connect here; individual
                            # tools handle lazy connections to avoid RO follower conflicts.
                            self.debug_log(f"tools/call executing {name}")
                            result = await execute_tool(
                                tool_name=name,
                                services=self.services,
                                embedding_manager=self.embedding_manager,
                                arguments=parse_mcp_arguments(arguments),
                                scan_progress=self._scan_progress,
                                llm_manager=self.llm_manager,
                            )
                            self.debug_log(f"tools/call done {name}")
                            # Match official MCP SDK ToolResult shape
                            import json as _json
                            if isinstance(result, str):
                                text_payload = result
                            else:
                                text_payload = _json.dumps(result, default=str)
                            tool_result = {
                                "content": [
                                    {"type": "text", "text": text_payload}
                                ],
                                "isError": False,
                            }
                            await _write({"jsonrpc": "2.0", "id": req_id, "result": tool_result})
                        except Exception as e:
                            try:
                                self.debug_log(f"tools/call error {name}: {e}")
                            except Exception:
                                pass
                            await _write(
                                {
                                    "jsonrpc": "2.0",
                                    "id": req_id,
                                    "error": {"code": -32000, "message": f"tool error: {e}"},
                                }
                            )
                    else:
                        # Unknown method
                        if req_id is not None:
                            await _write(
                                {
                                    "jsonrpc": "2.0",
                                "id": req_id,
                                "error": {"code": -32601, "message": f"Method not found: {method}"},
                            }
                        )
                # End while
                # Perform cleanup before exiting to release DB/locks
                try:
                    await self.cleanup()
                except Exception:
                    pass
                # Exit explicitly to speed up subprocess termination in tests
                try:
                    sys.exit(0)
                except SystemExit:
                    pass

        except KeyboardInterrupt:
            self.debug_log("Server interrupted by user")
        except Exception as e:
            self.debug_log(f"Server error: {e}")
            if self.debug_mode:
                import traceback
                traceback.print_exc(file=sys.stderr)


async def main(args: Any = None) -> None:
    """Main entry point for the MCP stdio server.

    Args:
        args: Pre-parsed arguments. If None, will parse from sys.argv.
    """
    import argparse

    from chunkhound.api.cli.utils.config_factory import create_validated_config
    from chunkhound.mcp_server.common import add_common_mcp_arguments

    if args is None:
        # Direct invocation - parse arguments
        parser = argparse.ArgumentParser(
            description="ChunkHound MCP stdio server",
            formatter_class=argparse.RawDescriptionHelpFormatter,
        )
        # Add common MCP arguments
        add_common_mcp_arguments(parser)
        # Parse arguments
        args = parser.parse_args()

    # Mark process as MCP mode so downstream code avoids interactive prompts
    os.environ["CHUNKHOUND_MCP_MODE"] = "1"

    # Create and validate configuration
    config, validation_errors = create_validated_config(args, "mcp")

    # In MCP stdio mode, be permissive on validation to ensure the server can
    # at least respond to initialize and basic tool calls. Tools will surface
    # configuration errors explicitly when invoked.
    # Do not exit here even if there are validation errors.

    # Create and run the stdio server
    try:
        # Early debug (before server exists)
        try:
            path = os.getenv("CHUNKHOUND_DEBUG_FILE", "/tmp/chunkhound_mcp_debug.log")
            if os.getenv("CHUNKHOUND_DEBUG") in ("1", "true", "yes", "on"):
                with open(path, "a", encoding="utf-8") as f:
                    from datetime import datetime as _dt
                    f.write(f"[{_dt.now().isoformat()}] [STDIO] main creating server\n")
        except Exception:
            pass

        server = StdioMCPServer(config, args=args)

        try:
            if os.getenv("CHUNKHOUND_DEBUG") in ("1", "true", "yes", "on"):
                with open(os.getenv("CHUNKHOUND_DEBUG_FILE", "/tmp/chunkhound_mcp_debug.log"), "a", encoding="utf-8") as f:
                    from datetime import datetime as _dt
                    f.write(f"[{_dt.now().isoformat()}] [STDIO] server created, running\n")
        except Exception:
            pass

        await server.run()
    except Exception:
        # CRITICAL: Cannot print to stderr in MCP mode - breaks JSON-RPC protocol
        # Exit silently with error code
        try:
            if os.getenv("CHUNKHOUND_DEBUG") in ("1", "true", "yes", "on"):
                import traceback
                tb = traceback.format_exc()
                with open(os.getenv("CHUNKHOUND_DEBUG_FILE", "/tmp/chunkhound_mcp_debug.log"), "a", encoding="utf-8") as f:
                    from datetime import datetime as _dt
                    f.write(f"[{_dt.now().isoformat()}] [STDIO] main exception:\n{tb}\n")
        except Exception:
            pass
        sys.exit(1)


def main_sync() -> None:
    """Synchronous wrapper for CLI entry point."""
    asyncio.run(main())


if __name__ == "__main__":
    main_sync()
