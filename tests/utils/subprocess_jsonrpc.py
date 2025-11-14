"""Robust JSON-RPC subprocess communication helper.

This module provides a deadlock-free JSON-RPC client for subprocess communication
following asyncio best practices. It uses a background reader task pattern to avoid
the common deadlock issues that occur when mixing stdin.write() with stdout.readline().

Key features:
- Background reader task for non-blocking stdout reading
- Request/response matching by JSON-RPC ID
- Per-request timeout handling
- Graceful error handling for subprocess crashes and malformed JSON
- Clean shutdown with proper task cancellation

See: https://docs.python.org/3/library/asyncio-subprocess.html#asyncio-subprocess-streams
"""

import asyncio
import json
import logging
import os
import time
from typing import Any, cast

logger = logging.getLogger(__name__)


def _env_true(val: str | None) -> bool:
    if not val:
        return False
    return str(val).strip().lower() in ("1", "true", "yes", "on")


def _dbg(msg: str) -> None:
    try:
        if not (_env_true(os.getenv("CH_TEST_JSONRPC_DEBUG")) or _env_true(os.getenv("CHUNKHOUND_DEBUG"))):
            return
        path = os.getenv("CH_TEST_JSONRPC_DEBUG_FILE") or os.getenv("CHUNKHOUND_DEBUG_FILE") or "/tmp/chunkhound_jsonrpc_debug.log"
        with open(path, "a", encoding="utf-8") as f:
            f.write(f"[{time.time():.6f}] [JSONRPC] {msg}\n")
            f.flush()
    except Exception:
        pass


class SubprocessJsonRpcError(Exception):
    """Base exception for JSON-RPC subprocess communication errors."""

    pass


class SubprocessCrashError(SubprocessJsonRpcError):
    """Raised when the subprocess terminates unexpectedly."""

    pass


class JsonRpcTimeoutError(SubprocessJsonRpcError):
    """Raised when a JSON-RPC request times out."""

    pass


class JsonRpcResponseError(SubprocessJsonRpcError):
    """Raised when a JSON-RPC response contains an error."""

    def __init__(self, code: int, message: str, data: dict[str, Any] | None = None):
        self.code = code
        self.message = message
        self.data = data
        super().__init__(f"JSON-RPC error {code}: {message}")


class SubprocessJsonRpcClient:
    """Robust JSON-RPC client for subprocess communication.

    This class implements a deadlock-free pattern for JSON-RPC communication
    with a subprocess using a background reader task that continuously reads
    from stdout and queues responses for retrieval.

    Example usage:
        proc = await asyncio.create_subprocess_exec(
            "uv", "run", "chunkhound", "mcp", "stdio",
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )

        client = SubprocessJsonRpcClient(proc)
        await client.start()

        try:
            response = await client.send_request(
                "initialize",
                {"protocolVersion": "2024-11-05"},
                timeout=5.0
            )
            await client.send_notification("initialized", {})
            result = await client.send_request(
                "tools/call",
                {"name": "search_regex", "arguments": {"pattern": "hello"}}
            )
        finally:
            await client.close()
    """

    def __init__(self, process: asyncio.subprocess.Process):
        """Initialize the JSON-RPC client.

        Args:
            process: The subprocess to communicate with. Must have stdin and
                stdout pipes.

        Raises:
            ValueError: If process doesn't have stdin or stdout pipes.
        """
        if process.stdin is None:
            raise ValueError("Process must have stdin pipe")
        if process.stdout is None:
            raise ValueError("Process must have stdout pipe")

        self._process = process
        self._reader_task: asyncio.Task[None] | None = None
        self._pending_requests: dict[int, asyncio.Future[dict[str, Any]]] = {}
        self._request_id_lock = asyncio.Lock()
        self._next_request_id = 1
        self._closed = False

    async def start(self) -> None:
        """Start the background reader task.

        This must be called before sending any requests.
        """
        if self._reader_task is not None:
            raise RuntimeError("Client already started")

        _dbg("client.start -> spawning reader task")
        self._reader_task = asyncio.create_task(self._read_responses())

    async def send_request(
        self, method: str, params: dict[str, Any] | None = None, timeout: float = 5.0
    ) -> dict[str, Any]:
        """Send a JSON-RPC request and wait for the response.

        Args:
            method: The JSON-RPC method name.
            params: The method parameters (optional).
            timeout: Maximum time to wait for response in seconds.

        Returns:
            The JSON-RPC result object.

        Raises:
            JsonRpcTimeoutError: If the request times out.
            JsonRpcResponseError: If the response contains an error.
            SubprocessCrashError: If the subprocess terminates unexpectedly.
            RuntimeError: If the client is not started or is closed.
        """
        if self._reader_task is None:
            raise RuntimeError("Client not started - call start() first")
        if self._closed:
            raise RuntimeError("Client is closed")

        # Generate unique request ID (async-safe)
        async with self._request_id_lock:
            request_id = self._next_request_id
            self._next_request_id += 1

        # Create future for response
        response_future: asyncio.Future[dict[str, Any]] = asyncio.Future()
        self._pending_requests[request_id] = response_future

        # Build and send request
        request = {
            "jsonrpc": "2.0",
            "id": request_id,
            "method": method,
        }
        if params is not None:
            request["params"] = params

        try:
            request_json = json.dumps(request) + "\n"
            _dbg(f"send_request id={request_id} method={method} bytes={len(request_json)}")
            # stdin is guaranteed to exist by __init__ check
            assert self._process.stdin is not None
            self._process.stdin.write(request_json.encode("utf-8"))
            _dbg(f"send_request write ok id={request_id}")
            try:
                await self._process.stdin.drain()
                _dbg(f"send_request drain ok id={request_id}")
            except (BrokenPipeError, ConnectionResetError) as e:
                # Subprocess crashed before we could send
                _dbg(f"send_request drain crash id={request_id} err={e}")
                raise SubprocessCrashError(
                    f"Subprocess crashed during request send: {e}"
                ) from e

            # Wait for response with timeout
            try:
                _dbg(f"awaiting response id={request_id} timeout={timeout}")
                response = await asyncio.wait_for(response_future, timeout=timeout)
                _dbg(f"response received id={request_id}")
            except asyncio.TimeoutError:
                # Clean up pending request
                self._pending_requests.pop(request_id, None)
                _dbg(f"timeout id={request_id} method={method} after={timeout}s")
                raise JsonRpcTimeoutError(
                    f"Request {method} (id={request_id}) timed out after {timeout}s"
                )

            # Check for JSON-RPC error
            if "error" in response:
                error = response["error"]
                raise JsonRpcResponseError(
                    code=error.get("code", -1),
                    message=error.get("message", "Unknown error"),
                    data=error.get("data"),
                )

            # Return result
            if "result" not in response:
                _dbg(f"malformed response id={request_id}: {response}")
                raise SubprocessJsonRpcError(
                    f"Response missing 'result' field: {response}"
                )

            result_obj = response["result"]
            # Compatibility: unwrap MCP ToolResult shape into a plain dict when possible
            # Expected shape from official MCP SDK: { content: [ { type: 'text', text: '{json}' } ], isError: false }
            if (
                method == "tools/call"
                and isinstance(result_obj, dict)
                and isinstance(result_obj.get("content"), list)
                and result_obj["content"]
                and isinstance(result_obj["content"][0], dict)
                and "text" in result_obj["content"][0]
            ):
                try:
                    text = result_obj["content"][0]["text"]
                    parsed = json.loads(text)
                    if isinstance(parsed, dict):
                        # Compatibility: return parsed fields while preserving original content
                        merged: dict[str, Any] = {**parsed}
                        merged.setdefault("content", result_obj["content"])  # expose ToolResult for tests expecting it
                        return merged
                except Exception:
                    # Fall through to returning raw result
                    pass

            return cast(dict[str, Any], result_obj)

        except Exception:
            # Clean up pending request on any error
            self._pending_requests.pop(request_id, None)
            raise

    async def send_notification(
        self, method: str, params: dict[str, Any] | None = None
    ) -> None:
        """Send a JSON-RPC notification (no response expected).

        Args:
            method: The JSON-RPC method name.
            params: The method parameters (optional).

        Raises:
            RuntimeError: If the client is not started or is closed.
        """
        if self._reader_task is None:
            raise RuntimeError("Client not started - call start() first")
        if self._closed:
            raise RuntimeError("Client is closed")

        # Build and send notification (no id field)
        notification: dict[str, Any] = {
            "jsonrpc": "2.0",
            "method": method,
        }
        if params is not None:
            notification["params"] = params

        notification_json = json.dumps(notification) + "\n"
        # stdin is guaranteed to exist by __init__ check
        assert self._process.stdin is not None
        _dbg(f"send_notification method={method} bytes={len(notification_json)}")
        self._process.stdin.write(notification_json.encode("utf-8"))
        try:
            await self._process.stdin.drain()
            _dbg(f"send_notification drain ok method={method}")
        except (BrokenPipeError, ConnectionResetError) as e:
            # Subprocess crashed before we could send
            _dbg(f"send_notification drain crash method={method} err={e}")
            raise SubprocessCrashError(
                f"Subprocess crashed during notification send: {e}"
            ) from e

    async def close(self) -> None:
        """Clean shutdown of the client and subprocess.

        This cancels the background reader task, terminates the subprocess,
        and ensures all resources are properly cleaned up.
        """
        if self._closed:
            return

        self._closed = True

        # Proactively terminate subprocess first to unblock reader fast
        if self._process.returncode is None:
            self._process.terminate()
            try:
                await asyncio.wait_for(self._process.wait(), timeout=1.5)
            except asyncio.TimeoutError:
                logger.warning("Subprocess didn't terminate, killing it")
                self._process.kill()
                try:
                    await asyncio.wait_for(self._process.wait(), timeout=1.5)
                except asyncio.TimeoutError:
                    logger.warning("Subprocess didn't die after SIGKILL; continuing shutdown")

        # Cancel background reader task with timeout tolerance
        if self._reader_task is not None:
            self._reader_task.cancel()
            try:
                await asyncio.wait_for(self._reader_task, timeout=0.5)
            except (asyncio.CancelledError, asyncio.TimeoutError):
                pass

        # Cancel all pending requests
        for future in self._pending_requests.values():
            if not future.done():
                future.cancel()
        self._pending_requests.clear()

    async def _read_responses(self) -> None:
        """Background task that continuously reads responses from stdout.

        This task runs until cancelled or the subprocess terminates.
        It reads JSON-RPC responses and matches them to pending requests.
        """
        try:
            # stdout is guaranteed to exist by __init__ check
            assert self._process.stdout is not None

            while True:
                # Read line from stdout
                try:
                    line_bytes = await self._process.stdout.readline()
                except Exception as e:
                    logger.error(f"Error reading from subprocess: {e}")
                    _dbg(f"reader exception: {e}")
                    break

                # Check if subprocess terminated
                if not line_bytes:
                    # EOF - subprocess terminated
                    _dbg("reader EOF from subprocess")
                    self._handle_subprocess_terminated()
                    break

                # Parse JSON response
                try:
                    line = line_bytes.decode("utf-8").strip()
                    _dbg(f"reader line: {line[:200]}")
                    if not line:
                        continue

                    response = json.loads(line)
                except json.JSONDecodeError as e:
                    logger.error(f"Failed to parse JSON response: {line_bytes!r} - {e}")
                    _dbg(f"reader JSON decode error: {e}")
                    continue
                except Exception as e:
                    logger.error(f"Error processing response: {e}")
                    _dbg(f"reader processing error: {e}")
                    continue

                # Match response to pending request
                if "id" in response:
                    request_id = response["id"]
                    future = self._pending_requests.pop(request_id, None)

                    if future is not None and not future.done():
                        future.set_result(response)
                        _dbg(f"reader matched response id={request_id}")
                    else:
                        logger.warning(
                            "Received response for unknown or completed "
                            f"request id={request_id}"
                        )
                        _dbg(f"reader unknown response id={request_id}")
                else:
                    # Notification or other message without id
                    logger.debug(f"Received notification: {response}")
                    _dbg(f"reader notification: {response}")

        except asyncio.CancelledError:
            # Expected during shutdown
            pass
        except Exception as e:
            logger.error(f"Reader task crashed: {e}", exc_info=True)
            self._handle_subprocess_terminated()

    def _handle_subprocess_terminated(self) -> None:
        """Handle unexpected subprocess termination.

        This fails all pending requests with SubprocessCrashError.
        """
        error = SubprocessCrashError(
            "Subprocess terminated unexpectedly "
            f"(exit code: {self._process.returncode})"
        )
        _dbg(f"reader terminated exit={self._process.returncode}")

        for future in self._pending_requests.values():
            if not future.done():
                future.set_exception(error)

        self._pending_requests.clear()
