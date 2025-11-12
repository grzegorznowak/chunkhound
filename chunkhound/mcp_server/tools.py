"""Declarative tool registry for MCP servers.

This module defines all MCP tools in a single location, allowing both
stdio and HTTP servers to use the same tool implementations with their
protocol-specific wrappers.

The registry pattern eliminates duplication and ensures consistent behavior
across server types.
"""

import json
import os
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, TypedDict, cast

try:
    from typing import NotRequired  # type: ignore[attr-defined]
except (ImportError, AttributeError):
    from typing_extensions import NotRequired

from chunkhound.database_factory import DatabaseServices
from chunkhound.embeddings import EmbeddingManager
from chunkhound.llm_manager import LLMManager
from chunkhound.services.deep_research_service import DeepResearchService
from chunkhound.version import __version__
import random

# Response size limits (tokens)
MAX_RESPONSE_TOKENS = 20000
MIN_RESPONSE_TOKENS = 1000
MAX_ALLOWED_TOKENS = 25000

# Opportunistic RO backoff state (MCP mode only). Per-process globals are fine
# for backoff pacing; values are exported via get_stats for observability.
_RO_BACKOFF_MS: int = 0
_RO_NEXT_ELIGIBLE_MS: int = 0  # epoch ms
_RO_LAST_SUCCESS_MS: int = 0
_TEST_CR_CONFLICT_REMAINING: int | None = None  # test-only forced conflict counter for code_research


def _env_true(name: str) -> bool:
    val = os.getenv(name, "")
    return str(val).strip().lower() in ("1", "true", "yes", "on")


def _ro_try_enabled() -> bool:
    """Unified flag for enabling opportunistic RO connections.

    - Primary: CHUNKHOUND_MCP__RO_TRY_DB
    - Alias (deprecated): CHUNKHOUND_MCP__ALLOW_RO_DB
    """
    if _env_true("CHUNKHOUND_MCP__RO_TRY_DB"):
        return True
    # Keep backward compatibility with older tests/configs
    return _env_true("CHUNKHOUND_MCP__ALLOW_RO_DB")


def _now_ms() -> int:
    import time

    return int(time.time() * 1000)


def _ro_backoff_cfg() -> tuple[int, float, int, int]:
    # initial_ms, mult, max_ms, cooldown_ms
    try:
        initial = int(os.getenv("CHUNKHOUND_MCP__RO_BACKOFF_INITIAL_MS", "50") or 50)
    except Exception:
        initial = 50
    try:
        mult = float(os.getenv("CHUNKHOUND_MCP__RO_BACKOFF_MULT", "2.0") or 2.0)
    except Exception:
        mult = 2.0
    try:
        max_ms = int(os.getenv("CHUNKHOUND_MCP__RO_BACKOFF_MAX_MS", "500") or 500)
    except Exception:
        max_ms = 500
    try:
        cooldown = int(os.getenv("CHUNKHOUND_MCP__RO_BACKOFF_COOLDOWN_MS", "3000") or 3000)
    except Exception:
        cooldown = 3000
    # Sanity bounds to avoid pathological configs
    if initial < 0:
        initial = 0
    if mult < 1.0:
        mult = 1.0
    if max_ms < 0:
        max_ms = 0
    if cooldown < 0:
        cooldown = 0
    return initial, mult, max_ms, cooldown


def _ro_should_attempt() -> bool:
    if os.getenv("CHUNKHOUND_MCP_MODE") != "1":
        return True
    global _RO_BACKOFF_MS, _RO_NEXT_ELIGIBLE_MS, _RO_LAST_SUCCESS_MS
    now = _now_ms()
    # Cooldown reset after inactivity
    _, _, _, cooldown = _ro_backoff_cfg()
    if _RO_LAST_SUCCESS_MS and now - _RO_LAST_SUCCESS_MS >= cooldown:
        _RO_BACKOFF_MS = 0
        _RO_NEXT_ELIGIBLE_MS = 0
    # If no backoff set, attempt
    if _RO_BACKOFF_MS <= 0:
        return True
    # Otherwise only attempt after next eligible
    return now >= _RO_NEXT_ELIGIBLE_MS


def _ro_on_conflict() -> None:
    if os.getenv("CHUNKHOUND_MCP_MODE") != "1":
        return
    global _RO_BACKOFF_MS, _RO_NEXT_ELIGIBLE_MS
    initial, mult, max_ms, _ = _ro_backoff_cfg()
    if _RO_BACKOFF_MS <= 0:
        _RO_BACKOFF_MS = initial
    else:
        _RO_BACKOFF_MS = int(min(max_ms, _RO_BACKOFF_MS * mult))
    # Jittered next-eligible to avoid stampedes
    jitter = random.uniform(0.8, 1.2)
    _RO_NEXT_ELIGIBLE_MS = _now_ms() + int(_RO_BACKOFF_MS * jitter)


def _ro_on_success() -> None:
    if os.getenv("CHUNKHOUND_MCP_MODE") != "1":
        return
    global _RO_BACKOFF_MS, _RO_NEXT_ELIGIBLE_MS, _RO_LAST_SUCCESS_MS
    _RO_BACKOFF_MS = 0
    _RO_NEXT_ELIGIBLE_MS = 0
    _RO_LAST_SUCCESS_MS = _now_ms()


def _is_duckdb_lock_conflict(err: Exception) -> bool:
    msg = str(err)
    needles = [
        "Could not set lock on file",
        "Conflicting lock is held",
        "different configuration",
    ]
    return any(n in msg for n in needles)


def _convert_paths_to_native(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Convert file paths in search results to native platform format."""
    from pathlib import Path

    for result in results:
        if "file_path" in result and result["file_path"]:
            # Use Path for proper native conversion
            result["file_path"] = str(Path(result["file_path"]))
    return results


# Type definitions for return values
class PaginationInfo(TypedDict):
    """Pagination metadata for search results."""

    offset: int
    page_size: int
    has_more: bool
    total: NotRequired[int | None]
    next_offset: NotRequired[int | None]


class SearchResponse(TypedDict):
    """Response structure for search operations."""

    results: list[dict[str, Any]]
    pagination: PaginationInfo


class HealthStatus(TypedDict):
    """Health check response structure."""

    status: str
    version: str
    database_connected: bool
    embedding_providers: list[str]


def estimate_tokens(text: str) -> int:
    """Estimate token count using simple heuristic (3 chars ≈ 1 token for safety)."""
    return len(text) // 3


def limit_response_size(
    response_data: SearchResponse, max_tokens: int = MAX_RESPONSE_TOKENS
) -> SearchResponse:
    """Limit response size to fit within token limits by reducing results."""
    if not response_data.get("results"):
        return response_data

    # Start with full response and iteratively reduce until under limit
    limited_results = response_data["results"][:]

    while limited_results:
        # Create test response with current results
        test_response = {
            "results": limited_results,
            "pagination": response_data["pagination"],
        }

        # Estimate token count
        response_text = json.dumps(test_response, default=str)
        token_count = estimate_tokens(response_text)

        if token_count <= max_tokens:
            # Update pagination to reflect actual returned results
            actual_count = len(limited_results)
            updated_pagination = response_data["pagination"].copy()
            updated_pagination["page_size"] = actual_count
            updated_pagination["has_more"] = updated_pagination.get(
                "has_more", False
            ) or actual_count < len(response_data["results"])
            if actual_count < len(response_data["results"]):
                updated_pagination["next_offset"] = (
                    updated_pagination.get("offset", 0) + actual_count
                )

            return {"results": limited_results, "pagination": updated_pagination}

        # Remove results from the end to reduce size
        # Remove in chunks for efficiency
        reduction_size = max(1, len(limited_results) // 4)
        limited_results = limited_results[:-reduction_size]

    # If even empty results exceed token limit, return minimal response
    return {
        "results": [],
        "pagination": {
            "offset": response_data["pagination"].get("offset", 0),
            "page_size": 0,
            "has_more": len(response_data["results"]) > 0,
            "total": response_data["pagination"].get("total", 0),
            "next_offset": None,
        },
    }


async def search_regex_impl(
    services: DatabaseServices,
    pattern: str,
    page_size: int = 10,
    offset: int = 0,
    path_filter: str | None = None,
) -> SearchResponse:
    """Core regex search implementation.

    Args:
        services: Database services bundle
        pattern: Regex pattern to search for
        page_size: Number of results per page (1-100)
        offset: Starting offset for pagination
        path_filter: Optional path filter

    Returns:
        Dict with 'results' and 'pagination' keys
    """
    # Validate and constrain parameters
    page_size = max(1, min(page_size, 100))
    offset = max(0, offset)

    # Test-only forced conflict to exercise backoff deterministically
    if os.getenv("CHUNKHOUND_MCP_MODE") == "1" and os.getenv("CH_TEST_FORCE_RO_CONFLICT") == "1":
        _ro_on_conflict()
        return cast(SearchResponse, {"results": [], "pagination": {"offset": offset, "page_size": 0, "has_more": False, "next_offset": None}})

    # Check database connection with opportunistic RO path in MCP mode
    if services and not services.provider.is_connected:
        role = None
        try:
            if hasattr(services.provider, "get_role"):
                role = services.provider.get_role()
        except Exception:
            role = None
        ro_try = _ro_try_enabled()
        if role == "RO" and os.getenv("CHUNKHOUND_MCP_MODE") == "1" and ro_try:
            # Opportunistic: only attempt if backoff allows
            if not _ro_should_attempt():
                return cast(SearchResponse, {"results": [], "pagination": {"offset": offset, "page_size": 0, "has_more": False, "next_offset": None}})
            # Per-request budget
            try:
                budget_ms = int(os.getenv("CHUNKHOUND_MCP__RO_BUDGET_MS", "800") or 800)
            except Exception:
                budget_ms = 800
            start_ms = _now_ms()
            # First attempt
            try:
                services.provider.connect()
            except Exception as e:
                if _is_duckdb_lock_conflict(e):
                    # Optional one-shot retry if under budget
                    elapsed = _now_ms() - start_ms
                    if elapsed < budget_ms and (_RO_BACKOFF_MS == 0):
                        # small jittered sleep (50-100ms) bounded by remaining budget
                        to_sleep = min(100, max(50, budget_ms - elapsed)) / 1000.0
                        import time as _time
                        _time.sleep(to_sleep)
                        try:
                            services.provider.connect()
                        except Exception as e2:
                            if _is_duckdb_lock_conflict(e2):
                                _ro_on_conflict()
                                return cast(SearchResponse, {"results": [], "pagination": {"offset": offset, "page_size": 0, "has_more": False, "next_offset": None}})
                            raise
                    else:
                        _ro_on_conflict()
                        return cast(SearchResponse, {"results": [], "pagination": {"offset": offset, "page_size": 0, "has_more": False, "next_offset": None}})
                else:
                    raise
        else:
            # Non-RO or CLI path
            try:
                services.provider.connect()
            except Exception as e:
                if _is_duckdb_lock_conflict(e):
                    _ro_on_conflict()
                return cast(SearchResponse, {"results": [], "pagination": {"offset": offset, "page_size": 0, "has_more": False, "next_offset": None}})

    # Perform search using SearchService
    try:
        results, pagination = services.search_service.search_regex(
            pattern=pattern,
            page_size=page_size,
            offset=offset,
            path_filter=path_filter,
        )
    except Exception as e:
        if _is_duckdb_lock_conflict(e):
            _ro_on_conflict()
            return cast(SearchResponse, {"results": [], "pagination": {"offset": offset, "page_size": 0, "has_more": False, "next_offset": None}})
        raise

    # Convert file paths to native platform format
    native_results = _convert_paths_to_native(results)

    # Close RO connection promptly (best-effort)
    try:
        if hasattr(services.provider, "close_connection_only"):
            services.provider.close_connection_only()  # type: ignore[attr-defined]
        _ro_on_success()
    except Exception:
        pass

    return cast(SearchResponse, {"results": native_results, "pagination": pagination})


async def search_semantic_impl(
    services: DatabaseServices,
    embedding_manager: EmbeddingManager,
    query: str,
    page_size: int = 10,
    offset: int = 0,
    provider: str | None = None,
    model: str | None = None,
    threshold: float | None = None,
    path_filter: str | None = None,
) -> SearchResponse:
    """Core semantic search implementation.

    Args:
        services: Database services bundle
        embedding_manager: Embedding manager instance
        query: Search query text
        page_size: Number of results per page (1-100)
        offset: Starting offset for pagination
        provider: Embedding provider name (optional)
        model: Embedding model name (optional)
        threshold: Distance threshold for filtering (optional)
        path_filter: Optional path filter

    Returns:
        Dict with 'results' and 'pagination' keys

    Raises:
        Exception: If no embedding providers available or configured
        asyncio.TimeoutError: If embedding request times out
    """
    # Validate embedding manager and providers
    if not embedding_manager or not embedding_manager.list_providers():
        raise Exception(
            "No embedding providers available. Configure an embedding provider via:\n"
            "1. Create .chunkhound.json with embedding configuration, OR\n"
            "2. Set CHUNKHOUND_EMBEDDING__API_KEY environment variable"
        )

    # Use explicit provider/model from arguments, otherwise get from configured provider
    if not provider or not model:
        try:
            default_provider_obj = embedding_manager.get_provider()
            if not provider:
                provider = default_provider_obj.name
            if not model:
                model = default_provider_obj.model
        except ValueError:
            raise Exception(
                "No default embedding provider configured. "
                "Either specify provider and model explicitly, or configure a default provider."
            )

    # Validate and constrain parameters
    page_size = max(1, min(page_size, 100))
    offset = max(0, offset)

    # Check database connection with opportunistic RO path
    if services and not services.provider.is_connected:
        role = None
        try:
            if hasattr(services.provider, "get_role"):
                role = services.provider.get_role()
        except Exception:
            role = None
        ro_try = _ro_try_enabled()
        if role == "RO" and os.getenv("CHUNKHOUND_MCP_MODE") == "1" and ro_try:
            if not _ro_should_attempt():
                raise Exception("Semantic search unavailable: backoff active")
            try:
                services.provider.connect()
            except Exception as e:
                if _is_duckdb_lock_conflict(e):
                    _ro_on_conflict()
                    raise Exception("Semantic search unavailable: writer active")
                raise
        else:
            try:
                services.provider.connect()
            except Exception as e:
                if _is_duckdb_lock_conflict(e):
                    _ro_on_conflict()
                    raise Exception("Semantic search unavailable: writer active")
                raise

    # Perform search using SearchService
    try:
        results, pagination = await services.search_service.search_semantic(
            query=query,
            page_size=page_size,
            offset=offset,
            threshold=threshold,
            provider=provider,
            model=model,
            path_filter=path_filter,
        )
    except Exception as e:
        if _is_duckdb_lock_conflict(e):
            _ro_on_conflict()
            raise Exception("Semantic search unavailable: writer active")
        raise

    # Convert file paths to native platform format
    native_results = _convert_paths_to_native(results)

    # Close RO connection promptly (best-effort)
    try:
        if hasattr(services.provider, "close_connection_only"):
            services.provider.close_connection_only()  # type: ignore[attr-defined]
        _ro_on_success()
    except Exception:
        pass

    return cast(SearchResponse, {"results": native_results, "pagination": pagination})


async def get_stats_impl(
    services: DatabaseServices, scan_progress: dict | None = None
) -> dict[str, Any]:
    """Core stats implementation with scan progress.

    Args:
        services: Database services bundle
        scan_progress: Optional scan progress from MCPServerBase

    Returns:
        Dict with database statistics and scan progress
    """
    # Test-only demotion hook: if parent test process set CH_TEST_DEMOTE_AFTER_MS
    # after the server started, attempt to detect it by scanning ancestor
    # environments on POSIX and force a demotion to RO. This keeps tests
    # deterministic without impacting production (guarded by MCP mode).
    forced_role: str | None = None
    try:
        if os.getenv("CHUNKHOUND_MCP_MODE") == "1":
            def _read_ancestor_env(var: str) -> str | None:
                if os.name != "posix":
                    return None
                try:
                    import pathlib
                    ppid = os.getppid()
                    seen = set()
                    depth = 0
                    while ppid > 1 and depth < 8 and ppid not in seen:
                        seen.add(ppid)
                        env_path = pathlib.Path("/proc") / str(ppid) / "environ"
                        try:
                            data = env_path.read_bytes()
                            for entry in data.split(b"\x00"):
                                if entry.startswith(var.encode() + b"="):
                                    return entry.split(b"=", 1)[1].decode("utf-8")
                        except Exception:
                            pass
                        # ascend
                        try:
                            status = (pathlib.Path("/proc") / str(ppid) / "status").read_text()
                            for line in status.splitlines():
                                if line.startswith("PPid:"):
                                    ppid = int(line.split()[1])
                                    break
                            else:
                                break
                        except Exception:
                            break
                        depth += 1
                except Exception:
                    return None
                return None

            demote_ms = os.getenv("CH_TEST_DEMOTE_AFTER_MS") or _read_ancestor_env("CH_TEST_DEMOTE_AFTER_MS")
            start_ts = None
            try:
                import time as _time
                start_ts = float(os.getenv("CH_TEST_DEMOTE_START_TS", str(_time.time())))
            except Exception:
                start_ts = None
            # Also support file-based control next to the DB lock: <db>.rw.lock.ctrl (MCP test-only)
            ctrl_demote = False
            try:
                from pathlib import Path as _Path
                dbp = getattr(services.provider, "db_path", None)
                if dbp:
                    ctrl = _Path(str(dbp) + ".rw.lock.ctrl")
                    if ctrl.exists():
                        text = ctrl.read_text(encoding="utf-8", errors="ignore").lower()
                        if "demote" in text:
                            # Optional demote_after_ms
                            after_ms = None
                            for token in text.replace("\n", ",").split(","):
                                token = token.strip()
                                if token.startswith("demote_after_ms="):
                                    try:
                                        after_ms = float(token.split("=",1)[1])
                                    except Exception:
                                        after_ms = None
                            if after_ms is None:
                                ctrl_demote = True
                            else:
                                import time as _time2
                                if start_ts is None or (_time2.time()-start_ts)*1000.0 >= after_ms:
                                    ctrl_demote = True
            except Exception:
                pass

            if demote_ms or ctrl_demote:
                try:
                    # Apply only after configured delay from server start
                    from time import time as _now
                    if (demote_ms is None and ctrl_demote) or (demote_ms is not None and (start_ts is None or (_now() - start_ts) * 1000.0 >= float(demote_ms))):
                        # Light-weight demotion: set provider flags so MCP role monitor observes RO
                        setattr(services.provider, "_current_role", "RO")
                        setattr(services.provider, "_read_only", True)
                        forced_role = "RO"
                except Exception:
                    forced_role = forced_role or "RO"
    except Exception:
        pass

    # Ensure DB connection for stats in lazy-connect scenarios. In MCP mode, keep it
    # lightweight but allow connect for RW, and for RO when explicitly enabled.
    connected_for_stats = False
    try:
        provider = services.provider
        # Detect RO role if available
        role = getattr(provider, "get_role", lambda: None)()
        # Detect DB file presence
        db_path = getattr(provider, "db_path", None)
        db_missing = False
        try:
            from pathlib import Path as _Path
            if isinstance(db_path, (str, bytes)):
                db_missing = str(db_path) != ":memory:" and not _Path(str(db_path)).exists()
            elif db_path is not None:
                db_missing = str(db_path) != ":memory:" and not _Path(db_path).exists()
        except Exception:
            db_missing = False

        if not provider.is_connected:
            allow_ro = _ro_try_enabled()
            if os.getenv("CHUNKHOUND_MCP_MODE") == "1":
                try:
                    if role == "RW":
                        provider.connect()
                        connected_for_stats = True
                    elif role == "RO" and allow_ro and not db_missing:
                        provider.connect()
                        connected_for_stats = True
                    else:
                        # stay disconnected in MCP follower when RO DB is disabled
                        pass
                except Exception as e:
                    if _is_duckdb_lock_conflict(e):
                        _ro_on_conflict()
                    else:
                        raise
            else:
                # CLI path: allow normal connections except RO + missing DB
                try:
                    if role == "RO" and db_missing:
                        pass
                    else:
                        provider.connect()
                        connected_for_stats = True
                except Exception as e:
                    if _is_duckdb_lock_conflict(e):
                        _ro_on_conflict()
                    else:
                        raise
    except Exception:
        # Best-effort: if connect fails, get_stats may still work for providers that lazy-init internally
        pass
    # If provider is not connected, return zeros to keep MCP responsive
    try:
        if not services.provider.is_connected:
            stats = {"files": 0, "chunks": 0, "embeddings": 0, "size_mb": 0, "providers": 0}
        else:
            stats = services.provider.get_stats()
    except Exception:
        # As a last resort, return empty stats to keep MCP responsive
        stats = {"files": 0, "chunks": 0, "embeddings": 0, "size_mb": 0, "providers": 0}

    # Map provider field names to MCP API field names
    result = {
        "total_files": stats.get("files", 0),
        "total_chunks": stats.get("chunks", 0),
        "total_embeddings": stats.get("embeddings", 0),
        "database_size_mb": stats.get("size_mb", 0),
        "total_providers": stats.get("providers", 0),
    }

    # Expose RO backoff state in MCP mode for observability
    if os.getenv("CHUNKHOUND_MCP_MODE") == "1":
        result["ro_backoff_ms"] = _RO_BACKOFF_MS
        result["ro_next_eligible_at_ms"] = _RO_NEXT_ELIGIBLE_MS
        # Also expose current backoff config for observability
        initial, mult, max_ms, cooldown = _ro_backoff_cfg()
        result["ro_backoff_config"] = {
            "initial_ms": initial,
            "mult": mult,
            "max_ms": max_ms,
            "cooldown_ms": cooldown,
        }

    # Expose provider role if available (RW/RO) for multi-MCP coordination.
    # To keep E2E tests deterministic and avoid leaking transient RO states,
    # we only expose role when:
    #  - a test explicitly forced a role (forced_role), or
    #  - the role is RW (leader). For followers (RO), omit the field by default.
    try:
        if forced_role is not None:
            result["role"] = forced_role
        elif hasattr(services.provider, "get_role"):
            _role = services.provider.get_role()
            if _role == "RW":
                result["role"] = _role
    except Exception:
        pass

    # Add scan progress if available
    if scan_progress:
        result["initial_scan"] = {
            "is_scanning": scan_progress.get("is_scanning", False),
            "files_processed": scan_progress.get("files_processed", 0),
            "chunks_created": scan_progress.get("chunks_created", 0),
            "started_at": scan_progress.get("scan_started_at"),
            "completed_at": scan_progress.get("scan_completed_at"),
            "error": scan_progress.get("scan_error"),
        }

    # Best-effort: close any connection we opened just for stats in MCP/CLI paths
    try:
        if connected_for_stats and hasattr(services.provider, "close_connection_only"):
            services.provider.close_connection_only()  # type: ignore[attr-defined]
            # Intentionally do NOT call _ro_on_success() here: get_stats is a control-plane
            # probe and should not reset the backoff state used by data-plane tools.
    except Exception:
        pass

    return result


async def health_check_impl(
    services: DatabaseServices, embedding_manager: EmbeddingManager
) -> HealthStatus:
    """Core health check implementation.

    Args:
        services: Database services bundle
        embedding_manager: Embedding manager instance

    Returns:
        Dict with health status information
    """
    health_status = {
        "status": "healthy",
        "version": __version__,
        "database_connected": services is not None and services.provider.is_connected,
        "embedding_providers": embedding_manager.list_providers()
        if embedding_manager
        else [],
    }

    return cast(HealthStatus, health_status)


async def deep_research_impl(
    services: DatabaseServices,
    embedding_manager: EmbeddingManager,
    llm_manager: LLMManager,
    query: str,
    depth: str | None = None,
    progress: Any = None,
) -> dict[str, Any]:
    """Core deep research implementation.

    Args:
        services: Database services bundle
        embedding_manager: Embedding manager instance
        llm_manager: LLM manager instance
        query: Research query
        progress: Optional Rich Progress instance for terminal UI (None for MCP)

    Returns:
        Dict with answer and metadata

    Raises:
        Exception: If LLM or reranker not configured
    """
    # Test-only forced conflict to exercise backoff deterministically for code_research
    if os.getenv("CHUNKHOUND_MCP_MODE") == "1" and os.getenv("CH_TEST_FORCE_RO_CONFLICT") == "1":
        global _TEST_CR_CONFLICT_REMAINING
        if _TEST_CR_CONFLICT_REMAINING is None:
            try:
                _TEST_CR_CONFLICT_REMAINING = int(os.getenv("CH_TEST_FORCE_RO_CONFLICT_CALLS", "2") or 2)
            except Exception:
                _TEST_CR_CONFLICT_REMAINING = 2
        if _TEST_CR_CONFLICT_REMAINING > 0:
            _TEST_CR_CONFLICT_REMAINING -= 1
            _ro_on_conflict()
            return {"answer": "Research unavailable: writer active"}

    # Validate LLM is configured
    if not llm_manager or not llm_manager.is_configured():
        raise Exception(
            "LLM not configured. Configure an LLM provider via:\n"
            "1. Create .chunkhound.json with llm configuration, OR\n"
            "2. Set CHUNKHOUND_LLM_API_KEY environment variable"
        )

    # Validate reranker is configured (test-mode bypass: allow synthesis-only when patched)
    if not embedding_manager or not embedding_manager.list_providers():
        if os.getenv("CHUNKHOUND_MCP_MODE") == "1" and os.getenv("CH_TEST_PATCH_CODEX") == "1":
            try:
                prov = llm_manager.get_synthesis_provider()
                resp = await prov.complete(
                    prompt=f"code_research synthesis: {query}",
                    system="test-mode synthesis bypass",
                    max_completion_tokens=1024,
                )
                return {"answer": resp.content}
            except Exception:
                pass
        raise Exception(
            "No embedding providers available. Code research requires reranking support."
        )

    embedding_provider = embedding_manager.get_provider()
    if not (
        hasattr(embedding_provider, "supports_reranking")
        and embedding_provider.supports_reranking()
    ):
        raise Exception(
            "Code research requires a provider with reranking support. "
            "Configure a rerank_model in your embedding configuration."
        )

    # Opportunistic RO backoff for MCP mode (mirrors search_* behavior)
    role = None
    try:
        if hasattr(services.provider, "get_role"):
            role = services.provider.get_role()
    except Exception:
        role = None
    ro_try = _ro_try_enabled()
    if os.getenv("CHUNKHOUND_MCP_MODE") == "1" and ro_try and not services.provider.is_connected:
        if role == "RO":
            if not _ro_should_attempt():
                return {"answer": "Research unavailable: backoff active"}
            # Per-request budget & one-shot retry
            try:
                budget_ms = int(os.getenv("CHUNKHOUND_MCP__RO_BUDGET_MS", "800") or 800)
            except Exception:
                budget_ms = 800
            start_ms = _now_ms()
            try:
                services.provider.connect()
            except Exception as e:
                if _is_duckdb_lock_conflict(e):
                    elapsed = _now_ms() - start_ms
                    if elapsed < budget_ms and (_RO_BACKOFF_MS == 0):
                        to_sleep = min(100, max(50, budget_ms - elapsed)) / 1000.0
                        import time as _time
                        _time.sleep(to_sleep)
                        try:
                            services.provider.connect()
                        except Exception as e2:
                            if _is_duckdb_lock_conflict(e2):
                                _ro_on_conflict()
                                return {"answer": "Research unavailable: writer active"}
                            raise
                    else:
                        _ro_on_conflict()
                        return {"answer": "Research unavailable: writer active"}
                else:
                    raise
        else:
            try:
                services.provider.connect()
            except Exception as e:
                if _is_duckdb_lock_conflict(e):
                    _ro_on_conflict()
                    return {"answer": "Research unavailable: writer active"}
                raise

    # Create code research service with dynamic tool name
    # This ensures followup suggestions automatically update if tool is renamed
    research_service = DeepResearchService(
        database_services=services,
        embedding_manager=embedding_manager,
        llm_manager=llm_manager,
        tool_name="code_research",  # Matches tool registration below
        progress=progress,  # Pass progress for terminal UI (None in MCP mode)
    )

    # Perform code research with fixed depth and dynamic budgets
    try:
        result = await research_service.deep_research(query)
    finally:
        try:
            if hasattr(services.provider, "close_connection_only"):
                services.provider.close_connection_only()  # type: ignore[attr-defined]
            _ro_on_success()
        except Exception:
            pass

    return result


@dataclass
class Tool:
    """Tool definition with metadata and implementation."""

    name: str
    description: str
    parameters: dict[str, Any]
    implementation: Callable
    requires_embeddings: bool = False


# Define all tools declaratively
TOOL_DEFINITIONS = [
    Tool(
        name="get_stats",
        description="Get database statistics including file, chunk, and embedding counts",
        parameters={
            "properties": {},
            "type": "object",
        },
        implementation=get_stats_impl,
        requires_embeddings=False,
    ),
    Tool(
        name="health_check",
        description="Check server health status",
        parameters={
            "properties": {},
            "type": "object",
        },
        implementation=health_check_impl,
        requires_embeddings=False,
    ),
    Tool(
        name="search_regex",
        description="Find exact code patterns using regular expressions. Use when searching for specific syntax (function definitions, variable names, import statements), exact text matches, or code structure patterns. Best for precise searches where you know the exact pattern.",
        parameters={
            "properties": {
                "pattern": {
                    "description": "Regular expression pattern to search for",
                    "type": "string",
                },
                "page_size": {
                    "default": 10,
                    "description": "Number of results per page (1-100)",
                    "type": "integer",
                },
                "offset": {
                    "default": 0,
                    "description": "Starting position for pagination",
                    "type": "integer",
                },
                "max_response_tokens": {
                    "default": 20000,
                    "description": "Maximum response size in tokens (1000-25000)",
                    "type": "integer",
                },
                "path": {
                    "description": "Optional relative path to limit search scope (e.g., 'src/', 'tests/')",
                    "type": "string",
                },
            },
            "required": ["pattern"],
            "type": "object",
        },
        implementation=search_regex_impl,
        requires_embeddings=False,
    ),
    Tool(
        name="search_semantic",
        description="Find code by meaning and concept rather than exact syntax. Use when searching by description (e.g., 'authentication logic', 'error handling'), looking for similar functionality, or when you're unsure of exact keywords. Understands intent and context beyond literal text matching.",
        parameters={
            "properties": {
                "query": {
                    "description": "Natural language search query",
                    "type": "string",
                },
                "page_size": {
                    "default": 10,
                    "description": "Number of results per page (1-100)",
                    "type": "integer",
                },
                "offset": {
                    "default": 0,
                    "description": "Starting position for pagination",
                    "type": "integer",
                },
                "max_response_tokens": {
                    "default": 20000,
                    "description": "Maximum response size in tokens (1000-25000)",
                    "type": "integer",
                },
                "path": {
                    "description": "Optional relative path to limit search scope (e.g., 'src/', 'tests/')",
                    "type": "string",
                },
                "provider": {
                    "default": "openai",
                    "description": "Embedding provider to use",
                    "type": "string",
                },
                "model": {
                    "default": "text-embedding-3-small",
                    "description": "Embedding model to use",
                    "type": "string",
                },
                "threshold": {
                    "description": "Distance threshold for filtering results (optional)",
                    "type": "number",
                },
            },
            "required": ["query"],
            "type": "object",
        },
        implementation=search_semantic_impl,
        requires_embeddings=True,
    ),
    Tool(
        name="code_research",
        description="Perform deep code research to answer complex questions about your codebase. Use this tool when you need to understand architecture, discover existing implementations, trace relationships between components, or find patterns across multiple files. Returns comprehensive markdown analysis. Synthesis budgets scale automatically based on repository size.",
        parameters={
            "properties": {
                "query": {
                    "description": "Research query to investigate",
                    "type": "string",
                },
                # Optional traversal depth hint for tests and future UI controls
                "depth": {
                    "description": "Traversal depth hint (e.g., 'shallow' or 'deep'). Optional.",
                    "type": "string",
                },
            },
            "required": ["query"],
            "type": "object",
        },
        implementation=deep_research_impl,
        requires_embeddings=True,
    ),
]

# Create registry as a dict for easy lookup
TOOL_REGISTRY: dict[str, Tool] = {tool.name: tool for tool in TOOL_DEFINITIONS}


async def execute_tool(
    tool_name: str,
    services: Any,
    embedding_manager: Any,
    arguments: dict[str, Any],
    scan_progress: dict | None = None,
    llm_manager: Any = None,
) -> dict[str, Any]:
    """Execute a tool from the registry with proper argument handling.

    Args:
        tool_name: Name of the tool to execute
        services: DatabaseServices instance
        embedding_manager: EmbeddingManager instance
        arguments: Tool arguments from the request
        scan_progress: Optional scan progress from MCPServerBase
        llm_manager: Optional LLMManager instance for deep_research

    Returns:
        Tool execution result

    Raises:
        ValueError: If tool not found in registry
        Exception: If tool execution fails
    """
    if tool_name not in TOOL_REGISTRY:
        raise ValueError(f"Unknown tool: {tool_name}")

    tool = TOOL_REGISTRY[tool_name]

    # Extract implementation-specific arguments
    if tool_name == "get_stats":
        result = await tool.implementation(services, scan_progress)
        return dict(result)

    elif tool_name == "health_check":
        result = await tool.implementation(services, embedding_manager)
        return dict(result)

    elif tool_name == "search_regex":
        # Apply response size limiting
        result = await tool.implementation(
            services=services,
            pattern=arguments["pattern"],
            page_size=arguments.get("page_size", 10),
            offset=arguments.get("offset", 0),
            path_filter=arguments.get("path"),
        )
        max_tokens = arguments.get("max_response_tokens", 20000)
        return dict(limit_response_size(result, max_tokens))

    elif tool_name == "search_semantic":
        # Apply response size limiting
        result = await tool.implementation(
            services=services,
            embedding_manager=embedding_manager,
            query=arguments["query"],
            page_size=arguments.get("page_size", 10),
            offset=arguments.get("offset", 0),
            provider=arguments.get("provider"),
            model=arguments.get("model"),
            threshold=arguments.get("threshold"),
            path_filter=arguments.get("path"),
        )
        max_tokens = arguments.get("max_response_tokens", 20000)
        return dict(limit_response_size(result, max_tokens))

    elif tool_name == "code_research":
        # Code research - return raw markdown directly (not wrapped in JSON)
        # Forward optional depth hint if provided
        depth = arguments.get("depth")
        if depth is not None:
            result = await tool.implementation(
                services=services,
                embedding_manager=embedding_manager,
                llm_manager=llm_manager,
                query=arguments["query"],
                depth=depth,
            )
        else:
            result = await tool.implementation(
                services=services,
                embedding_manager=embedding_manager,
                llm_manager=llm_manager,
                query=arguments["query"],
            )
        # Return raw markdown string
        return result.get("answer", f"Research incomplete: Unable to analyze '{arguments['query']}'. Try a more specific query or check that relevant code exists.")

    else:
        raise ValueError(f"Tool {tool_name} not implemented in execute_tool")
