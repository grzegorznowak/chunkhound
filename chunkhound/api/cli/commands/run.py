"""Run command module - handles directory indexing operations."""

import argparse
import os
import sys
from pathlib import Path
from typing import Any

from loguru import logger

from chunkhound.core.config.config import Config
from chunkhound.registry import configure_registry, create_indexing_coordinator
from chunkhound.services.directory_indexing_service import DirectoryIndexingService
from chunkhound.version import __version__

from ..parsers.run_parser import process_batch_arguments
from ..utils.rich_output import RichOutputFormatter
from ..utils.validation import (
    ensure_database_directory,
    validate_file_patterns,
    validate_path,
    validate_provider_args,
)


async def run_command(args: argparse.Namespace, config: Config) -> None:
    """Execute the run command using the service layer.

    Args:
        args: Parsed command-line arguments
        config: Pre-validated configuration instance
    """
    # Simulate mode
    if getattr(args, "simulate", False):
        # Ensure simulate doesn't require embeddings
        setattr(args, "no_embeddings", True)
        await _simulate_index(args, config)
        return

    # Initialize Rich output formatter
    formatter = RichOutputFormatter(verbose=args.verbose)

    # Check if local config was found (for logging purposes)
    project_dir = Path(args.path) if hasattr(args, "path") else Path.cwd()
    local_config_path = project_dir / ".chunkhound.json"
    if local_config_path.exists():
        formatter.info(f"Found local config: {local_config_path}")

    # Use database path from config
    db_path = Path(config.database.path)

    # Display modern startup information
    formatter.startup_info(
        version=__version__,
        directory=str(args.path),
        database=str(db_path),
        config=config.__dict__ if hasattr(config, "__dict__") else {},
    )

    # Process and validate batch arguments (includes deprecation warnings)
    process_batch_arguments(args)

    # Validate arguments - update args.db to use config value for validation
    args.db = db_path
    if not _validate_run_arguments(args, formatter, config):
        sys.exit(1)

    try:
        # Configure registry with the Config object
        configure_registry(config)

        formatter.success(f"Service layer initialized: {args.db}")

        # Create progress manager for modern UI
        with formatter.create_progress_display() as progress_manager:
            # Get the underlying Progress instance for service layers
            progress_instance = progress_manager.get_progress_instance()

            # Create indexing coordinator with Progress instance
            indexing_coordinator = create_indexing_coordinator()
            # Pass progress to the coordinator after creation
            if hasattr(indexing_coordinator, "progress"):
                indexing_coordinator.progress = progress_instance

            # Get initial stats
            initial_stats = await indexing_coordinator.get_stats()
            formatter.initial_stats_panel(initial_stats)

            # Simple progress callback for verbose output
            def progress_callback(message: str):
                if args.verbose:
                    formatter.verbose_info(message)

            # Create indexing service with Progress instance
            indexing_service = DirectoryIndexingService(
                indexing_coordinator=indexing_coordinator,
                config=config,
                progress_callback=progress_callback,
                progress=progress_instance,
            )

            # Process directory - service layers will add subtasks to progress_instance
            stats = await indexing_service.process_directory(
                Path(args.path), no_embeddings=args.no_embeddings
            )

        # Display results
        _print_completion_summary(stats, formatter)

        # Offer to add timed-out files to exclusion list in local config
        try:
            skipped_timeouts = []
            if hasattr(stats, "skipped_due_to_timeout"):
                skipped_timeouts = stats.skipped_due_to_timeout or []

            # Never prompt in MCP mode (stdio must not emit prompts/output)
            if skipped_timeouts and os.environ.get("CHUNKHOUND_MCP_MODE") == "1":
                formatter.info(
                    f"{len(skipped_timeouts)} files timed out. "
                    "Prompts are disabled in MCP mode. To exclude them, add to .chunkhound.json under indexing.exclude."
                )
                return

            # Respect explicit no-prompts
            if skipped_timeouts and os.environ.get("CHUNKHOUND_NO_PROMPTS") == "1":
                formatter.info(
                    f"{len(skipped_timeouts)} files timed out (prompts disabled)."
                )
                return

            # Only prompt in interactive TTY and when there are timeouts
            if skipped_timeouts and sys.stdin.isatty():
                base_dir = Path(args.path).resolve() if hasattr(args, "path") else Path.cwd().resolve()

                # Convert to unique relative paths within the project
                rel_paths: list[str] = []
                seen: set[str] = set()
                for p in skipped_timeouts:
                    try:
                        rel = Path(p).resolve().relative_to(base_dir).as_posix()
                    except Exception:
                        # If not under base_dir, keep as-is (rare)
                        rel = Path(p).as_posix()
                    if rel not in seen:
                        seen.add(rel)
                        rel_paths.append(rel)

                formatter.info(
                    f"{len(rel_paths)} timed-out files can be excluded from future runs."
                )
                reply = input("Add these to indexing.exclude in .chunkhound.json? [y/N]: ").strip().lower()
                if reply in ("y", "yes"):
                    local_config_path = base_dir / ".chunkhound.json"
                    # Load or initialize config data
                    data = {}
                    if local_config_path.exists():
                        import json

                        try:
                            data = json.loads(local_config_path.read_text())
                        except Exception:
                            data = {}

                    # Ensure structure exists
                    indexing = data.get("indexing") or {}
                    exclude_list = list(indexing.get("exclude") or [])

                    # Merge unique entries
                    existing = set(exclude_list)
                    added = 0
                    for rel in rel_paths:
                        if rel not in existing:
                            exclude_list.append(rel)
                            existing.add(rel)
                            added += 1

                    if added > 0:
                        indexing["exclude"] = exclude_list
                        data["indexing"] = indexing
                        import json

                        local_config_path.write_text(
                            json.dumps(data, indent=2, sort_keys=False) + "\n"
                        )
                        formatter.success(
                            f"Added {added} file(s) to indexing.exclude in {local_config_path}"
                        )
                    else:
                        formatter.info("All timed-out files already excluded.")
        except Exception as e:
            formatter.warning(f"Failed to offer exclusion prompt: {e}")

        formatter.success("Run command completed successfully")

    except KeyboardInterrupt:
        formatter.info("Interrupted by user")
        sys.exit(0)
    except Exception as e:
        formatter.error(f"Run command failed: {e}")
        logger.exception("Run command error details")
        sys.exit(1)
    finally:
        pass


def _print_completion_summary(stats, formatter: RichOutputFormatter) -> None:
    """Print completion summary from IndexingStats using Rich formatting."""
    # Convert stats object to dictionary for Rich display
    if hasattr(stats, "__dict__"):
        stats_dict = stats.__dict__
    else:
        stats_dict = stats if isinstance(stats, dict) else {}
    formatter.completion_summary(stats_dict, stats.processing_time)


def _validate_run_arguments(
    args: argparse.Namespace, formatter: RichOutputFormatter, config: Any = None
) -> bool:
    """Validate run command arguments.

    Args:
        args: Parsed arguments
        formatter: Output formatter
        config: Configuration (optional)

    Returns:
        True if valid, False otherwise
    """
    # Validate path
    if not validate_path(args.path, must_exist=True, must_be_dir=True):
        return False

    # Ensure database directory exists
    if not ensure_database_directory(args.db):
        return False

    # Validate provider arguments
    if not args.no_embeddings:
        # Use unified config values if available, fall back to CLI args
        if config and config.embedding:
            provider = config.embedding.provider
            api_key = (
                config.embedding.api_key.get_secret_value()
                if config.embedding.api_key
                else None
            )
            base_url = config.embedding.base_url
            model = config.embedding.model
        else:
            # Check if CLI args have provider info
            provider = getattr(args, "provider", None)
            api_key = getattr(args, "api_key", None)
            base_url = getattr(args, "base_url", None)
            model = getattr(args, "model", None)

            # If no provider info found, provide helpful error
            if not provider:
                formatter.error("No embedding provider configured.")
                formatter.info("To fix this, you can:")
                formatter.info(
                    "  1. Create .chunkhound.json config file with embeddings"
                )
                formatter.info("  2. Use --no-embeddings to skip embeddings")
                return False
        if not validate_provider_args(provider, api_key, base_url, model):
            return False

    # Validate file patterns
    if not validate_file_patterns(args.include, args.exclude):
        return False

    return True


__all__ = ["run_command"]


async def _simulate_index(args: argparse.Namespace, config: Config) -> None:
    """Dry-run discovery and print list of relative files.

    Minimal implementation: perform discovery via the coordinator and print
    the discovered files sorted. Later we may reflect change-detection.
    """
    base_dir = Path(args.path).resolve() if hasattr(args, "path") else Path.cwd().resolve()

    # Configure registry and create services like real run
    # Ensure database directory exists to allow provider to connect
    try:
        db_path = Path(config.database.path)
        db_dir = db_path.parent
        db_dir.mkdir(parents=True, exist_ok=True)
    except Exception:
        pass

    configure_registry(config)
    indexing_coordinator = create_indexing_coordinator()

    # Resolve patterns using the DirectoryIndexingService helper to keep logic aligned
    from chunkhound.services.directory_indexing_service import DirectoryIndexingService

    svc = DirectoryIndexingService(indexing_coordinator=indexing_coordinator, config=config)
    include_patterns, exclude_patterns = svc._resolve_file_patterns()

    # Normalize include patterns and call internal discovery
    from chunkhound.utils.file_patterns import normalize_include_patterns

    processed_patterns = normalize_include_patterns(include_patterns)

    files = await indexing_coordinator._discover_files(  # type: ignore[attr-defined]
        base_dir, processed_patterns, exclude_patterns
    )

    # Gather sizes and relative paths
    items: list[tuple[str, int]] = []
    for p in files:
        try:
            st = p.stat()
            size = int(st.st_size)
        except Exception:
            size = 0
        rel = p.resolve().relative_to(base_dir).as_posix()
        items.append((rel, size))

    # Sort
    sort_mode = getattr(args, "sort", "path") or "path"
    if sort_mode == "size":
        items.sort(key=lambda x: (x[1], x[0]))
    elif sort_mode == "size_desc":
        items.sort(key=lambda x: (-x[1], x[0]))
    else:
        items.sort(key=lambda x: x[0])

    import json as _json
    if getattr(args, "json", False):
        print(
            _json.dumps(
                {"files": [{"path": rel, "size_bytes": size} for rel, size in items]},
                indent=2,
            )
        )
    else:
        show_sizes = bool(getattr(args, "show_sizes", False))
        if show_sizes:
            for rel, size in items:
                print(f"{size:>10}  {rel}")
        else:
            for rel, _ in items:
                print(rel)
