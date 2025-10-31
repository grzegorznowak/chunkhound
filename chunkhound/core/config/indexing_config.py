"""Indexing configuration for ChunkHound.

This module provides configuration for the file indexing process including
batch processing, and pattern matching.
"""

import argparse
import os
from typing import Any

from pydantic import BaseModel, Field, field_validator, model_validator


def _get_default_include_patterns() -> list[str]:
    """Get complete default patterns from Language enum.

    Returns all supported file extensions as glob patterns.
    This is the single source of truth for default file discovery.
    """
    from chunkhound.core.types.common import Language

    patterns = []
    for ext in Language.get_all_extensions():
        patterns.append(f"**/*{ext}")
    # Add special filename patterns
    patterns.extend(["**/Makefile", "**/makefile", "**/GNUmakefile", "**/gnumakefile"])
    return patterns


class IndexingConfig(BaseModel):
    """Configuration for file indexing behavior.

    Controls how files are discovered and indexed.
    """

    # Indexing behavior
    force_reindex: bool = Field(
        default=False, description="Force re-indexing of all files"
    )

    # Internal settings - not exposed to users
    batch_size: int = Field(default=50, description="Internal batch size")
    db_batch_size: int = Field(default=100, description="Internal DB batch size")
    max_concurrent: int = Field(default=5, description="Internal concurrency")
    cleanup: bool = Field(default=True, description="Internal cleanup setting")
    ignore_gitignore: bool = Field(
        default=False, description="Internal gitignore setting"
    )
    max_file_size_mb: int = Field(default=10, description="Internal file size limit")
    config_file_size_threshold_kb: int = Field(
        default=20,
        description="Skip structured config files (JSON/YAML/TOML) larger than this (KB)",
    )
    chunk_overlap: int = Field(default=50, description="Internal chunk overlap")
    min_chunk_size: int = Field(default=50, description="Internal min chunk size")
    max_chunk_size: int = Field(default=2000, description="Internal max chunk size")

    # File parsing safety
    per_file_timeout_seconds: float = Field(
        default=3.0,
        description=
        "Maximum seconds to spend parsing a single file (0 disables timeout)",
    )
    per_file_timeout_min_size_kb: int = Field(
        default=128,
        description=(
            "Only apply timeout to files at or above this size (KB) to avoid "
            "overhead on small files"
        ),
    )
    mtime_epsilon_seconds: float = Field(
        default=0.01,
        description=(
            "Tolerance when comparing file mtimes with the database (seconds). "
            "Increase on filesystems with coarse timestamp precision."
        ),
    )

    # Parallel discovery settings
    parallel_discovery: bool = Field(
        default=True,
        description="Enable parallel directory traversal for large codebases (auto-disabled for <4 top-level dirs)",
    )
    min_dirs_for_parallel: int = Field(
        default=4,
        description="Minimum top-level directories required to activate parallel discovery",
    )
    max_discovery_workers: int = Field(
        default=16,
        description="Maximum worker processes for parallel directory discovery",
    )

    # File patterns
    include: list[str] = Field(
        default_factory=lambda: _get_default_include_patterns(),
        description="Glob patterns for files to include (all supported languages)",
    )

    # NOTE: For backwards-compatibility, users may set indexing.exclude to
    # the sentinel strings ".gitignore" or ".chignore" in .chunkhound.json.
    # We store the sentinel separately and keep `exclude` as a list to preserve
    # existing type semantics across the codebase.
    @staticmethod
    def _default_excludes() -> list[str]:
        return [
            # Virtual environments and package managers
            "**/node_modules/**",
            "**/.git/**",
            "**/__pycache__/**",
            "**/venv/**",
            "**/.venv/**",
            # ChunkHound config and working directory
            "**/.chunkhound/**",
            "**/.chunkhound.json",
            ".chunkhound.json",
            "**/.mypy_cache/**",
            # Build artifacts and distributions
            "**/dist/**",
            "**/build/**",
            "**/target/**",
            "**/.pytest_cache/**",
            # IDE and editor files
            "**/.vscode/**",
            "**/.idea/**",
            "**/.vs/**",
            # Cache and temporary directories
            "**/.cache/**",
            "tmp/**",
            "**/temp/**",
            # Static Site Generators (Docusaurus, Next.js, Gatsby, VuePress, Nuxt)
            "**/.docusaurus/**",
            "**/.docusaurus-cache/**",
            "**/.next/**",
            "**/out/**",
            "**/.nuxt/**",
            "**/.vuepress/dist/**",
            "**/.temp/**",
            # JavaScript bundler and build tool artifacts
            "**/.parcel-cache/**",
            "**/.serverless/**",
            "**/.fusebox/**",
            "**/.dynamodb/**",
            "**/.tern-port",
            "**/.vscode-test/**",
            # Yarn v2+ specific
            "**/.yarn/cache/**",
            "**/.yarn/unplugged/**",
            "**/.yarn/build-state.yml",
            "**/.yarn/install-state.gz",
            "**/.pnp.*",
            # Editor temporary file patterns
            # Vim patterns
            "**/*.swp",
            "**/*.swo",
            "**/.*.swp",
            "**/.*.swo",
            # VS Code / general patterns
            "**/*.tmp.*",
            "**/*.*.tmp",
            "**/*~.tmp",
            # Emacs patterns
            "**/.*#",
            "**/#*#",
            "**/.*~",
            # Generic temp patterns
            "**/*.tmp???",
            "**/*.???tmp",
            # Backup and old files
            "**/*.backup",
            "**/*.bak",
            "**/*~",
            "**/*.old",
            # Minified and generated files
            "**/*.min.js",
            "**/*.min.css",
            "**/*.min.html",
            "**/*.min.svg",
            "**/dist/*.js",
            "**/dist/*.css",
            "**/bundle.js",
            "**/vendor.js",
            "**/webpack.*.js",
            "**/*.bundle.js",
            "**/*.chunk.js",
            # JSON data files (not config)
            "**/*-lock.json",
            "**/package-lock.json",
            "**/yarn.lock",
            "**/composer.lock",
            "**/assets.json",
            "**/*.map.json",
            "**/*.min.json",
        ]

    exclude: list[str] = Field(
        default_factory=_default_excludes,
        description="Glob patterns for files to exclude",
    )

    # Private: sentinel value when user provides exclude as a string
    exclude_sentinel: str | None = Field(default=None, repr=False)

    # Root-level file name for ChunkHound-specific ignores (gitwildmatch syntax)
    chignore_file: str = Field(default=".chignore")

    # When true, also load the CH root's .gitignore as a global overlay in addition
    # to per-repo .gitignore files. This does not affect Git itself; it is a CH-only
    # convenience to apply workspace-wide rules across external repos.
    workspace_gitignore_overlay: bool = Field(
        default=False,
        description="Apply CH root .gitignore as global overlay across repos",
    )

    @field_validator("include", "exclude")
    def validate_patterns(cls, v: list[str]) -> list[str]:
        """Validate glob patterns."""
        if not isinstance(v, list):
            raise ValueError("Patterns must be a list")

        # Remove duplicates while preserving order
        seen = set()
        unique = []
        for pattern in v:
            if pattern not in seen:
                seen.add(pattern)
                unique.append(pattern)

        return unique

    def get_max_file_size_bytes(self) -> int:
        """Get maximum file size in bytes."""
        return self.max_file_size_mb * 1024 * 1024

    def should_index_file(self, file_path: str) -> bool:
        """Check if a file should be indexed based on patterns.

        Note: This is a simplified check. The actual implementation
        should use proper glob matching.
        """
        # This is a placeholder - actual implementation would use
        # pathlib and fnmatch for proper pattern matching
        return True

    @classmethod
    def add_cli_arguments(cls, parser: argparse.ArgumentParser) -> None:
        """Add indexing-related CLI arguments."""
        parser.add_argument(
            "--force-reindex",
            action="store_true",
            help="Force reindexing of all files, even if they haven't changed",
        )

        parser.add_argument(
            "--include",
            action="append",
            help="File patterns to include (can be specified multiple times)",
        )

        parser.add_argument(
            "--exclude",
            action="append",
            help="File patterns to exclude (can be specified multiple times)",
        )

        parser.add_argument(
            "--no-cleanup",
            action="store_true",
            help="Skip cleanup of orphaned database records for files no longer present",
        )

        parser.add_argument(
            "--max-concurrent",
            type=int,
            default=None,
            help=(
                "Maximum concurrent parser workers (processes). "
                "Overrides auto-detected concurrency and internal caps."
            ),
        )

        parser.add_argument(
            "--file-timeout",
            type=float,
            default=None,
            help=(
                "Maximum seconds to spend on any single file before skipping it "
                "(default: 0, disabled)"
            ),
        )
        parser.add_argument(
            "--file-timeout-min-size-kb",
            type=int,
            default=None,
            help=(
                "Only apply the per-file timeout to files >= this size (KB). "
                "Default: 128"
            ),
        )
        parser.add_argument(
            "--mtime-epsilon-seconds",
            type=float,
            default=None,
            help=(
                "Tolerance for mtime comparisons when skipping unchanged files. "
                "Default: 0.01"
            ),
        )
        parser.add_argument(
            "--config-file-size-threshold-kb",
            type=int,
            default=None,
            help=(
                "Structured config (JSON/YAML/TOML) larger than this KB are skipped. "
                "Set to 0 to disable. Default: 20"
            ),
        )

    @classmethod
    def load_from_env(cls) -> dict[str, Any]:
        """Load indexing config from environment variables."""
        config = {}

        if force_reindex := os.getenv("CHUNKHOUND_INDEXING__FORCE_REINDEX"):
            config["force_reindex"] = force_reindex.lower() in ("true", "1", "yes")

        # Handle comma-separated include/exclude patterns
        if include := os.getenv("CHUNKHOUND_INDEXING__INCLUDE"):
            config["include"] = include.split(",")
        if exclude := os.getenv("CHUNKHOUND_INDEXING__EXCLUDE"):
            config["exclude"] = exclude.split(",")

        # Per-file timeout (seconds)
        if per_file_timeout := os.getenv(
            "CHUNKHOUND_INDEXING__PER_FILE_TIMEOUT_SECONDS"
        ):
            try:
                config["per_file_timeout_seconds"] = float(per_file_timeout)
            except ValueError:
                # Ignore invalid env values and keep default
                pass
        if per_file_timeout_min := os.getenv(
            "CHUNKHOUND_INDEXING__PER_FILE_TIMEOUT_MIN_SIZE_KB"
        ):
            try:
                config["per_file_timeout_min_size_kb"] = int(per_file_timeout_min)
            except ValueError:
                pass
        if mtime_eps := os.getenv("CHUNKHOUND_INDEXING__MTIME_EPSILON_SECONDS"):
            try:
                config["mtime_epsilon_seconds"] = float(mtime_eps)
            except ValueError:
                pass
        # Structured config file size threshold
        if cfg_sz := os.getenv("CHUNKHOUND_INDEXING__CONFIG_FILE_SIZE_THRESHOLD_KB"):
            try:
                config["config_file_size_threshold_kb"] = int(cfg_sz)
            except ValueError:
                pass

        # Cleanup orphaned records toggle
        if cleanup := os.getenv("CHUNKHOUND_INDEXING__CLEANUP"):
            config["cleanup"] = cleanup.lower() in ("true", "1", "yes")

        # Concurrency cap for parser workers
        if maxc := os.getenv("CHUNKHOUND_INDEXING__MAX_CONCURRENT"):
            try:
                config["max_concurrent"] = int(maxc)
            except ValueError:
                pass

        return config

    @model_validator(mode="before")
    @classmethod
    def _coerce_exclude_sentinel(cls, data: Any) -> Any:
        """Allow exclude to be a sentinel string ('.gitignore' or '.chignore').

        We convert it into `exclude_sentinel` and normalize `exclude` to a list
        to keep runtime type expectations stable.
        """
        try:
            if isinstance(data, dict) and "exclude" in (data.get("indexing") or data):
                # Support both nested under indexing and direct construction
                if "indexing" in data:
                    idx = data["indexing"] or {}
                    exc = idx.get("exclude")
                    if isinstance(exc, str) and exc in {".gitignore", ".chignore"}:
                        idx["exclude_sentinel"] = exc
                        idx["exclude"] = []
                        data["indexing"] = idx
                else:
                    exc = data.get("exclude")
                    if isinstance(exc, str) and exc in {".gitignore", ".chignore"}:
                        data["exclude_sentinel"] = exc
                        data["exclude"] = []
        except Exception:
            # Be permissive; validation will catch true errors later
            pass
        return data

    # Convenience helper for callers to interpret ignore source selection
    def resolve_ignore_sources(self) -> list[str]:
        if self.exclude_sentinel == ".gitignore":
            return ["gitignore"]
        if self.exclude_sentinel == ".chignore":
            return ["chignore"]
        # Default: combine config array and gitignore
        return ["config", "gitignore"]

    def get_effective_config_excludes(self) -> list[str]:
        """Return the config-driven exclude globs to apply.

        If a sentinel is set (e.g., ".gitignore"), we still apply the default
        exclude set to preserve expected behavior (e.g., node_modules).
        """
        if self.exclude_sentinel in {".gitignore", ".chignore"}:
            return self._default_excludes()
        return list(self.exclude or self._default_excludes())

    @classmethod
    def extract_cli_overrides(cls, args: Any) -> dict[str, Any]:
        """Extract indexing config from CLI arguments."""
        overrides = {}

        if hasattr(args, "force_reindex") and args.force_reindex:
            overrides["force_reindex"] = args.force_reindex

        # Include/exclude patterns
        if hasattr(args, "include") and args.include:
            overrides["include"] = args.include
        if hasattr(args, "exclude") and args.exclude:
            overrides["exclude"] = args.exclude

        # Per-file timeout override
        if hasattr(args, "file_timeout") and args.file_timeout is not None:
            try:
                overrides["per_file_timeout_seconds"] = float(args.file_timeout)
            except (TypeError, ValueError):
                # Ignore invalid values; validation not strict here
                pass
        if (
            hasattr(args, "file_timeout_min_size_kb")
            and args.file_timeout_min_size_kb is not None
        ):
            try:
                overrides["per_file_timeout_min_size_kb"] = int(
                    args.file_timeout_min_size_kb
                )
            except (TypeError, ValueError):
                pass
        if hasattr(args, "mtime_epsilon_seconds") and args.mtime_epsilon_seconds is not None:
            try:
                overrides["mtime_epsilon_seconds"] = float(args.mtime_epsilon_seconds)
            except (TypeError, ValueError):
                pass
        # Structured config file size threshold override via CLI
        if hasattr(args, "config_file_size_threshold_kb") and args.config_file_size_threshold_kb is not None:
            try:
                overrides["config_file_size_threshold_kb"] = int(args.config_file_size_threshold_kb)
            except (TypeError, ValueError):
                pass

        # Cleanup toggle via CLI
        if hasattr(args, "no_cleanup") and args.no_cleanup:
            overrides["cleanup"] = False

        # Concurrency override via CLI
        if hasattr(args, "max_concurrent") and args.max_concurrent is not None:
            try:
                overrides["max_concurrent"] = int(args.max_concurrent)
            except (TypeError, ValueError):
                pass

        return overrides

    def __repr__(self) -> str:
        """String representation of indexing configuration."""
        return (
            f"IndexingConfig("
            f"force_reindex={self.force_reindex}, "
            f"patterns={len(self.include)} includes, {len(self.exclude)} excludes)"
        )
