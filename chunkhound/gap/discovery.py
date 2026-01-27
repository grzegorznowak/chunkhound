"""Filesystem discovery and classification for `chunkhound gap`."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import xxhash

from chunkhound.core.config.indexing_config import IndexingConfig
from chunkhound.gap.models import GapInputRef, GapWarning
from chunkhound.utils.file_patterns import (
    normalize_include_patterns,
    walk_directory_tree,
)
from chunkhound.utils.ignore_engine import build_repo_aware_ignore_engine

FileClass = Literal["text", "binary", "too_large", "decode_error", "ignored"]


@dataclass(frozen=True)
class DiscoveredFile:
    rel_path: str
    abs_path: Path
    size_bytes: int
    file_hash: str
    file_class: FileClass


@dataclass(frozen=True)
class SourceSnapshot:
    root: Path
    files: dict[str, DiscoveredFile]
    input_ref: GapInputRef


def _xxh3_64_hexdigest_iter(pairs: list[tuple[str, str]]) -> str:
    h = xxhash.xxh3_64()
    for rel_path, file_hash in pairs:
        h.update(rel_path.encode("utf-8"))
        h.update(b"\x00")
        h.update(file_hash.encode("utf-8"))
        h.update(b"\x00")
    return h.hexdigest()


def compute_scope_hash(
    *,
    a_source_hash: str,
    b_source_hash: str,
    include_patterns: list[str],
    config_excludes: list[str],
    ignore_sources: list[str],
    gitignore_backend: str,
    max_file_size_bytes: int,
) -> str:
    """Compute a deterministic scope hash for this gap run."""
    h = xxhash.xxh3_64()
    h.update(b"gap.v1\x00")
    h.update(a_source_hash.encode("utf-8"))
    h.update(b"\x00")
    h.update(b_source_hash.encode("utf-8"))
    h.update(b"\x00")
    h.update(str(max_file_size_bytes).encode("utf-8"))
    h.update(b"\x00")
    h.update(gitignore_backend.encode("utf-8"))
    h.update(b"\x00")
    for p in sorted(include_patterns):
        h.update(b"inc\x00")
        h.update(p.encode("utf-8"))
        h.update(b"\x00")
    for p in sorted(config_excludes):
        h.update(b"exc\x00")
        h.update(p.encode("utf-8"))
        h.update(b"\x00")
    for s in sorted(ignore_sources):
        h.update(b"src\x00")
        h.update(s.encode("utf-8"))
        h.update(b"\x00")
    return h.hexdigest()


def _classify_and_hash_file(
    *,
    path: Path,
    max_file_size_bytes: int,
) -> tuple[FileClass, str, int]:
    """Return (file_class, file_hash, size_bytes) for a file."""
    try:
        size_bytes = int(path.stat().st_size)
    except OSError:
        # Caller decides how to warn/skip.
        raise

    h = xxhash.xxh3_64()
    seen_first = False
    is_binary = False
    decode_error = False

    should_check_decode = size_bytes <= max_file_size_bytes

    # Fast UTF-8 strict check (deterministic) while hashing.
    decoder = None
    if should_check_decode:
        try:
            import codecs

            decoder = codecs.getincrementaldecoder("utf-8")("strict")
        except Exception:
            decoder = None

    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            if not seen_first:
                seen_first = True
                if b"\x00" in chunk:
                    is_binary = True
                    # If binary, stop spending cycles on decode validation.
                    decoder = None

            h.update(chunk)

            if decoder is not None:
                try:
                    decoder.decode(chunk)
                except UnicodeDecodeError:
                    decode_error = True
                    decoder = None

    file_hash = h.hexdigest()

    if is_binary:
        return "binary", file_hash, size_bytes
    if size_bytes > max_file_size_bytes:
        return "too_large", file_hash, size_bytes
    if decode_error:
        return "decode_error", file_hash, size_bytes
    return "text", file_hash, size_bytes


def discover_source(
    *,
    root: Path,
    indexing: IndexingConfig,
    source_ref_user: str | None = None,
    source_ref_resolved: str | None = None,
) -> tuple[SourceSnapshot, list[GapWarning]]:
    """Discover files under root using the current indexing scope rules."""
    warnings: list[GapWarning] = []
    root = root.resolve()
    resolved_ref = source_ref_resolved or str(root)
    user_ref = source_ref_user or resolved_ref

    include_patterns = normalize_include_patterns(list(indexing.include))
    ignore_sources = list(indexing.resolve_ignore_sources())
    config_excludes = list(indexing.get_effective_config_excludes())

    workspace_root_only_gitignore = getattr(
        indexing, "workspace_gitignore_nonrepo", None
    )

    ignore_engine = build_repo_aware_ignore_engine(
        root=root,
        sources=ignore_sources,
        chignore_file=getattr(indexing, "chignore_file", ".chignore"),
        config_exclude=config_excludes,
        backend=getattr(indexing, "gitignore_backend", "python"),
        workspace_root_only_gitignore=workspace_root_only_gitignore,
    )

    files, _ = walk_directory_tree(
        root,
        root,
        include_patterns,
        [],
        {root: []},
        ignore_engine=ignore_engine,
    )

    # Convert to stable, POSIX relpaths.
    rel_paths: list[tuple[str, Path]] = []
    for p in files:
        try:
            rel_paths.append((p.relative_to(root).as_posix(), p))
        except Exception:
            # Outside root (shouldn't happen), skip deterministically.
            continue

    rel_paths.sort(key=lambda x: x[0])

    max_file_size_bytes = int(indexing.get_max_file_size_bytes())

    discovered: dict[str, DiscoveredFile] = {}
    for rel, abs_path in rel_paths:
        try:
            file_class, file_hash, size_bytes = _classify_and_hash_file(
                path=abs_path, max_file_size_bytes=max_file_size_bytes
            )
        except OSError as e:
            warnings.append(
                GapWarning(
                    code="FILE_READ_ERROR",
                    message="Failed to read file during discovery; skipping",
                    meta={
                        "path": rel,
                        "error_type": type(e).__name__,
                    },
                )
            )
            continue

        discovered[rel] = DiscoveredFile(
            rel_path=rel,
            abs_path=abs_path,
            size_bytes=size_bytes,
            file_hash=file_hash,
            file_class=file_class,
        )

    source_hash = _xxh3_64_hexdigest_iter(
        [(rel, info.file_hash) for rel, info in sorted(discovered.items())]
    )

    snapshot = SourceSnapshot(
        root=root,
        files=discovered,
        input_ref=GapInputRef(
            source_kind="path",
            source_ref=user_ref,
            source_ref_user=user_ref,
            source_ref_resolved=resolved_ref,
            source_hash=source_hash,
        ),
    )
    return snapshot, warnings
