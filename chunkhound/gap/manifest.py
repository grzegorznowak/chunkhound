"""Symbol manifest extraction for `chunkhound gap`."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import xxhash

from chunkhound.core.types.common import ChunkType, FileId
from chunkhound.gap.discovery import DiscoveredFile
from chunkhound.gap.identity import SymbolIdentity, build_symbol_identity
from chunkhound.gap.models import GapSymbolHandle, GapWarning
from chunkhound.parsers.parser_factory import ParserFactory
from chunkhound.parsers.universal_parser import CASTConfig
from chunkhound.utils.normalization import normalize_content


@dataclass(frozen=True)
class SymbolEntry:
    primary_key_hash: str
    primary_key_kind: Literal["symbol_key", "stable_key"]
    key_strength: int
    handle: GapSymbolHandle


@dataclass(frozen=True)
class ParsedFileSymbols:
    rel_path: str
    symbols: list[SymbolEntry]
    parse_failed: bool


def create_gap_parser_factory() -> ParserFactory:
    """Create a ParserFactory tuned for gap runs (symbol granularity, no splitting)."""
    # Goal: keep one chunk per semantic symbol by avoiding emergency splitting.
    # We still keep greedy_merge disabled to avoid merging sibling definitions.
    cast_config = CASTConfig(
        max_chunk_size=1_000_000,
        min_chunk_size=50,
        merge_threshold=0.8,
        preserve_structure=True,
        greedy_merge=False,
        safe_token_limit=1_000_000,
    )
    return ParserFactory(default_cast_config=cast_config)


def _include_chunk(
    chunk_type: ChunkType,
    *,
    include_comments: bool,
    include_docs: bool,
) -> bool:
    if chunk_type == ChunkType.COMMENT:
        return include_comments
    if chunk_type.is_documentation:
        return include_docs
    if chunk_type.is_code:
        return True
    return chunk_type in {ChunkType.TABLE, ChunkType.KEY_VALUE, ChunkType.ARRAY}


def _xxh3_64_text_hash(text: str) -> str:
    return xxhash.xxh3_64(normalize_content(text).encode("utf-8")).hexdigest()


def extract_file_symbols(
    *,
    file: DiscoveredFile,
    parser_factory: ParserFactory,
    include_comments: bool,
    include_docs: bool,
) -> tuple[ParsedFileSymbols, list[GapWarning]]:
    """Parse a single text file and extract symbol entries."""
    warnings: list[GapWarning] = []

    try:
        parser = parser_factory.create_parser_for_file(file.abs_path)
        chunks = parser.parse_file(file.abs_path, FileId(0))
    except Exception as e:
        warnings.append(
            GapWarning(
                code="PARSE_FALLBACK",
                message="Failed to parse file; falling back to file-level change item",
                meta={
                    "path": file.rel_path,
                    "error_type": type(e).__name__,
                },
            )
        )
        return (
            ParsedFileSymbols(rel_path=file.rel_path, symbols=[], parse_failed=True),
            warnings,
        )

    if not chunks:
        # Distinguish between empty file vs unexpected empty parse result.
        if file.size_bytes > 0:
            warnings.append(
                GapWarning(
                    code="PARSE_FALLBACK",
                    message=(
                        "Parser produced no chunks for a non-empty file; "
                        "falling back to file-level change item"
                    ),
                    meta={"path": file.rel_path},
                )
            )
            return (
                ParsedFileSymbols(
                    rel_path=file.rel_path, symbols=[], parse_failed=True
                ),
                warnings,
            )
        return (
            ParsedFileSymbols(rel_path=file.rel_path, symbols=[], parse_failed=False),
            [],
        )

    # Filter chunks deterministically.
    filtered = [
        c
        for c in chunks
        if _include_chunk(
            c.chunk_type,
            include_comments=include_comments,
            include_docs=include_docs,
        )
        # Exclude synthetic file-level structure chunks (import lists, etc.) which are
        # not semantic entities and collide across files (e.g., `file_structure`).
        and not (
            c.chunk_type == ChunkType.NAMESPACE
            and (getattr(c, "symbol", None) or "") == "file_structure"
        )
    ]

    # Assign deterministic ordinals within the file.
    pre: list[
        tuple[tuple[int, int, str, str, str], SymbolIdentity, GapSymbolHandle]
    ] = []
    for c in filtered:
        identity = build_symbol_identity(c)
        text_hash = _xxh3_64_text_hash(c.code)

        handle = GapSymbolHandle(
            path=file.rel_path,
            start_line=int(c.start_line),
            end_line=int(c.end_line),
            symbol=identity.symbol,
            chunk_type=c.chunk_type.value,
            text_hash=text_hash,
            ordinal_in_file=0,
            parse_status="ok",
            stable_key_hash=identity.stable_key_hash,
            symbol_key_hash=identity.symbol_key_hash,
            start_byte=int(c.start_byte) if c.start_byte is not None else None,
            end_byte=int(c.end_byte) if c.end_byte is not None else None,
        )

        sort_key = (
            int(handle.start_line),
            int(handle.end_line),
            handle.chunk_type,
            handle.symbol,
            handle.text_hash,
        )
        pre.append((sort_key, identity, handle))

    pre.sort(key=lambda x: x[0])
    entries: list[SymbolEntry] = []
    for idx, (_, identity, handle) in enumerate(pre, start=1):
        handle2 = GapSymbolHandle(
            path=handle.path,
            start_line=handle.start_line,
            end_line=handle.end_line,
            symbol=handle.symbol,
            chunk_type=handle.chunk_type,
            text_hash=handle.text_hash,
            ordinal_in_file=idx,
            parse_status=handle.parse_status,
            stable_key_hash=handle.stable_key_hash,
            symbol_key_hash=handle.symbol_key_hash,
            start_byte=handle.start_byte,
            end_byte=handle.end_byte,
        )
        entries.append(
            SymbolEntry(
                primary_key_hash=identity.primary_key_hash,
                primary_key_kind=identity.primary_key_kind,
                key_strength=identity.key_strength,
                handle=handle2,
            )
        )

    return (
        ParsedFileSymbols(rel_path=file.rel_path, symbols=entries, parse_failed=False),
        warnings,
    )
