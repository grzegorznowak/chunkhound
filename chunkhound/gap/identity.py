"""Symbol identity and key construction for `chunkhound gap`."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Literal

import xxhash

from chunkhound.core.models.chunk import Chunk

PrimaryKeyKind = Literal["symbol_key", "stable_key"]


_PART_SUFFIX_RE = re.compile(r"(?:_part\d+)+$")
_LINE_BASED_SYMBOL_RE = re.compile(r"(?:_line_\d+|heading_\d+)$")


def _strip_chunk_part_suffix(symbol: str) -> str:
    return _PART_SUFFIX_RE.sub("", symbol)


def _is_line_based_symbol(symbol: str) -> bool:
    return bool(_LINE_BASED_SYMBOL_RE.search(symbol))


def _normalize_key_component(value: str) -> str:
    return " ".join(value.replace("\n", " ").split()).strip()


def _first_nonempty(*values: Any) -> str | None:
    for v in values:
        if v is None:
            continue
        if isinstance(v, str):
            s = _normalize_key_component(v)
            if s:
                return s
        else:
            s = _normalize_key_component(str(v))
            if s:
                return s
    return None


def _canonicalize_parameters(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, str):
        s = _normalize_key_component(value)
        return s or None
    if isinstance(value, list):
        parts: list[str] = []
        for item in value:
            if isinstance(item, str):
                s = _normalize_key_component(item)
                if s:
                    parts.append(s)
            elif isinstance(item, dict):
                name = _first_nonempty(item.get("name"))
                typ = _first_nonempty(item.get("type"))
                if name and typ:
                    parts.append(f"{name}:{typ}")
                elif name:
                    parts.append(name)
                elif typ:
                    parts.append(typ)
            else:
                s = _normalize_key_component(str(item))
                if s:
                    parts.append(s)
        joined = ",".join(parts)
        return joined or None
    return None


def _xxh3_64_hexdigest(value: str) -> str:
    return xxhash.xxh3_64(value.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class SymbolIdentity:
    stable_key_text: str
    stable_key_hash: str
    symbol_key_text: str | None
    symbol_key_hash: str | None
    primary_key_kind: PrimaryKeyKind
    primary_key_hash: str
    key_strength: int
    symbol: str


def build_symbol_identity(chunk: Chunk) -> SymbolIdentity:
    """Build identity keys for a chunk using mapping metadata when available."""
    metadata = chunk.metadata if isinstance(chunk.metadata, dict) else {}

    raw_symbol = chunk.symbol or "__unnamed__"
    clean_symbol = _strip_chunk_part_suffix(str(raw_symbol))
    symbol_is_line_based = _is_line_based_symbol(clean_symbol)

    language = getattr(chunk.language, "value", str(chunk.language))

    kind = (
        _first_nonempty(
            metadata.get("kind"),
            metadata.get("node_type"),
            chunk.chunk_type.value,
        )
        or "unknown"
    )

    params = _canonicalize_parameters(metadata.get("parameters"))

    type_hints = metadata.get("type_hints")
    ret_hint = None
    if isinstance(type_hints, dict):
        ret_hint = type_hints.get("return")
    ret = _first_nonempty(metadata.get("return_type"), ret_hint)

    stable_sig = f"{clean_symbol}({params or ''})"
    stable_key_text = "::".join(
        [
            _normalize_key_component(language),
            kind,
            stable_sig,
            ret or "",
        ]
    )
    stable_key_hash = _xxh3_64_hexdigest(stable_key_text)

    ns = _first_nonempty(
        metadata.get("namespace"),
        metadata.get("package_name"),
        metadata.get("module"),
        metadata.get("module_name"),
    )
    recv = _first_nonempty(metadata.get("receiver_type"))
    access_modifiers = metadata.get("access_modifiers")
    vis = _first_nonempty(
        metadata.get("visibility"),
        metadata.get("access"),
        ",".join(access_modifiers) if isinstance(access_modifiers, list) else None,
    )

    async_flag = None
    if metadata.get("is_async") or metadata.get("async"):
        async_flag = "async"

    symbol_key_text = None
    symbol_key_hash = None
    if ns or recv or vis or async_flag:
        parts = [
            _normalize_key_component(language),
            ns,
            kind,
            vis,
            async_flag,
            recv,
            stable_sig,
            ret,
        ]
        symbol_key_text = "::".join([p for p in parts if p])
        symbol_key_hash = _xxh3_64_hexdigest(symbol_key_text)

    key_strength = 0
    if ns:
        key_strength += 2
    if recv:
        key_strength += 2
    if params:
        key_strength += 2
    if ret:
        key_strength += 1
    if vis:
        key_strength += 1
    if not symbol_is_line_based:
        key_strength += 1

    # Primary key selection rule (v1): only use symbol_key when it materially reduces
    # collisions (namespace/module or receiver_type) and avoid line-based symbols.
    if (ns or recv) and (not symbol_is_line_based) and symbol_key_hash is not None:
        primary_kind: PrimaryKeyKind = "symbol_key"
        primary_hash = symbol_key_hash
    else:
        primary_kind = "stable_key"
        primary_hash = stable_key_hash

    return SymbolIdentity(
        stable_key_text=stable_key_text,
        stable_key_hash=stable_key_hash,
        symbol_key_text=symbol_key_text,
        symbol_key_hash=symbol_key_hash,
        primary_key_kind=primary_kind,
        primary_key_hash=primary_hash,
        key_strength=key_strength,
        symbol=clean_symbol,
    )
