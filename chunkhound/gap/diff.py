"""Deterministic symbol diffing for `chunkhound gap` (v1 anchor stage)."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable

import xxhash

from chunkhound.gap.manifest import SymbolEntry
from chunkhound.gap.models import (
    GapChangeItem,
    GapFileHandle,
    GapHandle,
    GapSymbolHandle,
)


def _op_sort(op: str) -> int:
    if op == "update":
        return 0
    if op == "add":
        return 1
    return 2


def _locator_tuple(handle: GapHandle) -> tuple[int, int, int]:
    if isinstance(handle, GapSymbolHandle):
        return (
            int(handle.start_line),
            int(handle.end_line),
            int(handle.ordinal_in_file),
        )
    return (0, 0, 0)


def _path_for_change(change: GapChangeItem) -> str:
    if change.new is not None:
        return change.new.path
    if change.old is not None:
        return change.old.path
    return ""


def _primary_key_hash_for_change(change: GapChangeItem) -> str:
    return change.primary_key_hash


def sort_gap_changes(changes: list[GapChangeItem]) -> None:
    """Sort changes in the stable contract order (in-place)."""
    changes.sort(
        key=lambda ch: (
            _op_sort(ch.op),
            ch.entity_kind,
            _path_for_change(ch),
            _primary_key_hash_for_change(ch),
            _locator_tuple(ch.new if ch.new is not None else ch.old),  # type: ignore[arg-type]
        )
    )


def _symbol_locator_sort_key(entry: SymbolEntry) -> tuple[str, int, int, int]:
    h = entry.handle
    return (h.path, int(h.start_line), int(h.end_line), int(h.ordinal_in_file))


def diff_symbol_entries_anchor(
    *,
    a_entries: Iterable[SymbolEntry],
    b_entries: Iterable[SymbolEntry],
) -> list[GapChangeItem]:
    """Compute deterministic anchor diff for symbol entries.

    This stage only uses the selected (primary) identity keys and never makes
    probabilistic inferences.
    """
    grouped_a: dict[tuple[str, str], list[SymbolEntry]] = defaultdict(list)
    grouped_b: dict[tuple[str, str], list[SymbolEntry]] = defaultdict(list)

    for e in a_entries:
        grouped_a[(e.primary_key_kind, e.primary_key_hash)].append(e)
    for e in b_entries:
        grouped_b[(e.primary_key_kind, e.primary_key_hash)].append(e)

    all_groups = sorted(set(grouped_a.keys()) | set(grouped_b.keys()))
    out: list[GapChangeItem] = []

    for key_kind, key_hash in all_groups:
        a_list = sorted(
            grouped_a.get((key_kind, key_hash), []),
            key=_symbol_locator_sort_key,
        )
        b_list = sorted(
            grouped_b.get((key_kind, key_hash), []),
            key=_symbol_locator_sort_key,
        )

        collision_group_size = max(len(a_list), len(b_list)) or 1
        had_collision = collision_group_size > 1

        pairs = min(len(a_list), len(b_list))
        for idx in range(pairs):
            old_entry = a_list[idx]
            new_entry = b_list[idx]
            old = old_entry.handle
            new = new_entry.handle

            moved = old.path != new.path
            renamed = old.symbol != new.symbol
            content_changed = old.text_hash != new.text_hash

            if not (moved or renamed or content_changed):
                continue

            out.append(
                GapChangeItem(
                    entity_kind="symbol",
                    op="update",
                    moved=moved,
                    renamed=renamed,
                    content_changed=content_changed,
                    reason="symbol_anchor",
                    confidence=1.0,
                    primary_key_kind=key_kind,  # type: ignore[arg-type]
                    primary_key_hash=key_hash,
                    key_strength=max(old_entry.key_strength, new_entry.key_strength),
                    had_collision=had_collision,
                    collision_group_size=collision_group_size,
                    old=old,
                    new=new,
                )
            )

        for entry in b_list[pairs:]:
            out.append(
                GapChangeItem(
                    entity_kind="symbol",
                    op="add",
                    moved=False,
                    renamed=False,
                    content_changed=True,
                    reason="symbol_anchor",
                    confidence=1.0,
                    primary_key_kind=key_kind,  # type: ignore[arg-type]
                    primary_key_hash=key_hash,
                    key_strength=entry.key_strength,
                    had_collision=had_collision,
                    collision_group_size=collision_group_size,
                    old=None,
                    new=entry.handle,
                )
            )

        for entry in a_list[pairs:]:
            out.append(
                GapChangeItem(
                    entity_kind="symbol",
                    op="remove",
                    moved=False,
                    renamed=False,
                    content_changed=True,
                    reason="symbol_anchor",
                    confidence=1.0,
                    primary_key_kind=key_kind,  # type: ignore[arg-type]
                    primary_key_hash=key_hash,
                    key_strength=entry.key_strength,
                    had_collision=had_collision,
                    collision_group_size=collision_group_size,
                    old=entry.handle,
                    new=None,
                )
            )

    return out


def make_parse_fallback_file_change(
    *,
    op: str,
    rel_path: str,
    old_file: GapFileHandle | None,
    new_file: GapFileHandle | None,
) -> GapChangeItem:
    """Create a deterministic file-level ChangeItem for parse fallback."""
    if op not in {"add", "remove", "update"}:
        raise ValueError(f"Unsupported op for parse fallback: {op}")
    return GapChangeItem(
        entity_kind="file",
        op=op,  # type: ignore[arg-type]
        moved=False,
        renamed=False,
        content_changed=True,
        reason="parse_fallback",
        confidence=1.0,
        primary_key_kind="path_key",
        primary_key_hash=xxhash.xxh3_64(rel_path.encode("utf-8")).hexdigest(),
        key_strength=0,
        had_collision=False,
        collision_group_size=1,
        old=old_file,
        new=new_file,
    )
