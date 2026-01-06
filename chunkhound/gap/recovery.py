"""Deterministic recovery pairing for `chunkhound gap` (v1)."""

from __future__ import annotations

from collections import defaultdict

from chunkhound.gap.models import GapChangeItem, GapSymbolHandle, GapWarning


def _symbol_handle_sort_key(handle: GapSymbolHandle) -> tuple[str, int, int, int]:
    return (
        handle.path,
        int(handle.start_line),
        int(handle.end_line),
        int(handle.ordinal_in_file),
    )


def recover_safe_text_hash(
    changes: list[GapChangeItem],
) -> tuple[list[GapChangeItem], list[GapWarning]]:
    """Collapse leftover symbol add/remove pairs via exact `(chunk_type, text_hash)`."""
    warnings: list[GapWarning] = []

    kept: list[GapChangeItem] = []
    adds: dict[tuple[str, str], list[GapChangeItem]] = defaultdict(list)
    removes: dict[tuple[str, str], list[GapChangeItem]] = defaultdict(list)

    for ch in changes:
        if ch.entity_kind != "symbol":
            kept.append(ch)
            continue

        if ch.op == "add" and isinstance(ch.new, GapSymbolHandle):
            adds[(ch.new.chunk_type, ch.new.text_hash)].append(ch)
            continue

        if ch.op == "remove" and isinstance(ch.old, GapSymbolHandle):
            removes[(ch.old.chunk_type, ch.old.text_hash)].append(ch)
            continue

        kept.append(ch)

    out: list[GapChangeItem] = list(kept)

    all_keys = sorted(set(adds.keys()) | set(removes.keys()))
    for key in all_keys:
        add_list = sorted(
            adds.get(key, []),
            key=lambda ch: _symbol_handle_sort_key(ch.new),  # type: ignore[arg-type]
        )
        rem_list = sorted(
            removes.get(key, []),
            key=lambda ch: _symbol_handle_sort_key(ch.old),  # type: ignore[arg-type]
        )

        collision_group_size = max(len(add_list), len(rem_list)) or 1
        had_collision = collision_group_size > 1

        pairs = min(len(add_list), len(rem_list))
        for idx in range(pairs):
            add_item = add_list[idx]
            rem_item = rem_list[idx]
            old = rem_item.old
            new = add_item.new
            if not isinstance(old, GapSymbolHandle) or not isinstance(
                new, GapSymbolHandle
            ):
                continue

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
                    reason="recovery_safe_text_hash",
                    confidence=1.0,
                    primary_key_kind="stable_key",
                    key_strength=0,
                    had_collision=had_collision,
                    collision_group_size=collision_group_size,
                    old=old,
                    new=new,
                )
            )

        out.extend(add_list[pairs:])
        out.extend(rem_list[pairs:])

    return out, warnings
