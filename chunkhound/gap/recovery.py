"""Deterministic recovery pairing for `chunkhound gap` (v1)."""

from __future__ import annotations

from collections import defaultdict
import re

import xxhash

from chunkhound.gap.models import GapChangeItem, GapSymbolHandle, GapWarning


def _symbol_handle_sort_key(handle: GapSymbolHandle) -> tuple[str, int, int, int]:
    return (
        handle.path,
        int(handle.start_line),
        int(handle.end_line),
        int(handle.ordinal_in_file),
    )


_PLACEHOLDER_SYMBOL_BASE_RE = re.compile(r"^(?P<base>(?:.*_line_|heading_))\d+$")


def _placeholder_symbol_base(symbol: str) -> str | None:
    match = _PLACEHOLDER_SYMBOL_BASE_RE.match(symbol)
    if match is None:
        return None
    return match.group("base")


def recover_safe_text_hash(
    changes: list[GapChangeItem],
    *,
    recovery_max_bucket_size: int = 8,
    recovery_max_total_recovered: int = 1000,
) -> tuple[list[GapChangeItem], list[GapWarning]]:
    """Collapse leftover symbol add/remove pairs via exact `(chunk_type, text_hash)`."""
    warnings: list[GapWarning] = []

    def _primary_key_hash_for_symbol_update(
        *, old: GapSymbolHandle, new: GapSymbolHandle
    ) -> str:
        if new.stable_key_hash:
            return new.stable_key_hash
        if old.stable_key_hash:
            return old.stable_key_hash
        if new.symbol_key_hash:
            return new.symbol_key_hash
        if old.symbol_key_hash:
            return old.symbol_key_hash
        path = new.path or old.path
        return xxhash.xxh3_64(path.encode("utf-8")).hexdigest()

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

    attempted_pairs_total = 0
    applied_total = 0

    all_keys = sorted(set(adds.keys()) | set(removes.keys()))
    for key in all_keys:
        chunk_type, text_hash = key
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

        if collision_group_size > recovery_max_bucket_size:
            warnings.append(
                GapWarning(
                    code="RECOVERY_SAFE_BUCKET_SKIPPED",
                    message=(
                        "Safe recovery bucket exceeded cap; skipping recovery for this"
                        " (chunk_type,text_hash) bucket"
                    ),
                    meta={
                        "chunk_type": chunk_type,
                        "text_hash": text_hash,
                        "adds": len(add_list),
                        "removes": len(rem_list),
                        "cap": recovery_max_bucket_size,
                    },
                )
            )
            out.extend(add_list)
            out.extend(rem_list)
            continue

        pairs_possible = min(len(add_list), len(rem_list))
        attempted_pairs_total += pairs_possible

        remaining_capacity = max(recovery_max_total_recovered - applied_total, 0)
        pairs = min(pairs_possible, remaining_capacity)

        for idx in range(pairs):
            add_item = add_list[idx]
            rem_item = rem_list[idx]
            old = rem_item.old
            new = add_item.new
            if not isinstance(old, GapSymbolHandle) or not isinstance(
                new, GapSymbolHandle
            ):
                continue
            applied_total += 1

            moved = old.path != new.path
            renamed = old.symbol != new.symbol
            if renamed:
                old_base = _placeholder_symbol_base(old.symbol)
                if old_base is not None and old_base == _placeholder_symbol_base(
                    new.symbol
                ):
                    renamed = False
            content_changed = old.text_hash != new.text_hash

            if not (moved or renamed or content_changed):
                continue

            confidence = 1.0
            if had_collision:
                confidence = max(0.1, 1.0 / float(collision_group_size))

            out.append(
                GapChangeItem(
                    entity_kind="symbol",
                    op="update",
                    moved=moved,
                    renamed=renamed,
                    content_changed=content_changed,
                    reason="recovery_safe_text_hash",
                    confidence=confidence,
                    primary_key_kind="stable_key",
                    primary_key_hash=_primary_key_hash_for_symbol_update(
                        old=old, new=new
                    ),
                    key_strength=0,
                    had_collision=had_collision,
                    collision_group_size=collision_group_size,
                    old=old,
                    new=new,
                )
            )

        out.extend(add_list[pairs:])
        out.extend(rem_list[pairs:])

    if attempted_pairs_total > recovery_max_total_recovered:
        warnings.append(
            GapWarning(
                code="RECOVERY_SAFE_CAPPED_TOTAL",
                message=(
                    "Safe recovery exceeded total cap; skipped some recovery pairs"
                ),
                meta={
                    "cap": recovery_max_total_recovered,
                    "attempted": attempted_pairs_total,
                    "applied": applied_total,
                },
            )
        )

    return out, warnings
