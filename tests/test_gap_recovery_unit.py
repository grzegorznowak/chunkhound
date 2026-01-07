#!/usr/bin/env python3
"""Unit tests for safe recovery in `chunkhound gap`."""

from chunkhound.gap.models import GapChangeItem, GapSymbolHandle
from chunkhound.gap.recovery import recover_safe_text_hash


def _sym_handle(
    *,
    path: str,
    symbol: str,
    chunk_type: str = "code",
    text_hash: str = "t1",
    stable_key_hash: str | None = None,
    ordinal_in_file: int = 1,
) -> GapSymbolHandle:
    return GapSymbolHandle(
        path=path,
        start_line=1,
        end_line=1,
        symbol=symbol,
        chunk_type=chunk_type,
        text_hash=text_hash,
        ordinal_in_file=ordinal_in_file,
        parse_status="ok",
        stable_key_hash=stable_key_hash,
        symbol_key_hash=None,
        start_byte=None,
        end_byte=None,
    )


def test_safe_recovery_collapses_remove_add_into_update_by_text_hash() -> None:
    removed = GapChangeItem(
        entity_kind="symbol",
        op="remove",
        moved=False,
        renamed=False,
        content_changed=True,
        reason="symbol_anchor",
        confidence=1.0,
        primary_key_kind="stable_key",
        key_strength=1,
        had_collision=False,
        collision_group_size=1,
        old=_sym_handle(path="a.py", symbol="foo", text_hash="t1", stable_key_hash="a"),
        new=None,
    )
    added = GapChangeItem(
        entity_kind="symbol",
        op="add",
        moved=False,
        renamed=False,
        content_changed=True,
        reason="symbol_anchor",
        confidence=1.0,
        primary_key_kind="stable_key",
        key_strength=1,
        had_collision=False,
        collision_group_size=1,
        old=None,
        new=_sym_handle(path="b.py", symbol="foo", text_hash="t1", stable_key_hash="b"),
    )

    recovered, warnings = recover_safe_text_hash([removed, added])
    assert warnings == []
    assert len(recovered) == 1
    ch = recovered[0]
    assert ch.op == "update"
    assert ch.reason == "recovery_safe_text_hash"
    assert ch.confidence == 1.0
    assert ch.primary_key_kind == "stable_key"
    assert ch.key_strength == 0
    assert ch.moved is True
    assert ch.renamed is False
    assert ch.content_changed is False


def test_safe_recovery_is_multiset_deterministic_under_collisions() -> None:
    removed1 = GapChangeItem(
        entity_kind="symbol",
        op="remove",
        moved=False,
        renamed=False,
        content_changed=True,
        reason="symbol_anchor",
        confidence=1.0,
        primary_key_kind="stable_key",
        key_strength=1,
        had_collision=False,
        collision_group_size=1,
        old=_sym_handle(path="a.py", symbol="foo", text_hash="t1", stable_key_hash="a"),
        new=None,
    )
    removed2 = GapChangeItem(
        entity_kind="symbol",
        op="remove",
        moved=False,
        renamed=False,
        content_changed=True,
        reason="symbol_anchor",
        confidence=1.0,
        primary_key_kind="stable_key",
        key_strength=1,
        had_collision=False,
        collision_group_size=1,
        old=_sym_handle(
            path="b.py",
            symbol="foo",
            text_hash="t1",
            stable_key_hash="a2",
            ordinal_in_file=2,
        ),
        new=None,
    )
    added1 = GapChangeItem(
        entity_kind="symbol",
        op="add",
        moved=False,
        renamed=False,
        content_changed=True,
        reason="symbol_anchor",
        confidence=1.0,
        primary_key_kind="stable_key",
        key_strength=1,
        had_collision=False,
        collision_group_size=1,
        old=None,
        new=_sym_handle(path="c.py", symbol="foo", text_hash="t1", stable_key_hash="b"),
    )
    added2 = GapChangeItem(
        entity_kind="symbol",
        op="add",
        moved=False,
        renamed=False,
        content_changed=True,
        reason="symbol_anchor",
        confidence=1.0,
        primary_key_kind="stable_key",
        key_strength=1,
        had_collision=False,
        collision_group_size=1,
        old=None,
        new=_sym_handle(
            path="d.py",
            symbol="foo",
            text_hash="t1",
            stable_key_hash="b2",
            ordinal_in_file=2,
        ),
    )

    recovered, warnings = recover_safe_text_hash([removed2, added2, added1, removed1])
    assert warnings == []
    assert len(recovered) == 2
    assert all(c.op == "update" for c in recovered)
    assert all(c.had_collision is True for c in recovered)
    assert all(c.collision_group_size == 2 for c in recovered)
    assert all(c.confidence == 0.5 for c in recovered)


def test_safe_recovery_skips_oversized_bucket_and_emits_warning() -> None:
    removed = [
        GapChangeItem(
            entity_kind="symbol",
            op="remove",
            moved=False,
            renamed=False,
            content_changed=True,
            reason="symbol_anchor",
            confidence=1.0,
            primary_key_kind="stable_key",
            key_strength=1,
            had_collision=False,
            collision_group_size=1,
            old=_sym_handle(
                path=f"a{i}.py",
                symbol="foo",
                text_hash="t1",
                stable_key_hash=f"ra{i}",
                ordinal_in_file=i,
            ),
            new=None,
        )
        for i in range(1, 10)
    ]
    added = [
        GapChangeItem(
            entity_kind="symbol",
            op="add",
            moved=False,
            renamed=False,
            content_changed=True,
            reason="symbol_anchor",
            confidence=1.0,
            primary_key_kind="stable_key",
            key_strength=1,
            had_collision=False,
            collision_group_size=1,
            old=None,
            new=_sym_handle(
                path=f"b{i}.py",
                symbol="foo",
                text_hash="t1",
                stable_key_hash=f"aa{i}",
                ordinal_in_file=i,
            ),
        )
        for i in range(1, 10)
    ]

    recovered, warnings = recover_safe_text_hash([*removed, *added])
    assert len(recovered) == len(removed) + len(added)
    assert all(c.op in {"add", "remove"} for c in recovered)

    assert len(warnings) == 1
    assert warnings[0].code == "RECOVERY_SAFE_BUCKET_SKIPPED"
    assert (warnings[0].meta or {}).get("cap") == 8
    assert (warnings[0].meta or {}).get("adds") == 9
    assert (warnings[0].meta or {}).get("removes") == 9


def test_safe_recovery_caps_total_pairs_and_emits_warning() -> None:
    removed = [
        GapChangeItem(
            entity_kind="symbol",
            op="remove",
            moved=False,
            renamed=False,
            content_changed=True,
            reason="symbol_anchor",
            confidence=1.0,
            primary_key_kind="stable_key",
            key_strength=1,
            had_collision=False,
            collision_group_size=1,
            old=_sym_handle(
                path=f"a{i}.py",
                symbol="foo",
                text_hash="t1",
                stable_key_hash=f"ra{i}",
                ordinal_in_file=i,
            ),
            new=None,
        )
        for i in range(1, 6)
    ]
    added = [
        GapChangeItem(
            entity_kind="symbol",
            op="add",
            moved=False,
            renamed=False,
            content_changed=True,
            reason="symbol_anchor",
            confidence=1.0,
            primary_key_kind="stable_key",
            key_strength=1,
            had_collision=False,
            collision_group_size=1,
            old=None,
            new=_sym_handle(
                path=f"b{i}.py",
                symbol="foo",
                text_hash="t1",
                stable_key_hash=f"aa{i}",
                ordinal_in_file=i,
            ),
        )
        for i in range(1, 6)
    ]

    recovered, warnings = recover_safe_text_hash(
        [*removed, *added],
        recovery_max_total_recovered=3,
    )

    assert len([c for c in recovered if c.op == "update"]) == 3
    assert len([c for c in recovered if c.op == "add"]) == 2
    assert len([c for c in recovered if c.op == "remove"]) == 2

    assert any(w.code == "RECOVERY_SAFE_CAPPED_TOTAL" for w in warnings)
    capped = next(w for w in warnings if w.code == "RECOVERY_SAFE_CAPPED_TOTAL")
    assert (capped.meta or {}).get("cap") == 3
    assert (capped.meta or {}).get("attempted") == 5
    assert (capped.meta or {}).get("applied") == 3


def test_safe_recovery_placeholder_numeric_drift_does_not_count_as_rename() -> None:
    removed = GapChangeItem(
        entity_kind="symbol",
        op="remove",
        moved=False,
        renamed=False,
        content_changed=True,
        reason="symbol_anchor",
        confidence=1.0,
        primary_key_kind="stable_key",
        key_strength=1,
        had_collision=False,
        collision_group_size=1,
        old=_sym_handle(
            path="a.py",
            symbol="block_line_224",
            text_hash="t1",
            stable_key_hash="ra",
        ),
        new=None,
    )
    added = GapChangeItem(
        entity_kind="symbol",
        op="add",
        moved=False,
        renamed=False,
        content_changed=True,
        reason="symbol_anchor",
        confidence=1.0,
        primary_key_kind="stable_key",
        key_strength=1,
        had_collision=False,
        collision_group_size=1,
        old=None,
        new=_sym_handle(
            path="b.py",
            symbol="block_line_230",
            text_hash="t1",
            stable_key_hash="aa",
        ),
    )

    recovered, warnings = recover_safe_text_hash([removed, added])
    assert warnings == []
    assert len(recovered) == 1
    assert recovered[0].op == "update"
    assert recovered[0].moved is True
    assert recovered[0].renamed is False
