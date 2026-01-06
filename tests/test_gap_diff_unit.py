#!/usr/bin/env python3
"""Unit tests for deterministic anchor diff pairing in `chunkhound gap`."""

from chunkhound.gap.diff import diff_symbol_entries_anchor
from chunkhound.gap.manifest import SymbolEntry
from chunkhound.gap.models import GapSymbolHandle


def _sym(
    *,
    path: str,
    symbol: str,
    text_hash: str,
    stable_key_hash: str,
    start_line: int = 1,
    end_line: int = 1,
    ordinal_in_file: int = 1,
) -> SymbolEntry:
    handle = GapSymbolHandle(
        path=path,
        start_line=start_line,
        end_line=end_line,
        symbol=symbol,
        chunk_type="code",
        text_hash=text_hash,
        ordinal_in_file=ordinal_in_file,
        parse_status="ok",
        stable_key_hash=stable_key_hash,
        symbol_key_hash=None,
        start_byte=None,
        end_byte=None,
    )
    return SymbolEntry(
        primary_key_hash=stable_key_hash,
        primary_key_kind="stable_key",
        key_strength=3,
        handle=handle,
    )


def test_anchor_diff_skips_unchanged_pairs() -> None:
    a = [_sym(path="a.py", symbol="foo", text_hash="t1", stable_key_hash="k1")]
    b = [_sym(path="a.py", symbol="foo", text_hash="t1", stable_key_hash="k1")]
    assert diff_symbol_entries_anchor(a_entries=a, b_entries=b) == []


def test_anchor_diff_detects_move_without_content_change() -> None:
    a = [_sym(path="a.py", symbol="foo", text_hash="t1", stable_key_hash="k1")]
    b = [_sym(path="b.py", symbol="foo", text_hash="t1", stable_key_hash="k1")]
    changes = diff_symbol_entries_anchor(a_entries=a, b_entries=b)
    assert len(changes) == 1
    assert changes[0].op == "update"
    assert changes[0].moved is True
    assert changes[0].content_changed is False
    assert changes[0].renamed is False


def test_anchor_diff_marks_collisions_and_pairs_by_sorted_locator() -> None:
    a = [
        _sym(path="a.py", symbol="foo", text_hash="t1", stable_key_hash="k1"),
        _sym(path="b.py", symbol="foo", text_hash="t1", stable_key_hash="k1"),
    ]
    b = [
        _sym(path="c.py", symbol="foo", text_hash="t1", stable_key_hash="k1"),
        _sym(path="d.py", symbol="foo", text_hash="t1", stable_key_hash="k1"),
    ]
    changes = diff_symbol_entries_anchor(a_entries=a, b_entries=b)
    assert len(changes) == 2
    assert all(c.op == "update" for c in changes)
    assert all(c.had_collision is True for c in changes)
    assert all(c.collision_group_size == 2 for c in changes)

