"""Advisory move/update pairing suggestions for `chunkhound gap`.

This module emits a separate, non-contractual artifact (`gap.suggestions.v1`) that
references `gap.v1` changes by `change_index`.

Design constraint: never mutate `gap.json` output.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

from chunkhound.gap.models import GapChangeItem, GapReport, GapSymbolHandle


@dataclass(frozen=True)
class MoveSuggestion:
    pair_id: int
    remove_change_index: int
    add_change_index: int
    method: str
    confidence: float


def _is_symbol_add(change: GapChangeItem) -> bool:
    return change.entity_kind == "symbol" and change.op == "add" and isinstance(
        change.new, GapSymbolHandle
    )


def _is_symbol_remove(change: GapChangeItem) -> bool:
    return change.entity_kind == "symbol" and change.op == "remove" and isinstance(
        change.old, GapSymbolHandle
    )


def collect_unresolved(
    report: GapReport,
    *,
    include_blocks: bool,
) -> tuple[list[tuple[int, GapChangeItem]], list[tuple[int, GapChangeItem]]]:
    # TODO: consider adding caps/filters for block-heavy diffs to avoid exploding
    # candidate pools. For now we intentionally include blocks and do not cap.
    adds: list[tuple[int, GapChangeItem]] = []
    removes: list[tuple[int, GapChangeItem]] = []

    for idx, ch in enumerate(report.changes):
        if not isinstance(ch, GapChangeItem):
            continue

        if _is_symbol_add(ch):
            handle = ch.new
            assert isinstance(handle, GapSymbolHandle)
            if not include_blocks and handle.chunk_type == "block":
                continue
            adds.append((int(idx), ch))
            continue

        if _is_symbol_remove(ch):
            handle = ch.old
            assert isinstance(handle, GapSymbolHandle)
            if not include_blocks and handle.chunk_type == "block":
                continue
            removes.append((int(idx), ch))
            continue

    return adds, removes


def build_move_suggestions_payload(
    *,
    report: GapReport,
    include_blocks: bool,
) -> dict[str, Any]:
    # v0: heuristics/embeddings/LLM stages implemented in later steps.
    _adds, _removes = collect_unresolved(report, include_blocks=include_blocks)

    payload: dict[str, Any] = {
        "schema_version": "gap.suggestions.v1",
        "source_schema_version": report.schema_version,
        "direction": report.direction,
        "source_scope_hash": report.scope.scope_hash,
        "config_snapshot": {
            "include_blocks": bool(include_blocks),
        },
        "counts_by_method": {},
        "skipped_reasons": {},
        "suggestions": [],
        "llm": {
            "status": "not_run",
            "incomplete": False,
        },
    }
    return payload


def write_move_suggestions(*, out_dir: Path, payload: dict[str, Any]) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "move_suggestions.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


__all__ = [
    "MoveSuggestion",
    "build_move_suggestions_payload",
    "collect_unresolved",
    "write_move_suggestions",
]
