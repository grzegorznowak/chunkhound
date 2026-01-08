#!/usr/bin/env python3
"""Unit tests for `chunkhound.gap.move_suggestions`."""

from __future__ import annotations

import pytest

import math
from dataclasses import dataclass

from chunkhound.gap.models import (
    GapChangeItem,
    GapCounts,
    GapInputRef,
    GapInputs,
    GapInvariants,
    GapNormalizationInvariants,
    GapReport,
    GapScope,
    GapStats,
    GapSymbolHandle,
    GapTimings,
)
from chunkhound.gap.move_suggestions import build_move_suggestions_payload

pytestmark = pytest.mark.asyncio


def _make_report(*, changes: list[GapChangeItem]) -> GapReport:
    return GapReport(
        schema_version="gap.v1",
        direction="A->B",
        invariants=GapInvariants(
            hash_alg="xxh3_64",
            normalization=GapNormalizationInvariants(
                id="normalize_content.v1", include_comments=False, include_docs=False
            ),
            chunker_version="cast@v1",
            recovery_mode="off",
            deterministic=True,
            embed_model_id=None,
            forced=False,
        ),
        inputs=GapInputs(
            a=GapInputRef(source_kind="path", source_ref="a", source_hash="a"),
            b=GapInputRef(source_kind="path", source_ref="b", source_hash="b"),
        ),
        scope=GapScope(
            scope_mode="full",
            scope_hash="x",
            changed_files_count=1,
            rename_hints_count=0,
        ),
        warnings=[],
        stats=GapStats(counts=GapCounts(), timings=GapTimings()),
        changes=changes,
    )


def _handle(*, path: str, chunk_type: str, symbol: str, text_hash: str) -> GapSymbolHandle:
    return GapSymbolHandle(
        path=path,
        start_line=1,
        end_line=1,
        symbol=symbol,
        chunk_type=chunk_type,
        text_hash=text_hash,
        ordinal_in_file=1,
        parse_status="ok",
    )


async def test_build_move_suggestions_payload_pairs_unique_same_path() -> None:
    old = _handle(path="a.py", chunk_type="function", symbol="foo", text_hash="old")
    new = _handle(path="a.py", chunk_type="function", symbol="foo", text_hash="new")
    report = _make_report(
        changes=[
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
                old=old,
                new=None,
            ),
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
                new=new,
            ),
        ]
    )

    payload = await build_move_suggestions_payload(report=report, include_blocks=True)
    suggestions = payload.get("suggestions")
    assert isinstance(suggestions, list)
    assert len(suggestions) == 1

    s0 = suggestions[0]
    assert isinstance(s0, dict)
    assert s0["pair_id"] == 1
    assert s0["remove_change_index"] == 0
    assert s0["add_change_index"] == 1
    assert s0["method"] == "heuristic_same_path"
    assert isinstance(s0.get("rationale"), str)

    counts = payload.get("counts_by_method")
    assert counts == {"heuristic_same_path": 1}


async def test_build_move_suggestions_payload_skips_ambiguous_same_path() -> None:
    old = _handle(path="a.py", chunk_type="function", symbol="foo", text_hash="old")
    new1 = _handle(path="a.py", chunk_type="function", symbol="foo", text_hash="new1")
    new2 = _handle(path="a.py", chunk_type="function", symbol="foo", text_hash="new2")
    report = _make_report(
        changes=[
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
                old=old,
                new=None,
            ),
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
                new=new1,
            ),
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
                new=new2,
            ),
        ]
    )

    payload = await build_move_suggestions_payload(report=report, include_blocks=True)
    suggestions = payload.get("suggestions")
    assert isinstance(suggestions, list)
    assert not suggestions


@dataclass(frozen=True)
class _Vec:
    x: float
    y: float

    def as_list(self) -> list[float]:
        return [float(self.x), float(self.y)]


class _TagEmbeddingProvider:
    async def embed(self, texts: list[str]) -> list[list[float]]:
        out: list[list[float]] = []
        for t in texts:
            if "PAIR1" in t:
                out.append(_Vec(1.0, 0.0).as_list())
                continue
            if "PAIR2" in t:
                out.append(_Vec(0.0, 1.0).as_list())
                continue
            if "BASE" in t:
                out.append(_Vec(1.0, 0.0).as_list())
                continue
            if "SIM085" in t:
                # Unit vector whose dot with [1,0] is 0.85.
                out.append(_Vec(0.85, math.sqrt(1.0 - 0.85**2)).as_list())
                continue
            if "ORTHO" in t:
                out.append(_Vec(0.0, 1.0).as_list())
                continue
            out.append(_Vec(0.0, 0.0).as_list())
        return out


class _StubLLMProvider:
    def __init__(self, *, chosen_add_change_index: int | None):
        self._chosen_add_change_index = chosen_add_change_index

    async def complete_structured(  # type: ignore[override]
        self,
        prompt: str,
        json_schema: dict[str, object],
        system: str | None = None,
        max_completion_tokens: int = 4096,
        timeout: int | None = None,
    ) -> dict[str, object]:
        _ = (prompt, json_schema, system, max_completion_tokens, timeout)
        return {
            "add_change_index": self._chosen_add_change_index,
            "confidence": 0.77,
            "rationale": "stub choice for unit test",
        }


@pytest.mark.asyncio
async def test_build_move_suggestions_payload_embedding_crosscheck_accepts_clear_match() -> None:
    provider = _TagEmbeddingProvider()

    old = _handle(path="a.py", chunk_type="function", symbol="old_sym", text_hash="x")
    new1 = _handle(path="b.py", chunk_type="function", symbol="new_sym_1", text_hash="y")
    new2 = _handle(path="c.py", chunk_type="function", symbol="new_sym_2", text_hash="z")
    report = _make_report(
        changes=[
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
                old=old,
                new=None,
            ),
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
                new=new1,
            ),
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
                new=new2,
            ),
        ]
    )

    payload = await build_move_suggestions_payload(
        report=report,
        include_blocks=True,
        embedding_provider=provider,  # type: ignore[arg-type]
        texts_a_by_path_ordinal={("a.py", 1): "PAIR1"},
        texts_b_by_path_ordinal={("b.py", 1): "PAIR1", ("c.py", 1): "PAIR2"},
    )

    suggestions = payload.get("suggestions")
    assert isinstance(suggestions, list)
    assert len(suggestions) == 1
    assert suggestions[0]["method"] == "embedding_crosscheck"
    assert suggestions[0]["remove_change_index"] == 0
    assert suggestions[0]["add_change_index"] == 1
    assert suggestions[0]["pair_id"] == 1


@pytest.mark.asyncio
async def test_build_move_suggestions_payload_embedding_crosscheck_uses_block_thresholds() -> None:
    provider = _TagEmbeddingProvider()

    old = _handle(path="a.py", chunk_type="block", symbol="old_sym", text_hash="x")
    new1 = _handle(path="b.py", chunk_type="block", symbol="new_sym_1", text_hash="y")
    new2 = _handle(path="c.py", chunk_type="block", symbol="new_sym_2", text_hash="z")
    report = _make_report(
        changes=[
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
                old=old,
                new=None,
            ),
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
                new=new1,
            ),
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
                new=new2,
            ),
        ]
    )

    payload = await build_move_suggestions_payload(
        report=report,
        include_blocks=True,
        embedding_provider=provider,  # type: ignore[arg-type]
        texts_a_by_path_ordinal={("a.py", 1): "BASE"},
        texts_b_by_path_ordinal={("b.py", 1): "SIM085", ("c.py", 1): "ORTHO"},
    )

    suggestions = payload.get("suggestions")
    assert isinstance(suggestions, list)
    assert not suggestions

    skipped = payload.get("skipped_reasons")
    assert skipped.get("embedding_Crosscheck_below_min_score") == 1


@pytest.mark.asyncio
async def test_build_move_suggestions_payload_embedding_crosscheck_rejects_ambiguous_margin() -> None:
    provider = _TagEmbeddingProvider()

    old = _handle(path="a.py", chunk_type="function", symbol="old_sym", text_hash="x")
    new1 = _handle(path="b.py", chunk_type="function", symbol="new_sym_1", text_hash="y")
    new2 = _handle(path="c.py", chunk_type="function", symbol="new_sym_2", text_hash="z")
    report = _make_report(
        changes=[
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
                old=old,
                new=None,
            ),
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
                new=new1,
            ),
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
                new=new2,
            ),
        ]
    )

    payload = await build_move_suggestions_payload(
        report=report,
        include_blocks=True,
        embedding_provider=provider,  # type: ignore[arg-type]
        texts_a_by_path_ordinal={("a.py", 1): "PAIR1"},
        texts_b_by_path_ordinal={("b.py", 1): "PAIR1", ("c.py", 1): "PAIR1"},
    )

    suggestions = payload.get("suggestions")
    assert isinstance(suggestions, list)
    assert not suggestions

    skipped = payload.get("skipped_reasons")
    assert skipped.get("embedding_Crosscheck_below_min_margin") == 1


@pytest.mark.asyncio
async def test_build_move_suggestions_payload_embedding_crosscheck_requires_mutual_best() -> None:
    provider = _TagEmbeddingProvider()

    old1 = _handle(path="a.py", chunk_type="function", symbol="old_sym_1", text_hash="x")
    old2 = _handle(path="b.py", chunk_type="function", symbol="old_sym_2", text_hash="y")
    new = _handle(path="c.py", chunk_type="function", symbol="new_sym", text_hash="z")
    report = _make_report(
        changes=[
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
                old=old1,
                new=None,
            ),
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
                old=old2,
                new=None,
            ),
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
                new=new,
            ),
        ]
    )

    payload = await build_move_suggestions_payload(
        report=report,
        include_blocks=True,
        embedding_provider=provider,  # type: ignore[arg-type]
        texts_a_by_path_ordinal={("a.py", 1): "PAIR1", ("b.py", 1): "PAIR1"},
        texts_b_by_path_ordinal={("c.py", 1): "PAIR1"},
    )

    suggestions = payload.get("suggestions")
    assert isinstance(suggestions, list)
    assert len(suggestions) == 1
    assert suggestions[0]["method"] == "embedding_crosscheck"
    assert suggestions[0]["remove_change_index"] == 0
    assert suggestions[0]["add_change_index"] == 2

    skipped = payload.get("skipped_reasons")
    assert skipped.get("embedding_Crosscheck_not_mutual_best") == 1


async def test_build_move_suggestions_payload_pairs_unique_same_dir() -> None:
    old = _handle(path="src/a.py", chunk_type="function", symbol="foo", text_hash="old")
    new = _handle(path="src/b.py", chunk_type="function", symbol="foo", text_hash="new")
    report = _make_report(
        changes=[
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
                old=old,
                new=None,
            ),
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
                new=new,
            ),
        ]
    )

    payload = await build_move_suggestions_payload(report=report, include_blocks=True)
    suggestions = payload.get("suggestions")
    assert isinstance(suggestions, list)
    assert len(suggestions) == 1
    assert suggestions[0]["method"] == "heuristic_same_dir"
    assert suggestions[0]["remove_change_index"] == 0
    assert suggestions[0]["add_change_index"] == 1


async def test_build_move_suggestions_payload_excludes_blocks_when_disabled() -> None:
    old = _handle(path="a.py", chunk_type="block", symbol="block_line_1", text_hash="x")
    new = _handle(path="a.py", chunk_type="block", symbol="block_line_1", text_hash="y")
    report = _make_report(
        changes=[
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
                old=old,
                new=None,
            ),
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
                new=new,
            ),
        ]
    )

    payload = await build_move_suggestions_payload(report=report, include_blocks=False)
    suggestions = payload.get("suggestions")
    assert isinstance(suggestions, list)
    assert not suggestions


@pytest.mark.asyncio
async def test_build_move_suggestions_payload_llm_tiebreak_runs_when_requested_and_embedding_candidates_exist() -> None:
    embedding_provider = _TagEmbeddingProvider()
    llm_provider = _StubLLMProvider(chosen_add_change_index=1)

    old = _handle(path="a.py", chunk_type="function", symbol="old_sym", text_hash="x")
    new1 = _handle(path="b.py", chunk_type="function", symbol="new_sym_1", text_hash="y")
    new2 = _handle(path="c.py", chunk_type="function", symbol="new_sym_2", text_hash="z")
    report = _make_report(
        changes=[
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
                old=old,
                new=None,
            ),
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
                new=new1,
            ),
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
                new=new2,
            ),
        ]
    )

    payload = await build_move_suggestions_payload(
        report=report,
        include_blocks=True,
        embedding_provider=embedding_provider,  # type: ignore[arg-type]
        texts_a_by_path_ordinal={("a.py", 1): "PAIR1"},
        texts_b_by_path_ordinal={("b.py", 1): "PAIR1", ("c.py", 1): "PAIR2"},
        embed_min_score=1.1,  # force embedding stage to NOT auto-accept
        embed_min_margin=1.1,
        llm_provider=llm_provider,  # type: ignore[arg-type]
        llm_enabled=True,
    )

    suggestions = payload.get("suggestions")
    assert isinstance(suggestions, list)
    assert len(suggestions) == 1
    assert suggestions[0]["method"] == "llm_tiebreak"
    assert suggestions[0]["remove_change_index"] == 0
    assert suggestions[0]["add_change_index"] == 1
    assert isinstance(suggestions[0].get("rationale"), str)

    llm_meta = payload.get("llm")
    assert isinstance(llm_meta, dict)
    assert llm_meta.get("status") == "complete"
    assert llm_meta.get("incomplete") is False


@pytest.mark.asyncio
async def test_build_move_suggestions_payload_llm_tiebreak_skips_when_disabled() -> None:
    embedding_provider = _TagEmbeddingProvider()
    llm_provider = _StubLLMProvider(chosen_add_change_index=1)

    old = _handle(path="a.py", chunk_type="function", symbol="old_sym", text_hash="x")
    new = _handle(path="b.py", chunk_type="function", symbol="new_sym", text_hash="y")
    report = _make_report(
        changes=[
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
                old=old,
                new=None,
            ),
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
                new=new,
            ),
        ]
    )

    payload = await build_move_suggestions_payload(
        report=report,
        include_blocks=True,
        embedding_provider=embedding_provider,  # type: ignore[arg-type]
        texts_a_by_path_ordinal={("a.py", 1): "PAIR1"},
        texts_b_by_path_ordinal={("b.py", 1): "PAIR1"},
        embed_min_score=1.1,
        embed_min_margin=1.1,
        llm_provider=llm_provider,  # type: ignore[arg-type]
        llm_enabled=False,
    )

    suggestions = payload.get("suggestions")
    assert isinstance(suggestions, list)
    assert not suggestions

    llm_meta = payload.get("llm")
    assert isinstance(llm_meta, dict)
    assert llm_meta.get("status") == "disabled"
    assert llm_meta.get("incomplete") is False


@pytest.mark.asyncio
async def test_build_move_suggestions_payload_llm_dry_run_collects_prompts_without_calling_llm() -> None:
    embedding_provider = _TagEmbeddingProvider()
    dry_run_calls: list[tuple[str, str]] = []

    old = _handle(path="a.py", chunk_type="function", symbol="old_sym", text_hash="x")
    new1 = _handle(path="b.py", chunk_type="function", symbol="new_sym_1", text_hash="y")
    new2 = _handle(path="c.py", chunk_type="function", symbol="new_sym_2", text_hash="z")
    report = _make_report(
        changes=[
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
                old=old,
                new=None,
            ),
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
                new=new1,
            ),
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
                new=new2,
            ),
        ]
    )

    payload = await build_move_suggestions_payload(
        report=report,
        include_blocks=True,
        embedding_provider=embedding_provider,  # type: ignore[arg-type]
        texts_a_by_path_ordinal={("a.py", 1): "PAIR1"},
        texts_b_by_path_ordinal={("b.py", 1): "PAIR1", ("c.py", 1): "PAIR2"},
        embed_min_score=1.1,
        embed_min_margin=1.1,
        llm_enabled=True,
        llm_provider=None,
        llm_dry_run=True,
        llm_dry_run_collector=dry_run_calls,
    )

    suggestions = payload.get("suggestions")
    assert isinstance(suggestions, list)
    assert not suggestions

    llm_meta = payload.get("llm")
    assert isinstance(llm_meta, dict)
    assert llm_meta.get("status") == "dry_run"
    assert llm_meta.get("incomplete") is True

    assert len(dry_run_calls) == 1
    filename, content = dry_run_calls[0]
    assert filename.startswith("llm_call_")
    assert "## Prompt" in content


@pytest.mark.asyncio
async def test_build_move_suggestions_payload_llm_dry_run_skips_when_top_score_below_min_score() -> None:
    embedding_provider = _TagEmbeddingProvider()
    dry_run_calls: list[tuple[str, str]] = []

    old = _handle(path="a.py", chunk_type="function", symbol="old_sym", text_hash="x")
    new1 = _handle(path="b.py", chunk_type="function", symbol="new_sym_1", text_hash="y")
    report = _make_report(
        changes=[
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
                old=old,
                new=None,
            ),
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
                new=new1,
            ),
        ]
    )

    payload = await build_move_suggestions_payload(
        report=report,
        include_blocks=True,
        embedding_provider=embedding_provider,  # type: ignore[arg-type]
        texts_a_by_path_ordinal={("a.py", 1): "PAIR1"},
        texts_b_by_path_ordinal={("b.py", 1): "SIM085"},
        embed_min_score=1.1,
        embed_min_margin=1.1,
        llm_enabled=True,
        llm_provider=None,
        llm_dry_run=True,
        llm_dry_run_collector=dry_run_calls,
        llm_min_score=0.90,
    )

    suggestions = payload.get("suggestions")
    assert isinstance(suggestions, list)
    assert not suggestions
    assert not dry_run_calls

    skipped = payload.get("skipped_reasons")
    assert isinstance(skipped, dict)
    assert skipped.get("llm_top_score_below_min_score") == 1


@pytest.mark.asyncio
async def test_build_move_suggestions_payload_llm_dry_run_trims_to_fit_token_budget() -> None:
    embedding_provider = _TagEmbeddingProvider()
    dry_run_calls: list[tuple[str, str]] = []

    old = _handle(path="a.py", chunk_type="function", symbol="old_sym", text_hash="x")
    new1 = _handle(path="b.py", chunk_type="function", symbol="new_sym_1", text_hash="y")
    new2 = _handle(path="c.py", chunk_type="function", symbol="new_sym_2", text_hash="z")
    report = _make_report(
        changes=[
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
                old=old,
                new=None,
            ),
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
                new=new1,
            ),
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
                new=new2,
            ),
        ]
    )

    payload = await build_move_suggestions_payload(
        report=report,
        include_blocks=True,
        embedding_provider=embedding_provider,  # type: ignore[arg-type]
        texts_a_by_path_ordinal={("a.py", 1): "PAIR1 " + ("x" * 3000)},
        texts_b_by_path_ordinal={
            ("b.py", 1): "PAIR1 " + ("x" * 3000),
            ("c.py", 1): "PAIR2 " + ("x" * 3000),
        },
        embed_min_score=1.1,
        embed_min_margin=1.1,
        llm_enabled=True,
        llm_provider=None,
        llm_dry_run=True,
        llm_dry_run_collector=dry_run_calls,
        llm_min_score=0.0,
        llm_top_k=2,
        llm_max_prompt_tokens=1700,  # force trimming
    )

    llm_meta = payload.get("llm")
    assert isinstance(llm_meta, dict)
    assert llm_meta.get("status") == "dry_run"

    assert len(dry_run_calls) == 1
    _, content = dry_run_calls[0]
    assert "- included: 1" in content
    assert "- available: 2" in content
