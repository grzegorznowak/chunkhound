#!/usr/bin/env python3
"""Unit tests for `chunkhound.gap.themes`."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path

import pytest

import chunkhound.gap.themes as gap_themes

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
from chunkhound.gap.themes import (
    IsotopePairing,
    ThemeItem,
    build_fallback_theme_output,
    build_theme_documents,
    build_theme_output,
    cluster_embeddings_hdbscan,
    embed_in_batches,
    render_themes_markdown,
)


@dataclass(frozen=True)
class _StubProvider:
    name: str = "stub"
    model: str = "stub-model"
    dims: int = 3


@dataclass(frozen=True)
class _StubEmbeddingProvider:
    name: str = "stub"
    model: str = "stub-model"
    dims: int = 3

    async def embed(self, texts: list[str]) -> list[list[float]]:
        return [[0.0, 0.0, 0.0] for _ in texts]


@dataclass(frozen=True)
class _BadEmbeddingProvider:
    name: str = "bad"
    model: str = "bad-model"
    dims: int = 3

    async def embed(self, texts: list[str]) -> list[list[float]]:
        _ = texts
        return [[0.0, 0.0, 0.0]]


def test_cluster_embeddings_hdbscan_finds_two_clusters() -> None:
    # Two clearly separated clusters, 3 points each (min_cluster_size=3).
    embeddings = [
        [1.0, 0.0, 0.0],
        [1.01, 0.0, 0.0],
        [1.0, 0.01, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 1.01, 0.0],
        [0.01, 1.0, 0.0],
    ]
    labels = cluster_embeddings_hdbscan(
        embeddings, min_cluster_size=3, min_samples=1, allow_single_cluster=True
    )
    assert len(labels) == len(embeddings)
    clusters = {l for l in labels if l != -1}
    assert len(clusters) == 2


def test_build_theme_output_groups_items_and_renders_markdown() -> None:
    items = [
        ThemeItem(
            change_index=0,
            entity_kind="symbol",
            op="update",
            reason="symbol_anchor",
            confidence=1.0,
            chunk_type="function",
            path="a.py",
            symbol="foo",
        ),
        ThemeItem(
            change_index=1,
            entity_kind="symbol",
            op="add",
            reason="symbol_anchor",
            confidence=1.0,
            chunk_type="function",
            path="b.py",
            symbol="bar",
        ),
    ]
    labels = [0, 1]
    output = build_theme_output(
        provider=_StubProvider(), items=items, labels=labels, min_cluster_size=3
    )
    md = render_themes_markdown(output=output)
    assert "Gap themes" in md
    assert "Theme 0" in md
    assert "Theme 1" in md


def test_fallback_theme_output_is_stable() -> None:
    out = build_fallback_theme_output(
        items=[
            ThemeItem(
                change_index=0,
                entity_kind="symbol",
                op="update",
                reason="symbol_anchor",
                confidence=1.0,
                chunk_type="function",
                path="a.py",
                symbol="foo",
            )
        ],
        label="fallback",
    )
    assert out.provider == "none"
    assert out.model == "none"
    assert out.dims == 0
    assert out.theme_labels[0] == "fallback"


def test_theme_label_ties_are_deterministic() -> None:
    items = [
        ThemeItem(
            change_index=0,
            entity_kind="symbol",
            op="add",
            reason="x",
            confidence=1.0,
            chunk_type=None,
            path=None,
            symbol="beta",
        ),
        ThemeItem(
            change_index=1,
            entity_kind="symbol",
            op="add",
            reason="x",
            confidence=1.0,
            chunk_type=None,
            path=None,
            symbol="alpha",
        ),
    ]
    labels = [0, 0]
    out = build_theme_output(provider=_StubProvider(), items=items, labels=labels)
    assert out.theme_labels[0] == "alpha, beta"


def test_build_theme_documents_truncates_to_max_tokens_per_doc() -> None:
    old_handle = GapSymbolHandle(
        path="a.py",
        start_line=1,
        end_line=2,
        symbol="foo",
        chunk_type="function",
        text_hash="x",
        ordinal_in_file=1,
        parse_status="ok",
    )
    new_handle = GapSymbolHandle(
        path="b.py",
        start_line=3,
        end_line=4,
        symbol="foo",
        chunk_type="function",
        text_hash="y",
        ordinal_in_file=1,
        parse_status="ok",
    )
    report = GapReport(
        schema_version="gap.v1",
        schema_revision="2026-01-09",
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
        changes=[
            GapChangeItem(
                entity_kind="symbol",
                op="update",
                moved=True,
                renamed=False,
                content_changed=True,
                reason="symbol_anchor",
                confidence=1.0,
                primary_key_kind="symbol_key",
                primary_key_hash="aaaaaaaaaaaaaaaa",
                key_strength=1,
                had_collision=False,
                collision_group_size=1,
                old=old_handle,
                new=new_handle,
            )
        ],
    )

    old_text = "OLD0\n" + "\n".join(f"old_line_{i} " + ("x" * 50) for i in range(200))
    new_text = "NEW0\n" + "\n".join(f"new_line_{i} " + ("y" * 50) for i in range(200))
    docs, items = build_theme_documents(
        report=report,
        texts_a_by_path_ordinal={("a.py", 1): old_text},
        texts_b_by_path_ordinal={("b.py", 1): new_text},
        max_tokens_per_doc=200,
    )
    assert len(docs) == 1
    assert len(items) == 1
    assert len(docs[0]) <= 700
    assert "op=update" in docs[0]
    assert "\nOLD " in docs[0]
    assert "\nNEW " in docs[0]
    assert "OLD0" in docs[0]
    assert "NEW0" in docs[0]
    assert docs[0].count("…") == 1


def test_render_themes_markdown_includes_isotope_pairing_when_provided() -> None:
    old = GapSymbolHandle(
        path="a.py",
        start_line=1,
        end_line=1,
        symbol="foo",
        chunk_type="function",
        text_hash="x",
        ordinal_in_file=1,
        parse_status="ok",
    )
    new = GapSymbolHandle(
        path="b.py",
        start_line=1,
        end_line=1,
        symbol="foo",
        chunk_type="function",
        text_hash="y",
        ordinal_in_file=1,
        parse_status="ok",
    )
    report = GapReport(
        schema_version="gap.v1",
        schema_revision="2026-01-09",
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
                primary_key_hash="bbbbbbbbbbbbbbbb",
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
                primary_key_hash="cccccccccccccccc",
                key_strength=1,
                had_collision=False,
                collision_group_size=1,
                old=None,
                new=new,
            ),
        ],
    )

    out = build_fallback_theme_output(
        items=[
            ThemeItem(
                change_index=0,
                entity_kind="symbol",
                op="remove",
                reason="symbol_anchor",
                confidence=1.0,
                chunk_type="function",
                path="a.py",
                symbol="foo",
            ),
            ThemeItem(
                change_index=1,
                entity_kind="symbol",
                op="add",
                reason="symbol_anchor",
                confidence=1.0,
                chunk_type="function",
                path="b.py",
                symbol="foo",
            ),
        ],
        label="x",
    )

    isotope_pairs = {
        0: IsotopePairing(
            pair_id=7,
            role="remove",
            counterpart_change_index=1,
            method="llm_tiebreak",
            confidence=0.72,
            rationale="matched by sentinel",
        ),
        1: IsotopePairing(
            pair_id=7,
            role="add",
            counterpart_change_index=0,
            method="llm_tiebreak",
            confidence=0.72,
            rationale="matched by sentinel",
        ),
    }

    md = render_themes_markdown(output=out, report=report, isotope_pairs=isotope_pairs)
    assert "paired_by=llm_tiebreak" in md
    assert "[ISO:llm_tiebreak]" in md
    assert "a.py#L1-L1" in md
    assert "b.py#L1-L1" in md
    assert "rationale=" in md
    assert "matched by sentinel" in md


def test_render_themes_markdown_compare_header_is_stable_and_ordered() -> None:
    report = GapReport(
        schema_version="gap.v1",
        schema_revision="2026-01-09",
        direction="A->B",
        invariants=GapInvariants(
            hash_alg="xxh3_64",
            normalization=GapNormalizationInvariants(
                id="normalize_content.v1", include_comments=False, include_docs=False
            ),
            chunker_version="cast@v1",
            recovery_mode="safe",
            deterministic=True,
            embed_model_id=None,
            forced=False,
        ),
        inputs=GapInputs(
            a=GapInputRef(source_kind="path", source_ref="A", source_hash="ha"),
            b=GapInputRef(source_kind="path", source_ref="B", source_hash="hb"),
        ),
        scope=GapScope(
            scope_mode="full",
            scope_hash="scope",
            changed_files_count=3,
            rename_hints_count=0,
        ),
        warnings=[],
        stats=GapStats(
            counts=GapCounts(
                files_a_total=1,
                files_b_total=2,
                changes_total=3,
                file_changes_total=0,
                symbol_changes_total=3,
            ),
            timings=GapTimings(),
        ),
        changes=[],
    )
    out = build_fallback_theme_output(items=[], label="no_symbol_changes")
    md = render_themes_markdown(output=out, report=report)
    lines = md.splitlines()

    def _idx(prefix: str) -> int:
        return next(i for i, l in enumerate(lines) if l.startswith(prefix))

    assert _idx("## Compare") < _idx("- schema_version:") < _idx("- A:") < _idx("- B:")
    assert (
        _idx("- B:")
        < _idx("- scope:")
        < _idx("- invariants:")
        < _idx("- stats.counts:")
    )
    assert _idx("- stats.counts:") < _idx("- warnings:")


def test_render_themes_markdown_update_lines_include_old_and_new_locators() -> None:
    old = GapSymbolHandle(
        path="a.py",
        start_line=10,
        end_line=12,
        symbol="foo",
        chunk_type="function",
        text_hash="x",
        ordinal_in_file=2,
        parse_status="ok",
    )
    new = GapSymbolHandle(
        path="b.py",
        start_line=20,
        end_line=22,
        symbol="foo",
        chunk_type="function",
        text_hash="y",
        ordinal_in_file=3,
        parse_status="ok",
    )
    report = GapReport(
        schema_version="gap.v1",
        schema_revision="2026-01-09",
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
        changes=[
            GapChangeItem(
                entity_kind="symbol",
                op="update",
                moved=True,
                renamed=False,
                content_changed=True,
                reason="symbol_anchor",
                confidence=1.0,
                primary_key_kind="stable_key",
                primary_key_hash="dddddddddddddddd",
                key_strength=1,
                had_collision=False,
                collision_group_size=1,
                old=old,
                new=new,
            )
        ],
    )
    out = build_fallback_theme_output(
        items=[
            ThemeItem(
                change_index=0,
                entity_kind="symbol",
                op="update",
                reason="symbol_anchor",
                confidence=1.0,
                chunk_type="function",
                path="b.py",
                symbol="foo",
            )
        ],
        label="x",
    )
    md = render_themes_markdown(output=out, report=report)
    assert "update function" in md
    assert "OLD" in md and "NEW" in md
    assert "a.py#L10-L12" in md
    assert "b.py#L20-L22" in md
    assert "ord=2" in md
    assert "ord=3" in md


async def test_embed_in_batches_returns_one_embedding_per_text() -> None:
    provider = _StubEmbeddingProvider()
    texts = ["a", "b", "c"]
    out = await embed_in_batches(provider=provider, texts=texts, batch_size=2)
    assert len(out) == len(texts)


async def test_embed_in_batches_raises_on_length_mismatch() -> None:
    provider = _BadEmbeddingProvider()
    with pytest.raises(ValueError, match="returned 1 vectors"):
        await embed_in_batches(provider=provider, texts=["a", "b"], batch_size=10)


@pytest.mark.parametrize(
    "items", [[], [ThemeItem(0, "symbol", "add", "x", 1.0, None, None, None)]]
)
def test_render_themes_markdown_does_not_crash(items: list[ThemeItem]) -> None:
    out = build_fallback_theme_output(items=items, label="x")
    md = render_themes_markdown(output=out)
    assert md.endswith("\n")


def test_compute_outliers_nearest_ranks_candidates_and_is_deterministic() -> None:
    items = [
        ThemeItem(
            change_index=0,
            entity_kind="symbol",
            op="add",
            reason="x",
            confidence=1.0,
            chunk_type="function",
            path="theme0.py",
            symbol="t0",
        ),
        ThemeItem(
            change_index=1,
            entity_kind="symbol",
            op="add",
            reason="x",
            confidence=1.0,
            chunk_type="function",
            path="theme1.py",
            symbol="t1",
        ),
        ThemeItem(
            change_index=2,
            entity_kind="symbol",
            op="add",
            reason="x",
            confidence=1.0,
            chunk_type="function",
            path="a.py",
            symbol="near_theme0",
        ),
        ThemeItem(
            change_index=3,
            entity_kind="symbol",
            op="add",
            reason="x",
            confidence=1.0,
            chunk_type="function",
            path="b.py",
            symbol="between_themes",
        ),
    ]
    labels = [0, 1, -1, -1]
    embeddings = [
        [1.0, 0.0, 0.0],  # theme 0 centroid
        [0.0, 1.0, 0.0],  # theme 1 centroid
        [0.99, 0.1, 0.0],  # outlier near theme 0
        [0.6, 0.6, 0.0],  # outlier between themes (tie at 2dp)
    ]
    output = gap_themes.ThemeOutput(
        provider="stub",
        model="stub-model",
        dims=3,
        min_cluster_size=3,
        min_samples=1,
        allow_single_cluster=True,
        theme_labels={0: "alpha", 1: "beta"},
        themes={
            0: [items[0]],
            1: [items[1]],
            -1: [items[2], items[3]],
        },
    )

    out1 = gap_themes.compute_outliers_nearest(
        embeddings=embeddings,
        labels=labels,
        items=items,
        output=output,
    )
    out2 = gap_themes.compute_outliers_nearest(
        embeddings=embeddings,
        labels=labels,
        items=items,
        output=output,
    )
    assert out1 == out2

    assert [it["change_index"] for it in out1] == [2, 3]

    near = out1[0]
    assert near["assigned_theme_id"] == 0
    assert near["assigned_similarity"] is not None
    assert near["candidates"][0]["theme_id"] == 0
    assert near["candidates"][0]["label"] == "alpha"
    assert near["candidates"][1]["theme_id"] == 1
    assert near["candidates"][1]["label"] == "beta"

    between = out1[1]
    assert between["assigned_theme_id"] is None
    assert between["assigned_similarity"] is None
    # Tie at 2dp (0.71 vs 0.71) should break by theme_id.
    assert between["candidates"][0]["theme_id"] == 0
    assert between["candidates"][1]["theme_id"] == 1


def test_render_themes_markdown_appends_outliers_nearest_section_after_outliers_theme() -> (
    None
):
    items = [
        ThemeItem(0, "symbol", "add", "x", 1.0, "function", "a.py", "t0"),
        ThemeItem(1, "symbol", "add", "x", 1.0, "function", "b.py", "t1"),
        ThemeItem(2, "symbol", "add", "x", 1.0, "function", "c.py", "outlier"),
    ]
    output = gap_themes.ThemeOutput(
        provider="stub",
        model="stub-model",
        dims=3,
        min_cluster_size=3,
        min_samples=1,
        allow_single_cluster=True,
        theme_labels={0: "alpha", 1: "beta"},
        themes={0: [items[0]], 1: [items[1]], -1: [items[2]]},
    )
    outliers_nearest = [
        {
            "change_index": 2,
            "assigned_theme_id": 0,
            "assigned_similarity": 0.812345,
            "candidates": [
                {"theme_id": 0, "label": "alpha", "similarity": 0.812345},
                {"theme_id": 1, "label": "beta", "similarity": 0.123456},
            ],
        }
    ]
    md = gap_themes.render_themes_markdown(
        output=output, outliers_nearest=outliers_nearest
    )
    lines = md.splitlines()

    idx_outliers_theme = next(i for i, l in enumerate(lines) if l.startswith("## Theme -1:"))
    idx_section = next(
        i for i, l in enumerate(lines) if l.startswith("## Outliers: Nearest Themes")
    )
    assert idx_outliers_theme < idx_section
    assert "### Assigned" in md
    assert "### Unassigned" in md


def test_write_theme_artifacts_includes_outliers_nearest_when_provided(
    tmp_path: Path,
) -> None:
    report = GapReport(
        schema_version="gap.v1",
        schema_revision="2026-01-09",
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
        changes=[],
    )
    output = gap_themes.ThemeOutput(
        provider="stub",
        model="stub-model",
        dims=3,
        min_cluster_size=3,
        min_samples=1,
        allow_single_cluster=True,
        theme_labels={},
        themes={},
    )
    outliers_nearest = [
        {
            "change_index": 123,
            "assigned_theme_id": 0,
            "assigned_similarity": 0.812345,
            "candidates": [
                {"theme_id": 0, "label": "alpha", "similarity": 0.812345},
            ],
        }
    ]
    gap_themes.write_theme_artifacts(
        out_dir=tmp_path,
        report=report,
        output=output,
        isotope_pairs=None,
        outliers_nearest=outliers_nearest,
    )
    payload = json.loads((tmp_path / "themes.json").read_text(encoding="utf-8"))
    assert payload["outliers_nearest"] == outliers_nearest


def test_outliers_nearest_is_omitted_when_embeddings_disabled(tmp_path: Path) -> None:
    report = GapReport(
        schema_version="gap.v1",
        schema_revision="2026-01-09",
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
        changes=[],
    )
    output = build_fallback_theme_output(items=[], label="no_symbol_changes")
    outliers_nearest = [
        {
            "change_index": 123,
            "assigned_theme_id": 0,
            "assigned_similarity": 0.812345,
            "candidates": [
                {"theme_id": 0, "label": "alpha", "similarity": 0.812345},
            ],
        }
    ]
    gap_themes.write_theme_artifacts(
        out_dir=tmp_path,
        report=report,
        output=output,
        isotope_pairs=None,
        outliers_nearest=outliers_nearest,
    )
    payload = json.loads((tmp_path / "themes.json").read_text(encoding="utf-8"))
    assert "outliers_nearest" not in payload
    md = (tmp_path / "themes.md").read_text(encoding="utf-8")
    assert "## Outliers: Nearest Themes (advisory)" not in md
