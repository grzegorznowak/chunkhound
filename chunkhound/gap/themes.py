"""Theme clustering for `chunkhound gap` (embedding-based, v1)."""

from __future__ import annotations

from dataclasses import dataclass
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Literal

import numpy as np
import hdbscan  # type: ignore[import-untyped]

from chunkhound.interfaces.embedding_provider import EmbeddingProvider
from chunkhound.gap.models import GapChangeItem, GapReport, GapSymbolHandle


@dataclass(frozen=True)
class ThemeItem:
    change_index: int
    entity_kind: str
    op: str
    reason: str
    confidence: float
    chunk_type: str | None
    path: str | None
    symbol: str | None


@dataclass(frozen=True)
class ThemeOutput:
    provider: str
    model: str
    dims: int
    min_cluster_size: int
    min_samples: int
    allow_single_cluster: bool
    theme_labels: dict[int, str]
    themes: dict[int, list[ThemeItem]]


@dataclass(frozen=True)
class IsotopePairing:
    pair_id: int
    role: Literal["add", "remove"]
    counterpart_change_index: int
    method: str
    confidence: float
    rationale: str | None = None


_TERM_RE = re.compile(r"[A-Za-z][A-Za-z0-9_]{2,}")
_STOP = {
    "the",
    "and",
    "for",
    "with",
    "from",
    "into",
    "this",
    "that",
    "true",
    "false",
    "none",
    "return",
    "class",
    "def",
}

ELLIPSIS_MARKER = "…"
MAX_DISPLAY_CHARS = 200

_WS_RE = re.compile(r"\s+")
_BACKTICK_RUN_RE = re.compile(r"`+")

OUTLIERS_NEAREST_TOP_K = 3
OUTLIERS_NEAREST_THRESHOLD = 0.80


def _normalize_display_text(text: str) -> str:
    text = text.replace("\r", " ").replace("\n", " ").replace("\t", " ")
    return _WS_RE.sub(" ", text).strip()


def _truncate_display_text(text: str, *, max_chars: int = MAX_DISPLAY_CHARS) -> str:
    if len(text) <= max_chars:
        return text
    if max_chars <= 1:
        return ELLIPSIS_MARKER
    return text[: max_chars - 1].rstrip() + ELLIPSIS_MARKER


def _markdown_code_span(text: str) -> str:
    text = _normalize_display_text(text)
    runs = _BACKTICK_RUN_RE.findall(text)
    fence_len = max((len(r) for r in runs), default=0) + 1
    fence = "`" * fence_len
    return f"{fence}{text}{fence}"


def _format_confidence_2dp(confidence: float) -> str:
    s = f"{float(confidence):.2f}"
    return "0.00" if s == "-0.00" else s


def _format_similarity_2dp(similarity: float) -> str:
    s = f"{float(similarity):.2f}"
    return "0.00" if s == "-0.00" else s


def _round_similarity_6dp(similarity: float) -> float:
    return round(float(similarity), 6)


def _symbol_handle_for_theme_item(
    *, report: GapReport, change_index: int, prefer_new: bool
) -> GapSymbolHandle | None:
    if change_index < 0 or change_index >= len(report.changes):
        return None
    ch = report.changes[int(change_index)]
    if not isinstance(ch, GapChangeItem) or ch.entity_kind != "symbol":
        return None
    if prefer_new and isinstance(ch.new, GapSymbolHandle):
        return ch.new
    if isinstance(ch.old, GapSymbolHandle):
        return ch.old
    if isinstance(ch.new, GapSymbolHandle):
        return ch.new
    return None


def _locator_display(handle: GapSymbolHandle | None) -> tuple[str, str]:
    if handle is None:
        return ("(unknown)#L?", "ord=?")
    path = _normalize_display_text(handle.path or "(unknown)")
    start_line = int(handle.start_line)
    end_line = int(handle.end_line)
    if start_line <= 0 or end_line <= 0 or end_line < start_line:
        loc = f"{path}#L?"
    else:
        loc = f"{path}#L{start_line}-L{end_line}"

    ordinal = int(handle.ordinal_in_file)
    ord_part = f"ord={ordinal}" if ordinal > 0 else "ord=?"
    return (loc, ord_part)


def _themes_compare_object(*, report: GapReport) -> dict[str, Any]:
    counts = report.stats.counts
    return {
        "schema_version": report.schema_version,
        "schema_revision": report.schema_revision,
        "direction": report.direction,
        "a": {
            "source_hash": report.inputs.a.source_hash,
            "source_ref": report.inputs.a.source_ref,
        },
        "b": {
            "source_hash": report.inputs.b.source_hash,
            "source_ref": report.inputs.b.source_ref,
        },
        "scope": {
            "scope_hash": report.scope.scope_hash,
            "changed_files_count": int(report.scope.changed_files_count),
        },
        "invariants": {
            "deterministic": bool(report.invariants.deterministic),
            "recovery_mode": report.invariants.recovery_mode,
        },
        "stats": {
            "counts": {
                "files_a_total": int(counts.files_a_total),
                "files_b_total": int(counts.files_b_total),
                "changes_total": int(counts.changes_total),
                "file_changes_total": int(counts.file_changes_total),
                "symbol_changes_total": int(counts.symbol_changes_total),
            }
        },
        "warnings": {"total": int(len(report.warnings))},
    }


def _themes_markdown_compare_lines(*, report: GapReport) -> list[str]:
    compare = _themes_compare_object(report=report)
    counts = compare["stats"]["counts"]
    lines: list[str] = []
    lines.append("## Compare")
    lines.append("")
    lines.append(
        f"- schema_version: {_markdown_code_span(str(compare['schema_version']))} "
        f"schema_revision: {_markdown_code_span(str(compare['schema_revision']))} "
        f"direction: {_markdown_code_span(str(compare['direction']))}"
    )
    lines.append(
        f"- A: source_hash={_markdown_code_span(str(compare['a']['source_hash']))} "
        f"source_ref={_markdown_code_span(str(compare['a']['source_ref']))}"
    )
    lines.append(
        f"- B: source_hash={_markdown_code_span(str(compare['b']['source_hash']))} "
        f"source_ref={_markdown_code_span(str(compare['b']['source_ref']))}"
    )
    lines.append(
        f"- scope: scope_hash={_markdown_code_span(str(compare['scope']['scope_hash']))} "
        f"changed_files_count={int(compare['scope']['changed_files_count'])}"
    )
    lines.append(
        f"- invariants: deterministic={bool(compare['invariants']['deterministic'])} "
        f"recovery_mode={_markdown_code_span(str(compare['invariants']['recovery_mode']))}"
    )
    lines.append(
        "- stats.counts: "
        f"files_a_total={int(counts['files_a_total'])} "
        f"files_b_total={int(counts['files_b_total'])} "
        f"changes_total={int(counts['changes_total'])} "
        f"file_changes_total={int(counts['file_changes_total'])} "
        f"symbol_changes_total={int(counts['symbol_changes_total'])}"
    )
    lines.append(f"- warnings: total={int(compare['warnings']['total'])}")
    lines.append("")
    return lines


def _safe_handle_text(
    handle: GapSymbolHandle, texts_by_path_ordinal: dict[tuple[str, int], str]
) -> str | None:
    return texts_by_path_ordinal.get((handle.path, int(handle.ordinal_in_file)))


def build_theme_documents(
    *,
    report: GapReport,
    texts_a_by_path_ordinal: dict[tuple[str, int], str],
    texts_b_by_path_ordinal: dict[tuple[str, int], str],
    max_tokens_per_doc: int = 1800,
) -> tuple[list[str], list[ThemeItem]]:
    """Create 1 document per change item for theme clustering."""
    docs: list[str] = []
    items: list[ThemeItem] = []

    max_chars = max(1, (int(max_tokens_per_doc) * 35) // 10)

    for idx, change in enumerate(report.changes):
        if change.entity_kind != "symbol":
            continue

        old = change.old if isinstance(change.old, GapSymbolHandle) else None
        new = change.new if isinstance(change.new, GapSymbolHandle) else None

        header_line = (
            f"op={change.op} reason={change.reason} moved={change.moved} "
            f"renamed={change.renamed} content_changed={change.content_changed}"
        )

        chunk_type = None
        path = None
        symbol = None

        old_header = None
        old_lines: list[str] = []
        if old is not None:
            chunk_type = old.chunk_type
            path = old.path
            symbol = old.symbol
            old_header = f"OLD {old.chunk_type} {old.symbol} path={old.path}"
            t = _safe_handle_text(old, texts_a_by_path_ordinal)
            if t:
                old_lines = t.splitlines()

        new_header = None
        new_lines: list[str] = []
        if new is not None:
            chunk_type = new.chunk_type
            path = new.path
            symbol = new.symbol
            new_header = f"NEW {new.chunk_type} {new.symbol} path={new.path}"
            t = _safe_handle_text(new, texts_b_by_path_ordinal)
            if t:
                new_lines = t.splitlines()

        def _assemble(*, old_keep: int, new_keep: int) -> str:
            out_parts: list[str] = [header_line]
            if old_header is not None:
                out_parts.append(old_header)
                out_parts.extend(old_lines[:old_keep])
            if new_header is not None:
                out_parts.append(new_header)
                out_parts.extend(new_lines[:new_keep])
            return "\n".join(out_parts).strip()

        full_doc = _assemble(old_keep=len(old_lines), new_keep=len(new_lines))
        if not full_doc:
            continue

        if len(full_doc) <= max_chars:
            doc = full_doc
        else:
            min_per_side = 1
            old_min = min_per_side if old_lines else 0
            new_min = min_per_side if new_lines else 0
            old_keep = len(old_lines)
            new_keep = len(new_lines)

            doc = full_doc
            while len(doc) > max_chars and (old_keep > old_min or new_keep > new_min):
                if old_keep > old_min and (old_keep >= new_keep or new_keep <= new_min):
                    old_keep = max(old_min, old_keep // 2)
                elif new_keep > new_min:
                    new_keep = max(new_min, new_keep // 2)
                doc = _assemble(old_keep=old_keep, new_keep=new_keep)

            if len(doc) > max_chars - 1:
                doc = doc[: max_chars - 1].rstrip()
            doc = doc + ELLIPSIS_MARKER

        docs.append(doc)
        items.append(
            ThemeItem(
                change_index=idx,
                entity_kind=change.entity_kind,
                op=change.op,
                reason=change.reason,
                confidence=float(change.confidence),
                chunk_type=chunk_type,
                path=path,
                symbol=symbol,
            )
        )

    return docs, items


async def embed_in_batches(
    *,
    provider: EmbeddingProvider,
    texts: list[str],
    batch_size: int,
    progress: Any | None = None,
    progress_task_id: Any | None = None,
) -> list[list[float]]:
    if not texts:
        return []
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    task_id = progress_task_id
    if task_id is None and progress is not None:
        try:
            task_id = progress.add_task(
                "Gap: embedding themes",
                total=len(texts),
                info="",
                speed="",
            )
        except Exception:
            task_id = None
    out: list[list[float]] = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        out.extend(await provider.embed(batch))
        if task_id is not None and progress is not None:
            try:
                progress.update(
                    task_id,
                    advance=len(batch),
                    info=f"{min(i + len(batch), len(texts))}/{len(texts)}",
                )
            except Exception:
                pass
    if len(out) != len(texts):
        raise ValueError(
            f"Embedding provider returned {len(out)} vectors for {len(texts)} texts"
        )
    return out


def cluster_embeddings_hdbscan(
    embeddings: list[list[float]],
    *,
    min_cluster_size: int = 3,
    min_samples: int = 1,
    allow_single_cluster: bool = True,
) -> list[int]:
    if not embeddings:
        return []
    if len(embeddings) == 1:
        return [0]
    arr = np.asarray(embeddings, dtype=float)
    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    norms[norms == 0.0] = 1.0
    arr = arr / norms

    effective_min_cluster_size = min(max(2, int(min_cluster_size)), len(embeddings))
    effective_min_samples = min(max(1, int(min_samples)), len(embeddings) - 1)

    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=effective_min_cluster_size,
        min_samples=effective_min_samples,
        metric="euclidean",
        cluster_selection_method="leaf",
        allow_single_cluster=allow_single_cluster,
    )
    labels = clusterer.fit_predict(arr)
    return [int(x) for x in labels.tolist()]


def _label_theme(items: list[ThemeItem]) -> str:
    terms: Counter[str] = Counter()
    for it in items:
        for s in (it.symbol or "", it.path or "", it.chunk_type or ""):
            for m in _TERM_RE.findall(s):
                t = m.lower()
                if t in _STOP:
                    continue
                terms[t] += 1
    top = [t for t, _ in sorted(terms.items(), key=lambda kv: (-kv[1], kv[0]))[:6]]
    if top:
        return ", ".join(top)
    return "theme"


def _theme_label(*, output: ThemeOutput, theme_id: int) -> str:
    if theme_id == -1:
        return "Outliers/Misc"
    return output.theme_labels.get(theme_id, "theme")


def _theme_sort_key(theme_id: int, items: list[ThemeItem]) -> tuple[int, int, int]:
    if theme_id == -1:
        return (1, 0, theme_id)
    return (0, -len(items), theme_id)


def _theme_item_sort_key(item: ThemeItem) -> tuple[str, str, str, str, int]:
    return (
        item.path or "",
        item.op,
        item.chunk_type or "",
        item.symbol or "",
        int(item.change_index),
    )


def build_theme_output(
    *,
    provider: EmbeddingProvider,
    items: list[ThemeItem],
    labels: list[int],
    min_cluster_size: int = 3,
    min_samples: int = 1,
    allow_single_cluster: bool = True,
) -> ThemeOutput:
    by_theme: dict[int, list[ThemeItem]] = defaultdict(list)
    for it, lab in zip(items, labels):
        by_theme[int(lab)].append(it)

    theme_labels = {tid: _label_theme(its) for tid, its in by_theme.items()}
    return ThemeOutput(
        provider=provider.name,
        model=provider.model,
        dims=provider.dims,
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        allow_single_cluster=allow_single_cluster,
        theme_labels=theme_labels,
        themes=dict(by_theme),
    )


def build_fallback_theme_output(*, items: list[ThemeItem], label: str) -> ThemeOutput:
    by_theme: dict[int, list[ThemeItem]] = {}
    if items:
        by_theme[0] = list(items)
    theme_labels = {0: label} if items else {}
    return ThemeOutput(
        provider="none",
        model="none",
        dims=0,
        min_cluster_size=0,
        min_samples=0,
        allow_single_cluster=True,
        theme_labels=theme_labels,
        themes=by_theme,
    )


def compute_outliers_nearest(
    *,
    embeddings: list[list[float]],
    labels: list[int],
    items: list[ThemeItem],
    output: ThemeOutput,
) -> list[dict[str, Any]]:
    if not embeddings:
        return []
    if len(embeddings) != len(labels) or len(embeddings) != len(items):
        raise ValueError(
            "embeddings, labels, and items must have identical length "
            f"(got {len(embeddings)}, {len(labels)}, {len(items)})"
        )

    outlier_indices = [i for i, lab in enumerate(labels) if int(lab) == -1]
    if not outlier_indices:
        return []

    theme_to_indices: dict[int, list[int]] = defaultdict(list)
    for i, lab in enumerate(labels):
        theme_id = int(lab)
        if theme_id == -1:
            continue
        theme_to_indices[theme_id].append(i)
    if not theme_to_indices:
        return []

    arr = np.asarray(embeddings, dtype=float)
    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    norms[norms == 0.0] = 1.0
    arr = arr / norms

    theme_ids = sorted(theme_to_indices.keys())
    centroids: list[np.ndarray] = []
    for tid in theme_ids:
        idxs = theme_to_indices[tid]
        cluster_arr = arr[idxs]
        centroid = cluster_arr.mean(axis=0)
        centroid_norm = float(np.linalg.norm(centroid))
        if centroid_norm == 0.0:
            centroid_norm = 1.0
        centroids.append(centroid / centroid_norm)
    centroids_arr = np.stack(centroids, axis=0)

    outlier_indices.sort(key=lambda i: _theme_item_sort_key(items[i]))

    out: list[dict[str, Any]] = []
    for i in outlier_indices:
        out_vec = arr[i]
        sims = centroids_arr @ out_vec

        candidates: list[dict[str, Any]] = []
        for tid, sim in zip(theme_ids, sims.tolist()):
            sim6 = _round_similarity_6dp(float(sim))
            candidates.append(
                {
                    "theme_id": int(tid),
                    "label": _theme_label(output=output, theme_id=int(tid)),
                    "similarity": sim6,
                }
            )

        candidates.sort(
            key=lambda c: (
                -float(_format_similarity_2dp(float(c["similarity"]))),
                int(c["theme_id"]),
            )
        )
        candidates = candidates[: min(OUTLIERS_NEAREST_TOP_K, len(candidates))]

        assigned_theme_id: int | None = None
        assigned_similarity: float | None = None
        if candidates:
            best_sim = _round_similarity_6dp(float(candidates[0]["similarity"]))
            if best_sim >= OUTLIERS_NEAREST_THRESHOLD:
                assigned_theme_id = int(candidates[0]["theme_id"])
                assigned_similarity = best_sim

        out.append(
            {
                "change_index": int(items[i].change_index),
                "assigned_theme_id": assigned_theme_id,
                "assigned_similarity": assigned_similarity,
                "candidates": candidates,
            }
        )

    return out


def render_themes_markdown(
    *,
    output: ThemeOutput,
    report: GapReport | None = None,
    isotope_pairs: dict[int, IsotopePairing] | None = None,
    outliers_nearest: list[dict[str, Any]] | None = None,
) -> str:
    theme_items = sorted(
        output.themes.items(),
        key=lambda kv: _theme_sort_key(kv[0], kv[1]),
    )
    lines: list[str] = []
    lines.append("# Gap themes")
    lines.append("")
    if report is not None:
        lines.extend(_themes_markdown_compare_lines(report=report))
    lines.append(
        f"- provider: `{output.provider}` model: `{output.model}` dims: `{output.dims}`"
    )
    lines.append(
        f"- hdbscan: min_cluster_size={output.min_cluster_size} min_samples={output.min_samples} allow_single_cluster={output.allow_single_cluster}"
    )
    lines.append("")

    emitted_pairs: set[int] = set()

    for theme_id, items in theme_items:
        label = _theme_label(output=output, theme_id=theme_id)
        lines.append(f"## Theme {theme_id}: {label} ({len(items)})")
        lines.append("")

        by_path: dict[str, list[ThemeItem]] = defaultdict(list)
        for it in items:
            if report is None:
                by_path[it.path or "(unknown)"].append(it)
                continue
            h = _symbol_handle_for_theme_item(
                report=report, change_index=int(it.change_index), prefer_new=True
            )
            by_path[(h.path if h is not None else (it.path or "(unknown)"))].append(it)
        for path in sorted(by_path.keys()):
            lines.append(f"- `{path}`")

            def _sort_key(it: ThemeItem) -> tuple[int, int, int, int]:
                if report is None:
                    return (0, 0, 0, int(it.change_index))
                h = _symbol_handle_for_theme_item(
                    report=report, change_index=int(it.change_index), prefer_new=True
                )
                if h is None:
                    return (10**9, 10**9, 10**9, int(it.change_index))
                sl = int(h.start_line) if int(h.start_line) > 0 else 10**9
                el = int(h.end_line) if int(h.end_line) > 0 else 10**9
                ord_ = int(h.ordinal_in_file) if int(h.ordinal_in_file) > 0 else 10**9
                return (sl, el, ord_, int(it.change_index))

            for it in sorted(by_path[path], key=_sort_key):
                pairing = (
                    isotope_pairs.get(int(it.change_index))
                    if isotope_pairs is not None
                    else None
                )
                if report is None:
                    sym = _markdown_code_span(
                        _truncate_display_text(it.symbol or "(unknown)")
                    )
                    ct = _normalize_display_text(it.chunk_type or "symbol")
                    conf = _format_confidence_2dp(float(it.confidence))
                    base = (
                        f"{it.op} {ct} {sym} "
                        f"(reason={_normalize_display_text(it.reason)}, confidence={conf}, change_index={int(it.change_index)})"
                    )
                    if pairing is not None:
                        base = (
                            base[:-1]
                            + f", paired_by={_normalize_display_text(pairing.method)}, pair_id={int(pairing.pair_id)}, pair_role={pairing.role}, paired_with={int(pairing.counterpart_change_index)})"
                        )
                    lines.append(f"  - {base}")
                else:
                    idx = int(it.change_index)
                    ch = report.changes[idx] if 0 <= idx < len(report.changes) else None
                    old = (
                        ch.old
                        if isinstance(ch, GapChangeItem)
                        and isinstance(ch.old, GapSymbolHandle)
                        else None
                    )
                    new = (
                        ch.new
                        if isinstance(ch, GapChangeItem)
                        and isinstance(ch.new, GapSymbolHandle)
                        else None
                    )
                    ct = _normalize_display_text(
                        (
                            new.chunk_type
                            if new is not None
                            else (
                                old.chunk_type
                                if old is not None
                                else (it.chunk_type or "symbol")
                            )
                        )
                    )
                    sym_raw = (
                        new.symbol
                        if new is not None
                        else (
                            old.symbol
                            if old is not None
                            else (it.symbol or "(unknown)")
                        )
                    )
                    sym = _markdown_code_span(
                        _truncate_display_text(_normalize_display_text(sym_raw))
                    )
                    reason = _normalize_display_text(it.reason)
                    conf = _format_confidence_2dp(float(it.confidence))

                    if isinstance(ch, GapChangeItem) and ch.op == "update":
                        old_loc, old_ord = _locator_display(old)
                        new_loc, new_ord = _locator_display(new)
                        line = (
                            f"update {ct} {sym} OLD {_markdown_code_span(old_loc)} {old_ord} -> "
                            f"NEW {_markdown_code_span(new_loc)} {new_ord} "
                            f"(reason={reason}, confidence={conf}, change_index={idx})"
                        )
                    else:
                        h = new if new is not None else old
                        loc, ord_part = _locator_display(h)
                        line = (
                            f"{it.op} {ct} {sym} @ {_markdown_code_span(loc)} {ord_part} "
                            f"(reason={reason}, confidence={conf}, change_index={idx})"
                        )
                        if pairing is not None:
                            line = (
                                line[:-1]
                                + f", paired_by={_normalize_display_text(pairing.method)}, pair_id={int(pairing.pair_id)}, pair_role={pairing.role}, paired_with={int(pairing.counterpart_change_index)}, pair_confidence={_format_confidence_2dp(pairing.confidence)})"
                            )
                    lines.append(f"  - {line}")

                if (
                    report is None
                    or pairing is None
                    or pairing.role != "remove"
                    or pairing.pair_id in emitted_pairs
                ):
                    continue

                emitted_pairs.add(pairing.pair_id)

                rem_idx = int(it.change_index)
                add_idx = int(pairing.counterpart_change_index)
                if (
                    rem_idx < 0
                    or rem_idx >= len(report.changes)
                    or add_idx < 0
                    or add_idx >= len(report.changes)
                ):
                    continue

                rem_change = report.changes[rem_idx]
                add_change = report.changes[add_idx]

                old = (
                    rem_change.old
                    if isinstance(rem_change, GapChangeItem)
                    and isinstance(rem_change.old, GapSymbolHandle)
                    else None
                )
                new = (
                    add_change.new
                    if isinstance(add_change, GapChangeItem)
                    and isinstance(add_change.new, GapSymbolHandle)
                    else None
                )

                old_desc = "(unknown old)"
                new_desc = "(unknown new)"
                if old is not None:
                    loc, ord_part = _locator_display(old)
                    old_desc = f"{_normalize_display_text(old.chunk_type)} {_markdown_code_span(_truncate_display_text(_normalize_display_text(old.symbol)))} {_markdown_code_span(loc)} {ord_part}"
                if new is not None:
                    loc, ord_part = _locator_display(new)
                    new_desc = f"{_normalize_display_text(new.chunk_type)} {_markdown_code_span(_truncate_display_text(_normalize_display_text(new.symbol)))} {_markdown_code_span(loc)} {ord_part}"

                rationale = ""
                if pairing.rationale:
                    rationale = f" rationale={_markdown_code_span(_truncate_display_text(_normalize_display_text(pairing.rationale)))}"

                lines.append(
                    "  - "
                    f"[ISO:{_normalize_display_text(pairing.method)}] suggest_update "
                    f"{old_desc} -> {new_desc} "
                    f"(remove_change_index={rem_idx}, add_change_index={add_idx}, confidence={_format_confidence_2dp(pairing.confidence)}, pair_id={pairing.pair_id}){rationale}"
                )
            lines.append("")

    if outliers_nearest is not None and int(output.dims) > 0:
        by_change_index: dict[int, ThemeItem] = {}
        for theme_id, theme_items in output.themes.items():
            _ = theme_id
            for it in theme_items:
                by_change_index.setdefault(int(it.change_index), it)

        assigned: dict[int, list[dict[str, Any]]] = defaultdict(list)
        unassigned: list[dict[str, Any]] = []
        for entry in outliers_nearest:
            if not isinstance(entry, dict):
                continue
            assigned_theme_id = entry.get("assigned_theme_id")
            if isinstance(assigned_theme_id, int):
                assigned[int(assigned_theme_id)].append(entry)
            else:
                unassigned.append(entry)

        lines.append("## Outliers: Nearest Themes (advisory)")
        lines.append("")
        lines.append(
            f"- params: top_k={OUTLIERS_NEAREST_TOP_K} threshold={OUTLIERS_NEAREST_THRESHOLD:.2f}"
        )
        lines.append("")

        def _brief_for_change_index(change_index: int) -> str:
            it = by_change_index.get(int(change_index))
            if it is None:
                return f"change_index={int(change_index)}"

            if report is not None:
                h = _symbol_handle_for_theme_item(
                    report=report, change_index=int(it.change_index), prefer_new=True
                )
                ct = _normalize_display_text(
                    (h.chunk_type if h is not None else (it.chunk_type or "symbol"))
                )
                sym_raw = h.symbol if h is not None else (it.symbol or "(unknown)")
                sym = _markdown_code_span(
                    _truncate_display_text(_normalize_display_text(sym_raw))
                )
                loc, ord_part = _locator_display(h)
                return (
                    f"{it.op} {ct} {sym} @ {_markdown_code_span(loc)} {ord_part} "
                    f"(change_index={int(it.change_index)})"
                )

            path = _normalize_display_text(it.path or "(unknown)")
            ct = _normalize_display_text(it.chunk_type or "symbol")
            sym = _markdown_code_span(
                _truncate_display_text(_normalize_display_text(it.symbol or "(unknown)"))
            )
            return (
                f"{it.op} {ct} {sym} (path={_markdown_code_span(path)}, "
                f"change_index={int(it.change_index)})"
            )

        lines.append(f"### Assigned (similarity >= {OUTLIERS_NEAREST_THRESHOLD:.2f})")
        lines.append("")
        if not assigned:
            lines.append("- (none)")
        else:
            for theme_id in sorted(assigned.keys()):
                label = _theme_label(output=output, theme_id=int(theme_id))
                lines.append(f"- Theme {int(theme_id)}: {label}")
                for entry in assigned[int(theme_id)]:
                    change_index = int(entry.get("change_index", -1))
                    sim = entry.get("assigned_similarity")
                    sim_part = (
                        _format_similarity_2dp(float(sim))
                        if isinstance(sim, (int, float))
                        else "?"
                    )
                    lines.append(
                        f"  - {_brief_for_change_index(change_index)} "
                        f"(similarity={sim_part})"
                    )
        lines.append("")

        lines.append(f"### Unassigned (similarity < {OUTLIERS_NEAREST_THRESHOLD:.2f})")
        lines.append("")
        if not unassigned:
            lines.append("- (none)")
        else:
            for entry in unassigned:
                change_index = int(entry.get("change_index", -1))
                candidates = entry.get("candidates")
                cand_str = ""
                if isinstance(candidates, list) and candidates:
                    parts: list[str] = []
                    for c in candidates:
                        if not isinstance(c, dict):
                            continue
                        tid = c.get("theme_id")
                        label = c.get("label")
                        sim = c.get("similarity")
                        if not isinstance(tid, int) or not isinstance(label, str):
                            continue
                        sim_part = (
                            _format_similarity_2dp(float(sim))
                            if isinstance(sim, (int, float))
                            else "?"
                        )
                        parts.append(f"Theme {int(tid)}: {label} ({sim_part})")
                    if parts:
                        cand_str = " nearest: " + "; ".join(parts)
                lines.append(f"- {_brief_for_change_index(change_index)}{cand_str}")
        lines.append("")

    if report is not None:
        file_changes: list[tuple[int, GapChangeItem]] = [
            (int(idx), ch)
            for idx, ch in enumerate(report.changes)
            if isinstance(ch, GapChangeItem) and ch.entity_kind == "file"
        ]
        file_changes.sort(
            key=lambda kv: (
                (kv[1].new.path if kv[1].new is not None else kv[1].old.path)  # type: ignore[union-attr]
                if (kv[1].new is not None or kv[1].old is not None)
                else "",
                kv[1].op,
                kv[1].reason,
                kv[0],
            )
        )
        if file_changes:
            lines.append("## Non-symbol file changes")
            lines.append("")
            lines.append(f"- total: {len(file_changes)}")
            lines.append("")
            for change_index, ch in file_changes:
                path = ""
                if ch.new is not None:
                    path = ch.new.path
                elif ch.old is not None:
                    path = ch.old.path
                old_class = (
                    getattr(ch.old, "file_class", None) if ch.old is not None else None
                )
                new_class = (
                    getattr(ch.new, "file_class", None) if ch.new is not None else None
                )
                if (
                    old_class is not None
                    and new_class is not None
                    and old_class != new_class
                ):
                    file_class = f"{old_class}->{new_class}"
                else:
                    file_class = str(new_class or old_class or "unknown")
                lines.append(
                    f"- `{path}`: {ch.op} (reason={ch.reason}, file_class={file_class}, change_index={change_index})"
                )
            lines.append("")

    return "\n".join(lines).rstrip() + "\n"


def write_theme_artifacts(
    *,
    out_dir: Path,
    report: GapReport,
    output: ThemeOutput,
    isotope_pairs: dict[int, IsotopePairing] | None = None,
    outliers_nearest: list[dict[str, Any]] | None = None,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    file_change_indexes = [
        int(idx)
        for idx, ch in enumerate(report.changes)
        if isinstance(ch, GapChangeItem) and ch.entity_kind == "file"
    ]
    file_change_indexes.sort()

    def _locator_fields(
        *, handle: GapSymbolHandle | None, prefix: str = ""
    ) -> dict[str, Any]:
        if handle is None:
            return {}
        return {
            f"{prefix}path": str(handle.path),
            f"{prefix}start_line": int(handle.start_line),
            f"{prefix}end_line": int(handle.end_line),
            f"{prefix}ordinal_in_file": int(handle.ordinal_in_file),
        }

    themes_payload: dict[str, Any] = {
        "schema_version": report.schema_version,
        "schema_revision": report.schema_revision,
        "direction": report.direction,
        "compare": _themes_compare_object(report=report),
        "provider": output.provider,
        "model": output.model,
        "dims": int(output.dims),
        "min_cluster_size": int(output.min_cluster_size),
        "min_samples": int(output.min_samples),
        "allow_single_cluster": bool(output.allow_single_cluster),
        "file_change_indexes": file_change_indexes,
        "themes": [],
    }
    if outliers_nearest is not None and int(output.dims) > 0:
        themes_payload["outliers_nearest"] = outliers_nearest

    ordered_themes = sorted(
        output.themes.items(),
        key=lambda kv: _theme_sort_key(kv[0], kv[1]),
    )
    for theme_id, items in ordered_themes:
        theme_dict = {
            "theme_id": int(theme_id),
            "label": _theme_label(output=output, theme_id=int(theme_id)),
            "items": [],
        }
        for it in sorted(items, key=_theme_item_sort_key):
            idx = int(it.change_index)
            item_dict: dict[str, Any] = {
                "change_index": idx,
                "entity_kind": it.entity_kind,
                "op": it.op,
                "reason": it.reason,
                "confidence": float(it.confidence),
                "chunk_type": it.chunk_type,
                "path": it.path,
                "symbol": it.symbol,
            }

            ch = report.changes[idx] if 0 <= idx < len(report.changes) else None
            if isinstance(ch, GapChangeItem) and ch.entity_kind == "symbol":
                old = ch.old if isinstance(ch.old, GapSymbolHandle) else None
                new = ch.new if isinstance(ch.new, GapSymbolHandle) else None

                if ch.op == "update":
                    item_dict.update(_locator_fields(handle=old, prefix="old_"))
                    item_dict.update(_locator_fields(handle=new, prefix="new_"))
                    preferred = new if new is not None else old
                else:
                    preferred = new if new is not None else old

                item_dict.update(_locator_fields(handle=preferred))

                if isotope_pairs is not None:
                    pairing = isotope_pairs.get(idx)
                    if pairing is not None:
                        item_dict.update(
                            {
                                "paired_by": pairing.method,
                                "pair_id": int(pairing.pair_id),
                                "pair_role": pairing.role,
                                "paired_with": int(pairing.counterpart_change_index),
                                "pair_confidence": float(pairing.confidence),
                                "pair_rationale": pairing.rationale,
                            }
                        )

            theme_dict["items"].append(item_dict)
        themes_payload["themes"].append(theme_dict)

    (out_dir / "themes.json").write_text(
        json.dumps(themes_payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (out_dir / "themes.md").write_text(
        render_themes_markdown(
            output=output,
            report=report,
            isotope_pairs=isotope_pairs,
            outliers_nearest=outliers_nearest if int(output.dims) > 0 else None,
        ),
        encoding="utf-8",
    )

    run_info: dict[str, Any] = {
        "schema_version": report.schema_version,
        "schema_revision": report.schema_revision,
        "direction": report.direction,
        "scope_hash": report.scope.scope_hash,
        "embedding_provider": output.provider,
        "embedding_model": output.model,
        "embedding_dims": output.dims,
        "hdbscan": {
            "min_cluster_size": output.min_cluster_size,
            "min_samples": output.min_samples,
            "allow_single_cluster": output.allow_single_cluster,
            "cluster_selection_method": "leaf",
        },
        "warnings_total": len(report.warnings),
    }
    (out_dir / "run.json").write_text(
        json.dumps(run_info, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


__all__ = [
    "IsotopePairing",
    "ThemeItem",
    "ThemeOutput",
    "build_theme_documents",
    "embed_in_batches",
    "cluster_embeddings_hdbscan",
    "build_theme_output",
    "build_fallback_theme_output",
    "compute_outliers_nearest",
    "render_themes_markdown",
    "write_theme_artifacts",
]
