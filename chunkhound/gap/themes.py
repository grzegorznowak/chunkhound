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

from chunkhound.core.utils import estimate_tokens
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

    for idx, change in enumerate(report.changes):
        if change.entity_kind != "symbol":
            continue

        old = change.old if isinstance(change.old, GapSymbolHandle) else None
        new = change.new if isinstance(change.new, GapSymbolHandle) else None

        parts: list[str] = []
        parts.append(
            f"op={change.op} reason={change.reason} moved={change.moved} renamed={change.renamed} content_changed={change.content_changed}"
        )

        chunk_type = None
        path = None
        symbol = None

        if old is not None:
            chunk_type = old.chunk_type
            path = old.path
            symbol = old.symbol
            parts.append(f"OLD {old.chunk_type} {old.symbol} path={old.path}")
            t = _safe_handle_text(old, texts_a_by_path_ordinal)
            if t:
                parts.append(t)

        if new is not None:
            chunk_type = new.chunk_type
            path = new.path
            symbol = new.symbol
            parts.append(f"NEW {new.chunk_type} {new.symbol} path={new.path}")
            t = _safe_handle_text(new, texts_b_by_path_ordinal)
            if t:
                parts.append(t)

        doc = "\n".join(parts).strip()
        if not doc:
            continue

        # Ensure 1:1 mapping: truncate documents proactively to avoid provider-side splitting.
        if estimate_tokens(doc) > max_tokens_per_doc:
            words = doc.split()
            while words and estimate_tokens(" ".join(words)) > max_tokens_per_doc:
                words = words[: int(len(words) * 0.9)]
            doc = " ".join(words)

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
    arr = np.asarray(embeddings, dtype=float)
    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    norms[norms == 0.0] = 1.0
    arr = arr / norms

    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=max(2, min_cluster_size),
        min_samples=min_samples,
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


def render_themes_markdown(
    *,
    output: ThemeOutput,
    report: GapReport | None = None,
    isotope_pairs: dict[int, IsotopePairing] | None = None,
) -> str:
    theme_items = sorted(
        output.themes.items(),
        key=lambda kv: _theme_sort_key(kv[0], kv[1]),
    )
    lines: list[str] = []
    lines.append("# Gap themes")
    lines.append("")
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
            by_path[it.path or "(unknown)"].append(it)
        for path in sorted(by_path.keys()):
            lines.append(f"- `{path}`")
            for it in sorted(by_path[path], key=_theme_item_sort_key):
                sym = it.symbol or "(unknown)"
                ct = it.chunk_type or "symbol"
                pairing = (
                    isotope_pairs.get(int(it.change_index))
                    if isotope_pairs is not None
                    else None
                )
                if pairing is None:
                    lines.append(
                        f"  - {it.op} {ct} `{sym}` (reason={it.reason}, confidence={it.confidence:.2f}, change_index={it.change_index})"
                    )
                    continue

                lines.append(
                    f"  - {it.op} {ct} `{sym}` (reason={it.reason}, confidence={it.confidence:.2f}, change_index={it.change_index}, paired_by={pairing.method}, pair_id={pairing.pair_id}, pair_role={pairing.role}, paired_with={pairing.counterpart_change_index})"
                )

                if (
                    report is None
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
                    old_desc = (
                        f"{old.chunk_type} `{old.symbol}` {old.path}:{old.start_line}-{old.end_line} ord={old.ordinal_in_file}"
                    )
                if new is not None:
                    new_desc = (
                        f"{new.chunk_type} `{new.symbol}` {new.path}:{new.start_line}-{new.end_line} ord={new.ordinal_in_file}"
                    )

                rationale = ""
                if pairing.rationale:
                    rationale = f" rationale={pairing.rationale!r}"

                lines.append(
                    f"  - [ISO:{pairing.method}] suggest_update {old_desc} -> {new_desc} (remove_change_index={rem_idx}, add_change_index={add_idx}, confidence={pairing.confidence:.2f}, pair_id={pairing.pair_id}){rationale}"
                )
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
                old_class = getattr(ch.old, "file_class", None) if ch.old is not None else None
                new_class = getattr(ch.new, "file_class", None) if ch.new is not None else None
                if old_class is not None and new_class is not None and old_class != new_class:
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
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    file_change_indexes = [
        int(idx)
        for idx, ch in enumerate(report.changes)
        if isinstance(ch, GapChangeItem) and ch.entity_kind == "file"
    ]
    file_change_indexes.sort()

    themes_payload: dict[str, Any] = {
        "provider": output.provider,
        "model": output.model,
        "dims": int(output.dims),
        "min_cluster_size": int(output.min_cluster_size),
        "min_samples": int(output.min_samples),
        "allow_single_cluster": bool(output.allow_single_cluster),
        "file_change_indexes": file_change_indexes,
        "themes": [],
    }

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
            theme_dict["items"].append(
                {
                    "change_index": int(it.change_index),
                    "entity_kind": it.entity_kind,
                    "op": it.op,
                    "reason": it.reason,
                    "confidence": float(it.confidence),
                    "chunk_type": it.chunk_type,
                    "path": it.path,
                    "symbol": it.symbol,
                }
            )
        themes_payload["themes"].append(theme_dict)

    (out_dir / "themes.json").write_text(
        json.dumps(themes_payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (out_dir / "themes.md").write_text(
        render_themes_markdown(
            output=output, report=report, isotope_pairs=isotope_pairs
        ),
        encoding="utf-8",
    )

    run_info: dict[str, Any] = {
        "schema_version": report.schema_version,
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
    "render_themes_markdown",
    "write_theme_artifacts",
]
