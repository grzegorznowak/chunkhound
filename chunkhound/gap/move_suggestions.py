"""Advisory move/update pairing suggestions for `chunkhound gap`.

This module emits a separate, non-contractual artifact (`gap.suggestions.v1`) that
references `gap.v1` changes by `change_index`.

Design constraint: never mutate `gap.json` output.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
import json
from pathlib import Path
from pathlib import PurePosixPath
from typing import Any

import numpy as np

from chunkhound.gap.models import GapChangeItem, GapReport, GapSymbolHandle
from chunkhound.interfaces.embedding_provider import EmbeddingProvider
from chunkhound.interfaces.llm_provider import LLMProvider


@dataclass(frozen=True)
class MoveSuggestion:
    pair_id: int
    remove_change_index: int
    add_change_index: int
    method: str
    confidence: float
    rationale: str | None = None


@dataclass(frozen=True)
class _Candidate:
    change_index: int
    handle: GapSymbolHandle


def _handle_sort_key(cand: _Candidate) -> tuple[str, int, int, int, int]:
    h = cand.handle
    return (
        str(h.path),
        int(h.start_line),
        int(h.end_line),
        int(h.ordinal_in_file),
        int(cand.change_index),
    )


def _dir_of(path: str) -> str:
    return PurePosixPath(path).parent.as_posix()


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


def _suggest_heuristic_pairs(
    *,
    adds: list[_Candidate],
    removes: list[_Candidate],
) -> tuple[list[MoveSuggestion], Counter[str]]:
    """Suggest 1:1 move/update pairings using deterministic heuristics only."""
    suggestions: list[MoveSuggestion] = []
    skipped: Counter[str] = Counter()

    used_adds: set[int] = set()
    used_removes: set[int] = set()

    def _pair(
        *,
        remove_idx: int,
        add_idx: int,
        method: str,
        confidence: float,
        rationale: str,
    ) -> None:
        suggestions.append(
            MoveSuggestion(
                pair_id=0,  # filled deterministically after sorting
                remove_change_index=int(remove_idx),
                add_change_index=int(add_idx),
                method=method,
                confidence=float(confidence),
                rationale=rationale,
            )
        )
        used_adds.add(int(add_idx))
        used_removes.add(int(remove_idx))

    def _unpaired_adds() -> list[_Candidate]:
        return [c for c in adds if c.change_index not in used_adds]

    def _unpaired_removes() -> list[_Candidate]:
        return [c for c in removes if c.change_index not in used_removes]

    # Heuristic 1: same path + same chunk_type + same symbol.
    adds_by_key: dict[tuple[str, str, str], list[_Candidate]] = defaultdict(list)
    removes_by_key: dict[tuple[str, str, str], list[_Candidate]] = defaultdict(list)
    for c in _unpaired_adds():
        k = (str(c.handle.path), str(c.handle.chunk_type), str(c.handle.symbol))
        adds_by_key[k].append(c)
    for c in _unpaired_removes():
        k = (str(c.handle.path), str(c.handle.chunk_type), str(c.handle.symbol))
        removes_by_key[k].append(c)

    for k in sorted(set(adds_by_key.keys()) | set(removes_by_key.keys())):
        a = sorted(adds_by_key.get(k, []), key=_handle_sort_key)
        r = sorted(removes_by_key.get(k, []), key=_handle_sort_key)
        if len(a) == 1 and len(r) == 1:
            _pair(
                remove_idx=r[0].change_index,
                add_idx=a[0].change_index,
                method="heuristic_same_path",
                confidence=0.90,
                rationale="unique 1:1 match on (path, chunk_type, symbol)",
            )
        elif len(a) > 1 or len(r) > 1:
            skipped["heuristic_same_path_ambiguous"] += 1

    # Heuristic 2: same directory + same chunk_type + same symbol.
    adds_by_key = defaultdict(list)
    removes_by_key = defaultdict(list)
    for c in _unpaired_adds():
        k = (_dir_of(str(c.handle.path)), str(c.handle.chunk_type), str(c.handle.symbol))
        adds_by_key[k].append(c)
    for c in _unpaired_removes():
        k = (_dir_of(str(c.handle.path)), str(c.handle.chunk_type), str(c.handle.symbol))
        removes_by_key[k].append(c)

    for k in sorted(set(adds_by_key.keys()) | set(removes_by_key.keys())):
        a = sorted(adds_by_key.get(k, []), key=_handle_sort_key)
        r = sorted(removes_by_key.get(k, []), key=_handle_sort_key)
        if len(a) == 1 and len(r) == 1:
            _pair(
                remove_idx=r[0].change_index,
                add_idx=a[0].change_index,
                method="heuristic_same_dir",
                confidence=0.80,
                rationale="unique 1:1 match on (dir, chunk_type, symbol)",
            )
        elif len(a) > 1 or len(r) > 1:
            skipped["heuristic_same_dir_ambiguous"] += 1

    # Deterministic pair_id assignment.
    suggestions.sort(
        key=lambda s: (
            int(s.remove_change_index),
            int(s.add_change_index),
            str(s.method),
        )
    )
    out: list[MoveSuggestion] = []
    for i, s in enumerate(suggestions, start=1):
        out.append(
            MoveSuggestion(
                pair_id=int(i),
                remove_change_index=int(s.remove_change_index),
                add_change_index=int(s.add_change_index),
                method=str(s.method),
                confidence=float(s.confidence),
                rationale=s.rationale,
            )
        )

    return out, skipped


def _candidate_doc(
    *,
    cand: _Candidate,
    side: str,
    texts_by_path_ordinal: dict[tuple[str, int], str],
) -> str:
    h = cand.handle
    parts = [
        f"side={side}",
        f"chunk_type={h.chunk_type}",
        f"symbol={h.symbol}",
        f"path={h.path}",
        f"lines={h.start_line}-{h.end_line}",
    ]
    t = texts_by_path_ordinal.get((h.path, int(h.ordinal_in_file)))
    if t:
        parts.append(t)
    return "\n".join(parts).strip()


async def _embed_in_batches(
    *, provider: EmbeddingProvider, texts: list[str], batch_size: int
) -> list[list[float]]:
    if not texts:
        return []
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    out: list[list[float]] = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        out.extend(await provider.embed(batch))
    if len(out) != len(texts):
        raise ValueError(
            f"Embedding provider returned {len(out)} vectors for {len(texts)} texts"
        )
    return out


async def _suggest_embedding_pairs_crosscheck(
    *,
    embedding_provider: EmbeddingProvider,
    adds: list[_Candidate],
    removes: list[_Candidate],
    texts_a_by_path_ordinal: dict[tuple[str, int], str],
    texts_b_by_path_ordinal: dict[tuple[str, int], str],
    min_score: float,
    min_margin: float,
    min_score_block: float,
    min_margin_block: float,
    batch_size: int,
    want_candidates: bool,
) -> tuple[list[MoveSuggestion], Counter[str], dict[int, list[tuple[int, float]]]]:
    """
    Suggest 1:1 pairings using embedding similarity with cross-check + margin gating.

    Strategy:
    - Compare within the same chunk_type only.
    - For each remove, pick best add and runner-up add.
    - Accept only if:
      - best score >= min_score (or block-specific threshold)
      - best score - runner-up >= min_margin (or block-specific threshold)
      - mutual-best: the chosen add also prefers this remove
    """
    suggestions: list[MoveSuggestion] = []
    skipped: Counter[str] = Counter()
    candidates_by_remove: dict[int, list[tuple[int, float]]] = {}

    adds_by_chunk: dict[str, list[_Candidate]] = defaultdict(list)
    removes_by_chunk: dict[str, list[_Candidate]] = defaultdict(list)
    for c in adds:
        adds_by_chunk[str(c.handle.chunk_type)].append(c)
    for c in removes:
        removes_by_chunk[str(c.handle.chunk_type)].append(c)

    for chunk_type in sorted(set(adds_by_chunk.keys()) | set(removes_by_chunk.keys())):
        add_group = sorted(adds_by_chunk.get(chunk_type, []), key=_handle_sort_key)
        rem_group = sorted(removes_by_chunk.get(chunk_type, []), key=_handle_sort_key)
        if not add_group or not rem_group:
            continue

        add_docs = [
            _candidate_doc(
                cand=c, side="add", texts_by_path_ordinal=texts_b_by_path_ordinal
            )
            for c in add_group
        ]
        rem_docs = [
            _candidate_doc(
                cand=c, side="remove", texts_by_path_ordinal=texts_a_by_path_ordinal
            )
            for c in rem_group
        ]

        try:
            add_vecs = await _embed_in_batches(
                provider=embedding_provider, texts=add_docs, batch_size=batch_size
            )
            rem_vecs = await _embed_in_batches(
                provider=embedding_provider, texts=rem_docs, batch_size=batch_size
            )
        except Exception:
            skipped["embedding_Crosscheck_embed_failed"] += 1
            continue

        add_arr = np.asarray(add_vecs, dtype=float)
        rem_arr = np.asarray(rem_vecs, dtype=float)
        if add_arr.ndim != 2 or rem_arr.ndim != 2:
            skipped["embedding_Crosscheck_bad_embedding_shape"] += 1
            continue
        if add_arr.shape[1] != rem_arr.shape[1]:
            skipped["embedding_Crosscheck_bad_embedding_dims"] += 1
            continue

        # Normalize for cosine similarity (dot product on unit vectors).
        add_norms = np.linalg.norm(add_arr, axis=1, keepdims=True)
        add_norms[add_norms == 0.0] = 1.0
        add_arr = add_arr / add_norms

        rem_norms = np.linalg.norm(rem_arr, axis=1, keepdims=True)
        rem_norms[rem_norms == 0.0] = 1.0
        rem_arr = rem_arr / rem_norms

        add_change_indexes = np.asarray(
            [int(c.change_index) for c in add_group], dtype=int
        )
        rem_change_indexes = np.asarray(
            [int(c.change_index) for c in rem_group], dtype=int
        )

        m = int(add_arr.shape[0])
        n = int(rem_arr.shape[0])
        if m <= 0 or n <= 0:
            continue

        best_add_pos = np.full(n, -1, dtype=int)
        best_scores = np.full(n, float("-inf"), dtype=float)
        second_scores = np.full(n, float("-inf"), dtype=float)

        best_score_for_add = np.full(m, float("-inf"), dtype=float)
        best_remove_for_add = np.full(m, -1, dtype=int)

        for i in range(n):
            r_idx = int(rem_change_indexes[i])
            sims = add_arr @ rem_arr[i]

            if want_candidates:
                order = np.argsort(-sims, kind="mergesort")
                candidates_by_remove[r_idx] = [
                    (int(add_change_indexes[int(j)]), float(sims[int(j)]))
                    for j in order
                ]

            j1 = int(np.argmax(sims))
            s1 = float(sims[j1])

            if m > 1:
                saved = float(sims[j1])
                sims[j1] = float("-inf")
                j2 = int(np.argmax(sims))
                s2 = float(sims[j2])
                sims[j1] = saved
            else:
                s2 = float("-inf")

            best_add_pos[i] = j1
            best_scores[i] = s1
            second_scores[i] = s2

            better = sims > best_score_for_add
            tie = (sims == best_score_for_add) & (r_idx < best_remove_for_add)
            mask = better | tie
            if np.any(mask):
                best_score_for_add[mask] = sims[mask]
                best_remove_for_add[mask] = r_idx

        if chunk_type == "block":
            effective_min_score = float(min_score_block)
            effective_min_margin = float(min_margin_block)
        else:
            effective_min_score = float(min_score)
            effective_min_margin = float(min_margin)

        for i in range(n):
            r_idx = int(rem_change_indexes[i])
            j1 = int(best_add_pos[i])
            if j1 < 0:
                skipped["embedding_Crosscheck_no_add_candidates"] += 1
                continue

            a_idx = int(add_change_indexes[j1])
            s1 = float(best_scores[i])
            s2 = float(second_scores[i])

            if s1 < effective_min_score:
                skipped["embedding_Crosscheck_below_min_score"] += 1
                continue
            if (s1 - s2) < effective_min_margin:
                skipped["embedding_Crosscheck_below_min_margin"] += 1
                continue
            if int(best_remove_for_add[j1]) != r_idx:
                skipped["embedding_Crosscheck_not_mutual_best"] += 1
                continue

            if np.isfinite(s2):
                runner_up = f"{s2:.3f}"
            else:
                runner_up = "none"

            suggestions.append(
                MoveSuggestion(
                    pair_id=0,
                    remove_change_index=int(r_idx),
                    add_change_index=int(a_idx),
                    method="embedding_crosscheck",
                    confidence=float(s1),
                    rationale=(
                        "mutual-best + margin passed "
                        f"(score={s1:.3f}, runner_up={runner_up}, "
                        f"min_score={effective_min_score:.2f}, "
                        f"min_margin={effective_min_margin:.2f})"
                    ),
                )
            )

    return suggestions, skipped, candidates_by_remove


def _truncate_for_llm(text: str, *, max_chars: int) -> str:
    if max_chars <= 0:
        return ""
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 1] + "…"


def _estimate_tokens_for_prompt(text: str) -> int:
    # Deterministic, provider-agnostic estimation to keep gating stable across
    # environments (do not consult registry config).
    if not text:
        return 0
    return max(1, int(len(text) / 3.0))


def _llm_tiebreak_schema() -> dict[str, Any]:
    return {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "add_change_index": {
                "anyOf": [
                    {"type": "integer"},
                    {"type": "null"},
                ]
            },
            "confidence": {"type": "number"},
            "rationale": {"type": "string"},
        },
        "required": ["add_change_index", "confidence", "rationale"],
    }


def _build_llm_tiebreak_prompt(
    *,
    remove: _Candidate,
    candidates: list[tuple[_Candidate, float]],
    texts_a_by_path_ordinal: dict[tuple[str, int], str],
    texts_b_by_path_ordinal: dict[tuple[str, int], str],
    max_chars_per_text: int,
) -> str:
    rem_doc = _candidate_doc(
        cand=remove, side="remove", texts_by_path_ordinal=texts_a_by_path_ordinal
    )
    rem_doc = _truncate_for_llm(rem_doc, max_chars=max_chars_per_text)

    lines: list[str] = []
    lines.append("You are resolving an ambiguous move/update pairing in a code diff.")
    lines.append("")
    lines.append(
        "Task: choose the best add candidate that corresponds to the given remove, "
        "or return null if none match."
    )
    lines.append("")
    lines.append("Return ONLY JSON matching the provided schema.")
    lines.append("")
    lines.append("REMOVE:")
    lines.append(rem_doc)
    lines.append("")
    lines.append("CANDIDATES:")
    for cand, score in candidates:
        doc = _candidate_doc(
            cand=cand, side="add", texts_by_path_ordinal=texts_b_by_path_ordinal
        )
        doc = _truncate_for_llm(doc, max_chars=max_chars_per_text)
        lines.append(
            f"- add_change_index={cand.change_index} score={score:.3f}\n{doc}".strip()
        )
        lines.append("")
    return "\n".join(lines).strip()


def _render_llm_call_markdown(
    *,
    call_index: int,
    remove: _Candidate,
    candidates: list[tuple[_Candidate, float]],
    candidates_available: int,
    prompt: str,
    schema: dict[str, Any],
) -> str:
    h = remove.handle
    lines: list[str] = []
    lines.append(f"# LLM Call {call_index}")
    lines.append("")
    lines.append("## Remove")
    lines.append(
        f"- change_index: {remove.change_index}\n"
        f"- chunk_type: {h.chunk_type}\n"
        f"- symbol: {h.symbol}\n"
        f"- path: {h.path}\n"
        f"- lines: {h.start_line}-{h.end_line}\n"
        f"- ordinal_in_file: {h.ordinal_in_file}"
    )
    lines.append("")
    lines.append("## Candidates")
    lines.append(f"- included: {len(candidates)}")
    lines.append(f"- available: {int(candidates_available)}")
    if candidates:
        lines.append(
            f"- top_score: {candidates[0][1]:.3f} (add_change_index={candidates[0][0].change_index})"
        )
        if len(candidates) > 1:
            lines.append(
                f"- runner_up_score: {candidates[1][1]:.3f} (add_change_index={candidates[1][0].change_index})"
            )
    lines.append("")
    lines.append("## JSON Schema")
    lines.append("```json")
    lines.append(json.dumps(schema, indent=2, sort_keys=True))
    lines.append("```")
    lines.append("")
    lines.append("## Prompt")
    lines.append(f"- estimated_tokens: {_estimate_tokens_for_prompt(prompt)}")
    lines.append("")
    lines.append("```")
    lines.append(prompt)
    lines.append("```")
    lines.append("")
    return "\n".join(lines)


async def _suggest_llm_tiebreak_pairs(
    *,
    llm_provider: LLMProvider | None,
    adds_by_change_index: dict[int, _Candidate],
    removes: list[_Candidate],
    candidates_by_remove: dict[int, list[tuple[int, float]]],
    texts_a_by_path_ordinal: dict[tuple[str, int], str],
    texts_b_by_path_ordinal: dict[tuple[str, int], str],
    used_adds: set[int],
    used_removes: set[int],
    dry_run_collector: list[tuple[str, str]] | None,
    dry_run: bool,
    min_score: float,
    top_k: int,
    max_prompt_tokens: int,
) -> tuple[list[MoveSuggestion], Counter[str], str, bool]:
    suggestions: list[MoveSuggestion] = []
    skipped: Counter[str] = Counter()
    had_error = False
    schema = _llm_tiebreak_schema()

    for rem in sorted(removes, key=_handle_sort_key):
        if rem.change_index in used_removes:
            continue
        raw = candidates_by_remove.get(int(rem.change_index))
        if not raw:
            skipped["llm_no_embedding_candidates"] += 1
            continue

        candidates: list[tuple[_Candidate, float]] = []
        for add_idx, score in raw:
            if int(add_idx) in used_adds:
                continue
            cand = adds_by_change_index.get(int(add_idx))
            if cand is None:
                continue
            candidates.append((cand, float(score)))

        if not candidates:
            skipped["llm_no_remaining_candidates"] += 1
            continue

        candidates_available = len(candidates)
        candidates.sort(key=lambda kv: (-float(kv[1]), int(kv[0].change_index)))

        top_score = float(candidates[0][1])
        if top_score < float(min_score):
            skipped["llm_top_score_below_min_score"] += 1
            continue

        effective_top_k = max(1, int(top_k))
        candidates = candidates[:effective_top_k]

        # Token-budget-based trimming: drop lowest-ranked candidates until prompt fits.
        prompt = _build_llm_tiebreak_prompt(
            remove=rem,
            candidates=candidates,
            texts_a_by_path_ordinal=texts_a_by_path_ordinal,
            texts_b_by_path_ordinal=texts_b_by_path_ordinal,
            max_chars_per_text=2000,
        )
        while (
            candidates
            and _estimate_tokens_for_prompt(prompt) > int(max_prompt_tokens)
            and len(candidates) > 1
        ):
            candidates = candidates[:-1]
            prompt = _build_llm_tiebreak_prompt(
                remove=rem,
                candidates=candidates,
                texts_a_by_path_ordinal=texts_a_by_path_ordinal,
                texts_b_by_path_ordinal=texts_b_by_path_ordinal,
                max_chars_per_text=2000,
            )
        if candidates and _estimate_tokens_for_prompt(prompt) > int(max_prompt_tokens):
            skipped["llm_prompt_over_token_limit"] += 1
            continue

        if dry_run:
            if dry_run_collector is not None:
                filename = f"llm_call_{len(dry_run_collector)+1:04d}.md"
                dry_run_collector.append(
                    (
                        filename,
                        _render_llm_call_markdown(
                            call_index=len(dry_run_collector) + 1,
                            remove=rem,
                            candidates=candidates,
                            candidates_available=candidates_available,
                            prompt=prompt,
                            schema=schema,
                        ),
                    )
                )
            skipped["llm_dry_run_no_call"] += 1
            continue

        if llm_provider is None:
            skipped["llm_provider_missing"] += 1
            had_error = True
            continue

        try:
            result = await llm_provider.complete_structured(
                prompt=prompt, json_schema=schema
            )
        except Exception:
            skipped["llm_call_failed"] += 1
            had_error = True
            continue

        if not isinstance(result, dict):
            skipped["llm_invalid_response_type"] += 1
            continue

        add_choice = result.get("add_change_index")
        confidence = result.get("confidence")
        rationale = result.get("rationale")

        if add_choice is not None and not isinstance(add_choice, int):
            skipped["llm_invalid_add_change_index_type"] += 1
            continue
        if isinstance(confidence, int):
            confidence = float(confidence)
        if not isinstance(confidence, float):
            skipped["llm_invalid_confidence_type"] += 1
            continue
        if not isinstance(rationale, str):
            skipped["llm_invalid_rationale_type"] += 1
            continue

        if add_choice is None:
            skipped["llm_selected_null"] += 1
            continue

        if int(add_choice) not in {int(c.change_index) for c, _ in candidates}:
            skipped["llm_selected_non_candidate"] += 1
            continue

        if int(add_choice) in used_adds:
            skipped["llm_selected_already_used_add"] += 1
            continue

        suggestions.append(
            MoveSuggestion(
                pair_id=0,
                remove_change_index=int(rem.change_index),
                add_change_index=int(add_choice),
                method="llm_tiebreak",
                confidence=float(confidence),
                rationale=str(rationale).strip() or "llm tiebreak",
            )
        )
        used_adds.add(int(add_choice))
        used_removes.add(int(rem.change_index))

    if had_error and not suggestions:
        status = "failed"
        incomplete = True
    elif had_error and suggestions:
        status = "partial"
        incomplete = True
    else:
        status = "dry_run" if dry_run else "complete"
        incomplete = bool(dry_run)

    return suggestions, skipped, status, incomplete


async def build_move_suggestions_payload(
    *,
    report: GapReport,
    include_blocks: bool,
    texts_a_by_path_ordinal: dict[tuple[str, int], str] | None = None,
    texts_b_by_path_ordinal: dict[tuple[str, int], str] | None = None,
    embedding_provider: EmbeddingProvider | None = None,
    embed_min_score: float = 0.82,
    embed_min_margin: float = 0.06,
    embed_min_score_block: float = 0.88,
    embed_min_margin_block: float = 0.10,
    embed_batch_size: int = 100,
    llm_provider: LLMProvider | None = None,
    llm_enabled: bool = True,
    llm_dry_run: bool = False,
    llm_dry_run_collector: list[tuple[str, str]] | None = None,
    llm_min_score: float = 0.90,
    llm_top_k: int = 8,
    llm_max_prompt_tokens: int = 8000,
) -> dict[str, Any]:
    adds_raw, removes_raw = collect_unresolved(report, include_blocks=include_blocks)
    adds: list[_Candidate] = []
    removes: list[_Candidate] = []
    for idx, ch in adds_raw:
        handle = ch.new
        if isinstance(handle, GapSymbolHandle):
            adds.append(_Candidate(change_index=int(idx), handle=handle))
    for idx, ch in removes_raw:
        handle = ch.old
        if isinstance(handle, GapSymbolHandle):
            removes.append(_Candidate(change_index=int(idx), handle=handle))

    heuristic_suggestions, skipped = _suggest_heuristic_pairs(adds=adds, removes=removes)

    used_adds = {int(s.add_change_index) for s in heuristic_suggestions}
    used_removes = {int(s.remove_change_index) for s in heuristic_suggestions}
    adds_left = [c for c in adds if c.change_index not in used_adds]
    removes_left = [c for c in removes if c.change_index not in used_removes]

    embed_suggestions: list[MoveSuggestion] = []
    embed_skipped: Counter[str] = Counter()
    candidates_by_remove: dict[int, list[tuple[int, float]]] = {}
    if embedding_provider is not None and adds_left and removes_left:
        (
            embed_suggestions,
            embed_skipped,
            candidates_by_remove,
        ) = await _suggest_embedding_pairs_crosscheck(
            embedding_provider=embedding_provider,
            adds=adds_left,
            removes=removes_left,
            texts_a_by_path_ordinal=texts_a_by_path_ordinal or {},
            texts_b_by_path_ordinal=texts_b_by_path_ordinal or {},
            min_score=float(embed_min_score),
            min_margin=float(embed_min_margin),
            min_score_block=float(embed_min_score_block),
            min_margin_block=float(embed_min_margin_block),
            batch_size=max(100, int(embed_batch_size)),
            want_candidates=bool(llm_dry_run or (llm_enabled and llm_provider is not None)),
        )

    used_adds = {int(s.add_change_index) for s in heuristic_suggestions + embed_suggestions}
    used_removes = {
        int(s.remove_change_index) for s in heuristic_suggestions + embed_suggestions
    }

    llm_suggestions: list[MoveSuggestion] = []
    llm_skipped: Counter[str] = Counter()
    llm_status = "not_run"
    llm_incomplete = False

    remaining_removes = [c for c in removes_left if c.change_index not in used_removes]
    if not llm_enabled:
        llm_status = "disabled"
        llm_incomplete = False
    elif not remaining_removes:
        llm_status = "not_needed"
        llm_incomplete = False
    elif embedding_provider is None:
        llm_status = "skipped_no_embedding_candidates"
        llm_incomplete = True
    elif not candidates_by_remove:
        llm_status = "skipped_no_embedding_candidates"
        llm_incomplete = True
    elif (not llm_dry_run) and llm_provider is None:
        llm_status = "not_configured"
        llm_incomplete = True
    else:
        adds_by_change_index = {int(c.change_index): c for c in adds_left}
        (
            llm_suggestions,
            llm_skipped,
            llm_status,
            llm_incomplete,
        ) = await _suggest_llm_tiebreak_pairs(
            llm_provider=llm_provider,
            adds_by_change_index=adds_by_change_index,
            removes=remaining_removes,
            candidates_by_remove=candidates_by_remove,
            texts_a_by_path_ordinal=texts_a_by_path_ordinal or {},
            texts_b_by_path_ordinal=texts_b_by_path_ordinal or {},
            used_adds=used_adds,
            used_removes=used_removes,
            dry_run_collector=llm_dry_run_collector,
            dry_run=bool(llm_dry_run),
            min_score=float(llm_min_score),
            top_k=int(llm_top_k),
            max_prompt_tokens=int(llm_max_prompt_tokens),
        )

    all_suggestions = (
        list(heuristic_suggestions) + list(embed_suggestions) + list(llm_suggestions)
    )
    all_suggestions.sort(
        key=lambda s: (
            int(s.remove_change_index),
            int(s.add_change_index),
            str(s.method),
        )
    )
    numbered: list[MoveSuggestion] = []
    for i, s in enumerate(all_suggestions, start=1):
        numbered.append(
            MoveSuggestion(
                pair_id=int(i),
                remove_change_index=int(s.remove_change_index),
                add_change_index=int(s.add_change_index),
                method=str(s.method),
                confidence=float(s.confidence),
                rationale=s.rationale,
            )
        )

    skipped_total = Counter(skipped)
    skipped_total.update(embed_skipped)
    skipped_total.update(llm_skipped)
    counts_by_method = Counter(s.method for s in numbered)

    payload: dict[str, Any] = {
        "schema_version": "gap.suggestions.v1",
        "source_schema_version": report.schema_version,
        "direction": report.direction,
        "source_scope_hash": report.scope.scope_hash,
        "config_snapshot": {
            "include_blocks": bool(include_blocks),
            "embed": {
                "strategy": "crosscheck+margin+min_score",
                "min_score": float(embed_min_score),
                "min_margin": float(embed_min_margin),
                "min_score_block": float(embed_min_score_block),
                "min_margin_block": float(embed_min_margin_block),
                "batch_size": int(max(100, int(embed_batch_size))),
            },
            "llm": {
                "enabled": bool(llm_enabled),
                "dry_run": bool(llm_dry_run),
                "min_score": float(llm_min_score),
                "top_k": int(llm_top_k),
                "max_prompt_tokens": int(llm_max_prompt_tokens),
            },
        },
        "counts_by_method": dict(counts_by_method),
        "skipped_reasons": dict(skipped_total),
        "suggestions": [asdict(s) for s in numbered],
        "llm": {
            "status": str(llm_status),
            "incomplete": bool(llm_incomplete),
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
