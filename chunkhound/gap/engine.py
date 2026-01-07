"""Stateless `chunkhound gap` engine (v1)."""

from collections import Counter, defaultdict
from dataclasses import replace
from pathlib import Path

from chunkhound.core.config.indexing_config import IndexingConfig
from chunkhound.gap.diff import (
    diff_symbol_entries_anchor,
    make_parse_fallback_file_change,
    sort_gap_changes,
)
from chunkhound.gap.discovery import compute_scope_hash, discover_source
from chunkhound.gap.manifest import create_gap_parser_factory, extract_file_symbols
from chunkhound.gap.models import (
    GapChangeItem,
    GapCounts,
    GapFileHandle,
    GapInputs,
    GapInvariants,
    GapNormalizationInvariants,
    GapReport,
    GapScope,
    GapStats,
    GapTimings,
    GapWarning,
)
from chunkhound.gap.recovery import recover_safe_text_hash
from chunkhound.gap.stats_report import GapStatsDetails


class GapEngine:
    """Compute a deterministic `gap.v1` changeset between two directory trees."""

    def __init__(self) -> None:
        pass

    def _run_pipeline(
        self,
        *,
        a_root: Path,
        b_root: Path,
        indexing: IndexingConfig,
        recovery_mode: str,
        deterministic: bool,
        warnings: list[GapWarning] | None,
        capture_symbol_texts: bool,
    ) -> tuple[
        GapReport,
        GapStatsDetails,
        dict[tuple[str, int], str],
        dict[tuple[str, int], str],
    ]:
        a_root = a_root.resolve()
        b_root = b_root.resolve()

        warnings_out: list[GapWarning] = list(warnings or [])

        a_source, a_warnings = discover_source(root=a_root, indexing=indexing)
        b_source, b_warnings = discover_source(root=b_root, indexing=indexing)
        warnings_out.extend(a_warnings)
        warnings_out.extend(b_warnings)

        inputs = GapInputs(a=a_source.input_ref, b=b_source.input_ref)

        invariants = GapInvariants(
            hash_alg="xxh3_64",
            normalization=GapNormalizationInvariants(
                id="normalize_content.v1",
                include_comments=False,
                include_docs=False,
            ),
            chunker_version="cast@v1",
            recovery_mode=recovery_mode,  # type: ignore[arg-type]
            deterministic=deterministic,
            embed_model_id=None,
            forced=False,
        )

        scope_hash = compute_scope_hash(
            a_source_hash=inputs.a.source_hash,
            b_source_hash=inputs.b.source_hash,
            include_patterns=list(indexing.include),
            config_excludes=indexing.get_effective_config_excludes(),
            ignore_sources=indexing.resolve_ignore_sources(),
            gitignore_backend=indexing.gitignore_backend,
            max_file_size_bytes=indexing.get_max_file_size_bytes(),
        )

        changed_files: set[str] = set()
        files_added = 0
        files_removed = 0
        files_modified = 0
        files_unchanged = 0

        file_changes: list[GapChangeItem] = []
        a_symbol_entries = []
        b_symbol_entries = []
        a_texts_by_path_ordinal: dict[tuple[str, int], str] = {}
        b_texts_by_path_ordinal: dict[tuple[str, int], str] = {}

        parser_factory = create_gap_parser_factory()

        a_paths = set(a_source.files.keys())
        b_paths = set(b_source.files.keys())
        all_paths = sorted(a_paths | b_paths)
        for rel_path in all_paths:
            a_file = a_source.files.get(rel_path)
            b_file = b_source.files.get(rel_path)

            if a_file is None:
                files_added += 1
                changed_files.add(rel_path)
                if b_file.file_class != "text":
                    file_changes.append(
                        GapChangeItem(
                            entity_kind="file",
                            op="add",
                            moved=False,
                            renamed=False,
                            content_changed=True,
                            reason="file_anchor",
                            confidence=1.0,
                            primary_key_kind="path_key",
                            key_strength=0,
                            had_collision=False,
                            collision_group_size=1,
                            old=None,
                            new=GapFileHandle(
                                path=rel_path,
                                file_hash=b_file.file_hash,
                                size_bytes=b_file.size_bytes,
                                file_class=b_file.file_class,
                            ),
                        )
                    )
                else:
                    parsed, parse_warnings = extract_file_symbols(
                        file=b_file,
                        parser_factory=parser_factory,
                        include_comments=invariants.normalization.include_comments,
                        include_docs=invariants.normalization.include_docs,
                    )
                    warnings_out.extend(parse_warnings)
                    if parsed.parse_failed:
                        file_changes.append(
                            make_parse_fallback_file_change(
                                op="add",
                                rel_path=rel_path,
                                old_file=None,
                                new_file=GapFileHandle(
                                    path=rel_path,
                                    file_hash=b_file.file_hash,
                                    size_bytes=b_file.size_bytes,
                                    file_class=b_file.file_class,
                                ),
                            )
                        )
                    else:
                        b_symbol_entries.extend(parsed.symbols)
                        if capture_symbol_texts:
                            for ordinal, txt in parsed.texts_by_ordinal.items():
                                b_texts_by_path_ordinal[(rel_path, int(ordinal))] = txt
                continue

            if b_file is None:
                files_removed += 1
                changed_files.add(rel_path)
                if a_file.file_class != "text":
                    file_changes.append(
                        GapChangeItem(
                            entity_kind="file",
                            op="remove",
                            moved=False,
                            renamed=False,
                            content_changed=True,
                            reason="file_anchor",
                            confidence=1.0,
                            primary_key_kind="path_key",
                            key_strength=0,
                            had_collision=False,
                            collision_group_size=1,
                            old=GapFileHandle(
                                path=rel_path,
                                file_hash=a_file.file_hash,
                                size_bytes=a_file.size_bytes,
                                file_class=a_file.file_class,
                            ),
                            new=None,
                        )
                    )
                else:
                    parsed, parse_warnings = extract_file_symbols(
                        file=a_file,
                        parser_factory=parser_factory,
                        include_comments=invariants.normalization.include_comments,
                        include_docs=invariants.normalization.include_docs,
                    )
                    warnings_out.extend(parse_warnings)
                    if parsed.parse_failed:
                        file_changes.append(
                            make_parse_fallback_file_change(
                                op="remove",
                                rel_path=rel_path,
                                old_file=GapFileHandle(
                                    path=rel_path,
                                    file_hash=a_file.file_hash,
                                    size_bytes=a_file.size_bytes,
                                    file_class=a_file.file_class,
                                ),
                                new_file=None,
                            )
                        )
                    else:
                        a_symbol_entries.extend(parsed.symbols)
                        if capture_symbol_texts:
                            for ordinal, txt in parsed.texts_by_ordinal.items():
                                a_texts_by_path_ordinal[(rel_path, int(ordinal))] = txt
                continue

            # Present in both
            if a_file.file_hash != b_file.file_hash:
                files_modified += 1
                changed_files.add(rel_path)
                if a_file.file_class != "text" or b_file.file_class != "text":
                    file_changes.append(
                        GapChangeItem(
                            entity_kind="file",
                            op="update",
                            moved=False,
                            renamed=False,
                            content_changed=True,
                            reason="file_anchor",
                            confidence=1.0,
                            primary_key_kind="path_key",
                            key_strength=0,
                            had_collision=False,
                            collision_group_size=1,
                            old=GapFileHandle(
                                path=rel_path,
                                file_hash=a_file.file_hash,
                                size_bytes=a_file.size_bytes,
                                file_class=a_file.file_class,
                            ),
                            new=GapFileHandle(
                                path=rel_path,
                                file_hash=b_file.file_hash,
                                size_bytes=b_file.size_bytes,
                                file_class=b_file.file_class,
                            ),
                        )
                    )
                    continue

                parsed_a, warnings_a = extract_file_symbols(
                    file=a_file,
                    parser_factory=parser_factory,
                    include_comments=invariants.normalization.include_comments,
                    include_docs=invariants.normalization.include_docs,
                )
                parsed_b, warnings_b = extract_file_symbols(
                    file=b_file,
                    parser_factory=parser_factory,
                    include_comments=invariants.normalization.include_comments,
                    include_docs=invariants.normalization.include_docs,
                )
                warnings_out.extend(warnings_a)
                warnings_out.extend(warnings_b)

                if parsed_a.parse_failed or parsed_b.parse_failed:
                    file_changes.append(
                        make_parse_fallback_file_change(
                            op="update",
                            rel_path=rel_path,
                            old_file=GapFileHandle(
                                path=rel_path,
                                file_hash=a_file.file_hash,
                                size_bytes=a_file.size_bytes,
                                file_class=a_file.file_class,
                            ),
                            new_file=GapFileHandle(
                                path=rel_path,
                                file_hash=b_file.file_hash,
                                size_bytes=b_file.size_bytes,
                                file_class=b_file.file_class,
                            ),
                        )
                    )
                else:
                    a_symbol_entries.extend(parsed_a.symbols)
                    b_symbol_entries.extend(parsed_b.symbols)
                    if capture_symbol_texts:
                        for ordinal, txt in parsed_a.texts_by_ordinal.items():
                            a_texts_by_path_ordinal[(rel_path, int(ordinal))] = txt
                        for ordinal, txt in parsed_b.texts_by_ordinal.items():
                            b_texts_by_path_ordinal[(rel_path, int(ordinal))] = txt
            else:
                files_unchanged += 1

        collision_counts_a: dict[tuple[str, str], int] = defaultdict(int)
        collision_counts_b: dict[tuple[str, str], int] = defaultdict(int)
        for entry in a_symbol_entries:
            collision_counts_a[(entry.primary_key_kind, entry.primary_key_hash)] += 1
        for entry in b_symbol_entries:
            collision_counts_b[(entry.primary_key_kind, entry.primary_key_hash)] += 1
        collision_groups = 0
        collision_group_max_size = 0
        for key in sorted(set(collision_counts_a) | set(collision_counts_b)):
            size = max(collision_counts_a.get(key, 0), collision_counts_b.get(key, 0))
            if size > 1:
                collision_groups += 1
                collision_group_max_size = max(collision_group_max_size, size)

        symbol_changes = diff_symbol_entries_anchor(
            a_entries=a_symbol_entries,
            b_entries=b_symbol_entries,
        )

        warnings_recovery: list[GapWarning] = []
        if recovery_mode == "safe":
            symbol_changes, warnings_recovery = recover_safe_text_hash(symbol_changes)
            warnings_out.extend(warnings_recovery)

        symbol_changes_add = sum(1 for c in symbol_changes if c.op == "add")
        symbol_changes_remove = sum(1 for c in symbol_changes if c.op == "remove")
        symbol_changes_update = sum(1 for c in symbol_changes if c.op == "update")
        symbol_updates_moved = sum(
            1 for c in symbol_changes if c.op == "update" and c.moved
        )
        symbol_updates_renamed = sum(
            1 for c in symbol_changes if c.op == "update" and c.renamed
        )
        symbol_updates_content_changed = sum(
            1 for c in symbol_changes if c.op == "update" and c.content_changed
        )

        recovery_recovered_updates = sum(
            1
            for c in symbol_changes
            if c.op == "update" and c.reason == "recovery_safe_text_hash"
        )
        recovery_bucket_skipped = sum(
            1 for w in warnings_recovery if w.code == "RECOVERY_SAFE_BUCKET_SKIPPED"
        )
        recovery_cap_warning = next(
            (
                w
                for w in warnings_recovery
                if w.code == "RECOVERY_SAFE_CAPPED_TOTAL"
            ),
            None,
        )
        recovery_total_cap = None
        recovery_total_attempted = None
        recovery_total_applied = None
        if recovery_cap_warning is not None and recovery_cap_warning.meta is not None:
            recovery_total_cap = int(recovery_cap_warning.meta.get("cap"))
            recovery_total_attempted = int(recovery_cap_warning.meta.get("attempted"))
            recovery_total_applied = int(recovery_cap_warning.meta.get("applied"))

        changes = list(file_changes) + list(symbol_changes)
        sort_gap_changes(changes)

        scope = GapScope(
            scope_mode="full",
            scope_hash=scope_hash,
            changed_files_count=len(changed_files),
            rename_hints_count=0,
        )

        stats = GapStats(
            counts=GapCounts(
                files_a_total=len(a_source.files),
                files_b_total=len(b_source.files),
                changes_total=len(changes),
                file_changes_total=len(file_changes),
                symbol_changes_total=len(symbol_changes),
            ),
            timings=GapTimings(),
        )

        if deterministic:
            stats = replace(stats, timings=GapTimings())

        report = GapReport(
            schema_version="gap.v1",
            direction="A->B",
            invariants=invariants,
            inputs=inputs,
            scope=scope,
            warnings=warnings_out,
            stats=stats,
            changes=changes,
        )

        a_class_counts = Counter(f.file_class for f in a_source.files.values())
        b_class_counts = Counter(f.file_class for f in b_source.files.values())
        file_class_counts_a = {
            k: int(a_class_counts.get(k, 0))
            for k in ["text", "binary", "too_large", "decode_error"]
        }
        file_class_counts_b = {
            k: int(b_class_counts.get(k, 0))
            for k in ["text", "binary", "too_large", "decode_error"]
        }
        details = GapStatsDetails(
            file_class_counts_a=file_class_counts_a,
            file_class_counts_b=file_class_counts_b,
            file_diff_counts={
                "added": files_added,
                "removed": files_removed,
                "modified": files_modified,
                "unchanged": files_unchanged,
            },
            symbols_a_total=len(a_symbol_entries),
            symbols_b_total=len(b_symbol_entries),
            symbol_changes_add=symbol_changes_add,
            symbol_changes_remove=symbol_changes_remove,
            symbol_changes_update=symbol_changes_update,
            symbol_updates_moved=symbol_updates_moved,
            symbol_updates_renamed=symbol_updates_renamed,
            symbol_updates_content_changed=symbol_updates_content_changed,
            collision_groups=collision_groups,
            collision_group_max_size=collision_group_max_size,
            recovery_recovered_updates=recovery_recovered_updates,
            recovery_bucket_skipped=recovery_bucket_skipped,
            recovery_total_cap=recovery_total_cap,
            recovery_total_attempted=recovery_total_attempted,
            recovery_total_applied=recovery_total_applied,
        )

        return report, details, a_texts_by_path_ordinal, b_texts_by_path_ordinal

    def run(
        self,
        *,
        a_root: Path,
        b_root: Path,
        indexing: IndexingConfig,
        recovery_mode: str,
        deterministic: bool,
        warnings: list[GapWarning] | None = None,
    ) -> GapReport:
        report, _, _, _ = self._run_pipeline(
            a_root=a_root,
            b_root=b_root,
            indexing=indexing,
            recovery_mode=recovery_mode,
            deterministic=deterministic,
            warnings=warnings,
            capture_symbol_texts=False,
        )
        return report

    def run_with_details(
        self,
        *,
        a_root: Path,
        b_root: Path,
        indexing: IndexingConfig,
        recovery_mode: str,
        deterministic: bool,
        warnings: list[GapWarning] | None = None,
    ) -> tuple[GapReport, GapStatsDetails]:
        report, details, _, _ = self._run_pipeline(
            a_root=a_root,
            b_root=b_root,
            indexing=indexing,
            recovery_mode=recovery_mode,
            deterministic=deterministic,
            warnings=warnings,
        )
        return report, details

    def run_with_details_and_symbol_texts(
        self,
        *,
        a_root: Path,
        b_root: Path,
        indexing: IndexingConfig,
        recovery_mode: str,
        deterministic: bool,
        warnings: list[GapWarning] | None = None,
    ) -> tuple[
        GapReport,
        GapStatsDetails,
        dict[tuple[str, int], str],
        dict[tuple[str, int], str],
    ]:
        return self._run_pipeline(
            a_root=a_root,
            b_root=b_root,
            indexing=indexing,
            recovery_mode=recovery_mode,
            deterministic=deterministic,
            warnings=warnings,
            capture_symbol_texts=True,
        )
