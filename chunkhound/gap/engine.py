"""Stateless `chunkhound gap` engine (v1)."""

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


class GapEngine:
    """Compute a deterministic `gap.v1` changeset between two directory trees."""

    def __init__(self) -> None:
        pass

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
        file_changes: list[GapChangeItem] = []
        a_symbol_entries = []
        b_symbol_entries = []
        parser_factory = create_gap_parser_factory()

        a_paths = set(a_source.files.keys())
        b_paths = set(b_source.files.keys())
        all_paths = sorted(a_paths | b_paths)
        for rel_path in all_paths:
            a_file = a_source.files.get(rel_path)
            b_file = b_source.files.get(rel_path)

            if a_file is None:
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
                continue

            if b_file is None:
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
                continue

            # Present in both
            if a_file.file_hash != b_file.file_hash:
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

        symbol_changes = diff_symbol_entries_anchor(
            a_entries=a_symbol_entries,
            b_entries=b_symbol_entries,
        )

        warnings_recovery: list[GapWarning] = []
        if recovery_mode == "safe":
            symbol_changes, warnings_recovery = recover_safe_text_hash(symbol_changes)
            warnings_out.extend(warnings_recovery)

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

        return GapReport(
            schema_version="gap.v1",
            direction="A->B",
            invariants=invariants,
            inputs=inputs,
            scope=scope,
            warnings=warnings_out,
            stats=stats,
            changes=changes,
        )
