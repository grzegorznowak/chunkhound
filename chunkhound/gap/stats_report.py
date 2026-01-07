"""Human-readable stats report helpers for `chunkhound gap`."""

from __future__ import annotations

from dataclasses import dataclass

from chunkhound.gap.models import GapReport


@dataclass(frozen=True)
class GapStatsDetails:
    file_class_counts_a: dict[str, int]
    file_class_counts_b: dict[str, int]
    file_diff_counts: dict[str, int]
    symbols_a_total: int
    symbols_b_total: int
    symbol_changes_add: int
    symbol_changes_remove: int
    symbol_changes_update: int
    symbol_updates_moved: int
    symbol_updates_renamed: int
    symbol_updates_content_changed: int
    collision_groups: int
    collision_group_max_size: int
    recovery_recovered_updates: int
    recovery_bucket_skipped: int
    recovery_total_cap: int | None
    recovery_total_attempted: int | None
    recovery_total_applied: int | None


def render_gap_stats_report(*, report: GapReport, details: GapStatsDetails) -> str:
    """Render a deterministic multi-line stats report for `chunkhound gap --stats`."""
    files = report.stats.counts

    a_classes = details.file_class_counts_a
    b_classes = details.file_class_counts_b
    file_diff = details.file_diff_counts

    lines: list[str] = []
    lines.append(f"{report.schema_version} {report.direction}")
    lines.append(
        f"scope: {report.scope.scope_mode} changed_files={report.scope.changed_files_count} scope_hash={report.scope.scope_hash}"
    )

    lines.append(
        "files: "
        f"a_total={files.files_a_total} b_total={files.files_b_total} "
        f"added={file_diff['added']} removed={file_diff['removed']} "
        f"modified={file_diff['modified']} unchanged={file_diff['unchanged']}"
    )
    lines.append(
        "file_classes_a: "
        f"text={a_classes['text']} binary={a_classes['binary']} "
        f"too_large={a_classes['too_large']} decode_error={a_classes['decode_error']}"
    )
    lines.append(
        "file_classes_b: "
        f"text={b_classes['text']} binary={b_classes['binary']} "
        f"too_large={b_classes['too_large']} decode_error={b_classes['decode_error']}"
    )

    lines.append(
        "symbols(parsed_changed_files): "
        f"a_total={details.symbols_a_total} b_total={details.symbols_b_total}"
    )

    lines.append(
        "symbol_changes: "
        f"add={details.symbol_changes_add} remove={details.symbol_changes_remove} "
        f"update={details.symbol_changes_update} "
        f"moved={details.symbol_updates_moved} renamed={details.symbol_updates_renamed} "
        f"content_changed={details.symbol_updates_content_changed}"
    )

    lines.append(
        "identity_collisions(parsed_changed_files): "
        f"groups={details.collision_groups} max_group_size={details.collision_group_max_size}"
    )

    recovery_line = (
        "recovery: "
        f"mode={report.invariants.recovery_mode} "
        f"recovered_updates={details.recovery_recovered_updates} "
        f"bucket_skipped={details.recovery_bucket_skipped}"
    )
    if (
        details.recovery_total_cap is not None
        and details.recovery_total_attempted is not None
        and details.recovery_total_applied is not None
    ):
        recovery_line += (
            " total_cap_hit="
            f"{details.recovery_total_applied}/{details.recovery_total_attempted}"
            f" (cap={details.recovery_total_cap})"
        )
    lines.append(recovery_line)

    lines.append(f"warnings: total={len(report.warnings)}")

    return "\n".join(lines) + "\n"


__all__ = ["GapStatsDetails", "render_gap_stats_report"]

