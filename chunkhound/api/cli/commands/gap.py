"""Gap command module - stateless semantic gap analysis between two folders."""

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from loguru import logger

from chunkhound.core.config.config import Config
from chunkhound.core.config.embedding_factory import EmbeddingProviderFactory
from chunkhound.gap.engine import GapEngine
from chunkhound.gap.models import GapWarning
from chunkhound.gap.stats_report import render_gap_stats_report
from chunkhound.gap.themes import (
    IsotopePairing,
    build_fallback_theme_output,
    build_theme_documents,
    build_theme_output,
    cluster_embeddings_hdbscan,
    embed_in_batches,
    render_themes_markdown,
    write_theme_artifacts,
)
from chunkhound.gap.move_suggestions import (
    build_move_suggestions_payload,
    write_move_suggestions,
)


def _write_json(text: str, out: str) -> None:
    if out == "-":
        try:
            sys.stdout.write(text)
        except BrokenPipeError:
            try:
                sys.stdout.close()
            except Exception:
                pass
        return

    Path(out).write_text(text, encoding="utf-8")


def _write_text(text: str, out: Path) -> None:
    out.write_text(text, encoding="utf-8")


def _cleanup_theme_artifacts(out_dir: Path) -> None:
    for name in ("themes.json", "themes.md", "run.json"):
        try:
            (out_dir / name).unlink()
        except FileNotFoundError:
            pass


def _build_isotope_pairs_from_suggestions_payload(
    payload: dict[str, Any],
) -> dict[int, IsotopePairing]:
    out: dict[int, IsotopePairing] = {}
    suggestions = payload.get("suggestions")
    if not isinstance(suggestions, list):
        return out

    for item in suggestions:
        if not isinstance(item, dict):
            continue
        pair_id = item.get("pair_id")
        rem_idx = item.get("remove_change_index")
        add_idx = item.get("add_change_index")
        method = item.get("method")
        confidence = item.get("confidence")
        rationale = item.get("rationale")

        if not isinstance(pair_id, int):
            continue
        if not isinstance(rem_idx, int) or not isinstance(add_idx, int):
            continue
        if not isinstance(method, str):
            continue
        if isinstance(confidence, int):
            confidence = float(confidence)
        if not isinstance(confidence, float):
            continue
        if rationale is not None and not isinstance(rationale, str):
            rationale = None

        out[int(rem_idx)] = IsotopePairing(
            pair_id=int(pair_id),
            role="remove",
            counterpart_change_index=int(add_idx),
            method=method,
            confidence=float(confidence),
            rationale=rationale,
        )
        out[int(add_idx)] = IsotopePairing(
            pair_id=int(pair_id),
            role="add",
            counterpart_change_index=int(rem_idx),
            method=method,
            confidence=float(confidence),
            rationale=rationale,
        )

    return out


async def gap_command(args: argparse.Namespace, config: Config) -> None:
    """Execute the gap command (v1)."""
    a_root = Path(args.a).resolve()
    b_root = Path(args.b).resolve()

    if not a_root.exists() or not a_root.is_dir():
        logger.error(f"Path A must be an existing directory: {a_root}")
        sys.exit(1)
    if not b_root.exists() or not b_root.is_dir():
        logger.error(f"Path B must be an existing directory: {b_root}")
        sys.exit(1)

    out = getattr(args, "out", None)
    out_dir = getattr(args, "out_dir", None)
    want_stats = bool(getattr(args, "stats", False))
    want_json_only = bool(getattr(args, "json", False))

    if out is None and out_dir is None and not want_stats and not want_json_only:
        logger.error("Specify --out, --out-dir, --json, and/or --stats")
        sys.exit(1)

    requested_recovery = str(getattr(args, "recovery", "off") or "off")
    deterministic = bool(getattr(args, "deterministic", False))

    warnings: list[GapWarning] = []

    effective_recovery = requested_recovery
    if requested_recovery == "aggressive":
        effective_recovery = "safe"
        warnings.append(
            GapWarning(
                code="RECOVERY_DOWNGRADED_AGGRESSIVE_TO_SAFE",
                message=(
                    "recovery=aggressive is not available in v1; "
                    "downgraded to safe"
                ),
                meta={"requested": requested_recovery, "effective": effective_recovery},
            )
        )

    if deterministic and effective_recovery == "aggressive":
        # Defensive (should not happen after downgrade); keep behavior stable.
        effective_recovery = "safe"

    engine = GapEngine()
    capture_texts = (out_dir is not None) or (not want_json_only and out != "-")
    if want_stats or out_dir is not None or capture_texts:
        report, details, texts_a_by_path_ordinal, texts_b_by_path_ordinal = (
            engine.run_with_details_and_symbol_texts(
                a_root=a_root,
                b_root=b_root,
                indexing=config.indexing,
                recovery_mode=effective_recovery,
                deterministic=deterministic,
                warnings=warnings,
            )
        )
    else:
        report = engine.run(
            a_root=a_root,
            b_root=b_root,
            indexing=config.indexing,
            recovery_mode=effective_recovery,
            deterministic=deterministic,
            warnings=warnings,
        )
        details = None
        texts_a_by_path_ordinal = {}
        texts_b_by_path_ordinal = {}

    json_text = None
    if out is not None or want_json_only:
        report_dict = report.to_dict()
        json_text = json.dumps(report_dict, indent=2, sort_keys=True) + "\n"

        if out is not None:
            _write_json(json_text, str(out))

    out_dir_path = Path(out_dir).resolve() if out_dir is not None else None
    if out_dir_path is not None:
        out_dir_path.mkdir(parents=True, exist_ok=True)
        if json_text is None:
            report_dict = report.to_dict()
            json_text = json.dumps(report_dict, indent=2, sort_keys=True) + "\n"
        _write_text(json_text, out_dir_path / "gap.json")

        if details is not None:
            _write_text(
                render_gap_stats_report(report=report, details=details),
                out_dir_path / "stats.txt",
            )

    # Advisory move/update suggestions (do not mutate gap.json); only written under --out-dir.
    # We compute early so downstream renderers can use it, but we write it only after
    # theme artifact generation succeeds to preserve existing failure semantics.
    move_suggestions_payload: dict[str, Any] | None = None
    isotope_pairs: dict[int, IsotopePairing] | None = None
    if out_dir_path is not None:
        move_suggestions_payload = build_move_suggestions_payload(
            report=report,
            # TODO: add CLI flags to tune block inclusion and caps for block-heavy diffs.
            include_blocks=True,
        )
        isotope_pairs = _build_isotope_pairs_from_suggestions_payload(
            move_suggestions_payload
        )

    # Theme clustering (symbols only); always produce when it won't corrupt JSON stdout,
    # and always emit artifacts under --out-dir.
    want_themes = (out_dir_path is not None) or (not want_json_only and out != "-")
    if want_themes:
        docs, theme_items = build_theme_documents(
            report=report,
            texts_a_by_path_ordinal=texts_a_by_path_ordinal,
            texts_b_by_path_ordinal=texts_b_by_path_ordinal,
        )

        embedding_configured = config.embedding is not None and not config.embeddings_disabled

        provider = None
        if theme_items and embedding_configured:
            try:
                provider = EmbeddingProviderFactory.create_provider(config.embedding)
            except Exception as e:
                if out_dir_path is not None:
                    _cleanup_theme_artifacts(out_dir_path)
                logger.error(f"Theme embedding provider creation failed: {e}")
                sys.exit(1)

        if not theme_items:
            theme_output = build_fallback_theme_output(
                items=theme_items, label="no_symbol_changes"
            )
        elif provider is None:
            theme_output = build_fallback_theme_output(
                items=theme_items, label="embedding_provider_unavailable"
            )
        else:
            await provider.initialize()
            try:
                batch_size = max(100, int(getattr(config.embedding, "batch_size", 100)))
                embeddings = await embed_in_batches(
                    provider=provider,
                    texts=docs,
                    batch_size=batch_size,
                )
                labels = cluster_embeddings_hdbscan(
                    embeddings,
                    min_cluster_size=3,
                    min_samples=1,
                    allow_single_cluster=True,
                )
                theme_output = build_theme_output(
                    provider=provider,
                    items=theme_items,
                    labels=labels,
                    min_cluster_size=3,
                    min_samples=1,
                    allow_single_cluster=True,
                )
            except Exception as e:
                if out_dir_path is not None:
                    _cleanup_theme_artifacts(out_dir_path)
                logger.error(f"Theme embedding/clustering failed: {e}")
                sys.exit(1)
            finally:
                await provider.shutdown()

        if out_dir_path is not None:
            write_theme_artifacts(
                out_dir=out_dir_path,
                report=report,
                output=theme_output,
                isotope_pairs=isotope_pairs,
            )

        if out_dir_path is not None and move_suggestions_payload is not None:
            write_move_suggestions(out_dir=out_dir_path, payload=move_suggestions_payload)

        if out_dir_path is None and not want_json_only:
            if want_stats and details is not None:
                sys.stdout.write(render_gap_stats_report(report=report, details=details))
            sys.stdout.write(render_themes_markdown(output=theme_output, report=report))
            return

    if want_stats and not want_json_only:
        assert details is not None
        sys.stdout.write(render_gap_stats_report(report=report, details=details))

    # If user asked for JSON-only and no --out path, default to stdout
    if want_json_only and out is None and json_text is not None:
        _write_json(json_text, "-")
