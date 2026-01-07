"""Gap command module - stateless semantic gap analysis between two folders."""

import argparse
import json
import sys
from pathlib import Path

from loguru import logger

from chunkhound.core.config.config import Config
from chunkhound.gap.engine import GapEngine
from chunkhound.gap.models import GapWarning
from chunkhound.gap.stats_report import render_gap_stats_report


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
    want_stats = bool(getattr(args, "stats", False))
    want_json_only = bool(getattr(args, "json", False))

    if out is None and not want_stats:
        logger.error("Specify either --out (to write JSON) and/or --stats")
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
    details = None
    if want_stats and not want_json_only:
        report, details = engine.run_with_details(
            a_root=a_root,
            b_root=b_root,
            indexing=config.indexing,
            recovery_mode=effective_recovery,
            deterministic=deterministic,
            warnings=warnings,
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

    json_text = None
    if out is not None or want_json_only:
        report_dict = report.to_dict()
        json_text = json.dumps(report_dict, indent=2, sort_keys=True) + "\n"

        if out is not None:
            _write_json(json_text, str(out))

    if want_stats and not want_json_only:
        assert details is not None
        sys.stdout.write(render_gap_stats_report(report=report, details=details))

    # If user asked for JSON-only and no --out path, default to stdout
    if want_json_only and out is None and json_text is not None:
        _write_json(json_text, "-")
