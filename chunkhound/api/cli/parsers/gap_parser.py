"""Gap command argument parser for ChunkHound CLI."""

import argparse
from pathlib import Path
from typing import Any, cast

from .common_arguments import add_common_arguments, add_config_arguments


def add_gap_subparser(subparsers: Any) -> argparse.ArgumentParser:
    """Add gap command subparser to the main parser."""
    gap_parser = subparsers.add_parser(
        "gap",
        help="Compare two folders and emit a semantic changeset (gap.v1)",
        description=(
            "Stateless, git-agnostic semantic diff engine. Compares two directory "
            "trees (A and B) and emits a deterministic `gap.v1` JSON contract."
        ),
    )

    gap_parser.add_argument("a", type=Path, help="Path to folder A (baseline)")
    gap_parser.add_argument("b", type=Path, help="Path to folder B (target)")

    add_common_arguments(gap_parser)

    # Gap reuses indexing include/exclude + ignore settings, and supports disabling
    # embeddings via --no-embeddings to keep deterministic/no-network runs stable.
    add_config_arguments(gap_parser, ["indexing", "embedding"])

    gap_parser.add_argument(
        "--out",
        type=str,
        default=None,
        help="Write JSON to this path, or '-' for stdout",
    )
    gap_parser.add_argument(
        "--out-dir",
        type=str,
        default=None,
        help="Write gap artifacts (gap.json, themes.*, stats.txt) into this folder",
    )
    gap_parser.add_argument(
        "--json",
        action="store_true",
        help="Emit JSON only (no human text)",
    )
    gap_parser.add_argument(
        "--stats",
        action="store_true",
        help="Print a stats report (human-readable by default)",
    )
    gap_parser.add_argument(
        "--deterministic",
        action="store_true",
        help="Force deterministic behavior (byte-stable JSON; clamps nondeterministic stages)",
    )
    gap_parser.add_argument(
        "--recovery",
        choices=["off", "safe", "aggressive"],
        default="off",
        help="Recovery mode for pairing leftover symbols (default: off)",
    )

    gap_parser.add_argument(
        "--no-move-suggestions",
        action="store_true",
        help="Disable advisory move_suggestions.json and isotope annotations under --out-dir",
    )
    gap_parser.add_argument(
        "--no-move-suggestions-llm",
        action="store_true",
        help="Disable the LLM tiebreak stage for move_suggestions (heuristics/embeddings only)",
    )
    gap_parser.add_argument(
        "--move-suggestions-llm-dry-run",
        action="store_true",
        help=(
            "Debug: write each LLM tiebreak prompt to llm_call_N.md under --out-dir "
            "without invoking any LLM provider"
        ),
    )
    gap_parser.add_argument(
        "--move-suggestions-llm-min-score",
        type=float,
        default=0.90,
        help=(
            "Minimum embedding similarity required to attempt LLM tiebreak for a remove "
            "(default: 0.90)"
        ),
    )
    gap_parser.add_argument(
        "--move-suggestions-llm-top-k",
        type=int,
        default=8,
        help=(
            "Maximum number of embedding-ranked candidates to include per LLM tiebreak prompt "
            "(default: 8; may be reduced to fit token budget)"
        ),
    )
    gap_parser.add_argument(
        "--move-suggestions-llm-max-prompt-tokens",
        type=int,
        default=8000,
        help=(
            "Maximum estimated prompt tokens for LLM tiebreak (default: 8000; candidates trimmed "
            "to fit)"
        ),
    )
    gap_parser.add_argument(
        "--move-suggestions-embed-min-score",
        type=float,
        default=0.82,
        help="Embedding auto-accept minimum similarity for non-block symbols (default: 0.82)",
    )
    gap_parser.add_argument(
        "--move-suggestions-embed-min-margin",
        type=float,
        default=0.06,
        help="Embedding auto-accept minimum best-vs-runner-up margin for non-block symbols (default: 0.06)",
    )
    gap_parser.add_argument(
        "--move-suggestions-embed-min-score-block",
        type=float,
        default=0.88,
        help="Embedding auto-accept minimum similarity for block symbols (default: 0.88)",
    )
    gap_parser.add_argument(
        "--move-suggestions-embed-min-margin-block",
        type=float,
        default=0.10,
        help="Embedding auto-accept minimum best-vs-runner-up margin for block symbols (default: 0.10)",
    )

    return cast(argparse.ArgumentParser, gap_parser)


__all__: list[str] = ["add_gap_subparser"]
