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

    # Gap reuses indexing include/exclude + ignore settings, but does not require DB/embeddings.
    add_config_arguments(gap_parser, ["indexing"])

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

    return cast(argparse.ArgumentParser, gap_parser)


__all__: list[str] = ["add_gap_subparser"]
