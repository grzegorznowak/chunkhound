"""Diagnose command argument parser for ChunkHound CLI."""

import argparse
from pathlib import Path
from typing import Any, cast


def add_diagnose_subparser(subparsers: Any) -> argparse.ArgumentParser:
    parser = subparsers.add_parser(
        "diagnose",
        help="Diagnose ignore rules vs Git and report issues",
        description=(
            "Compare ChunkHound ignore decisions with Git, report mismatches "
            "and suggest fixes."
        ),
    )

    parser.add_argument(
        "path",
        nargs="?",
        type=Path,
        default=Path("."),
        help="Directory to diagnose (default: current directory)",
    )

    parser.add_argument(
        "--json",
        action="store_true",
        help="Output JSON report",
    )

    return cast(argparse.ArgumentParser, parser)


__all__: list[str] = ["add_diagnose_subparser"]

