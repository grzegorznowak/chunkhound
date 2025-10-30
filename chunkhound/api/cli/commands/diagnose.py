"""Diagnose command: compare CH ignore decisions with Git and report mismatches."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path
from typing import Any

from chunkhound.core.config.config import Config


def _nearest_repo_root(path: Path, stop: Path) -> Path | None:
    p = path.resolve()
    stop = stop.resolve()
    while True:
        if (p / ".git").exists():
            return p
        if p == stop or p.parent == p:
            return None
        p = p.parent


def _git_ignored(repo_root: Path, rel_path: str) -> bool:
    try:
        proc = subprocess.run(
            ["git", "check-ignore", "-q", "--no-index", rel_path],
            cwd=str(repo_root),
            text=True,
            capture_output=True,
        )
        return proc.returncode == 0
    except Exception:
        return False


def _ch_ignored(root: Path, file_path: Path, config: Config) -> bool:
    try:
        from chunkhound.utils.ignore_engine import build_repo_aware_ignore_engine

        sources = config.indexing.resolve_ignore_sources()
        cfg_ex = config.indexing.get_effective_config_excludes()
        chf = config.indexing.chignore_file
        eng = build_repo_aware_ignore_engine(root, sources, chf, cfg_ex)
        return bool(eng.matches(file_path, is_dir=False))
    except Exception:
        return False


async def diagnose_command(args: argparse.Namespace, config: Config) -> None:
    base_dir = Path(args.path).resolve()

    # Collect candidate files without pruning (we want to see mismatches where CH would exclude)
    candidates: list[Path] = []
    for p in base_dir.rglob("*"):
        if p.is_file():
            candidates.append(p)

    mismatches: list[dict[str, Any]] = []
    for fp in candidates:
        # Git decision from nearest repo root (if any)
        repo = _nearest_repo_root(fp.parent, base_dir) or base_dir
        rel = fp.resolve().relative_to(repo if repo else base_dir).as_posix()
        git_decision = _git_ignored(repo, rel) if repo else False

        # CH decision
        ch_decision = _ch_ignored(base_dir, fp, config)

        if git_decision != ch_decision:
            mismatches.append({"path": fp.resolve().relative_to(base_dir).as_posix(), "git": git_decision, "ch": ch_decision})

    report = {"mismatches": mismatches, "total": len(candidates), "base": base_dir.as_posix()}
    if getattr(args, "json", False):
        print(json.dumps(report, indent=2))
    else:
        print(f"Base: {report['base']}")
        print(f"Paths scanned: {report['total']}")
        print(f"Mismatches: {len(mismatches)}")
        for m in mismatches[:20]:
            print(f" - {m['path']}: CH={m['ch']} Git={m['git']}")

