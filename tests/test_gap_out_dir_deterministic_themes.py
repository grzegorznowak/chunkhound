#!/usr/bin/env python3
"""Determinism tests for themed `chunkhound gap --out-dir` artifacts."""

from __future__ import annotations

import subprocess
import tempfile
from pathlib import Path

from tests.utils.windows_subprocess import get_safe_subprocess_env


def _run_gap(*, a_dir: Path, b_dir: Path, out_dir: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [
            "uv",
            "run",
            "chunkhound",
            "gap",
            str(a_dir),
            str(b_dir),
            "--out-dir",
            str(out_dir),
            "--deterministic",
        ],
        capture_output=True,
        text=True,
        timeout=30,
        env=get_safe_subprocess_env(),
    )


def test_gap_out_dir_themes_artifacts_are_byte_identical_without_embeddings() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        a_dir = root / "a"
        b_dir = root / "b"
        out1 = root / "out1"
        out2 = root / "out2"
        a_dir.mkdir()
        b_dir.mkdir()

        (a_dir / "a.py").write_text("def foo():\n    return 1\n", encoding="utf-8")
        (b_dir / "a.py").write_text("def foo():\n    return 2\n", encoding="utf-8")

        res1 = _run_gap(a_dir=a_dir, b_dir=b_dir, out_dir=out1)
        assert res1.returncode == 0, res1.stderr
        res2 = _run_gap(a_dir=a_dir, b_dir=b_dir, out_dir=out2)
        assert res2.returncode == 0, res2.stderr

        expected = {
            "gap.json",
            "stats.txt",
            "themes.json",
            "themes.md",
            "run.json",
            "move_suggestions.json",
        }
        assert {p.name for p in out1.iterdir()} == expected
        assert {p.name for p in out2.iterdir()} == expected

        for name in (
            "gap.json",
            "stats.txt",
            "themes.json",
            "themes.md",
            "run.json",
            "move_suggestions.json",
        ):
            b1 = (out1 / name).read_bytes()
            b2 = (out2 / name).read_bytes()
            assert b1 == b2, f"{name} differs between deterministic runs"
