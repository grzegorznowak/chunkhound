#!/usr/bin/env python3
"""End-to-end tests for `chunkhound gap --out-dir` artifact emission."""

import json
import subprocess
import tempfile
from pathlib import Path

from tests.utils.windows_subprocess import get_safe_subprocess_env


def test_gap_out_dir_writes_artifacts_without_embeddings() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        a_dir = root / "a"
        b_dir = root / "b"
        out_dir = root / "out"
        a_dir.mkdir()
        b_dir.mkdir()

        (a_dir / "a.py").write_text("def foo():\n    return 1\n", encoding="utf-8")
        (b_dir / "a.py").write_text("def foo():\n    return 2\n", encoding="utf-8")

        res = subprocess.run(
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
        assert res.returncode == 0, res.stderr

        assert (out_dir / "gap.json").exists()
        assert (out_dir / "stats.txt").exists()
        assert (out_dir / "themes.json").exists()
        assert (out_dir / "themes.md").exists()
        assert (out_dir / "run.json").exists()

        payload = json.loads((out_dir / "gap.json").read_text(encoding="utf-8"))
        assert payload["schema_version"] == "gap.v1"

