#!/usr/bin/env python3
"""End-to-end tests for embedding-configured theme failure semantics."""

from __future__ import annotations

import json
import subprocess
import tempfile
from pathlib import Path

from tests.utils.windows_subprocess import get_safe_subprocess_env


def test_gap_out_dir_with_embedding_config_failure_leaves_only_gap_and_stats() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        a_dir = root / "a"
        b_dir = root / "b"
        out_dir = root / "out"
        a_dir.mkdir()
        b_dir.mkdir()

        cfg_path = root / "cfg.json"
        cfg_path.write_text(
            json.dumps(
                {
                    "indexing": {"include": ["**/*.py"]},
                    "embedding": {"provider": "openai", "model": "text-embedding-3-small"},
                }
            ),
            encoding="utf-8",
        )

        (a_dir / "a.py").write_text("def foo():\n    return 1\n", encoding="utf-8")
        (b_dir / "a.py").write_text("def foo():\n    return 2\n", encoding="utf-8")

        env = get_safe_subprocess_env()
        env.pop("CHUNKHOUND_EMBEDDING__API_KEY", None)
        env.pop("CHUNKHOUND_EMBEDDING_API_KEY", None)

        res = subprocess.run(
            [
                "uv",
                "run",
                "chunkhound",
                "gap",
                str(a_dir),
                str(b_dir),
                "--config",
                str(cfg_path),
                "--out-dir",
                str(out_dir),
                "--deterministic",
            ],
            capture_output=True,
            text=True,
            timeout=30,
            env=env,
        )

        assert res.returncode != 0

        assert (out_dir / "gap.json").exists()
        assert (out_dir / "stats.txt").exists()
        assert not (out_dir / "themes.json").exists()
        assert not (out_dir / "themes.md").exists()
        assert not (out_dir / "run.json").exists()
