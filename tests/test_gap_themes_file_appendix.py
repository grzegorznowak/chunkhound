#!/usr/bin/env python3
"""End-to-end tests for `themes.md` including file-level appendix."""

from __future__ import annotations

import json
import subprocess
import tempfile
from pathlib import Path

from tests.utils.windows_subprocess import get_safe_subprocess_env


def test_themes_markdown_includes_non_symbol_file_changes_appendix() -> None:
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
                    "indexing": {
                        "include": ["**/*.py"],
                        "max_file_size_mb": 1,
                    }
                }
            ),
            encoding="utf-8",
        )

        # Symbol-level text change
        (a_dir / "a.py").write_text("def foo():\n    return 1\n", encoding="utf-8")
        (b_dir / "a.py").write_text("def foo():\n    return 2\n", encoding="utf-8")

        # Non-symbol file changes: binary + too_large
        (a_dir / "binary.py").write_bytes(b"\x00\x01hello")
        (b_dir / "binary.py").write_bytes(b"\x00\x02hello")

        a_big = b"a" * (1024 * 1024 + 1)
        b_big = b"b" * (1024 * 1024 + 1)
        (a_dir / "huge.py").write_bytes(a_big)
        (b_dir / "huge.py").write_bytes(b_big)

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
            env=get_safe_subprocess_env(),
        )
        assert res.returncode == 0, res.stderr

        md = (out_dir / "themes.md").read_text(encoding="utf-8")
        assert "Non-symbol file changes" in md
        assert "`binary.py`" in md
        assert "file_class=binary" in md
        assert "`huge.py`" in md
        assert "file_class=too_large" in md


def test_themes_markdown_includes_appendix_when_no_symbol_changes() -> None:
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
                    "indexing": {
                        "include": ["**/*.py"],
                        "max_file_size_mb": 1,
                    }
                }
            ),
            encoding="utf-8",
        )

        # File-only changes: binary + too_large (no text/symbol changes).
        (a_dir / "binary.py").write_bytes(b"\x00\x01hello")
        (b_dir / "binary.py").write_bytes(b"\x00\x02hello")

        a_big = b"a" * (1024 * 1024 + 1)
        b_big = b"b" * (1024 * 1024 + 1)
        (a_dir / "huge.py").write_bytes(a_big)
        (b_dir / "huge.py").write_bytes(b_big)

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
            env=get_safe_subprocess_env(),
        )
        assert res.returncode == 0, res.stderr

        md = (out_dir / "themes.md").read_text(encoding="utf-8")
        assert "Non-symbol file changes" in md
        assert "`binary.py`" in md
        assert "file_class=binary" in md
        assert "`huge.py`" in md
        assert "file_class=too_large" in md
