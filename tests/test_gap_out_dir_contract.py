#!/usr/bin/env python3
"""Contract tests for `chunkhound gap --out-dir` theme artifacts (v1)."""

from __future__ import annotations

import json
import subprocess
import tempfile
from pathlib import Path

from tests.utils.windows_subprocess import get_safe_subprocess_env


def test_gap_out_dir_emits_themes_json_contract() -> None:
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
                "--no-embeddings",
            ],
            capture_output=True,
            text=True,
            timeout=30,
            env=get_safe_subprocess_env(),
        )
        assert res.returncode == 0, res.stderr

        themes_payload = json.loads(
            (out_dir / "themes.json").read_text(encoding="utf-8")
        )
        for k in (
            "provider",
            "model",
            "dims",
            "min_cluster_size",
            "min_samples",
            "allow_single_cluster",
            "file_change_indexes",
            "themes",
        ):
            assert k in themes_payload

        assert isinstance(themes_payload["file_change_indexes"], list)
        assert themes_payload["file_change_indexes"] == sorted(
            themes_payload["file_change_indexes"]
        )
        assert all(isinstance(i, int) for i in themes_payload["file_change_indexes"])

        assert isinstance(themes_payload["themes"], list)
        assert themes_payload["themes"]
        for theme in themes_payload["themes"]:
            assert isinstance(theme, dict)
            assert isinstance(theme.get("theme_id"), int)
            assert isinstance(theme.get("label"), str)
            items = theme.get("items")
            assert isinstance(items, list)
            for item in items:
                assert isinstance(item, dict)
                assert isinstance(item.get("change_index"), int)

        run_payload = json.loads((out_dir / "run.json").read_text(encoding="utf-8"))
        for k in (
            "schema_version",
            "direction",
            "scope_hash",
            "embedding_provider",
            "embedding_model",
            "embedding_dims",
            "hdbscan",
            "warnings_total",
        ):
            assert k in run_payload

        suggestions_payload = json.loads(
            (out_dir / "move_suggestions.json").read_text(encoding="utf-8")
        )
        assert suggestions_payload["schema_version"] == "gap.suggestions.v1"
        assert suggestions_payload["source_schema_version"] == "gap.v1"
        assert suggestions_payload["source_scope_hash"] == run_payload["scope_hash"]
        assert isinstance(suggestions_payload.get("suggestions"), list)
