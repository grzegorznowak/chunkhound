#!/usr/bin/env python3
"""End-to-end tests for the `chunkhound gap` CLI command (contract-first)."""

import json
import subprocess
import tempfile
from pathlib import Path

from tests.utils.windows_subprocess import get_safe_subprocess_env


def _run_gap(a_dir: Path, b_dir: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [
            "uv",
            "run",
            "chunkhound",
            "gap",
            str(a_dir),
            str(b_dir),
            "--out",
            "-",
            "--json",
            "--deterministic",
        ],
        capture_output=True,
        text=True,
        timeout=20,
        env=get_safe_subprocess_env(),
    )


def test_gap_emits_valid_gap_v1_json_contract() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        a_dir = root / "a"
        b_dir = root / "b"
        a_dir.mkdir()
        b_dir.mkdir()

        # Minimal content (must still be valid JSON)
        (a_dir / "a.py").write_text("def a():\n    return 1\n")
        (b_dir / "b.py").write_text("def b():\n    return 2\n")

        res = _run_gap(a_dir, b_dir)
        assert res.returncode == 0, res.stderr

        payload = json.loads(res.stdout)
        assert payload["schema_version"] == "gap.v1"
        assert payload["direction"] == "A->B"

        inv = payload["invariants"]
        assert inv["hash_alg"] == "xxh3_64"
        assert inv["chunker_version"] == "cast@v1"
        assert inv["deterministic"] is True
        assert inv["recovery_mode"] in {"off", "safe", "aggressive"}

        norm = inv["normalization"]
        assert norm["id"] == "normalize_content.v1"
        assert norm["include_comments"] is False
        assert norm["include_docs"] is False

        assert "inputs" in payload
        assert payload["inputs"]["a"]["source_kind"] == "path"
        assert payload["inputs"]["b"]["source_kind"] == "path"

        assert "scope" in payload
        assert "warnings" in payload and isinstance(payload["warnings"], list)
        assert "stats" in payload
        assert "changes" in payload and isinstance(payload["changes"], list)

        # Deterministic contract: timings must be present and zeroed
        timings = payload["stats"]["timings"]
        assert all(float(v) == 0.0 for v in timings.values())

        # Gap excludes synthetic file-structure placeholders to avoid collisions
        assert all(
            (c.get("new") or c.get("old") or {}).get("symbol") != "file_structure"
            for c in payload["changes"]
            if c.get("entity_kind") == "symbol"
        )


def test_gap_deterministic_runs_are_byte_identical() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        a_dir = root / "a"
        b_dir = root / "b"
        a_dir.mkdir()
        b_dir.mkdir()
        (a_dir / "x.txt").write_text("hello\n")
        (b_dir / "x.txt").write_text("hello\n")

        res1 = _run_gap(a_dir, b_dir)
        assert res1.returncode == 0, res1.stderr
        res2 = _run_gap(a_dir, b_dir)
        assert res2.returncode == 0, res2.stderr

        assert res1.stdout == res2.stdout


def test_gap_emits_file_level_changes_for_non_text_classes() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        a_dir = root / "a"
        b_dir = root / "b"
        a_dir.mkdir()
        b_dir.mkdir()

        # Force a small max file size for manageable too_large fixtures.
        cfg_path = root / "cfg.json"
        cfg_path.write_text(
            json.dumps(
                {
                    "indexing": {
                        "include": ["**/*.py"],
                        "max_file_size_mb": 1,
                    }
                }
            )
        )

        # Binary (NUL present)
        (a_dir / "binary.py").write_bytes(b"\x00\x01hello")
        (b_dir / "binary.py").write_bytes(b"\x00\x02hello")

        # Decode error (invalid UTF-8, no NUL)
        (a_dir / "bad.py").write_bytes(b"\xff\xfe\xfd")
        (b_dir / "bad.py").write_bytes(b"\xff\xfe\xfc")

        # Too large (over 1MB), valid UTF-8
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
                "--out",
                "-",
                "--json",
                "--deterministic",
            ],
            capture_output=True,
            text=True,
            timeout=30,
            env=get_safe_subprocess_env(),
        )
        assert res.returncode == 0, res.stderr

        payload = json.loads(res.stdout)
        changes = payload["changes"]
        assert len(changes) == 3

        by_path = {c["old"]["path"]: c for c in changes}
        assert by_path["binary.py"]["old"]["file_class"] == "binary"
        assert by_path["bad.py"]["old"]["file_class"] == "decode_error"
        assert by_path["huge.py"]["old"]["file_class"] == "too_large"

        for item in changes:
            assert item["entity_kind"] == "file"
            assert item["op"] == "update"
            assert item["moved"] is False
            assert item["renamed"] is False
            assert item["content_changed"] is True
            assert item["primary_key_kind"] == "path_key"
            assert item["had_collision"] is False
            assert item["collision_group_size"] == 1
            assert item["old"]["path"] == item["new"]["path"]


def test_gap_detects_symbol_move_across_files() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        a_dir = root / "a"
        b_dir = root / "b"
        a_dir.mkdir()
        b_dir.mkdir()

        (a_dir / "a.py").write_text("def foo():\n    return 1\n", encoding="utf-8")
        (b_dir / "b.py").write_text("def foo():\n    return 1\n", encoding="utf-8")

        res = _run_gap(a_dir, b_dir)
        assert res.returncode == 0, res.stderr

        payload = json.loads(res.stdout)
        moves = [
            c
            for c in payload["changes"]
            if c.get("entity_kind") == "symbol"
            and c.get("op") == "update"
            and c.get("moved") is True
            and (c.get("old") or {}).get("path") == "a.py"
            and (c.get("new") or {}).get("path") == "b.py"
            and str((c.get("old") or {}).get("symbol", "")).startswith("foo")
        ]
        assert len(moves) == 1
        assert moves[0]["content_changed"] is False
        assert moves[0]["renamed"] is False


def test_gap_detects_symbol_content_change_in_place() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        a_dir = root / "a"
        b_dir = root / "b"
        a_dir.mkdir()
        b_dir.mkdir()

        (a_dir / "a.py").write_text("def foo():\n    return 1\n", encoding="utf-8")
        (b_dir / "a.py").write_text("def foo():\n    return 2\n", encoding="utf-8")

        res = _run_gap(a_dir, b_dir)
        assert res.returncode == 0, res.stderr

        payload = json.loads(res.stdout)
        updates = [
            c
            for c in payload["changes"]
            if c.get("entity_kind") == "symbol"
            and c.get("op") == "update"
            and c.get("moved") is False
            and (c.get("old") or {}).get("path") == "a.py"
            and (c.get("new") or {}).get("path") == "a.py"
            and str((c.get("old") or {}).get("symbol", "")).startswith("foo")
        ]
        assert updates
        assert any(u["content_changed"] is True for u in updates)


def test_gap_stats_prints_rich_human_report() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        a_dir = root / "a"
        b_dir = root / "b"
        a_dir.mkdir()
        b_dir.mkdir()

        # Unchanged
        (a_dir / "same.py").write_text("def same():\n    return 1\n", encoding="utf-8")
        (b_dir / "same.py").write_text("def same():\n    return 1\n", encoding="utf-8")

        # Modified
        (a_dir / "mod.py").write_text("def mod():\n    return 1\n", encoding="utf-8")
        (b_dir / "mod.py").write_text("def mod():\n    return 2\n", encoding="utf-8")

        # Removed / Added
        (a_dir / "removed.py").write_text(
            "def removed():\n    return 1\n", encoding="utf-8"
        )
        (b_dir / "added.py").write_text(
            "def added():\n    return 1\n", encoding="utf-8"
        )

        res = subprocess.run(
            [
                "uv",
                "run",
                "chunkhound",
                "gap",
                str(a_dir),
                str(b_dir),
                "--stats",
                "--deterministic",
            ],
            capture_output=True,
            text=True,
            timeout=20,
            env=get_safe_subprocess_env(),
        )
        assert res.returncode == 0, res.stderr

        out = res.stdout
        assert "gap.v1 A->B" in out
        assert "scope: full" in out
        assert "files: a_total=3 b_total=3 added=1 removed=1 modified=1 unchanged=1" in out
        assert "symbols(parsed_changed_files):" in out
        assert "symbol_changes:" in out
        assert "identity_collisions(parsed_changed_files):" in out
        assert "recovery: mode=" in out
        assert "warnings: total=" in out
