#!/usr/bin/env python3
"""Local-only E2E characterization for LLM move suggestions.

This test is intentionally skipped in CI and requires a repo-root
`.chunkhound.json` with working embedding + LLM configuration.
"""

from __future__ import annotations

import json
import os
import subprocess
import tempfile
from pathlib import Path

import pytest

from tests.utils.windows_subprocess import get_safe_subprocess_env


def _repo_root_config() -> Path:
    return Path(".chunkhound.json")


def _fixture_root() -> Path:
    return Path(__file__).parent / "fixtures" / "gap_llm_tiebreak"


def _load_root_config_or_skip() -> dict[str, object]:
    cfg_path = _repo_root_config()
    if not cfg_path.exists():
        pytest.skip("Missing repo-root .chunkhound.json (required for local live LLM test)")
    try:
        return json.loads(cfg_path.read_text(encoding="utf-8"))
    except Exception as e:
        pytest.skip(f"Unable to load repo-root .chunkhound.json: {e}")


def _run_gap(*, a_dir: Path, b_dir: Path, out_dir: Path, extra_args: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [
            "uv",
            "run",
            "chunkhound",
            "gap",
            str(a_dir),
            str(b_dir),
            "--config",
            str(_repo_root_config()),
            "--out-dir",
            str(out_dir),
            "--deterministic",
            "--recovery",
            "off",
            *extra_args,
        ],
        capture_output=True,
        text=True,
        timeout=120,
        env=get_safe_subprocess_env(),
    )


@pytest.mark.skipif(os.environ.get("CI") == "true", reason="Local-only live LLM test")
def test_gap_move_suggestions_llm_tiebreak_isotope_present_when_enabled() -> None:
    cfg = _load_root_config_or_skip()
    if not isinstance(cfg.get("embedding"), dict):
        pytest.skip("No embedding config in repo-root .chunkhound.json")
    if not isinstance(cfg.get("llm"), dict):
        pytest.skip("No llm config in repo-root .chunkhound.json")

    fixture = _fixture_root()
    a_dir = fixture / "a"
    b_dir = fixture / "b"

    with tempfile.TemporaryDirectory() as tmp:
        out1 = Path(tmp) / "baseline"
        out2 = Path(tmp) / "llm"

        # 1) Baseline: suggestions disabled => no llm isotopes.
        res1 = _run_gap(
            a_dir=a_dir,
            b_dir=b_dir,
            out_dir=out1,
            extra_args=["--no-move-suggestions"],
        )
        assert res1.returncode == 0, res1.stderr + res1.stdout
        themes1 = (out1 / "themes.md").read_text(encoding="utf-8")
        assert "[ISO:llm_tiebreak]" not in themes1

        # 2) Suggestions enabled (default), force embedding auto-accept to fail so LLM must decide.
        res2 = _run_gap(
            a_dir=a_dir,
            b_dir=b_dir,
            out_dir=out2,
            extra_args=[
                "--move-suggestions-embed-min-score",
                "1.1",
                "--move-suggestions-embed-min-margin",
                "1.1",
                "--move-suggestions-embed-min-score-block",
                "1.1",
                "--move-suggestions-embed-min-margin-block",
                "1.1",
            ],
        )
        assert res2.returncode == 0, res2.stderr + res2.stdout

        suggestions = json.loads((out2 / "move_suggestions.json").read_text(encoding="utf-8"))
        assert isinstance(suggestions.get("suggestions"), list)
        llm_pairs = [s for s in suggestions["suggestions"] if s.get("method") == "llm_tiebreak"]
        assert llm_pairs, "Expected at least one llm_tiebreak suggestion"

        themes2 = (out2 / "themes.md").read_text(encoding="utf-8")
        assert "[ISO:llm_tiebreak]" in themes2

        gap = json.loads((out2 / "gap.json").read_text(encoding="utf-8"))
        changes = gap.get("changes")
        assert isinstance(changes, list)
        max_idx = len(changes) - 1
        for s in llm_pairs:
            rem = s.get("remove_change_index")
            add = s.get("add_change_index")
            assert isinstance(rem, int) and 0 <= rem <= max_idx
            assert isinstance(add, int) and 0 <= add <= max_idx

