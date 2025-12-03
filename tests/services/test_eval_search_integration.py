from __future__ import annotations

import json
from pathlib import Path

import pytest

from chunkhound.tools.eval.search import _async_main


@pytest.mark.asyncio
async def test_eval_search_regex_smoke(tmp_path: Path) -> None:
    """End-to-end smoke test for eval.search in regex mode.

    Runs the harness against a tiny synthetic corpus (single language)
    and asserts that a JSON payload is written with basic structure.
    This stays in regex mode so no embedding provider is required.
    """

    output_path = tmp_path / "eval_search_regex.json"
    argv = [
        "--mode",
        "mixed",
        "--search-mode",
        "regex",
        "--languages",
        "python",
        "--k",
        "1",
        "--output",
        str(output_path),
    ]

    exit_code = await _async_main(argv)
    assert exit_code == 0
    assert output_path.is_file()

    payload = json.loads(output_path.read_text(encoding="utf-8"))
    assert payload["mode"] == "mixed"
    assert payload["search_mode"] == "regex"
    assert payload["languages"] == ["python"]
    assert payload["ks"] == [1]
    assert payload["per_query"], "expected at least one query in eval output"

