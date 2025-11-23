import json
from pathlib import Path

from operations.agent_doc import _categorize_file_path, _compute_semantic_coverage_summary


def test_categorize_file_path_basic_buckets() -> None:
    assert _categorize_file_path("project/src/main.py") == "code"
    assert _categorize_file_path("project/tests/test_main.py") == "tests"
    assert _categorize_file_path("project/README.md") == "docs"
    assert _categorize_file_path("project/.github/workflows/ci.yml") == "config"
    assert _categorize_file_path("project/assets/logo.png") == "other"


def test_compute_semantic_coverage_summary_counts_files_and_categories(tmp_path: Path) -> None:
    project_root = tmp_path / "proj"
    scope_path = project_root / "pkg"
    scope_path.mkdir(parents=True)

    # Create a small, synthetic scoped project
    (scope_path / "src").mkdir()
    (scope_path / "tests").mkdir()

    code_file = scope_path / "src" / "main.py"
    test_file = scope_path / "tests" / "test_main.py"
    docs_file = scope_path / "README.md"

    for p in (code_file, test_file, docs_file):
        p.write_text("# stub", encoding="utf-8")

    # Prepare a debug directory with a semantic_results_merged.json that
    # references each of the scoped files at least once.
    debug_dir = tmp_path / "debug"
    debug_dir.mkdir()

    merged = [
        {
            "path": "pkg/src/main.py",
            "start_line": 1,
            "end_line": 10,
            "score": 0.9,
            "code_preview": "print('hello')",
        },
        {
            "path": "pkg/tests/test_main.py",
            "start_line": 1,
            "end_line": 5,
            "score": 0.8,
            "code_preview": "def test_something(): ...",
        },
        {
            "path": "pkg/README.md",
            "start_line": 1,
            "end_line": 5,
            "score": 0.7,
            "code_preview": "Project overview",
        },
        # Include an out-of-scope path to ensure it is ignored for file coverage
        {
            "path": "other_project/src/app.py",
            "start_line": 1,
            "end_line": 5,
            "score": 0.6,
            "code_preview": "print('other')",
        },
    ]

    (debug_dir / "semantic_results_merged.json").write_text(
        json.dumps(merged, indent=2),
        encoding="utf-8",
    )

    summary = _compute_semantic_coverage_summary(
        project_root=project_root,
        scope_path=scope_path,
        scope_label="pkg",
        debug_dir=debug_dir,
    )

    # We created exactly three scoped files, all of which are referenced.
    assert summary["total_files"] == 3
    assert summary["files_with_semantic_hits"] == 3
    assert summary["coverage_ratio"] == 1.0
    assert summary["total_snippets"] == len(merged)

    # Category summary should reflect at least one snippet and one file per bucket
    cats = summary["category_summary"]
    assert cats["code"]["snippets"] == 2  # main.py + other_project/app.py
    assert cats["code"]["files"] == 1  # only main.py is in the scoped folder
    assert cats["tests"]["snippets"] == 1
    assert cats["tests"]["files"] == 1
    assert cats["docs"]["snippets"] == 1
    assert cats["docs"]["files"] == 1

