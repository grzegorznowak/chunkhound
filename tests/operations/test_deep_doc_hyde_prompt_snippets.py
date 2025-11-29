import asyncio
from pathlib import Path

import pytest

import operations.deep_doc.deep_doc as deep_doc_mod
from operations.deep_doc.deep_doc import AgentDocMetadata, HydeConfig


@pytest.mark.asyncio
async def test_hyde_scope_prompt_includes_snippets_when_project_root_diff_cwd(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """HyDE scope prompt should include code snippets even when project_root != CWD.

    This guards against regressions where _build_hyde_scope_prompt reads files
    relative to the current working directory instead of the configured
    project_root path passed into deep-doc.
    """
    # Create a fake workspace outside the current CWD so relative paths differ.
    project_root = tmp_path / "workspace"
    scope_path = project_root / "arguseek"
    scope_path.mkdir(parents=True, exist_ok=True)

    target_file = scope_path / "foo.py"
    target_file.write_text(
        "def example() -> None:\n"
        "    x = 1\n"
        "    y = 2\n"
        "    return x + y\n",
        encoding="utf-8",
    )

    # Sanity check: ensure we're not accidentally running the test with CWD
    # equal to the fake project_root, otherwise the bug wouldn't reproduce.
    assert project_root.resolve() != Path.cwd().resolve()

    # Minimal metadata/config for HyDE.
    meta = AgentDocMetadata(
        created_from_sha="TEST_SHA",
        previous_target_sha="TEST_SHA",
        target_sha="TEST_SHA",
        generated_at="2025-01-01T00:00:00Z",
        llm_config={},
        generation_stats={},
    )
    hyde_cfg = HydeConfig.from_env()

    captured: dict[str, str] = {}

    async def fake_run_hyde_only_query(
        llm_manager,
        prompt: str,
        provider_override=None,
        hyde_cfg: HydeConfig | None = None,
    ) -> str:  # type: ignore[override]
        captured["prompt"] = prompt
        return "TEST_PLAN"

    monkeypatch.setattr(
        deep_doc_mod,
        "_run_hyde_only_query",
        fake_run_hyde_only_query,
        raising=True,
    )

    # Run the HyDE bootstrap path; this will build the scope prompt and call
    # fake_run_hyde_only_query with it.
    await deep_doc_mod._run_hyde_bootstrap(
        hyde_only=False,
        project_root=project_root,
        scope_path=scope_path,
        scope_label="arguseek",
        meta=meta,
        hyde_cfg=hyde_cfg,
        llm_manager=None,
        assembly_provider=None,
        out_dir=None,
    )

    prompt = captured.get("prompt", "")
    # The scope prompt should mention the file and include a code fence with
    # the file's contents, not the generic "no sample code snippets" marker.
    assert "File: arguseek/foo.py" in prompt
    assert "(no sample code snippets available)" not in prompt

