import textwrap

from operations.deep_doc.deep_doc import AgentDocMetadata, _build_research_prompt
from chunkhound.interfaces.llm_provider import LLMProvider, LLMResponse


def _make_meta(created: str = "AAA", previous: str = "BBB", target: str = "CCC") -> AgentDocMetadata:
    return AgentDocMetadata(
        created_from_sha=created,
        previous_target_sha=previous,
        target_sha=target,
        generated_at="2025-01-01T00:00:00Z",
        llm_config={"provider": "codex-cli", "synthesis_model": "gpt-5"},
    )


def test_build_research_prompt_includes_scope_and_hyde_for_root_scope() -> None:
    meta = _make_meta()
    prompt = _build_research_prompt(
        meta=meta,
        diff_since_created=["M chunkhound/core/config/config.py"],
        diff_since_previous=[],
        scope_label="/",
    )

    # Scope should be rendered as "/" and mentioned explicitly
    assert "in-scope folder: /" in prompt

    # HyDE instructions should be present
    assert "HyDE-style synthesis backbone for this run:" in prompt
    assert "imagine an ideal, well-structured documentation set" in prompt


def test_build_research_prompt_includes_scope_and_hyde_for_subfolder_scope() -> None:
    meta = _make_meta()
    scope_label = "chunkhound/core"
    prompt = _build_research_prompt(
        meta=meta,
        diff_since_created=["M chunkhound/core/config/config.py"],
        diff_since_previous=[],
        scope_label=scope_label,
    )

    # Scope should be rendered as a relative folder
    assert f"in-scope folder: ./chunkhound/core" in prompt

    # HyDE text should still be included
    assert "HyDE-style synthesis backbone for this run:" in prompt
