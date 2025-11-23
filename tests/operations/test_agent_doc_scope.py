import textwrap

from operations.agent_doc import AgentDocMetadata, _build_research_prompt, _trim_doc_for_scope
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


class _FakeLLMProvider(LLMProvider):
    def __init__(self) -> None:
        self.last_prompt: str | None = None

    @property
    def name(self) -> str:  # pragma: no cover - trivial
        return "fake"

    @property
    def model(self) -> str:  # pragma: no cover - trivial
        return "fake-model"

    async def complete(
        self,
        prompt: str,
        system: str | None = None,
        max_completion_tokens: int = 4096,
    ) -> LLMResponse:
        self.last_prompt = prompt
        # Simple "trimming": keep only lines that mention the scope
        lines = []
        for line in prompt.splitlines():
            if "# Agent Doc" in line or "chunkhound/core" in line:
                lines.append(line)
        content = "\n".join(lines) or "# Agent Doc\n\n(trimmed)"
        return LLMResponse(content=content, tokens_used=1, model=self.model)

    async def batch_complete(  # pragma: no cover - unused in tests
        self,
        prompts: list[str],
        system: str | None = None,
        max_completion_tokens: int = 4096,
    ) -> list[LLMResponse]:
        return [
            await self.complete(prompt=p, system=system, max_completion_tokens=max_completion_tokens)
            for p in prompts
        ]

    def estimate_tokens(self, text: str) -> int:  # pragma: no cover - trivial
        return len(text.split())

    async def health_check(self) -> dict[str, object]:  # pragma: no cover - trivial
        return {"status": "ok"}

    def get_usage_stats(self) -> dict[str, object]:  # pragma: no cover - trivial
        return {}


class _FakeLLMManager:
    def __init__(self, provider: LLMProvider) -> None:
        self._provider = provider

    def is_configured(self) -> bool:
        return True

    def get_synthesis_provider(self) -> LLMProvider:
        return self._provider


async def test_trim_doc_for_scope_uses_scope_and_preserves_heading() -> None:
    body = "# Agent Doc\n\nThis mentions chunkhound/core and other parts."
    provider = _FakeLLMProvider()
    manager = _FakeLLMManager(provider)

    trimmed = await _trim_doc_for_scope(manager, body, scope_label="chunkhound/core")

    # Heading must be preserved
    assert "# Agent Doc" in trimmed
    # Our fake provider only keeps lines with 'chunkhound/core' or heading
    assert "chunkhound/core" in trimmed
