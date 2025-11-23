import json
from pathlib import Path

from chunkhound.interfaces.llm_provider import LLMProvider, LLMResponse
from operations.agent_doc import (
    AgentDocMetadata,
    _run_semantic_overview_query,
)


def _make_meta() -> AgentDocMetadata:
    return AgentDocMetadata(
        created_from_sha="AAA",
        previous_target_sha="BBB",
        target_sha="CCC",
        generated_at="2025-01-01T00:00:00Z",
        llm_config={"provider": "codex-cli", "synthesis_model": "gpt-5"},
    )


class _FakeSearchService:
    def __init__(self) -> None:
        self.semantic_calls: list[str] = []
        self.regex_calls: list[str] = []

    async def search_semantic(
        self,
        query: str,
        page_size: int,
        offset: int,
        threshold,
        path_filter,
    ):
        # Always return the same semantic hit from a single file
        self.semantic_calls.append(query)
        return (
            [
                {
                    "path": "scope/src/main.py",
                    "start_line": 1,
                    "end_line": 5,
                    "score": 0.9,
                    "code_preview": "print('semantic')",
                }
            ],
            1,
        )

    async def search_regex_async(
        self,
        pattern: str,
        page_size: int,
        offset: int,
        path_filter,
    ):
        # Gap fill should call this with a pattern based on 'main', but we
        # do not assert the exact value here.
        self.regex_calls.append(pattern)
        return (
            [
                {
                    "path": "scope/tests/test_main.py",
                    "start_line": 1,
                    "end_line": 5,
                    "code_preview": "def test_semantic(): ...",
                }
            ],
            1,
        )


class _FakeServices:
    def __init__(self) -> None:
        self.search_service = _FakeSearchService()


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
        content = "# Agent Doc\n\nGap-fill overview body."
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


async def test_semantic_gap_fill_adds_new_file_results(tmp_path: Path) -> None:
    services = _FakeServices()
    provider = _FakeLLMProvider()
    manager = _FakeLLMManager(provider)
    meta = _make_meta()

    debug_dir = tmp_path / "debug"
    result = await _run_semantic_overview_query(
        services=services,
        llm_manager=manager,
        meta=meta,
        scope_label="scope",
        hyde_plan=None,
        debug_dir=debug_dir,
        lexical_sidecar=False,
        gap_fill=True,
    )

    # We still get a valid Agent Doc body back
    assert "# Agent Doc" in result

    # Gap fill should have invoked regex at least once
    assert services.search_service.regex_calls

    # Merged debug results should include both the semantic file and the
    # new test file from regex-based gap filling.
    merged_path = debug_dir / "semantic_results_merged.json"
    merged = json.loads(merged_path.read_text(encoding="utf-8"))
    paths = {record["path"] for record in merged}
    assert "scope/src/main.py" in paths
    assert "scope/tests/test_main.py" in paths

