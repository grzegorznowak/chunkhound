import json
from pathlib import Path

from chunkhound.interfaces.llm_provider import LLMProvider, LLMResponse
from operations.agent_doc import (
    AgentDocMetadata,
    _get_debug_dir,
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
        self.calls: list[str] = []

    async def search_semantic(
        self,
        query: str,
        page_size: int,
        offset: int,
        threshold,
        path_filter,
        provider=None,
        model=None,
        force_strategy=None,
    ):
        self.calls.append(query)
        return (
            [
                {
                    "path": "tree-sitter-haskell/src/parser.c",
                    "start_line": 1,
                    "end_line": 10,
                    "score": 0.9,
                    "code_preview": "int main() { return 0; }",
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
        self.last_system: str | None = None

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
        self.last_system = system
        # Return a minimal, valid Agent Doc body
        content = "# Agent Doc\n\nSemantic overview body."
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


def test_get_debug_dir_uses_out_dir(tmp_path: Path) -> None:
    out_dir = tmp_path / "docs"
    debug_dir = _get_debug_dir(out_dir=out_dir, debug_dump=True)
    assert debug_dir == out_dir / "debug"

    # When debug is disabled or out_dir is None, no debug dir is used
    assert _get_debug_dir(out_dir=out_dir, debug_dump=False) is None
    assert _get_debug_dir(out_dir=None, debug_dump=True) is None


async def test_semantic_overview_writes_debug_artifacts(tmp_path: Path) -> None:
    services = _FakeServices()
    provider = _FakeLLMProvider()
    manager = _FakeLLMManager(provider)
    meta = _make_meta()

    debug_dir = tmp_path / "debug"
    result = await _run_semantic_overview_query(
        services=services,
        llm_manager=manager,
        meta=meta,
        scope_label="tree-sitter-haskell",
        hyde_plan=None,
        debug_dir=debug_dir,
    )

    # We still get an Agent Doc-style body back
    assert "# Agent Doc" in result

    # Core debug files should be written
    queries_path = debug_dir / "semantic_queries.json"
    raw_path = debug_dir / "semantic_results_raw.jsonl"
    merged_path = debug_dir / "semantic_results_merged.json"
    prompt_path = debug_dir / "semantic_prompt.txt"
    context_path = debug_dir / "semantic_context.md"
    raw_answer_path = debug_dir / "semantic_raw_answer.md"

    for path in (
        queries_path,
        raw_path,
        merged_path,
        prompt_path,
        context_path,
        raw_answer_path,
    ):
        assert path.is_file(), f"Expected debug artifact at {path}"

    # semantic_queries.json: list of query descriptors
    queries = json.loads(queries_path.read_text(encoding="utf-8"))
    assert isinstance(queries, list)
    assert any("tree-sitter-haskell" in q["text"] for q in queries)

    # semantic_results_raw.jsonl: per-query raw hits
    raw_lines = [line for line in raw_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    assert raw_lines, "Expected at least one raw semantic result"
    record = json.loads(raw_lines[0])
    assert record["path"] == "tree-sitter-haskell/src/parser.c"
    assert record["query_id"] == 0

    # semantic_results_merged.json: merged, deduplicated results
    merged = json.loads(merged_path.read_text(encoding="utf-8"))
    assert isinstance(merged, list)
    assert merged[0]["path"] == "tree-sitter-haskell/src/parser.c"

    # prompt/context/answer snapshots should contain key markers
    prompt_text = prompt_path.read_text(encoding="utf-8")
    assert "<<<CONTEXT>>>" in prompt_text
    context_text = context_path.read_text(encoding="utf-8")
    assert "tree-sitter-haskell/src/parser.c" in context_text
    raw_answer_text = raw_answer_path.read_text(encoding="utf-8")
    assert "# Agent Doc" in raw_answer_text
