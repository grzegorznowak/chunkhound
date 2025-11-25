import pytest

import operations.agent_doc as agent_doc_mod
from operations.agent_doc import _run_research_query


@pytest.mark.asyncio
async def test_run_research_query_passes_scope_as_path(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, object] = {}

    async def _fake_deep_research_impl(
        *,
        services,
        embedding_manager,
        llm_manager,
        query,
        progress=None,
        path=None,
    ):
        captured["services"] = services
        captured["embedding_manager"] = embedding_manager
        captured["llm_manager"] = llm_manager
        captured["query"] = query
        captured["path"] = path
        return {"answer": "OK"}

    monkeypatch.setattr(agent_doc_mod, "deep_research_impl", _fake_deep_research_impl, raising=True)

    services = object()
    embedding_manager = object()
    llm_manager = object()
    prompt = "TEST PROMPT"

    # Scoped folder should be passed through as path
    await _run_research_query(
        services=services,
        embedding_manager=embedding_manager,  # type: ignore[arg-type]
        llm_manager=llm_manager,  # type: ignore[arg-type]
        prompt=prompt,
        scope_label="tree-sitter-haskell",
    )
    assert captured["path"] == "tree-sitter-haskell"

    # Root scope should disable path scoping
    await _run_research_query(
        services=services,
        embedding_manager=embedding_manager,  # type: ignore[arg-type]
        llm_manager=llm_manager,  # type: ignore[arg-type]
        prompt=prompt,
        scope_label="/",
    )
    assert captured["path"] is None

    # None scope should also disable path scoping
    await _run_research_query(
        services=services,
        embedding_manager=embedding_manager,  # type: ignore[arg-type]
        llm_manager=llm_manager,  # type: ignore[arg-type]
        prompt=prompt,
        scope_label=None,
    )
    assert captured["path"] is None

