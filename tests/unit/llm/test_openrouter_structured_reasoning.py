"""Contracts for OpenRouter structured-call reasoning negotiation."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, patch

import httpx
import pytest
from openai import APIStatusError, APITimeoutError

from chunkhound.providers.llm.openai_compatible_provider import OpenAICompatibleProvider
from tests.fixtures.openai_compatible_server import (
    ChatCompletionScript,
    OpenAICompatibleTestServer,
)

PAYLOAD = {"reasoning": {"enabled": False}}
SCHEMA = {
    "type": "object",
    "properties": {"answer": {"type": "string"}},
    "required": ["answer"],
    "additionalProperties": False,
}


class FakeCapabilityStore:
    """Small persistence-boundary fake; provider memory remains real."""

    def __init__(self, state: str = "unknown") -> None:
        self.state = state
        self.get_calls: list[tuple[str, str]] = []
        self.set_calls: list[tuple[str, str, str]] = []

    def get(self, provider: str, model: str) -> str:
        self.get_calls.append((provider, model))
        return self.state

    def set(self, provider: str, model: str, state: str) -> None:
        self.set_calls.append((provider, model, state))
        self.state = state


@pytest.fixture
def mock_completion() -> AsyncMock:
    shared_create = AsyncMock()

    def client_factory(**kwargs: Any) -> SimpleNamespace:
        return SimpleNamespace(
            base_url=kwargs.get("base_url"),
            chat=SimpleNamespace(
                completions=SimpleNamespace(create=shared_create),
            ),
        )

    with patch(
        "chunkhound.providers.llm.openai_compatible_provider.AsyncOpenAI",
        side_effect=client_factory,
    ):
        yield shared_create


@pytest.mark.asyncio
async def test_unknown_success_caches_accepted_and_keeps_payload_sticky(
    mock_completion: AsyncMock,
) -> None:
    """A successful optimistic request becomes accepted for this instance."""
    store = FakeCapabilityStore()
    mock_completion.return_value = _response()
    provider = _provider(store)

    assert await provider.complete_structured("first", SCHEMA) == {"answer": "42"}
    assert await provider.complete_structured("second", SCHEMA) == {"answer": "42"}

    assert [call.kwargs["extra_body"] for call in mock_completion.call_args_list] == [
        PAYLOAD,
        PAYLOAD,
    ]
    assert store.get_calls == [("openrouter", "poolside/laguna-xs-2.1")]
    assert store.set_calls == [("openrouter", "poolside/laguna-xs-2.1", "accepted")]


@pytest.mark.asyncio
@pytest.mark.parametrize("status_code", [400, 422])
async def test_unknown_rejection_replays_once_and_caches_rejected(
    mock_completion: AsyncMock,
    status_code: int,
) -> None:
    """Only an unflagged success may turn a 400/422 into sticky rejection."""
    store = FakeCapabilityStore()
    mock_completion.side_effect = [
        _status_error(status_code),
        _response(),
        _response(),
    ]
    provider = _provider(store)

    assert await provider.complete_structured("probe", SCHEMA) == {"answer": "42"}
    assert await provider.complete_structured("later", SCHEMA) == {"answer": "42"}

    calls = mock_completion.call_args_list
    assert len(calls) == 3
    assert calls[0].kwargs["extra_body"] == PAYLOAD
    assert "extra_body" not in calls[1].kwargs
    assert "extra_body" not in calls[2].kwargs
    assert store.set_calls == [("openrouter", "poolside/laguna-xs-2.1", "rejected")]


@pytest.mark.asyncio
async def test_replay_failure_does_not_poison_capability_state(
    mock_completion: AsyncMock,
) -> None:
    """Two failed shapes leave the unknown state unwritten and surface the error."""
    store = FakeCapabilityStore()
    mock_completion.side_effect = [_status_error(400), _status_error(400)]
    provider = _provider(store)

    with pytest.raises(RuntimeError, match="structured completion failed"):
        await provider.complete_structured("probe", SCHEMA)

    calls = mock_completion.call_args_list
    assert len(calls) == 2
    assert calls[0].kwargs["extra_body"] == PAYLOAD
    assert "extra_body" not in calls[1].kwargs
    assert store.state == "unknown"
    assert store.set_calls == []


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "error_factory",
    [
        pytest.param(lambda: _status_error(429), id="http-429"),
        pytest.param(lambda: _status_error(500), id="http-500"),
        pytest.param(
            lambda: APITimeoutError(httpx.Request("POST", "https://example.test")),
            id="timeout",
        ),
    ],
)
async def test_transient_probe_failure_is_not_replayed_or_cached(
    mock_completion: AsyncMock,
    error_factory: Callable[[], Exception],
) -> None:
    """Rate limits, server errors, and timeouts use normal one-call handling."""
    store = FakeCapabilityStore()
    mock_completion.side_effect = error_factory()
    provider = _provider(store)

    with pytest.raises(RuntimeError, match="structured completion failed"):
        await provider.complete_structured("probe", SCHEMA)

    assert mock_completion.call_count == 1
    assert mock_completion.call_args.kwargs["extra_body"] == PAYLOAD
    assert store.state == "unknown"
    assert store.set_calls == []


@pytest.mark.asyncio
async def test_cached_accepted_rejection_replays_and_flips_state(
    mock_completion: AsyncMock,
) -> None:
    """A contradictory 400 immediately replaces stale accepted state."""
    store = FakeCapabilityStore("accepted")
    mock_completion.side_effect = [_status_error(400), _response()]
    provider = _provider(store)

    assert await provider.complete_structured("changed", SCHEMA) == {"answer": "42"}

    calls = mock_completion.call_args_list
    assert calls[0].kwargs["extra_body"] == PAYLOAD
    assert "extra_body" not in calls[1].kwargs
    assert store.set_calls == [("openrouter", "poolside/laguna-xs-2.1", "rejected")]


@pytest.mark.asyncio
async def test_concurrent_unknown_calls_single_flight_the_probe(
    mock_completion: AsyncMock,
) -> None:
    """Parallel first calls share one rejected probe before continuing unflagged."""
    store = FakeCapabilityStore()
    probe_started = asyncio.Event()
    release_probe = asyncio.Event()

    async def complete(**kwargs: Any) -> SimpleNamespace:
        if "extra_body" in kwargs:
            probe_started.set()
            await release_probe.wait()
            raise _status_error(400)
        return _response()

    mock_completion.side_effect = complete
    provider = _provider(store)

    first = asyncio.create_task(provider.complete_structured("first", SCHEMA))
    await asyncio.wait_for(probe_started.wait(), timeout=1)
    followers = [
        asyncio.create_task(provider.complete_structured(name, SCHEMA))
        for name in ("second", "third")
    ]
    await asyncio.sleep(0)
    release_probe.set()

    results = await asyncio.gather(first, *followers)

    assert results == [{"answer": "42"}] * 3
    calls = mock_completion.call_args_list
    assert len(calls) == 4
    assert sum("extra_body" in call.kwargs for call in calls) == 1
    assert store.set_calls == [("openrouter", "poolside/laguna-xs-2.1", "rejected")]


@pytest.mark.asyncio
async def test_complete_never_sends_structured_reasoning_payload(
    mock_completion: AsyncMock,
) -> None:
    """The non-structured public API keeps today's request shape."""
    store = FakeCapabilityStore("accepted")
    mock_completion.return_value = _response(content="plain answer")
    provider = _provider(store)

    response = await provider.complete("not structured")

    assert response.content == "plain answer"
    assert "extra_body" not in mock_completion.call_args.kwargs


@pytest.mark.asyncio
async def test_structured_call_without_payload_keeps_existing_shape(
    mock_completion: AsyncMock,
) -> None:
    """Providers and custom endpoints without metadata do not negotiate."""
    store = FakeCapabilityStore()
    mock_completion.return_value = _response()
    provider = _provider(store, structured_reasoning_disable_extra_body=None)

    assert await provider.complete_structured("unchanged", SCHEMA) == {"answer": "42"}

    assert "extra_body" not in mock_completion.call_args.kwargs
    assert store.set_calls == []


@pytest.mark.asyncio
async def test_structured_reasoning_payload_reaches_wire_body() -> None:
    """The SDK must serialize extra_body as OpenRouter's top-level reasoning key."""
    marker = "structured-reasoning-wire-marker"
    script = ChatCompletionScript(
        name="structured-reasoning",
        marker=marker,
        content='{"answer": "42"}',
    )
    store = FakeCapabilityStore("accepted")

    with OpenAICompatibleTestServer([script]) as server:
        async with _wire_provider(server, store) as provider:
            assert await provider.complete_structured(marker, SCHEMA) == {
                "answer": "42"
            }

            body = server.requests[0]["json"]
            assert body["reasoning"] == {"enabled": False}
            assert "extra_body" not in body
            server.assert_all_scripts_consumed()


def _provider(
    capability_store: FakeCapabilityStore,
    **overrides: Any,
) -> OpenAICompatibleProvider:
    kwargs: dict[str, Any] = {
        "provider_name": "openrouter",
        "api_key": "sk-test",
        "model": "poolside/laguna-xs-2.1",
        "default_base_url": "https://openrouter.ai/api/v1",
        "supports_structured_outputs": False,
        "structured_reasoning_disable_extra_body": PAYLOAD,
        "capability_store": capability_store,
        **overrides,
    }
    return OpenAICompatibleProvider(**kwargs)


def _response(content: str = '{"answer": "42"}') -> SimpleNamespace:
    return SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(content=content),
                finish_reason="stop",
            )
        ],
        usage=SimpleNamespace(
            prompt_tokens=3,
            completion_tokens=4,
            total_tokens=7,
        ),
    )


def _status_error(status_code: int) -> APIStatusError:
    request = httpx.Request("POST", "https://openrouter.ai/api/v1/chat/completions")
    response = httpx.Response(status_code, request=request)
    return APIStatusError(
        "opaque provider error",
        response=response,
        body={"error": {"message": "opaque provider error"}},
    )


@asynccontextmanager
async def _wire_provider(
    server: OpenAICompatibleTestServer,
    capability_store: FakeCapabilityStore,
) -> AsyncIterator[OpenAICompatibleProvider]:
    provider = _provider(
        capability_store,
        api_key="sk-local-fixture-not-a-real-credential",
        base_url=server.base_url,
        default_base_url=None,
        model="loopback-test-model",
        max_retries=0,
    )
    try:
        yield provider
    finally:
        await provider._client.close()
