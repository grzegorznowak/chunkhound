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

from chunkhound.providers.llm.capability_cache import LLMCapabilityStore
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
async def test_cached_rejected_first_call_never_sends_payload(
    mock_completion: AsyncMock,
) -> None:
    """A persisted rejection omits the payload on the first instance call."""
    store = FakeCapabilityStore("rejected")
    mock_completion.return_value = _response()
    provider = _provider(store)

    assert await provider.complete_structured("cached", SCHEMA) == {"answer": "42"}

    assert mock_completion.call_count == 1
    assert "extra_body" not in mock_completion.call_args.kwargs
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


@pytest.mark.asyncio
async def test_concurrent_cached_accepted_rejections_each_replay_once(
    mock_completion: AsyncMock,
) -> None:
    """Each flagged request may recover even after another flips the state."""
    store = FakeCapabilityStore("accepted")
    both_flagged = asyncio.Event()
    flagged_calls = 0

    async def complete(**kwargs: Any) -> SimpleNamespace:
        nonlocal flagged_calls
        if "extra_body" in kwargs:
            flagged_calls += 1
            if flagged_calls == 2:
                both_flagged.set()
            await asyncio.wait_for(both_flagged.wait(), timeout=5)
            raise _status_error(400)
        return _response()

    mock_completion.side_effect = complete
    provider = _provider(store)

    results = await asyncio.gather(
        provider.complete_structured("first", SCHEMA),
        provider.complete_structured("second", SCHEMA),
    )

    assert results == [{"answer": "42"}] * 2
    calls = mock_completion.call_args_list
    assert sum("extra_body" in call.kwargs for call in calls) == 2
    assert sum("extra_body" not in call.kwargs for call in calls) == 2
    assert store.state == "rejected"
    assert store.set_calls == [("openrouter", "poolside/laguna-xs-2.1", "rejected")]


@pytest.mark.asyncio
async def test_unknown_accepted_probe_releases_followers_before_network_io(
    mock_completion: AsyncMock,
) -> None:
    """Followers are only serialized while the single-flight probe resolves."""
    store = FakeCapabilityStore()
    leader_started = asyncio.Event()
    release_leader = asyncio.Event()
    follower_started = {name: asyncio.Event() for name in ("second", "third")}

    async def complete(**kwargs: Any) -> SimpleNamespace:
        prompt = kwargs["messages"][-1]["content"]
        assert "extra_body" in kwargs
        if prompt == "first":
            leader_started.set()
            await release_leader.wait()
        else:
            follower_started[prompt].set()
            other = "third" if prompt == "second" else "second"
            await asyncio.wait_for(follower_started[other].wait(), timeout=5)
        return _response()

    mock_completion.side_effect = complete
    provider = _provider(store)

    leader = asyncio.create_task(provider.complete_structured("first", SCHEMA))
    await asyncio.wait_for(leader_started.wait(), timeout=1)
    followers = [
        asyncio.create_task(provider.complete_structured(name, SCHEMA))
        for name in ("second", "third")
    ]
    release_leader.set()

    results = await asyncio.wait_for(asyncio.gather(leader, *followers), timeout=5)

    assert results == [{"answer": "42"}] * 3
    assert all(event.is_set() for event in follower_started.values())
    assert mock_completion.call_count == 3
    assert all("extra_body" in call.kwargs for call in mock_completion.call_args_list)
    assert store.set_calls == [("openrouter", "poolside/laguna-xs-2.1", "accepted")]


@pytest.mark.asyncio
@pytest.mark.parametrize("initial_state", ["unknown", "accepted"])
async def test_replay_failure_preserves_prior_state_and_surfaces_replay_error(
    mock_completion: AsyncMock, initial_state: str
) -> None:
    """Only a successful unflagged replay may replace the prior capability."""
    store = FakeCapabilityStore(initial_state)
    replay_error = APITimeoutError(httpx.Request("POST", "https://example.test"))
    mock_completion.side_effect = [_status_error(400), replay_error]
    provider = _provider(store)

    with pytest.raises(RuntimeError, match="structured completion failed") as raised:
        await provider.complete_structured("probe", SCHEMA)

    assert raised.value.__cause__ is replay_error
    assert store.state == initial_state
    assert store.set_calls == []


@pytest.mark.asyncio
async def test_transient_probe_failure_is_reprobed_by_next_call(
    mock_completion: AsyncMock,
) -> None:
    """A transient failure leaves unknown capability available for reprobe."""
    store = FakeCapabilityStore()
    mock_completion.side_effect = [_status_error(500), _response()]
    provider = _provider(store)

    with pytest.raises(RuntimeError, match="structured completion failed"):
        await provider.complete_structured("first", SCHEMA)
    assert await provider.complete_structured("second", SCHEMA) == {"answer": "42"}

    assert all("extra_body" in call.kwargs for call in mock_completion.call_args_list)
    assert store.set_calls == [("openrouter", "poolside/laguna-xs-2.1", "accepted")]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("initial_state", "payload", "error_factory", "has_payload"),
    [
        ("rejected", PAYLOAD, lambda: _status_error(400), False),
        ("accepted", PAYLOAD, lambda: _status_error(500), True),
        ("unknown", None, lambda: _status_error(400), False),
    ],
)
async def test_unflagged_and_nonrejection_failures_are_not_replayed(
    mock_completion: AsyncMock,
    initial_state: str,
    payload: dict[str, Any] | None,
    error_factory: Callable[[], Exception],
    has_payload: bool,
) -> None:
    """Only flagged 400/422 responses receive a single fallback attempt."""
    store = FakeCapabilityStore(initial_state)
    mock_completion.side_effect = error_factory()
    provider = _provider(store, structured_reasoning_disable_extra_body=payload)

    with pytest.raises(RuntimeError, match="structured completion failed"):
        await provider.complete_structured("probe", SCHEMA)

    assert mock_completion.call_count == 1
    assert ("extra_body" in mock_completion.call_args.kwargs) is has_payload
    assert store.set_calls == []


@pytest.mark.asyncio
async def test_probe_non_http_exception_is_not_replayed(
    mock_completion: AsyncMock,
) -> None:
    """Non-HTTP probe failures retain the normal one-call error behavior."""
    store = FakeCapabilityStore()
    mock_completion.side_effect = TypeError("boom")
    provider = _provider(store)

    with pytest.raises(RuntimeError, match="structured completion failed"):
        await provider.complete_structured("probe", SCHEMA)

    assert mock_completion.call_count == 1
    assert "extra_body" in mock_completion.call_args.kwargs
    assert store.set_calls == []


@pytest.mark.asyncio
async def test_structured_payload_is_deep_copied_per_request(
    mock_completion: AsyncMock,
) -> None:
    """SDK-side mutation cannot alter the configured payload or later calls."""
    source_payload = {"reasoning": {"enabled": False}}
    store = FakeCapabilityStore("accepted")
    mock_completion.return_value = _response()
    provider = _provider(store, structured_reasoning_disable_extra_body=source_payload)

    assert await provider.complete_structured("first", SCHEMA) == {"answer": "42"}
    first_payload = mock_completion.call_args.kwargs["extra_body"]
    first_payload["reasoning"]["enabled"] = True
    assert await provider.complete_structured("second", SCHEMA) == {"answer": "42"}
    second_payload = mock_completion.call_args.kwargs["extra_body"]

    assert source_payload == {"reasoning": {"enabled": False}}
    assert second_payload == {"reasoning": {"enabled": False}}
    assert first_payload is not second_payload


def test_capability_lock_rebinds_across_event_loops(mock_completion: AsyncMock) -> None:
    """A provider reused by separate asyncio.run calls gets a fresh lock."""
    store = FakeCapabilityStore()
    mock_completion.side_effect = [_status_error(500), _response()]
    provider = _provider(store)

    with pytest.raises(RuntimeError, match="structured completion failed"):
        asyncio.run(provider.complete_structured("first", SCHEMA))
    second = asyncio.run(provider.complete_structured("second", SCHEMA))
    assert second == {"answer": "42"}


def test_successful_structured_call_survives_cache_write_failure(
    mock_completion: AsyncMock, monkeypatch: pytest.MonkeyPatch, tmp_path: Any
) -> None:
    """Persistent-cache I/O failures do not discard a successful response."""
    blocked_parent = tmp_path / "not-a-directory"
    blocked_parent.write_text("blocked")
    monkeypatch.setenv(
        "CHUNKHOUND_LLM_CAPABILITY_CACHE", str(blocked_parent / "capabilities.json")
    )
    store = LLMCapabilityStore()
    mock_completion.return_value = _response()
    provider = _provider(store)

    first = asyncio.run(provider.complete_structured("first", SCHEMA))
    second = asyncio.run(provider.complete_structured("second", SCHEMA))

    assert first == {"answer": "42"}
    assert second == {"answer": "42"}
    assert provider._structured_reasoning_capability == "accepted"
    assert mock_completion.call_count == 2
    assert all("extra_body" in call.kwargs for call in mock_completion.call_args_list)


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
