"""Request-local structured reasoning failures for research caller diagnostics."""

from contextvars import ContextVar

# Mutable only within a deep_research run; child asyncio tasks inherit this binding.
# Each run creates its own dict, so unrelated requests sharing a provider stay isolated.
current_empty_failures: ContextVar[dict[object, int] | None] = ContextVar(
    "chunkhound_structured_reasoning_empty_failures", default=None
)


def record_empty_failure(provider: object) -> None:
    """Attribute a rejected-capability empty response to the active run, if any."""
    failures = current_empty_failures.get()
    if failures is not None:
        failures[provider] = failures.get(provider, 0) + 1
