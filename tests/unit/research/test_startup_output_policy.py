"""Research startup output reports the selected synthesis request-limit policy."""

import asyncio
from io import StringIO
from typing import Any

import pytest

from chunkhound.api.cli.utils.tree_progress import TreeProgressDisplay
from chunkhound.interfaces.llm_provider import (
    OutputLimitCapability,
    OutputLimitMetadata,
    OutputLimitPolicy,
)
from chunkhound.services.research.v1.pluggable_research_service import (
    PluggableResearchService,
)


class _StopAfterStartupError(Exception):
    """Sentinel preventing the contract test from running the research pipeline."""


class _Provider:
    def __init__(self, policy: OutputLimitPolicy) -> None:
        self.synthesis_output_limit_policy = policy


class _Manager:
    def __init__(self, policy: OutputLimitPolicy) -> None:
        self._provider = _Provider(policy)

    def get_synthesis_provider(self) -> _Provider:
        return self._provider


async def _stop_after_startup(*_: Any, **__: Any) -> list[dict[str, Any]]:
    raise _StopAfterStartupError


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("policy", "expected"),
    [
        (
            OutputLimitPolicy(
                output_limits_enabled=False,
                fallback_tokens=64_000,
                metadata=OutputLimitMetadata(omission=OutputLimitCapability.SUPPORTED),
            ),
            "Max depth: 1; synthesis request limits: provider-managed (cap omitted)",
        ),
        (
            OutputLimitPolicy(
                output_limits_enabled=False,
                fallback_tokens=64_000,
                metadata=OutputLimitMetadata(
                    omission=OutputLimitCapability.UNKNOWN,
                    declared_max_tokens=64_000,
                    declared_max_source="https://provider.example/output-limits",
                ),
            ),
            "Max depth: 1; synthesis request limits: provider-managed "
            "(provider-declared cap: 64,000 tokens)",
        ),
        (
            OutputLimitPolicy(
                output_limits_enabled=False,
                fallback_tokens=64_000,
                metadata=OutputLimitMetadata(omission=OutputLimitCapability.UNKNOWN),
            ),
            "Max depth: 1; synthesis request limits: provider-managed "
            "(fallback cap: 64,000 tokens)",
        ),
        (
            OutputLimitPolicy(
                output_limits_enabled=True,
                fallback_tokens=64_000,
                metadata=OutputLimitMetadata(),
            ),
            "Max depth: 1; synthesis request limits: legacy numeric "
            "(30,000-token single/reduce cap; computed per-map caps)",
        ),
    ],
    ids=["omission", "declaration", "fallback", "legacy"],
)
async def test_deep_research_renders_resolved_startup_output_policy(
    policy: OutputLimitPolicy,
    expected: str,
) -> None:
    output = StringIO()
    progress = TreeProgressDisplay(output=output)
    progress.start()

    service = object.__new__(PluggableResearchService)
    service._llm_manager = _Manager(policy)
    service._progress = progress
    service._progress_lock = asyncio.Lock()
    service._unified_search = _stop_after_startup

    try:
        with pytest.raises(_StopAfterStartupError):
            await service.deep_research("policy contract")
    finally:
        progress.stop()

    info_lines = [
        line for line in output.getvalue().splitlines() if "Max depth:" in line
    ]
    assert len(info_lines) == 1
    assert info_lines[0].endswith(expected)
    assert "output budget" not in output.getvalue()

    all_variants = {
        "Max depth: 1; synthesis request limits: provider-managed (cap omitted)",
        "Max depth: 1; synthesis request limits: provider-managed "
        "(provider-declared cap: 64,000 tokens)",
        "Max depth: 1; synthesis request limits: provider-managed "
        "(fallback cap: 64,000 tokens)",
        "Max depth: 1; synthesis request limits: legacy numeric "
        "(30,000-token single/reduce cap; computed per-map caps)",
    }
    for other in all_variants - {expected}:
        assert other not in output.getvalue()
