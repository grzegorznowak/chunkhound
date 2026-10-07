"""User-visible cache locations of independent persisted features."""

import sys
from pathlib import Path

import pytest

from chunkhound.providers.llm.capability_cache import LLMCapabilityStore
from chunkhound.watchman_runtime.loader import _default_runtime_cache_dir


@pytest.mark.parametrize(
    ("platform", "environment", "relative_root"),
    [
        ("linux", "xdg", Path("xdg/chunkhound")),
        ("linux", "home", Path("home/.cache/chunkhound")),
        ("win32", "local-appdata", Path("local-appdata/ChunkHound")),
        ("win32", "home", Path("home/AppData/Local/ChunkHound")),
    ],
)
def test_platform_default_cache_root_shared_by_capabilities_and_watchman(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    platform: str,
    environment: str,
    relative_root: Path,
) -> None:
    monkeypatch.delenv("CHUNKHOUND_LLM_CAPABILITY_CACHE", raising=False)
    monkeypatch.delenv("CHUNKHOUND_WATCHMAN_RUNTIME_CACHE_DIR", raising=False)
    monkeypatch.delenv("XDG_CACHE_HOME", raising=False)
    monkeypatch.delenv("LOCALAPPDATA", raising=False)
    monkeypatch.setattr(sys, "platform", platform)
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: tmp_path / "home"))
    if environment != "home":
        env_name = "XDG_CACHE_HOME" if environment == "xdg" else "LOCALAPPDATA"
        monkeypatch.setenv(env_name, str(tmp_path / environment))

    expected = tmp_path / relative_root
    assert LLMCapabilityStore()._path == expected / "llm-capabilities.json"
    assert _default_runtime_cache_dir() == expected / "watchman-runtime"


def test_feature_specific_cache_overrides_remain_independent(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    llm_path = tmp_path / "llm.json"
    watchman_path = tmp_path / "watchman"
    monkeypatch.setenv("CHUNKHOUND_LLM_CAPABILITY_CACHE", str(llm_path))
    monkeypatch.setenv("CHUNKHOUND_WATCHMAN_RUNTIME_CACHE_DIR", str(watchman_path))

    assert LLMCapabilityStore()._path == llm_path
    assert _default_runtime_cache_dir() == watchman_path
