"""Tests for sentinel strings in indexing.exclude (backwards-compatibility).

These tests define the desired behavior for using ".gitignore"
as the exclusion source via the existing `indexing.exclude` config key.
They will fail until the Config/IndexingConfig supports this union type and
provides a resolver method.
"""

from __future__ import annotations

from chunkhound.core.config.config import Config


def test_exclude_sentinel_gitignore_maps_to_source_gitignore() -> None:
    cfg = Config(**{"indexing": {"exclude": ".gitignore"}})

    # New helper to resolve ignore sources from config
    sources = cfg.indexing.resolve_ignore_sources()  # type: ignore[attr-defined]
    assert sources == ["gitignore"]


def test_exclude_sentinel_chignore_now_ignored() -> None:
    # .chignore sentinel is no longer supported; fallback to default (.gitignore only)
    cfg = Config(**{"indexing": {"exclude": ".chignore"}})
    sources = cfg.indexing.resolve_ignore_sources()  # type: ignore[attr-defined]
    assert sources == ["gitignore"]


def test_exclude_list_uses_config_only() -> None:
    cfg = Config(**{"indexing": {"exclude": ["**/dist/**", "**/*.min.js"]}})

    sources = cfg.indexing.resolve_ignore_sources()  # type: ignore[attr-defined]
    assert sources == ["config"]


def test_exclude_missing_defaults_to_gitignore_only() -> None:
    # When 'exclude' not provided explicitly, default behavior should be gitignore-only
    cfg = Config(**{"indexing": {}})
    sources = cfg.indexing.resolve_ignore_sources()  # type: ignore[attr-defined]
    assert sources == ["gitignore"]
