"""Backwards-compatible shim for the deep-doc splitter.

The splitter implementation now lives under ``operations.deep_doc.split_deep_doc``.
This module re-exports its public helpers so existing imports continue to work.
"""

from __future__ import annotations

from operations.deep_doc.split_deep_doc import *  # type: ignore[import-not-found]  # noqa: F401,F403


def main() -> None:  # pragma: no cover - thin wrapper
    from operations.deep_doc.split_deep_doc import main as _main

    _main()


if __name__ == "__main__":
    main()

