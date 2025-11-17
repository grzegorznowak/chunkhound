"""Database providers package for ChunkHound - concrete database implementations.

Use lazy import to avoid importing heavy backends (e.g., duckdb) on package import.
"""

__all__ = ["DuckDBProvider"]


def __getattr__(name: str):
    if name == "DuckDBProvider":
        from .duckdb_provider import DuckDBProvider  # lazy import

        return DuckDBProvider
    raise AttributeError(name)
