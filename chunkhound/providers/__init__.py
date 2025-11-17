"""Providers package for ChunkHound - concrete implementations of abstract interfaces.

Use lazy import to avoid importing heavy backends during package import.
"""

__all__ = [
    "DuckDBProvider",
    "OpenAIEmbeddingProvider",
]


def __getattr__(name: str):
    if name == "DuckDBProvider":
        from .database import DuckDBProvider  # lazy

        return DuckDBProvider
    if name == "OpenAIEmbeddingProvider":
        from .embeddings import OpenAIEmbeddingProvider  # lazy

        return OpenAIEmbeddingProvider
    raise AttributeError(name)
