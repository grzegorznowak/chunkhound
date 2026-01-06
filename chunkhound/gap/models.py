"""Typed models for the `gap.v1` JSON contract."""

from dataclasses import asdict, dataclass, field
from typing import Any, Literal


@dataclass(frozen=True)
class GapNormalizationInvariants:
    id: str
    include_comments: bool
    include_docs: bool


@dataclass(frozen=True)
class GapInvariants:
    hash_alg: Literal["xxh3_64"]
    normalization: GapNormalizationInvariants
    chunker_version: str
    recovery_mode: Literal["off", "safe", "aggressive"]
    deterministic: bool
    embed_model_id: str | None
    forced: bool


@dataclass(frozen=True)
class GapInputRef:
    source_kind: Literal["path"]
    source_ref: str
    source_hash: str


@dataclass(frozen=True)
class GapInputs:
    a: GapInputRef
    b: GapInputRef


@dataclass(frozen=True)
class GapScope:
    scope_mode: str
    scope_hash: str
    changed_files_count: int
    rename_hints_count: int


@dataclass(frozen=True)
class GapWarning:
    code: str
    message: str
    meta: dict[str, Any] | None = None


@dataclass(frozen=True)
class GapTimings:
    total_ms: float = 0.0
    discover_ms: float = 0.0
    manifest_ms: float = 0.0
    diff_ms: float = 0.0
    recovery_ms: float = 0.0


@dataclass(frozen=True)
class GapCounts:
    files_a_total: int = 0
    files_b_total: int = 0
    changes_total: int = 0
    file_changes_total: int = 0
    symbol_changes_total: int = 0


@dataclass(frozen=True)
class GapStats:
    counts: GapCounts = field(default_factory=GapCounts)
    timings: GapTimings = field(default_factory=GapTimings)


@dataclass(frozen=True)
class GapFileHandle:
    path: str
    file_hash: str
    size_bytes: int
    file_class: Literal["text", "binary", "too_large", "decode_error", "ignored"]


@dataclass(frozen=True)
class GapSymbolHandle:
    path: str
    start_line: int
    end_line: int
    symbol: str
    chunk_type: str
    text_hash: str
    ordinal_in_file: int
    parse_status: Literal["ok", "fallback"]
    stable_key_hash: str | None = None
    symbol_key_hash: str | None = None
    start_byte: int | None = None
    end_byte: int | None = None


GapHandle = GapFileHandle | GapSymbolHandle


@dataclass(frozen=True)
class GapChangeItem:
    entity_kind: Literal["symbol", "file"]
    op: Literal["add", "remove", "update"]
    moved: bool
    renamed: bool
    content_changed: bool
    reason: str
    confidence: float
    primary_key_kind: Literal["symbol_key", "stable_key", "path_key"]
    key_strength: int
    had_collision: bool
    collision_group_size: int
    old: GapHandle | None
    new: GapHandle | None


@dataclass(frozen=True)
class GapReport:
    schema_version: Literal["gap.v1"]
    direction: Literal["A->B"]
    invariants: GapInvariants
    inputs: GapInputs
    scope: GapScope
    warnings: list[GapWarning]
    stats: GapStats
    changes: list[GapChangeItem]

    def to_dict(self) -> dict[str, Any]:
        """Convert report to a plain JSON-serializable dict."""
        return asdict(self)

