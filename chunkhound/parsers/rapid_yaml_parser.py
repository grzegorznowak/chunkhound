"""RapidYAML-backed YAML parser."""

from __future__ import annotations

import logging
import os
import re
from collections import Counter
from contextlib import contextmanager
import os
from dataclasses import dataclass
from pathlib import Path
from typing import List, Sequence

from chunkhound.core.models.chunk import Chunk
from chunkhound.core.types.common import ChunkType, FileId, Language, LineNumber
from chunkhound.interfaces.language_parser import LanguageParser, ParseResult
from chunkhound.parsers.universal_parser import UniversalParser
from chunkhound.parsers.yaml_template_sanitizer import sanitize_helm_templates

logger = logging.getLogger(__name__)


def _env_wants_tree_sitter() -> bool:
    """Return True when RapidYAML should be disabled via env."""
    value = os.environ.get("CHUNKHOUND_YAML_ENGINE", "").strip().lower()
    if not value:
        value = os.environ.get("CHUNKHOUND_DISABLE_RAPIDYAML", "").strip().lower()
        if value in {"1", "true", "yes"}:
            return True
        return False
    return value in {"tree", "treesitter", "tree_sitter", "ts"}


class RapidYamlParser(LanguageParser):
    """LanguageParser implementation that prefers RapidYAML, with fallback."""

    _KEY_NODE_TYPES = {"KEYVAL", "KEYMAP", "KEYSEQ"}

    def __init__(self, fallback: UniversalParser) -> None:
        self._fallback = fallback
        self._enabled = not _env_wants_tree_sitter()
        self._ryml = None
        self._tree = None
        # Memoize paths that should not be parsed with RapidYAML again this process
        self._denylist_paths: set[str] = set()
        # Counters for one-line summary logging
        self._count_sanitized = 0
        self._count_pre_skip = 0
        self._count_complex_skip = 0
        self._count_ryml_ok = 0
        self._count_ryml_fail = 0
        self._count_fallback_ts = 0

        if self._enabled:
            try:
                import ryml  # type: ignore[import-not-found]

                self._ryml = ryml
                self._tree = ryml.Tree()
            except Exception as exc:  # pragma: no cover - import-time guard
                self._enabled = False
                logger.info(
                    "RapidYAML disabled (import failure): %s. Falling back to tree-sitter.",
                    exc,
                )

    # ------------------------------------------------------------------#
    # LanguageParser interface (delegating most behavior to fallback)
    # ------------------------------------------------------------------#
    @property
    def language(self) -> Language:
        return self._fallback.language

    @property
    def supported_extensions(self) -> set[str]:
        return self._fallback.supported_extensions

    @property
    def supported_chunk_types(self) -> set[ChunkType]:
        return self._fallback.supported_chunk_types

    @property
    def is_initialized(self) -> bool:
        return self._fallback.is_initialized

    @property
    def config(self):
        return self._fallback.config

    def parse_file(self, file_path: Path, file_id: FileId) -> list[Chunk]:
        if not self._can_use_rapid():
            return self._fallback.parse_file(file_path, file_id)

        try:
            content = file_path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            # Attempt to decode using fallback parser
            return self._fallback.parse_file(file_path, file_id)

        return self.parse_content(content, file_path, file_id)

    def parse_content(
        self,
        content: str,
        file_path: Path | None = None,
        file_id: FileId | None = None,
    ) -> list[Chunk]:
        # Denylist: skip ryml attempts for known-bad paths (no tree-sitter fallback)
        if file_path is not None and str(file_path) in getattr(self, "_denylist_paths", set()):
            self._count_fallback_ts += 1
            return []

        if not self._can_use_rapid():
            return self._fallback.parse_content(content, file_path, file_id)

        sanitized = sanitize_helm_templates(content)
        effective_content = sanitized.text
        fallback_source = effective_content if sanitized.changed else content
        if sanitized.changed:
            self._count_sanitized += 1
            summary = _summarize_rewrites(sanitized.rewrites)
            path_str = str(file_path) if file_path else "<memory>"
            logger.debug(
                "Sanitized templated YAML %s (%s)",
                path_str,
                summary,
            )

        # If sanitizer advises pre-skip (eg, non-YAML fragments), avoid ryml churn
        if getattr(sanitized, "pre_skip", False):
            self._count_pre_skip += 1
            self._count_fallback_ts += 1
            return []

        if _has_complex_keys(effective_content):
            path_str = str(file_path) if file_path else "<memory>"
            logger.debug(
                "RapidYAML skipped %s: detected complex YAML keys. Falling back to tree-sitter.",
                path_str,
            )
            self._count_complex_skip += 1
            self._count_fallback_ts += 1
            return []

        if not effective_content.strip():
            return []

        try:
            builder = _RapidYamlChunkBuilder(
                self._ryml,
                self._tree,
                effective_content,
                file_id or FileId(0),
            )
            chunks = builder.build_chunks()
            self._count_ryml_ok += 1
            return chunks
        except Exception as exc:
            logger.debug(
                "RapidYAML parser failed (%s). Falling back to tree-sitter.", exc
            )
            # Add to denylist to avoid repeated attempts
            if file_path is not None:
                self._denylist_paths.add(str(file_path))
            self._count_ryml_fail += 1
            self._count_fallback_ts += 1
            return []

    def parse_with_result(self, file_path: Path, file_id: FileId) -> ParseResult:
        if not self._can_use_rapid():
            return self._fallback.parse_with_result(file_path, file_id)
        # Reuse fallback's ParseResult structure but supply rapid chunks.
        chunks = self.parse_file(file_path, file_id)
        return ParseResult(
            chunks=[chunk.to_dict() for chunk in chunks],
            language=Language.YAML,
            total_chunks=len(chunks),
            parse_time=0.0,
            errors=[],
            warnings=[],
            metadata={"parser_type": "rapid_yaml"},
        )

    def supports_incremental_parsing(self) -> bool:
        return False

    def parse_incremental(
        self, file_path: Path, previous_chunks: list[dict[str, object]] | None = None
    ) -> list[Chunk]:
        return self.parse_file(file_path, FileId(0))

    def get_parse_tree(self, content: str):
        return self._fallback.get_parse_tree(content)

    def setup(self) -> None:
        self._fallback.setup()

    def cleanup(self) -> None:
        # Emit one-line summary for this parser instance
        logger.info(
            "RapidYAML summary: sanitized=%d pre_skip=%d complex_skip=%d ryml_ok=%d ryml_fail=%d fallback_ts=%d",
            self._count_sanitized,
            self._count_pre_skip,
            self._count_complex_skip,
            self._count_ryml_ok,
            self._count_ryml_fail,
            self._count_fallback_ts,
        )
        # Delegate cleanup if supported
        if hasattr(self._fallback, "cleanup"):
            try:
                self._fallback.cleanup()  # type: ignore[call-arg]
            except Exception:
                pass

    def reset(self) -> None:
        self._fallback.reset()

    def can_parse_file(self, file_path: Path) -> bool:
        return self._fallback.can_parse_file(file_path)

    def detect_language(self, file_path: Path) -> Language | None:
        return self._fallback.detect_language(file_path)

    def validate_syntax(self, content: str) -> list[str]:
        return self._fallback.validate_syntax(content)

    # ------------------------------------------------------------------#
    def _can_use_rapid(self) -> bool:
        return self._enabled and self._ryml is not None and self._tree is not None


@dataclass
class _LineLocator:
    """Utility to approximate line ranges for emitted YAML blocks."""

    lines: Sequence[str]
    depth_positions: List[int]
    fallback_line: int = 0

    def __init__(self, content: str) -> None:
        self.lines = content.splitlines()
        self.depth_positions = [0]
        self.fallback_line = 0

    def truncate(self, depth: int) -> None:
        if depth + 1 < len(self.depth_positions):
            self.depth_positions = self.depth_positions[: depth + 1]

    def locate(self, first_line: str, depth: int) -> tuple[int, int]:
        if not self.lines:
            self.fallback_line += 1
            return self.fallback_line, self.fallback_line

        while len(self.depth_positions) <= depth:
            self.depth_positions.append(self.depth_positions[-1])

        start_search = self.depth_positions[depth]
        idx = self._find_from(first_line, start_search)
        if idx is None:
            idx = self._find_from(first_line, 0)
        if idx is None:
            self.fallback_line += 1
            return self.fallback_line, self.fallback_line

        self.depth_positions[depth] = idx + 1
        for i in range(depth + 1, len(self.depth_positions)):
            self.depth_positions[i] = max(self.depth_positions[i], idx + 1)

        end_idx = self._compute_block_end(idx)
        return idx + 1, end_idx + 1

    def _find_from(self, target_line: str, start: int) -> int | None:
        stripped_target = target_line.strip()
        if not stripped_target:
            return None

        for idx in range(max(0, start), len(self.lines)):
            candidate = self.lines[idx].strip()
            if not candidate:
                continue
            if candidate == stripped_target or candidate.startswith(stripped_target):
                return idx
        return None

    def _compute_block_end(self, start_idx: int) -> int:
        base_indent = self._indent_level(self.lines[start_idx])
        end_idx = start_idx
        for idx in range(start_idx + 1, len(self.lines)):
            line = self.lines[idx]
            stripped = line.strip()
            if not stripped:
                end_idx = idx
                continue
            indent = self._indent_level(line)
            if indent <= base_indent and not stripped.startswith("- "):
                break
            end_idx = idx
        return end_idx

    @staticmethod
    def _indent_level(line: str) -> int:
        idx = 0
        for ch in line:
            if ch in (" ", "\t"):
                idx += 1
            else:
                break
        return idx


class _RapidYamlChunkBuilder:
    """Walks a RapidYAML tree and produces Chunk objects."""

    def __init__(self, ryml_module, tree, content: str, file_id: FileId) -> None:
        self.ryml = ryml_module
        self.tree = tree
        self.file_id = file_id
        self.content = content
        self._buffer = bytearray(content.encode("utf-8"))
        self.lines = content.splitlines()
        self.locator = _LineLocator(content)
        self._decoder = ryml_module.u

    def build_chunks(self) -> list[Chunk]:
        self.tree.clear()
        self.tree.clear_arena()
        with _suppress_c_output():
            self.ryml.parse_in_place(self._buffer, tree=self.tree)

        root = self.tree.root_id()
        chunks: list[Chunk] = []
        path: list[str] = []

        for node, depth in self.ryml.walk(self.tree, root):
            ancestor_len = max(depth - 1, 0)
            if len(path) != ancestor_len:
                path = path[:ancestor_len]
            self.locator.truncate(depth)

            key_text = self._key_text(node)
            if key_text:
                if len(path) == ancestor_len:
                    path.append(key_text)
                else:
                    path = path[:ancestor_len] + [key_text]

            node_type = self.tree.type_str(node)
            if node_type not in RapidYamlParser._KEY_NODE_TYPES:
                continue

            if path:
                symbol = ".".join(path)
            elif key_text:
                symbol = key_text
            else:
                symbol = f"yaml_node_{node}"
            chunk = self._create_chunk(node, node_type, symbol, depth)
            if chunk:
                chunks.append(chunk)

        return chunks

    def _create_chunk(
        self, node: int, node_type: str, symbol: str, depth: int
    ) -> Chunk | None:
        with _suppress_c_output():
            emitted = self.ryml.emit_yaml(self.tree, node)
        normalized = emitted.lstrip("\n")
        first_line = normalized.splitlines()[0] if normalized else symbol
        start_line, end_line = self.locator.locate(first_line, depth)

        file_snippet = self._slice_content(start_line, end_line)
        if not file_snippet.strip():
            file_snippet = normalized or first_line

        chunk_type = self._chunk_type_for(node_type)
        metadata = {
            "parser": "rapid_yaml",
            "node_type": node_type.lower(),
            "key_path": symbol,
            "value_kind": self._value_kind(node_type),
        }

        value = self.tree.val(node)
        if value is not None:
            metadata["scalar_value"] = self._decoder(value)

        return Chunk(
            symbol=symbol,
            start_line=LineNumber(start_line),
            end_line=LineNumber(max(start_line, end_line)),
            code=file_snippet,
            chunk_type=chunk_type,
            file_id=self.file_id,
            language=Language.YAML,
            metadata=metadata,
        )

    def _slice_content(self, start_line: int, end_line: int) -> str:
        if not self.content:
            return ""
        start_idx = max(0, start_line - 1)
        end_idx = min(len(self.lines), end_line)
        if start_idx >= len(self.lines):
            return ""
        return "\n".join(self.lines[start_idx:end_idx])

    def _key_text(self, node: int) -> str | None:
        key = self.tree.key(node)
        if key is None:
            return None
        text = self._decoder(key).strip()
        return text or None

    @staticmethod
    def _chunk_type_for(node_type: str) -> ChunkType:
        if node_type == "KEYVAL":
            return ChunkType.KEY_VALUE
        if node_type == "KEYSEQ":
            return ChunkType.ARRAY
        return ChunkType.BLOCK

    @staticmethod
    def _value_kind(node_type: str) -> str:
        if node_type == "KEYVAL":
            return "scalar"
        if node_type == "KEYSEQ":
            return "sequence"
        return "mapping"


_COMPLEX_KEY_RE = re.compile(r"^\s*\?\s*(?:\{|\[|$)")


def _has_complex_keys(content: str) -> bool:
    for line in content.splitlines():
        if _COMPLEX_KEY_RE.match(line):
            return True
    return False


def _summarize_rewrites(rewrites) -> str:
    if not rewrites:
        return "no rewrites"
    counts = Counter(rewrite.kind for rewrite in rewrites)
    parts = [f"{kind}={counts[kind]}" for kind in sorted(counts)]
    return ", ".join(parts)


@contextmanager
def _suppress_c_output():
    """Temporarily redirect C-level stdout/stderr to os.devnull during ryml calls."""
    try:
        devnull = os.open(os.devnull, os.O_WRONLY)
        saved_out = os.dup(1)
        saved_err = os.dup(2)
        os.dup2(devnull, 1)
        os.dup2(devnull, 2)
        try:
            yield
        finally:
            os.dup2(saved_out, 1)
            os.dup2(saved_err, 2)
            os.close(saved_out)
            os.close(saved_err)
            os.close(devnull)
    except Exception:
        # Fail open: if redirection fails, proceed without silencing
        yield
