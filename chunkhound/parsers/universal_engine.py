"""Universal tree-sitter parsing engine for all languages."""

from dataclasses import dataclass
from enum import Enum
from typing import Any

import os
from tree_sitter import Language, Node, Parser, Query, Tree


class UniversalConcept(Enum):
    """Universal semantic concepts found in ALL programming languages."""

    DEFINITION = "definition"  # Functions, classes, types, variables
    BLOCK = "block"  # Scoped regions, logical groupings
    COMMENT = "comment"  # Documentation, explanations
    IMPORT = "import"  # Dependencies, includes, modules
    STRUCTURE = "structure"  # Hierarchical organization (headers, sections)


@dataclass(frozen=True)
class UniversalChunk:
    """Language-agnostic representation of semantic code unit."""

    concept: UniversalConcept
    name: str
    content: str
    start_line: int
    end_line: int
    metadata: dict[str, Any]
    language_node_type: str  # Original tree-sitter type for debugging


class SetupError(Exception):
    """Error raised when parser setup fails."""

    def __init__(
        self,
        parser: str,
        missing_dependency: str,
        install_command: str,
        original_error: str,
    ) -> None:
        self.parser = parser
        self.missing_dependency = missing_dependency
        self.install_command = install_command
        self.original_error = original_error
        super().__init__(str(self))

    def __str__(self) -> str:
        return (
            f"Parser {self.parser} setup failed:\n"
            f"Missing: {self.missing_dependency}\n"
            f"Fix: {self.install_command}\n"
            f"Details: {self.original_error}"
        )


class QueryCompilationError(Exception):
    """Error raised when tree-sitter query compilation fails."""

    def __init__(
        self, concept: UniversalConcept, language: str, query: str, error: str
    ) -> None:
        self.concept = concept
        self.language = language
        self.query = query
        self.error = error
        super().__init__(str(self))

    def __str__(self) -> str:
        return (
            f"Query compilation failed for {self.concept} in {self.language}:\n"
            f"Query: {self.query}\n"
            f"Error: {self.error}"
        )


class TreeSitterEngine:
    """Universal tree-sitter parsing engine - works for ALL languages."""

    def __init__(self, language_name: str, tree_sitter_language: Language):
        """Initialize with pre-imported tree-sitter language object.

        Args:
            language_name: Name of the language (e.g., 'python', 'javascript')
            tree_sitter_language: Tree-sitter Language object for this language
        """
        self.language_name = language_name
        self._language = tree_sitter_language
        self._parser = Parser()
        self._parser.language = self._language
        # Apply a global parse timeout to prevent pathological hangs
        # Env override: CHUNKHOUND_PARSE_TIMEOUT_MS (0 disables timeout)
        try:
            timeout_ms = int(os.getenv("CHUNKHOUND_PARSE_TIMEOUT_MS", "3000"))
        except Exception:
            timeout_ms = 3000
        try:
            if timeout_ms and hasattr(self._parser, "timeout_micros"):
                # Deprecation warnings are acceptable; property still supported
                self._parser.timeout_micros = int(timeout_ms) * 1000
        except Exception:
            # If setting fails, continue without timeout
            pass

    def parse_to_ast(self, content: str) -> Tree:
        """Parse content to AST - universal across all languages."""
        return self._parser.parse(content.encode("utf-8"))

    def compile_query(self, query_string: str) -> Query:
        """Compile tree-sitter query - universal syntax."""
        try:
            return Query(self._language, query_string)
        except Exception as e:
            # Re-raise with more context
            raise QueryCompilationError(
                concept=UniversalConcept.DEFINITION,  # Will be overridden by caller
                language=self.language_name,
                query=query_string,
                error=str(e),
            )

    def get_node_text(self, node: Node, content: bytes) -> str:
        """Get text content of a node."""
        return content[node.start_byte : node.end_byte].decode("utf-8")
