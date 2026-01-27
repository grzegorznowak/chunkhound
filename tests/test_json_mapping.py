#!/usr/bin/env python3
"""Unit tests for JSON mapping chunk type hints."""

from __future__ import annotations

from chunkhound.core.types.common import ChunkType, FileId, Language
from chunkhound.parsers.parser_factory import ParserFactory
from chunkhound.parsers.universal_engine import UniversalConcept


def test_json_pair_nodes_hint_key_value_and_parse_as_key_value() -> None:
    parser = ParserFactory().create_parser(Language.JSON)

    sample = '{"name": "x"}'
    ast = parser.engine.parse_to_ast(sample)
    ucs = parser.extractor.extract_concept(
        ast.root_node, sample.encode("utf-8"), UniversalConcept.DEFINITION
    )
    pair_nodes = [uc for uc in ucs if uc.language_node_type == "pair"]
    assert pair_nodes
    assert all(
        (uc.metadata or {}).get("chunk_type_hint") == "key_value" for uc in pair_nodes
    )

    chunks = parser.parse_content(sample, "a.json", FileId(1))
    assert any(c.chunk_type == ChunkType.KEY_VALUE for c in chunks)


def test_json_object_and_array_nodes_hint_object_and_array() -> None:
    parser = ParserFactory().create_parser(Language.JSON)

    sample = '{"obj": {"a": 1}, "arr": [1, 2]}'
    ast = parser.engine.parse_to_ast(sample)
    ucs = parser.extractor.extract_concept(
        ast.root_node, sample.encode("utf-8"), UniversalConcept.BLOCK
    )
    object_nodes = [uc for uc in ucs if uc.language_node_type == "object"]
    array_nodes = [uc for uc in ucs if uc.language_node_type == "array"]
    assert object_nodes
    assert array_nodes
    assert all(
        (uc.metadata or {}).get("chunk_type_hint") == "object" for uc in object_nodes
    )
    assert all(
        (uc.metadata or {}).get("chunk_type_hint") == "array" for uc in array_nodes
    )
