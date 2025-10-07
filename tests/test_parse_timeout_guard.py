import os
from pathlib import Path

from chunkhound.parsers.parser_factory import create_parser_for_file
from chunkhound.core.types.common import FileId


def test_parse_timeout_guard_applies_env(tmp_path: Path, monkeypatch):
    # Force a very small timeout to ensure the property is applied
    monkeypatch.setenv("CHUNKHOUND_PARSE_TIMEOUT_MS", "100")

    # Create a moderately nested YAML to exercise the parser quickly
    content = "a: [" + ",".join(str(i) for i in range(1000)) + "]\n"
    p = tmp_path / "t.yaml"
    p.write_text(content, encoding="utf-8")

    parser = create_parser_for_file(p)

    # Access internal parser timeout via engine if available
    engine = getattr(parser, "engine", None)
    parser_obj = getattr(engine, "_parser", None)
    timeout_us = getattr(parser_obj, "timeout_micros", None)

    # The guard should set a non-zero timeout in microseconds
    assert isinstance(timeout_us, int)
    assert 50_000 <= timeout_us <= 200_000  # around 100ms in microseconds

    # Also ensure parsing completes without exceptions
    chunks = parser.parse_file(p, FileId(0))
    assert isinstance(chunks, list)
