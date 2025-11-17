"""
Unit test for realtime poke filtering.

Verifies that poke_for_recent_files enqueues only source files according to
include patterns and ignores internal paths like .chunkhound/ and .git/ as
well as files that are not matched by include patterns.
"""

import asyncio
from pathlib import Path

from chunkhound.core.config.config import Config
from chunkhound.core.config.indexing_config import IndexingConfig
from chunkhound.services.realtime_indexing_service import RealtimeIndexingService


def test_poke_filters_ignores_non_source_files(tmp_path: Path) -> None:
    async def _run() -> None:
        # Arrange: create a mix of files
        (tmp_path / ".chunkhound").mkdir(exist_ok=True)
        (tmp_path / ".git").mkdir(exist_ok=True)

        (tmp_path / ".chunkhound" / "test.db").write_text("db")
        (tmp_path / ".chunkhound" / "test.db.wal").write_text("wal")
        (tmp_path / ".git" / "config").write_text("[core]")
        (tmp_path / "notes.txt").write_text("ignore me")
        code_file = tmp_path / "main.py"
        code_file.write_text("print('ok')\n")

        # Config with explicit include to keep scope tight
        cfg = Config(target_dir=tmp_path, indexing=IndexingConfig(include=["**/*.py"]))

        svc = RealtimeIndexingService(services=None, config=cfg, debug_sink=None)
        # Set watch_path directly; we don't need to start the watcher thread
        svc.watch_path = tmp_path

        # Act: poke for recent files (enable burst to bypass debounce)
        svc.enable_burst_mode(2.0)
        await svc.poke_for_recent_files(seconds=60.0)

        # Assert: only the .py file should be queued
        queued_paths: list[Path] = []
        while not svc.file_queue.empty():
            prio, fp = await svc.file_queue.get()
            queued_paths.append(Path(fp))

        queued_names = {p.name for p in queued_paths}
        assert "main.py" in queued_names
        assert "test.db" not in queued_names
        assert "test.db.wal" not in queued_names
        assert "config" not in queued_names
        assert "notes.txt" not in queued_names

    asyncio.run(_run())
