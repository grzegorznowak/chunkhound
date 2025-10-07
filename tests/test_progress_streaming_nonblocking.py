import os
import time
import asyncio
from pathlib import Path

import pytest
from rich.progress import Progress

from chunkhound.services.indexing_coordinator import IndexingCoordinator
from chunkhound.core.types.common import Language, FilePath
from chunkhound.core.models.file import File


class _FakeDB:
    def __init__(self, base_dir: Path):
        self._base = base_dir
        self._files_by_path: dict[str, dict] = {}
        self._chunks_by_file_id: dict[int, list] = {}
        self._next_file_id = 1
        self._next_chunk_id = 1
        self._config = None

    # Transaction no-ops
    def begin_transaction(self):
        pass

    def commit_transaction(self):
        pass

    def rollback_transaction(self):
        pass

    # File ops
    def get_file_by_path(self, rel_path: str):
        return self._files_by_path.get(rel_path)

    def insert_file(self, file_model: File) -> int:
        file_id = self._next_file_id
        self._next_file_id += 1
        rel_path = str(file_model.path)
        self._files_by_path[rel_path] = {
            "id": file_id,
            "path": rel_path,
            "mtime": file_model.mtime,
            "size_bytes": file_model.size_bytes,
            "language": file_model.language.value,
        }
        self._chunks_by_file_id[file_id] = []
        return file_id

    def update_file(self, file_id: int, size_bytes: int, mtime: float):
        # Find by id and update (best-effort for test)
        for rec in self._files_by_path.values():
            if rec["id"] == file_id:
                rec["size_bytes"] = size_bytes
                rec["mtime"] = mtime
                break

    # Chunk ops
    def insert_chunks_batch(self, chunk_models: list) -> list[int]:
        ids = []
        for cm in chunk_models:
            cid = self._next_chunk_id
            self._next_chunk_id += 1
            ids.append(cid)
            # Attach to file_id from the model
            fid = cm.file_id
            self._chunks_by_file_id.setdefault(fid, []).append(cm)
        return ids

    def get_chunks_by_file_id(self, file_id: int, as_model: bool = False):
        return list(self._chunks_by_file_id.get(file_id, []))

    def delete_chunk(self, chunk_id: int):
        for fid, lst in self._chunks_by_file_id.items():
            for i, cm in enumerate(lst):
                # Our simple model doesn't carry IDs in tests; treat as no-op
                pass

    # Cleanup helpers used by coordinator
    def delete_file_completely(self, rel_path: str) -> bool:
        rec = self._files_by_path.pop(rel_path, None)
        return rec is not None

    def execute_query(self, query: str, params: list):
        # Support the simple SELECT id,path FROM files used in cleanup
        return list(self._files_by_path.values())

    def optimize_tables(self):
        pass

    # Stats
    def get_stats(self) -> dict:
        return {
            "files": len(self._files_by_path),
            "chunks": sum(len(v) for v in self._chunks_by_file_id.values()),
            "embeddings": 0,
        }


def _write_yaml_files(base: Path, n: int = 100):
    base.mkdir(parents=True, exist_ok=True)
    for i in range(n):
        (base / f"f{i:04d}.yaml").write_text(
            f"a: {i}\nlist: [1,2,3]\nmap: {{x: {i}, y: {i+1}}}\n",
            encoding="utf-8",
        )


def test_progress_streaming_nonblocking_large_batch(tmp_path: Path, monkeypatch):
    # Ensure streaming is enabled (kill-switch off)
    os.environ.pop("CHUNKHOUND_NO_PARSE_PROGRESS", None)

    # Prepare many small YAML files to overflow the progress queue capacity
    _write_yaml_files(tmp_path, n=100)

    db = _FakeDB(tmp_path)

    async def run():
        # Single progress bar (as in CLI)
        with Progress() as progress:
            coord = IndexingCoordinator(
                database_provider=db,
                base_directory=tmp_path,
                embedding_provider=None,
                language_parsers=None,
                progress=progress,
                config=None,
            )

            t0 = time.perf_counter()
            res = await coord.process_directory(
                tmp_path,
                patterns=["**/*.yaml"],
                exclude_patterns=[],
                config_file_size_threshold_kb=20,
            )
            dt = time.perf_counter() - t0

            # Should complete successfully and reasonably fast (no deadlock)
            assert res["status"] in ("success", "complete")
            assert res.get("files_processed", 0) == 100
            assert res.get("total_chunks", 0) >= 0
            assert dt < 30, f"Indexing took too long: {dt:.1f}s"
    # Monkeypatch the Manager so that Manager().Queue(maxsize=512) becomes size 1
    import multiprocessing as mp

    class _SmallManager:
        def __init__(self):
            self._mgr = mp.Manager()

        def Queue(self, maxsize=0):  # ignore requested maxsize
            return self._mgr.Queue(1)

        def shutdown(self):
            self._mgr.shutdown()

    monkeypatch.setattr(
        "chunkhound.services.indexing_coordinator.multiprocessing.Manager",
        lambda: _SmallManager(),
        raising=True,
    )

    asyncio.run(run())
