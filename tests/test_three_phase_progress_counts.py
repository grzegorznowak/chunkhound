import asyncio
from pathlib import Path

from rich.progress import Progress

from chunkhound.services.indexing_coordinator import IndexingCoordinator


class _FakeDBMini:
    def __init__(self, base: Path):
        self._base = base
        self._files = {}
        self._chunks = {}
        self._next_file_id = 1

    def begin_transaction(self):
        pass

    def commit_transaction(self):
        pass

    def rollback_transaction(self):
        pass

    def get_file_by_path(self, path: str):
        return self._files.get(path)

    def insert_file(self, file_model):
        fid = self._next_file_id
        self._next_file_id += 1
        self._files[file_model.path] = {"id": fid, "path": file_model.path}
        self._chunks[fid] = []
        return fid

    def update_file(self, *args, **kwargs):
        pass

    def get_chunks_by_file_id(self, file_id: int, as_model: bool = False):
        return list(self._chunks.get(file_id, []))

    def insert_chunks_batch(self, chunk_models):
        ids = []
        for cm in chunk_models:
            ids.append(len(ids) + 1)
        return ids

    def delete_chunk(self, chunk_id: int):
        pass

    def execute_query(self, q, p):
        # Used by cleanup; return current files list
        return [{"id": v["id"], "path": k} for k, v in self._files.items()]

    def delete_file_completely(self, rel_path: str) -> bool:
        return self._files.pop(rel_path, None) is not None

    def optimize_tables(self):
        pass

    def get_stats(self):
        return {"files": len(self._files), "chunks": 0, "embeddings": 0}


def _write_files(base: Path, n: int):
    base.mkdir(parents=True, exist_ok=True)
    for i in range(n):
        (base / f"f{i:04d}.py").write_text(f"def f{i}():\n    return {i}\n", encoding="utf-8")


def test_three_phase_progress_counts(tmp_path: Path):
    _write_files(tmp_path, 20)
    db = _FakeDBMini(tmp_path)

    async def run():
        with Progress() as progress:
            coord = IndexingCoordinator(
                database_provider=db,
                base_directory=tmp_path,
                embedding_provider=None,
                language_parsers=None,
                progress=progress,
            )

            await coord.process_directory(
                tmp_path, patterns=["**/*.py"], exclude_patterns=[]
            )

            # Find tasks by description
            read = next(t for t in progress.tasks if "Reading files" in t.description)
            calc = next(t for t in progress.tasks if "Calculating chunks" in t.description)

            assert read.completed == read.total == 20
            assert calc.completed == calc.total == 20

    asyncio.run(run())

