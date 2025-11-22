# Database Concurrency Operations

This note captures the current, experiment-backed stance on database concurrency for ChunkHound. It is informational for operators and maintainers; the production rule remains **single-owner, single-threaded access per database path** enforced via `SerialDatabaseProvider`.

## Summary

- **DuckDB**: Designed for a single owning process per database file. When a second process attempts to connect to the same `.duckdb` file, DuckDB fails fast with a locking error.
- **LanceDB**: In practice, can tolerate one writer process plus multiple reader processes, provided:
  - Each process uses its own LanceDB connection.
  - Processes are started with the `spawn` start method (not `fork`).
  - Only one process performs writes for a given database directory at any time.
- **ChunkHound policy**: Regardless of backend behavior, ChunkHound treats each database path as owned by a single process and routes all access through `SerialDatabaseProvider` to avoid lock contention and corruption.

## Experiments

Two standalone scripts under `operations/experiments/` probe the backends with ChunkHound-like chunk + embedding data:

- `operations/experiments/lancedb_concurrency_probe.py`
  - Seeds a LanceDB database with `files` and `chunks` tables, where `chunks.embedding` is a fixed-size float vector and `provider`/`model` columns mirror real usage.
  - Spawns **1 writer + N readers** using `multiprocessing.get_context("spawn")`.
  - Writer:
    - Re-embeds existing rows.
    - Inserts new rows with embeddings (new chunk IDs).
  - Readers:
    - Run vector search queries on the embedding column filtered by `provider`/`model`.
    - Count how often they see rows with `id > seed_rows` (freshly inserted data).
  - Observed behavior for representative runs (e.g., `--rows 5000 --readers 4 --writers 1 --duration 10`):
    - Non-zero `reads_ok` and `writes_ok`.
    - `fresh_read_iterations` and `fresh_read_results` > 0, meaning readers see new data while the writer runs.
    - `reads_err` and `writes_err` remained 0 in these tests.

- `operations/experiments/duckdb_concurrency_probe.py`
  - Seeds a DuckDB database file with a `chunks` table using an `embedding FLOAT[dims]` column and similar metadata columns.
  - Attempts the same **1 writer + N readers** pattern, again with `spawn`.
  - Writer connects and performs updates/inserts successfully.
  - Reader processes fail at connect time with:
    - `_duckdb.IOException: IO Error: Could not set lock on file "...": Conflicting lock is held ...`
  - Result: no successful concurrent reads while the writer holds the file lock.

These experiments are intentionally isolated: they do **not** use ChunkHound providers or services and should not be treated as production patterns.

## Operational Guidance

- Treat every configured database path as owned by a **single process** in production, regardless of backend (`duckdb` or `lancedb`).
- Rely on `SerialDatabaseProvider` and its single-threaded executor to:
  - Serialize all database operations inside the owning process.
  - Avoid concurrent writes and lock contention.
- If you need multiple MCP endpoints or CLIs:
  - Either route them through the same owning process (e.g., via HTTP/RPC).
  - Or give each process its own independent database path; do **not** share a single DuckDB/LanceDB directory across processes.
- Use the experimental probes only for investigation and regression checks, not as a justification to relax the single-owner rule in core ChunkHound paths.
