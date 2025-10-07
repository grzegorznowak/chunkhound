"""Batch file processor for parallel processing across CPU cores.

# FILE_CONTEXT: Worker function for ProcessPoolExecutor to parse files in parallel
# ROLE: Performs CPU-bound read→parse→chunk pipeline independently per batch
# CRITICAL: Must be picklable (top-level function, serializable arguments)
"""

import os
from dataclasses import dataclass
from typing import Any
from pathlib import Path

from chunkhound.core.types.common import FileId, Language
from chunkhound.parsers.parser_factory import create_parser_for_language


@dataclass
class ParsedFileResult:
    """Result from processing a single file in a batch."""

    file_path: Path
    chunks: list[dict]
    language: Language
    file_size: int
    file_mtime: float
    status: str
    error: str | None = None


def process_file_batch(
    file_paths: list[Path],
    config_dict: dict,
    progress_queue: Any | None = None,
    worker_index: int | None = None,
) -> list[ParsedFileResult]:
    """Process a batch of files in a worker process.

    This function runs in a separate process via ProcessPoolExecutor.
    Performs the complete read→parse→chunk pipeline for all files in the batch.

    Args:
        file_paths: List of file paths to process in this batch
        config_dict: Serialized configuration dictionary for parser initialization

    Returns:
        List of ParsedFileResult objects with parsed chunks and metadata
    """
    results = []

    for file_path in file_paths:
        try:
            # Emit progress signal before heavy work begins (safe, best-effort)
            if progress_queue is not None:
                try:
                    # Prefer put_nowait; fallback to put(block=False) for Manager proxies
                    if hasattr(progress_queue, "put_nowait"):
                        progress_queue.put_nowait(("start", worker_index, str(file_path)))
                    else:
                        progress_queue.put(("start", worker_index, str(file_path)), block=False)
                except Exception:
                    # Progress is best-effort; ignore cross-process issues
                    pass
            # Get file metadata
            file_stat = os.stat(file_path)

            # Detect language from file extension
            language = Language.from_file_extension(file_path)
            if language == Language.UNKNOWN:
                # Progress accounting for phases even if we skip
                if progress_queue is not None:
                    try:
                        if hasattr(progress_queue, "put_nowait"):
                            progress_queue.put_nowait(("read_done", worker_index, str(file_path)))
                            progress_queue.put_nowait(("parsed", worker_index, str(file_path)))
                        else:
                            progress_queue.put(("read_done", worker_index, str(file_path)), block=False)
                            progress_queue.put(("parsed", worker_index, str(file_path)), block=False)
                    except Exception:
                        pass
                results.append(
                    ParsedFileResult(
                        file_path=file_path,
                        chunks=[],
                        language=language,
                        file_size=file_stat.st_size,
                        file_mtime=file_stat.st_mtime,
                        status="skipped",
                        error="Unknown file type",
                    )
                )
                continue

            # Skip large config/data files (config files are typically < 20KB)
            if language.is_structured_config_language:
                file_size_kb = file_stat.st_size / 1024
                threshold_kb = config_dict.get("config_file_size_threshold_kb", 20)
                if file_size_kb > threshold_kb:
                    # Account for progress without parsing
                    if progress_queue is not None:
                        try:
                            if hasattr(progress_queue, "put_nowait"):
                                progress_queue.put_nowait(("read_done", worker_index, str(file_path)))
                                progress_queue.put_nowait(("parsed", worker_index, str(file_path)))
                            else:
                                progress_queue.put(("read_done", worker_index, str(file_path)), block=False)
                                progress_queue.put(("parsed", worker_index, str(file_path)), block=False)
                        except Exception:
                            pass
                    results.append(
                        ParsedFileResult(
                            file_path=file_path,
                            chunks=[],
                            language=language,
                            file_size=file_stat.st_size,
                            file_mtime=file_stat.st_mtime,
                            status="skipped",
                            error="large_config_file",
                        )
                    )
                    continue

            # Create parser for this language
            parser = create_parser_for_language(language)
            if not parser:
                if progress_queue is not None:
                    try:
                        if hasattr(progress_queue, "put_nowait"):
                            progress_queue.put_nowait(("read_done", worker_index, str(file_path)))
                            progress_queue.put_nowait(("parsed", worker_index, str(file_path)))
                        else:
                            progress_queue.put(("read_done", worker_index, str(file_path)), block=False)
                            progress_queue.put(("parsed", worker_index, str(file_path)), block=False)
                    except Exception:
                        pass
                results.append(
                    ParsedFileResult(
                        file_path=file_path,
                        chunks=[],
                        language=language,
                        file_size=file_stat.st_size,
                        file_mtime=file_stat.st_mtime,
                        status="error",
                        error=f"No parser available for {language}",
                    )
                )
                continue

            # Read content and emit read completion
            try:
                content = file_path.read_text(encoding="utf-8", errors="ignore")
            except Exception:
                try:
                    content = file_path.read_bytes().decode("utf-8", errors="ignore")
                except Exception:
                    content = ""

            if progress_queue is not None:
                try:
                    if hasattr(progress_queue, "put_nowait"):
                        progress_queue.put_nowait(("read_done", worker_index, str(file_path)))
                    else:
                        progress_queue.put(("read_done", worker_index, str(file_path)), block=False)
                except Exception:
                    pass

            # Signal parse start for UI separation
            if progress_queue is not None:
                try:
                    if hasattr(progress_queue, "put_nowait"):
                        progress_queue.put_nowait(("parse_start", worker_index, str(file_path)))
                    else:
                        progress_queue.put(("parse_start", worker_index, str(file_path)), block=False)
                except Exception:
                    pass

            # Parse content and generate chunks
            # Note: FileId(0) is placeholder - actual ID assigned during storage
            chunks = parser.parse_content(content, file_path, FileId(0))

            # Convert chunks to dictionaries for ProcessPoolExecutor serialization
            # Using standard Chunk.to_dict() method for consistent serialization
            chunks_data = [chunk.to_dict() for chunk in chunks]

            # Signal parse completion
            if progress_queue is not None:
                try:
                    if hasattr(progress_queue, "put_nowait"):
                        progress_queue.put_nowait(("parsed", worker_index, str(file_path)))
                    else:
                        progress_queue.put(("parsed", worker_index, str(file_path)), block=False)
                except Exception:
                    pass

            results.append(
                ParsedFileResult(
                    file_path=file_path,
                    chunks=chunks_data,
                    language=language,
                    file_size=file_stat.st_size,
                    file_mtime=file_stat.st_mtime,
                    status="success",
                )
            )

        except Exception as e:
            # Capture errors but continue processing other files in batch
            results.append(
                ParsedFileResult(
                    file_path=file_path,
                    chunks=[],
                    language=Language.UNKNOWN,
                    file_size=0,
                    file_mtime=0.0,
                    status="error",
                    error=str(e),
                )
            )
        finally:
            if progress_queue is not None:
                try:
                    if hasattr(progress_queue, "put_nowait"):
                        progress_queue.put_nowait(("done", worker_index, str(file_path)))
                    else:
                        progress_queue.put(("done", worker_index, str(file_path)), block=False)
                except Exception:
                    # Drop on full/closed
                    pass

    return results
