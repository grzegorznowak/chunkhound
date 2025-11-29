"""Split the monolithic agent doc into smaller, search-friendly files.

This script is intentionally lightweight and **does not** call any LLMs or
ChunkHound services directly. It simply reads:

    operations/chunkhound_agent_doc.md

and writes a set of focused markdown files under:

    operations/agent_docs/

The goal is to keep the original monolithic document as the single source of
truth while giving agents smaller, subsystem-scoped files that are easier to
search and load into context.

The monolithic doc is expected to have the stable heading structure enforced by
operations/agent_doc.py (sections 1–7 plus subsystem deep dives and Sources).
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any


SOURCE_DOC = Path("operations/chunkhound_agent_doc.md")


@dataclass
class SectionSpec:
    filename: str
    title: str
    description: str
    content: str


def _read_source() -> str:
    if not SOURCE_DOC.exists():
        raise SystemExit(
            f"Source agent doc not found at {SOURCE_DOC}. "
            "Generate it first with: make agent-doc"
        )
    return SOURCE_DOC.read_text(encoding="utf-8")


def _extract_meta(lines: list[str]) -> dict[str, Any]:
    """Extract simple metadata (created_from_sha, target_sha, generated_at, llm_config)."""
    meta: dict[str, Any] = {}
    llm_cfg: dict[str, str] = {}
    in_block = False
    in_llm = False

    for raw in lines:
        line = raw.strip()
        if line.startswith("agent_doc_metadata:"):
            in_block = True
            continue
        if not in_block:
            continue
        if line == "-->":
            break
        if not line:
            continue
        if line.startswith("llm_config:"):
            in_llm = True
            continue
        if ":" not in line:
            continue
        key, value = [part.strip() for part in line.split(":", 1)]
        if in_llm:
            llm_cfg[key] = value
        else:
            meta[key] = value

    if llm_cfg:
        meta["llm_config"] = llm_cfg
    return meta


def _write_sections_and_readme(
    sections: list[SectionSpec],
    meta: dict[str, Any],
    base_output_dir: Path | None = None,
) -> None:
    """Write section files and a README into the target agent_docs directory.

    This helper is shared between the legacy canonical splitter and the newer
    fluid splitter so that both produce a consistent on-disk layout and
    metadata summary.
    """

    # Determine output directory for split docs
    output_dir = (base_output_dir or Path("operations")).joinpath("agent_docs")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Best-effort cleanup: remove existing per-section markdown files (but keep
    # README) so that stale sections from previous runs do not linger when the
    # structure changes between canonical and fluid modes.
    for path in output_dir.glob("*.md"):
        if path.name == "README.md":
            continue
        try:
            path.unlink()
        except Exception:
            # Cleanup is best-effort; never fail splitting because a file
            # cannot be removed.
            pass

    # Write each section file with its existing headings preserved.
    for spec in sections:
        path = output_dir / spec.filename
        content = spec.content.strip()
        if not content:
            continue
        path.write_text(content + "\n", encoding="utf-8")

    # Generate a small README describing layout and metadata
    readme = output_dir / "README.md"
    created_from = meta.get("created_from_sha", "?")
    target_sha = meta.get("target_sha", "?")
    generated_at = meta.get("generated_at", "?")
    llm = meta.get("llm_config", {}) if isinstance(meta.get("llm_config"), dict) else {}

    synthesis_model = llm.get("synthesis_model", "?")
    utility_model = llm.get("utility_model", "?")

    lines = [
        "# Agent Docs (Split)",
        "",
        "These files are split from the monolithic agent document at "
        "`operations/chunkhound_agent_doc.md` to make it easier for agents to load "
        "and search focused topics instead of a single, very large document.",
        "",
        "Generation metadata (from the monolithic agent doc):",
        f"- created_from_sha: `{created_from}`",
        f"- target_sha: `{target_sha}`",
        f"- generated_at: `{generated_at}`",
        f"- llm.synthesis_model: `{synthesis_model}`",
        f"- llm.utility_model: `{utility_model}`",
        "",
        "## Layout",
        "",
    ]

    for spec in sections:
        path = output_dir / spec.filename
        if not path.exists():
            continue
        lines.append(f"- `{spec.filename}` – {spec.description}")

    readme.write_text("\n".join(lines) + "\n", encoding="utf-8")


def split_agent_doc_text_fluid(
    text: str,
    meta: dict[str, Any],
    base_output_dir: Path | None = None,
) -> None:
    """Split an agent doc body into sections based on actual headings.

    This variant is intended for the more fluid map-only merge mode where the
    document structure is not constrained to a fixed 1–6 heading schema. It
    groups content by `##` headings (plus a dedicated Sources file when
    present) and writes one markdown file per section.
    """

    lines = text.splitlines()

    sections_raw: list[tuple[str, list[str]]] = []
    current_title: str | None = None
    current_buf: list[str] = []

    for raw in lines:
        line = raw.rstrip("\n")
        stripped = line.strip()

        # Start a new section on each second-level heading.
        if stripped.startswith("## "):
            if current_title is not None and current_buf:
                sections_raw.append((current_title, current_buf.copy()))
            current_title = stripped[3:].strip()
            current_buf = [line]
        else:
            if current_title is not None:
                current_buf.append(line)

    if current_title is not None and current_buf:
        sections_raw.append((current_title, current_buf.copy()))

    if not sections_raw:
        # Nothing to split; fall back to a single synthetic section.
        sections = [
            SectionSpec(
                filename="01_agent_doc_full.md",
                title="Agent Doc (Full)",
                description="Full agent doc content (no section headings detected).",
                content=text,
            )
        ]
        _write_sections_and_readme(sections, meta, base_output_dir=base_output_dir)
        return

    def _slugify(title: str) -> str:
        # Basic, grep-friendly slug suitable for filenames.
        slug = title.strip().lower()
        # Replace non-alphanumeric characters with underscores.
        cleaned_chars = []
        for ch in slug:
            if ch.isalnum():
                cleaned_chars.append(ch)
            else:
                cleaned_chars.append("_")
        slug = "".join(cleaned_chars)
        # Collapse multiple underscores.
        while "__" in slug:
            slug = slug.replace("__", "_")
        slug = slug.strip("_")
        if not slug:
            slug = "section"
        # Keep filenames reasonably short.
        return slug[:60]

    sections: list[SectionSpec] = []
    index = 1
    sources_spec: SectionSpec | None = None

    for title, buf in sections_raw:
        content = "\n".join(buf).strip()
        if not content:
            continue

        # Treat Sources as a dedicated late-numbered section for consistency
        # with the canonical splitter.
        if title.lower().startswith("sources"):
            sources_spec = SectionSpec(
                filename="99_sources.md",
                title=title,
                description=(
                    "List of files and chunks that fed the agent doc synthesis."
                ),
                content=content,
            )
            continue

        slug = _slugify(title)
        filename = f"{index:02d}_{slug}.md"
        sections.append(
            SectionSpec(
                filename=filename,
                title=title,
                description=f"Section extracted from Agent Doc: {title}",
                content=content,
            )
        )
        index += 1

    if sources_spec is not None:
        sections.append(sources_spec)

    _write_sections_and_readme(sections, meta, base_output_dir=base_output_dir)


def split_agent_doc() -> None:
    """CLI entrypoint: read the monolithic doc from disk and split it."""
    text = _read_source()
    lines = text.splitlines()
    meta = _extract_meta(lines)
    # Always use the fluid splitter; the legacy canonical splitter with
    # hard-coded ChunkHound subsystems has been removed.
    split_agent_doc_text_fluid(text, meta, base_output_dir=Path("operations"))


def main() -> None:
    split_agent_doc()


if __name__ == "__main__":
    main()
