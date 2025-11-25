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


def _slice(text: str, start_marker: str, end_marker: str | None) -> str:
    start = text.find(start_marker)
    if start == -1:
        raise SystemExit(f"Expected heading not found: {start_marker!r}")
    if end_marker is None:
        end = len(text)
    else:
        end = text.find(end_marker, start)
        if end == -1:
            end = len(text)
    return text[start:end].strip() + "\n"


def _parse_h3_block(block: str) -> tuple[str, dict[str, str]]:
    """Split a '## ...' block into prefix and per-### sections."""
    lines = block.splitlines()
    prefix_lines: list[str] = []
    sections: dict[str, str] = {}
    current_title: str | None = None
    buf: list[str] = []

    for raw in lines:
        line = raw.rstrip("\n")
        if line.startswith("### "):
            if current_title is None:
                prefix_lines = buf.copy()
            else:
                sections[current_title] = "\n".join(buf).strip() + "\n"
            current_title = line[4:].strip()
            buf = [line]
        else:
            buf.append(line)

    if current_title is not None and buf:
        sections[current_title] = "\n".join(buf).strip() + "\n"

    prefix = "\n".join(prefix_lines).strip() + "\n" if prefix_lines else ""
    return prefix, sections


def split_agent_doc_text(text: str, meta: dict[str, Any], base_output_dir: Path | None = None) -> None:
    """Split a given agent doc body into section files and README.

    This function is tolerant of partial documents (e.g., HyDE-only runs that
    may omit some headings). Missing sections simply result in missing files
    rather than hard failures.
    """

    def maybe_slice(start_marker: str, end_marker: str | None) -> str:
        if start_marker not in text:
            return ""
        return _slice(text, start_marker, end_marker)

    # Locate top-level sections (best-effort)
    identity = maybe_slice(
        "## 1. Project Identity and Constraints",
        "## 2. Architecture Overview",
    )
    architecture = maybe_slice(
        "## 2. Architecture Overview",
        "## 3. Subsystem Guides",
    )
    subsystems_block = maybe_slice(
        "## 3. Subsystem Guides",
        "## 4. Operations, Commands, and Testing",
    )
    operations_cmds = maybe_slice(
        "## 4. Operations, Commands, and Testing",
        "## 5. Known Pitfalls and Debugging Patterns",
    )
    pitfalls = maybe_slice(
        "## 5. Known Pitfalls and Debugging Patterns",
        "## 7. Subsystem Deep Dives",
    )
    deep_dives_block = maybe_slice(
        "## 7. Subsystem Deep Dives",
        "## Sources",
    )
    sources = maybe_slice("## Sources", None)

    # Split subsystem guides and deep dives by ### headings
    subsys_prefix, subsys_sections = _parse_h3_block(subsystems_block)
    _, deep_sections = _parse_h3_block(deep_dives_block)

    # Compose per-subsystem content: guide + matching deep dive (when present)
    def combine(name: str, deep_key: str | None) -> str:
        parts: list[str] = []
        guide = subsys_sections.get(name)
        if guide:
            parts.append(guide.strip())
        if deep_key:
            deep = deep_sections.get(deep_key)
            if deep:
                parts.append(deep.strip())
        return "\n\n".join(parts).strip() + "\n"

    # Define output sections with filenames and descriptions
    sections: list[SectionSpec] = [
        SectionSpec(
            filename="01_project_identity_and_constraints.md",
            title="Project Identity and Constraints",
            description="Global identity, critical constraints, and performance guardrails.",
            content=identity,
        ),
        SectionSpec(
            filename="02_architecture_overview.md",
            title="Architecture Overview",
            description="High-level bootstrap sequence and layer relationships.",
            content=architecture,
        ),
        SectionSpec(
            filename="03_subsystems_overview.md",
            title="Subsystem Guides Overview",
            description="Overview text preceding per-subsystem guides.",
            content=subsys_prefix,
        ),
        SectionSpec(
            filename="04_subsystem_core.md",
            title="Core Subsystem",
            description="Config/models behavior and deep reasoning about the core layer.",
            content=combine("Core", "Core Configuration and Models – Deep Dive"),
        ),
        SectionSpec(
            filename="05_subsystem_providers.md",
            title="Providers Subsystem",
            description="Database and embedding providers, SerialDatabaseProvider, and related deep dive.",
            content=combine(
                "Providers",
                "Database and Embedding Providers – Deep Dive",
            ),
        ),
        SectionSpec(
            filename="06_subsystem_services.md",
            title="Services Subsystem",
            description="Indexing, search, embedding, and deep research services with their deep dive.",
            content=combine(
                "Services",
                "Services Layer and Deep Research – Deep Dive",
            ),
        ),
        SectionSpec(
            filename="07_subsystem_interfaces.md",
            title="Interfaces Subsystem",
            description="DatabaseProvider/EmbeddingProvider contracts and their role in the system.",
            content=combine("Interfaces", None),
        ),
        SectionSpec(
            filename="08_subsystem_api_cli.md",
            title="API / CLI Subsystem",
            description="CLI entrypoints, parsers, commands and their deep dive.",
            content=combine(
                "API / CLI",
                "CLI Surface Area – Deep Dive",
            ),
        ),
        SectionSpec(
            filename="09_subsystem_mcp_server.md",
            title="MCP Server Subsystem",
            description="MCP servers, transports, and tool registry with deep dive.",
            content=combine(
                "MCP Server",
                "MCP Servers and Tool Registry – Deep Dive",
            ),
        ),
        SectionSpec(
            filename="10_subsystem_utils.md",
            title="Utils Subsystem",
            description="Utility helpers and support modules referenced by other layers.",
            content=combine("Utils", None),
        ),
        SectionSpec(
            filename="11_subsystem_operations.md",
            title="Operations Subsystem",
            description="Operations docs, experiments, and agent-doc generation flow with deep dive.",
            content=combine(
                "Operations",
                "Operations, Experiments, and Agent Doc Flow – Deep Dive",
            ),
        ),
        SectionSpec(
            filename="12_subsystem_tests.md",
            title="Tests Subsystem",
            description="Test suite roles and how they enforce the architecture.",
            content=combine("Tests", None),
        ),
        SectionSpec(
            filename="20_operations_commands_and_testing.md",
            title="Operations, Commands, and Testing",
            description="CLI commands, testing strategy, and operational workflows.",
            content=operations_cmds,
        ),
        SectionSpec(
            filename="22_known_pitfalls_and_debugging_patterns.md",
            title="Known Pitfalls and Debugging Patterns",
            description="Common failure modes and how to debug them.",
            content=pitfalls,
        ),
        SectionSpec(
            filename="99_sources.md",
            title="Sources",
            description="List of files and chunks that fed the agent doc synthesis.",
            content=sources,
        ),
    ]

    # Determine output directory for split docs
    output_dir = (base_output_dir or Path("operations")).joinpath("agent_docs")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Write each section file with its existing headings preserved
    for spec in sections:
        path = output_dir / spec.filename
        # Only write non-empty content
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
        # Skip files that may not have been written (empty content)
        path = output_dir / spec.filename
        if not path.exists():
            continue
        lines.append(f"- `{spec.filename}` – {spec.description}")

    readme.write_text("\n".join(lines) + "\n", encoding="utf-8")


def split_agent_doc() -> None:
    """CLI entrypoint: read the monolithic doc from disk and split it."""
    text = _read_source()
    lines = text.splitlines()
    meta = _extract_meta(lines)
    split_agent_doc_text(text, meta, base_output_dir=Path("operations"))


def main() -> None:
    split_agent_doc()


if __name__ == "__main__":
    main()
