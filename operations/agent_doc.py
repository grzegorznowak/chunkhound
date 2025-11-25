"""Generate a single, agent-optimized documentation file for a scoped folder.

This script uses ChunkHound's deep research and search pipeline to build an
LLM-friendly architecture and operations document for any folder under a
configured workspace. It assumes:

- A ChunkHound workspace database already exists for the target path
  (run `chunkhound index <workspace-root>`).
- Embedding and LLM providers are configured in `.chunkhound.json`.

On first run for a given output file, it records the current commit SHA as the
documentation baseline. On subsequent runs, it reads the embedded metadata,
computes diffs between the original baseline and the current HEAD (if Git is
available), and asks the research pipeline to generate an updated document that
reflects the current state and summarizes changes.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import subprocess
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from chunkhound.core.config.config import Config
from chunkhound.core.config.embedding_factory import EmbeddingProviderFactory
from chunkhound.database_factory import create_services
from chunkhound.embeddings import EmbeddingManager
from chunkhound.llm_manager import LLMManager
from chunkhound.mcp_server.tools import deep_research_impl
from chunkhound.services.research.citation_manager import CitationManager
from operations.split_agent_doc import split_agent_doc_text


DEFAULT_OUTPUT_RELATIVE = Path("operations/chunkhound_agent_doc.md")


@dataclass
class AgentDocMetadata:
    """Commit metadata used to drive incremental regeneration."""

    created_from_sha: str
    previous_target_sha: str
    target_sha: str
    generated_at: str
    llm_config: dict[str, str] = field(default_factory=dict)
    generation_stats: dict[str, str] = field(default_factory=dict)


def _run_git(args: list[str], cwd: Path) -> str:
    """Run a git command and return stdout as text.

    This helper is intentionally forgiving: callers are expected to handle
    failures gracefully for workspaces that are not Git repositories.
    """
    result = subprocess.run(
        ["git", *args],
        cwd=str(cwd),
        text=True,
        capture_output=True,
    )
    if result.returncode != 0:
        raise subprocess.CalledProcessError(
            result.returncode, result.args, output=result.stdout, stderr=result.stderr
        )
    return result.stdout.strip()


def _get_head_sha(project_root: Path) -> str:
    """Return the current HEAD SHA for the project.

    For workspaces that are not Git repositories, this returns a stable
    placeholder instead of failing hard.
    """
    try:
        return _run_git(["rev-parse", "HEAD"], project_root)
    except subprocess.CalledProcessError:
        # Non-git workspace: downstream logic treats this as "no git metadata"
        return "NO_GIT_HEAD"


def _get_name_status_diff(base_sha: str, target_sha: str, project_root: Path) -> list[str]:
    """Return `git diff --name-status base..target` as a list of lines.

    For non-git workspaces, this returns an empty list.
    """
    if base_sha == target_sha:
        return []

    try:
        output = _run_git(
            ["diff", "--name-status", f"{base_sha}..{target_sha}"], project_root
        )
    except subprocess.CalledProcessError:
        return []

    lines = [line.strip() for line in output.splitlines() if line.strip()]
    return lines


def _parse_existing_metadata(doc_path: Path) -> AgentDocMetadata | None:
    """Parse the metadata comment block from an existing agent doc, if present.

    Expected format at the top of the file:

    <!--
    agent_doc_metadata:
      created_from_sha: <AAAA>
      previous_target_sha: <BBBB>
      target_sha: <CCCC>
      generated_at: 2025-11-22T...
    -->
    """
    if not doc_path.exists():
        return None

    created_from_sha: str | None = None
    previous_target_sha: str | None = None
    target_sha: str | None = None
    generated_at: str | None = None
    llm_cfg: dict[str, str] = {}
    gen_stats: dict[str, str] = {}

    in_metadata = False
    in_llm_block = False
    in_gen_stats_block = False
    lines_read = 0

    with doc_path.open("r", encoding="utf-8") as f:
        for raw_line in f:
            line = raw_line.strip()
            lines_read += 1

            # Stop scanning after a small header window
            if lines_read > 80:
                break

            if "agent_doc_metadata:" in line:
                in_metadata = True
                continue

            if in_metadata:
                if "-->" in line:
                    break
                if ":" not in line:
                    continue
                key, value = line.split(":", 1)
                key = key.strip()
                value = value.strip()
                if key == "llm_config":
                    in_llm_block = True
                    in_gen_stats_block = False
                    continue
                if key == "generation_stats":
                    in_gen_stats_block = True
                    in_llm_block = False
                    continue
                if in_llm_block:
                    llm_cfg[key] = value
                elif in_gen_stats_block:
                    gen_stats[key] = value
                elif key == "created_from_sha":
                    created_from_sha = value
                elif key == "previous_target_sha":
                    previous_target_sha = value
                elif key == "target_sha":
                    target_sha = value
                elif key == "generated_at":
                    generated_at = value

    if not created_from_sha or not target_sha:
        return None

    if not previous_target_sha:
        previous_target_sha = target_sha

    if not generated_at:
        generated_at = ""

    return AgentDocMetadata(
        created_from_sha=created_from_sha,
        previous_target_sha=previous_target_sha,
        target_sha=target_sha,
        generated_at=generated_at,
        llm_config=llm_cfg,
        generation_stats=gen_stats,
    )


def _format_metadata_block(meta: AgentDocMetadata) -> str:
    """Render the metadata comment block."""
    lines = [
        "<!--",
        "agent_doc_metadata:",
        f"  created_from_sha: {meta.created_from_sha}",
        f"  previous_target_sha: {meta.previous_target_sha}",
        f"  target_sha: {meta.target_sha}",
        f"  generated_at: {meta.generated_at}",
    ]
    if meta.llm_config:
        lines.append("  llm_config:")
        for key, value in meta.llm_config.items():
            lines.append(f"    {key}: {value}")
    if meta.generation_stats:
        lines.append("  generation_stats:")
        for key, value in meta.generation_stats.items():
            lines.append(f"    {key}: {value}")
    lines.append("-->")
    return "\n".join(lines) + "\n\n"


def _split_sources_footer(body: str) -> tuple[str, str | None]:
    """Split a deep-research answer into (main_body, sources_footer).

    Deep research synthesis appends a Sources footer that starts with a
    '## Sources' heading (usually preceded by a '---' separator). When we run
    a scope-trimming pass, the LLM may drop that footer. To keep behavior
    consistent, we extract it from the raw answer and re-attach it after
    trimming if necessary.
    """
    if not body:
        return body, None

    lines = body.splitlines()
    last_sources_idx: int | None = None

    for idx, line in enumerate(lines):
        if line.strip().startswith("## Sources"):
            last_sources_idx = idx

    if last_sources_idx is None:
        return body, None

    start_idx = last_sources_idx
    if (
        last_sources_idx >= 2
        and lines[last_sources_idx - 2].strip() == "---"
        and lines[last_sources_idx - 1].strip() == ""
    ):
        start_idx = last_sources_idx - 2

    footer_lines = lines[start_idx:]
    body_lines = lines[:start_idx]

    footer = "\n".join(footer_lines).strip()
    main_body = "\n".join(body_lines).rstrip()

    if not footer:
        return body, None

    return main_body, footer


def _extract_hyde_bullets(hyde_plan: str | None, max_bullets: int) -> list[str]:
    """Extract bullet-shaped HyDE lines as generic queries.

    This mirrors the heuristic used in semantic mode so that both semantic and
    code_research flows interpret the HyDE scaffold in the same way.
    """
    if not hyde_plan:
        return []

    bullets: list[str] = []
    for raw in hyde_plan.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if not line.startswith(("-", "*")):
            continue
        text = line.lstrip("-* ").strip()
        if len(text) < 20:
            continue
        bullets.append(text)
        if len(bullets) >= max_bullets:
            break
    return bullets


def _extract_hyde_sections(hyde_plan: str, max_sections: int) -> list[dict[str, str]]:
    """Extract grouped HyDE sections (title + body) for map passes.

    Sections are derived from the 'Global Hooks' and 'Subsystem Hooks' parts of
    the HyDE plan and are intentionally coarse-grained. This provides an
    alternative to per-bullet queries so we can run code_research on richer,
    paragraph-shaped contexts.
    """
    lines = hyde_plan.splitlines()
    sections: list[dict[str, str]] = []

    in_hooks = False
    current_title: str | None = None
    current_body: list[str] = []

    def flush() -> None:
        nonlocal current_title, current_body
        if current_title and current_body:
            sections.append(
                {
                    "title": current_title.strip(),
                    "body": "\n".join(current_body).strip(),
                }
            )
        current_title = None
        current_body = []

    for raw in lines:
        stripped = raw.rstrip("\n")
        l = stripped.strip()

        # Enter/exit hooks region based on top-level headings
        if l.startswith("## "):
            heading = l[3:].strip()
            if heading in {"Global Hooks", "Subsystem Hooks"}:
                # Starting hooks region; flush any prior section
                flush()
                in_hooks = True
                continue
            else:
                # Leaving hooks region
                if in_hooks:
                    flush()
                    in_hooks = False
                continue

        if not in_hooks:
            continue

        # Subsection title within hooks
        if l.startswith("### "):
            # New section; flush previous one
            flush()
            current_title = l[4:].strip()
            current_body = []
            continue

        # Accumulate body lines (bullets or text) when inside hooks region
        if l:
            # If no explicit subsection title, synthesize one from first bullet
            if current_title is None and (l.startswith("-") or l.startswith("*")):
                current_title = l.lstrip("-* ").strip()[:80]
            current_body.append(stripped)

        # Stop if we've collected enough sections
        if len(sections) >= max_sections:
            break

    # Flush last section
    if len(sections) < max_sections:
        flush()

    return sections[:max_sections]


def _split_sources_footer(body: str) -> tuple[str, str | None]:
    """Split a deep-research answer into (main_body, sources_footer).

    Deep research synthesis appends a Sources footer that starts with a
    '## Sources' heading (usually preceded by a '---' separator). When we run
    a scope-trimming pass, the LLM may drop that footer. To keep behavior
    consistent, we extract it from the raw answer and re-attach it after
    trimming if necessary.
    """
    if not body:
        return body, None

    lines = body.splitlines()
    last_sources_idx: int | None = None

    for idx, line in enumerate(lines):
        if line.strip().startswith("## Sources"):
            last_sources_idx = idx

    if last_sources_idx is None:
        return body, None

    start_idx = last_sources_idx
    if (
        last_sources_idx >= 2
        and lines[last_sources_idx - 2].strip() == "---"
        and lines[last_sources_idx - 1].strip() == ""
    ):
        start_idx = last_sources_idx - 2

    footer_lines = lines[start_idx:]
    body_lines = lines[:start_idx]

    footer = "\n".join(footer_lines).strip()
    main_body = "\n".join(body_lines).rstrip()

    if not footer:
        return body, None

    return main_body, footer


def _load_overview_prompt_template() -> str:
    """Load the overview deep research prompt template from disk.

    Keeping the large CTA prompt in a separate text file makes it easier to
    iterate on without touching Python source. The template uses str.format(...)
    placeholders such as {scope_display}, {created}, and {plan_block}.
    """
    template_path = Path(__file__).parent / "agent_doc_overview_prompt.txt"
    try:
        return template_path.read_text(encoding="utf-8").strip()
    except Exception:
        # Fallback to an empty string so callers can still build a prompt even
        # if the template file is missing or unreadable.
        return ""


def _build_research_prompt(
    meta: AgentDocMetadata,
    diff_since_created: list[str],
    diff_since_previous: list[str],
    scope_label: str,
    hyde_plan: str | None = None,
) -> str:
    """Construct the overview deep research query for the agent doc."""
    created = meta.created_from_sha
    previous = meta.previous_target_sha
    target = meta.target_sha

    diff_created_block = (
        "No changes detected (initial build; baseline and target are identical)."
        if not diff_since_created
        else "\n".join(f"- {line}" for line in diff_since_created)
    )

    diff_previous_block = (
        "No changes since previous agent doc build."
        if not diff_since_previous
        else "\n".join(f"- {line}" for line in diff_since_previous)
    )

    scope_display = "/" if scope_label == "/" else f"./{scope_label}"

    plan_block = ""
    if hyde_plan and hyde_plan.strip():
        plan_block = f"""

Planning context (from a separate HyDE-style heuristic scan).
Use this ONLY as a rough outline. Do NOT copy phrases or sentences from it;
re-derive all explanations from the actual code, tests, and docs.

<<<PLAN>>>
{hyde_plan}
<<<ENDPLAN>>>
""".rstrip()

    template = _load_overview_prompt_template()
    if not template:
        # Fallback to a minimal inline prompt if the external template is
        # unavailable for any reason. This preserves behavior without failing.
        return (
            f"Generate an architecture and operations overview for the scoped "
            f"folder {scope_display}, using git diffs and HyDE context where "
            f"available. created_from_sha={created}, previous_target_sha={previous}, "
            f"target_sha={target}.\n\n"
            f"Diff since created:\n{diff_created_block}\n\n"
            f"Diff since previous:\n{diff_previous_block}\n\n"
            f"{plan_block}\n"
            f"Return a markdown '# Agent Doc' with the standard sections."
        )

    return template.format(
        created=created,
        previous=previous,
        target=target,
        scope_display=scope_display,
        diff_created_block=diff_created_block,
        diff_previous_block=diff_previous_block,
        plan_block=plan_block,
    ).strip()



async def _run_research_query(
    services: Any,
    embedding_manager: EmbeddingManager,
    llm_manager: LLMManager | None,
    prompt: str,
    scope_label: str | None = None,
) -> str:
    """Run a single deep_research_impl call and return only the answer text."""
    answer, _ = await _run_research_query_with_metadata(
        services=services,
        embedding_manager=embedding_manager,
        llm_manager=llm_manager,
        prompt=prompt,
        scope_label=scope_label,
    )
    return answer


async def _run_research_query_with_metadata(
    services: Any,
    embedding_manager: EmbeddingManager,
    llm_manager: LLMManager | None,
    prompt: str,
    scope_label: str | None = None,
) -> tuple[str, dict[str, Any]]:
    """Run deep_research_impl and return (answer, metadata)."""
    path_filter = None if scope_label in (None, "/") else scope_label
    result: dict[str, Any] = await deep_research_impl(
        services=services,
        embedding_manager=embedding_manager,
        llm_manager=llm_manager,
        query=prompt,
        progress=None,
        path=path_filter,
    )

    answer = result.get(
        "answer",
        "Research incomplete: unable to generate this section for the current run.",
    )
    metadata = result.get("metadata", {}) or {}
    return answer, metadata


async def _run_semantic_overview_query(
    services: Any,
    llm_manager: LLMManager | None,
    meta: AgentDocMetadata,
    scope_label: str,
    hyde_plan: str | None = None,
    debug_dir: Path | None = None,
    lexical_sidecar: bool = False,
    gap_fill: bool = False,
) -> str:
    """Run a faster, semantic-search-only overview synthesis.

    This uses SearchService.search_semantic() scoped to the selected folder to
    retrieve representative chunks, then asks the synthesis LLM to write an
    architecture document based on those chunks (and optional HyDE plan).
    """
    if not llm_manager or not llm_manager.is_configured():
        return "LLM not configured for semantic overview mode."

    search_service = getattr(services, "search_service", None)
    if search_service is None:
        return "Search service unavailable for semantic overview mode."

    try:
        provider = llm_manager.get_synthesis_provider()
    except Exception:
        return "Synthesis provider unavailable for semantic overview mode."

    scope_display = "/" if scope_label == "/" else f"./{scope_label}"
    path_filter = None if scope_label == "/" else scope_label

    base_query = (
        f"Explain the architecture, responsibilities, and interactions of code under "
        f"{scope_display}, focusing on key modules, their roles, and how they collaborate."
    )
    grammar_query = (
        f"Identify the core grammar and parser implementation files under {scope_display} "
        f"(for example: grammar.js, grammar/*.js, src/parser.c, src/scanner.c, src/*.c). "
        f"Prefer chunks that show how language constructs are defined."
    )
    bindings_query = (
        f"Identify bindings and packaging files under {scope_display} "
        f"(for example: bindings/*, CMakeLists.txt, Package.swift, tree-sitter.json, setup.py). "
        f"Prefer chunks that show how the parser is exposed to other languages or tools."
    )
    tests_query = (
        f"Identify corpus and test fixtures under {scope_display} "
        f"(for example: test/corpus/*.txt, tests/*). Prefer chunks that show expected "
        f"syntax coverage or regression tests."
    )

    # If a HyDE plan is available, turn its bullets into additional semantic
    # queries so that retrieval follows the hallucinated outline point-by-point.
    max_hyde_queries = 12
    hyde_queries: list[str] = _extract_hyde_bullets(hyde_plan, max_hyde_queries)

    queries: list[str] = []
    if hyde_queries:
        queries.extend(hyde_queries)
    queries.extend([base_query, grammar_query, bindings_query, tests_query])

    # Run multiple semantic searches with different focuses and merge results.
    all_results: list[dict[str, Any]] = []
    per_query_results: list[list[dict[str, Any]]] = []
    for q in queries:
        combined_batch: list[dict[str, Any]] = []
        try:
            semantic_batch, _ = await search_service.search_semantic(
                query=q,
                page_size=40,
                offset=0,
                threshold=None,
                path_filter=path_filter,
            )
            combined_batch.extend(semantic_batch)
        except Exception:
            semantic_batch = []

        # Optional lexical sidecar: run a lightweight regex search derived from
        # the query text, scoped by the same path_filter. This is intended as a
        # generic experiment and must not change core semantics when disabled.
        if lexical_sidecar:
            pattern = _build_lexical_pattern_from_query(q)
            if pattern:
                try:
                    regex_batch, _ = await search_service.search_regex_async(
                        pattern=pattern,
                        page_size=20,
                        offset=0,
                        path_filter=path_filter,
                    )
                    combined_batch.extend(regex_batch)
                except Exception:
                    # Lexical sidecar is best-effort; ignore failures.
                    pass

        per_query_results.append(combined_batch)
        all_results.extend(combined_batch)

    if not all_results:
        return (
            "Semantic overview mode: no results found for the scoped folder. "
            "Try running with code_research mode instead."
        )

    # Deduplicate results by (path, start_line, end_line) and sort by score.
    # We may recompute this after optional gap-filling.
    def _dedup_and_sort(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
        local_dedup: dict[tuple[str, int | None, int | None], dict[str, Any]] = {}
        for res in results:
            path = res.get("path") or res.get("file_path") or ""
            key = (path, res.get("start_line"), res.get("end_line"))
            if not path:
                continue
            existing = local_dedup.get(key)
            if existing is None or res.get("score", 0.0) > existing.get("score", 0.0):
                local_dedup[key] = res
        return sorted(
            local_dedup.values(), key=lambda r: r.get("score", 0.0), reverse=True
        )

    merged_results = _dedup_and_sort(all_results)

    # Optional gap-filling pass: use tokens derived from the existing merged
    # landscape to run small, targeted regex searches and pull in snippets
    # from additional files within the same scoped path.
    if gap_fill:
        extra_results: list[dict[str, Any]] = []
        seen_paths: set[str] = set()
        for res in merged_results:
            path = (res.get("path") or res.get("file_path") or "").replace("\\", "/")
            if path:
                seen_paths.add(path)

        tokens = _extract_gap_fill_tokens(merged_results)
        max_new_files = 6
        new_files: set[str] = set()

        for token in tokens:
            if len(new_files) >= max_new_files:
                break
            pattern = rf"\\b{re.escape(token)}\\b"
            try:
                regex_batch, _ = await search_service.search_regex_async(
                    pattern=pattern,
                    page_size=20,
                    offset=0,
                    path_filter=path_filter,
                )
            except Exception:
                continue

            for res in regex_batch:
                path = (res.get("path") or res.get("file_path") or "").replace("\\", "/")
                if not path:
                    continue
                if path in seen_paths or path in new_files:
                    continue
                extra_results.append(res)
                new_files.add(path)
                if len(new_files) >= max_new_files:
                    break

        if extra_results:
            all_results.extend(extra_results)
            merged_results = _dedup_and_sort(all_results)

    # Build a compact context block from top results and capture provenance.
    # We cap the total number of snippets but also cap per-file to improve
    # coverage across different modules.
    snippets: list[str] = []
    chunks_for_footer: list[dict[str, Any]] = []
    files_for_footer: dict[str, str] = {}
    per_file_cap = 3
    per_file_counts: dict[str, int] = {}

    # Allow more snippets to broaden provenance, but still keep a hard cap
    # to avoid runaway prompt growth.
    max_snippets = 40
    merged_snapshot: list[dict[str, Any]] = []
    for result in merged_results:
        if len(snippets) >= max_snippets:
            break

        path = result.get("path") or result.get("file_path") or "unknown"
        if path == "unknown":
            continue

        count = per_file_counts.get(path, 0)
        if count >= per_file_cap:
            continue
        start = result.get("start_line")
        end = result.get("end_line")

        header = f"File: {path}"
        if isinstance(start, int) and isinstance(end, int):
            header += f" (lines {start}-{end})"
            # Record chunk for provenance footer
            if path != "unknown":
                chunks_for_footer.append(
                    {
                        "file_path": path,
                        "start_line": start,
                        "end_line": end,
                    }
                )

        code_preview = result.get("code_preview") or result.get("code") or ""
        # Use a slightly larger snippet window; this is still compact
        # compared to full files but gives the LLM more structure to work with.
        if len(code_preview) > 800:
            code_preview = code_preview[:800] + "..."

        snippets.append(header + "\n" + code_preview)

        # Track file presence and per-file usage
        files_for_footer.setdefault(path, "")
        per_file_counts[path] = count + 1

        merged_snapshot.append(
            {
                "path": path,
                "start_line": start,
                "end_line": end,
                "score": float(result.get("score", 0.0)),
                "code_preview": code_preview,
            }
        )

    context_block = "\n\n---\n\n".join(snippets)

    plan_block = ""
    if hyde_plan and hyde_plan.strip():
        plan_block = f"""

Planning context (from a separate HyDE-style heuristic scan).
Use this ONLY as a rough outline. Do NOT copy phrases or sentences from it;
re-derive all explanations directly from the code snippets provided below.

<<<PLAN>>>
{hyde_plan}
<<<ENDPLAN>>>
""".rstrip()

    prompt = f"""
You are generating an architecture and operations overview for a codebase,
scoped to a specific folder.

Scoped folder (relative to workspace root):
- {scope_display}

Workspace commit context (if available):
- created_from_sha: {meta.created_from_sha}
- previous_target_sha: {meta.previous_target_sha}
- target_sha: {meta.target_sha}

The following scoped code snippets were retrieved via semantic search:

<<<CONTEXT>>>
{context_block}
<<<ENDCONTEXT>>>

{plan_block}

Task:
- Using ONLY the code snippets in <<<CONTEXT>>> as ground truth, write a clear
  architecture and operations document for this scoped folder.
- You may use the PLAN block as a rough outline, but do NOT copy text from it.
  Re-derive all wording based on the actual snippets.

Output format:
- Produce a single markdown document with these headings:
  # Agent Doc
  ## 1. Project Identity and Constraints (brief summary tailored to this scope)
  ## 2. Architecture Overview (for the in-scope folder)
  ## 3. Subsystem Guides (for modules under this folder)
  ## 4. Operations, Commands, and Testing (if relevant for this scope)
  ## 5. Known Pitfalls and Debugging Patterns (for this scope)
    """.strip()

    system_message = (
        "You are an offline documentation synthesizer running inside a code "
        "analysis tool. The <<<CONTEXT>>> section of the user prompt "
        "contains real code snippets and file metadata from the workspace. "
        "You MUST base your answer solely on that context.\n\n"
        "If <<<CONTEXT>>> is non-empty, you must write the requested "
        "architecture/operations document. Do NOT say that you cannot see "
        "the workspace, cannot access files, or that folders/files are "
        "missing. Never claim that a file or directory does not exist; if "
        "it is not present in <<<CONTEXT>>>, simply do not mention it.\n\n"
        "Do not mention having limited access or being unable to browse; "
        "just describe what the context shows."
    )

    # Best-effort debug dumps of semantic search diagnostics and prompt context.
    if debug_dir is not None:
        try:
            debug_dir.mkdir(parents=True, exist_ok=True)

            queries_payload: list[dict[str, Any]] = []
            hyde_query_set = set(hyde_queries)
            for idx, q in enumerate(queries):
                source = "hyde" if q in hyde_query_set else "base"
                if q == grammar_query:
                    source = "grammar"
                elif q == bindings_query:
                    source = "bindings"
                elif q == tests_query:
                    source = "tests"
                queries_payload.append({"id": idx, "text": q, "source": source})

            (debug_dir / "semantic_queries.json").write_text(
                json.dumps(queries_payload, indent=2),
                encoding="utf-8",
            )

            raw_lines: list[str] = []
            for idx, (q, results) in enumerate(zip(queries, per_query_results)):
                for result in results:
                    raw_path = result.get("path") or result.get("file_path") or ""
                    start_line = result.get("start_line")
                    end_line = result.get("end_line")
                    score = float(result.get("score", 0.0))
                    code_preview = result.get("code_preview") or result.get("code") or ""
                    raw_lines.append(
                        json.dumps(
                            {
                                "query_id": idx,
                                "query_text": q,
                                "path": raw_path,
                                "start_line": start_line,
                                "end_line": end_line,
                                "score": score,
                                "code_preview": code_preview,
                            }
                        )
                    )

            (debug_dir / "semantic_results_raw.jsonl").write_text(
                "\n".join(raw_lines),
                encoding="utf-8",
            )

            (debug_dir / "semantic_results_merged.json").write_text(
                json.dumps(merged_snapshot, indent=2),
                encoding="utf-8",
            )

            # HyDE bullet-to-file mapping for later analysis.
            if hyde_queries:
                hyde_to_files: dict[str, list[str]] = {}
                hyde_set = set(hyde_queries)
                for idx, (q, results) in enumerate(zip(queries, per_query_results)):
                    if q not in hyde_set:
                        continue
                    paths: set[str] = set()
                    for result in results:
                        path = result.get("path") or result.get("file_path") or ""
                        if path:
                            paths.add(path)
                    if paths:
                        hyde_to_files[q] = sorted(paths)

                (debug_dir / "hyde_to_files.json").write_text(
                    json.dumps(hyde_to_files, indent=2),
                    encoding="utf-8",
                )

            (debug_dir / "semantic_prompt.txt").write_text(
                "SYSTEM:\n"
                + system_message
                + "\n\nPROMPT:\n"
                + prompt,
                encoding="utf-8",
            )
            (debug_dir / "semantic_context.md").write_text(
                context_block,
                encoding="utf-8",
            )
        except Exception:
            # Debug dumps must never break the main flow.
            pass

    try:
        response = await provider.complete(
            prompt=prompt,
            system=system_message,
            max_completion_tokens=4096,
        )
        if not response or not getattr(response, "content", None):
            return "Semantic overview synthesis returned no content."
        answer_text = response.content
    except Exception as exc:
        return f"Semantic overview synthesis failed: {exc}"

    if debug_dir is not None:
        try:
            debug_dir.mkdir(parents=True, exist_ok=True)
            (debug_dir / "semantic_raw_answer.md").write_text(
                answer_text,
                encoding="utf-8",
            )
        except Exception:
            pass

    # Append a Sources footer similar to deep research mode, using the
    # semantic search results as provenance. This keeps the final document
    # self-explanatory for agents even in fast semantic mode.
    sources_footer = ""
    if files_for_footer and chunks_for_footer:
        try:
            cm = CitationManager()
            # Build a simple reference map for stable numbering
            ref_map = cm.build_file_reference_map(chunks_for_footer, files_for_footer)
            sources_footer = cm.build_sources_footer(
                chunks_for_footer, files_for_footer, ref_map
            )
        except Exception:
            # Provenance is best-effort; if it fails, still return the main answer
            sources_footer = ""

    if sources_footer:
        return answer_text.strip() + "\n\n" + sources_footer

    return answer_text




def _get_debug_dir(out_dir: Path | None, debug_dump: bool) -> Path | None:
    """Return the debug directory for agent-doc runs, if enabled."""
    if not debug_dump or out_dir is None:
        return None
    return out_dir / "debug"


def _build_lexical_pattern_from_query(query: str) -> str | None:
    """Build a simple regex pattern from a natural-language query.

    This is intentionally conservative and project-agnostic: it extracts the
    first reasonably long alphanumeric token and uses it as a word-boundary
    pattern. If no such token exists, returns None.
    """
    tokens = []
    current = []
    for ch in query:
        if ch.isalnum() or ch == "_":
            current.append(ch)
        else:
            if current:
                tokens.append("".join(current))
                current = []
    if current:
        tokens.append("".join(current))

    for token in tokens:
        if len(token) >= 4:
            # Simple word-boundary style pattern; SearchService enforces safety.
            return rf"\\b{token}\\b"

    return None


def _extract_gap_fill_tokens(merged_results: list[dict[str, Any]]) -> list[str]:
    """Extract candidate tokens for gap-filling from merged semantic results.

    This inspects file paths to derive stem-like tokens that are already
    present in the scoped project (for example, 'scanner', 'precedences',
    'module'). It remains project-agnostic and does not rely on domain
    knowledge.
    """
    token_counts: Counter[str] = Counter()

    # Small, generic stopword set to avoid very common, low-signal tokens.
    stopwords = {
        "project",
        "source",
        "src",
        "test",
        "tests",
        "module",
        "import",
        "export",
        "file",
        "files",
        "config",
        "core",
        "data",
        "type",
        "types",
        "docs",
        "readme",
        "build",
        "setup",
        "package",
    }

    for result in merged_results:
        path = (result.get("path") or result.get("file_path") or "").replace("\\", "/")
        if not path:
            continue
        name = path.rsplit("/", 1)[-1]
        stem = name.split(".", 1)[0]
        # Split stem into tokens on non-alphanumeric boundaries
        for raw in re.split(r"[^A-Za-z0-9_]+", stem):
            token = raw.strip()
            if not token:
                continue
            lower = token.lower()
            if len(lower) < 4 or lower in stopwords:
                continue
            token_counts[lower] += 1

    # Return top tokens by frequency; keep the list small to bound extra work.
    return [tok for tok, _ in token_counts.most_common(8)]


def _categorize_file_path(path: str) -> str:
    """Roughly categorize a file path for coverage metrics.

    This is intentionally heuristic and project-agnostic. It is used only for
    diagnostics and must not affect retrieval behavior.
    """
    normalized = path.replace("\\", "/")
    lower = normalized.lower()
    name = lower.rsplit("/", 1)[-1]
    ext = ""
    if "." in name:
        ext = "." + name.split(".")[-1]

    code_exts = {
        ".c",
        ".cc",
        ".cpp",
        ".cxx",
        ".h",
        ".hpp",
        ".rs",
        ".go",
        ".py",
        ".rb",
        ".java",
        ".kt",
        ".ts",
        ".tsx",
        ".js",
        ".jsx",
        ".cs",
        ".swift",
        ".m",
        ".mm",
        ".php",
        ".scala",
        ".zig",
        ".sh",
        ".bash",
        ".ps1",
        ".pl",
        ".ex",
        ".exs",
        ".clj",
        ".cljs",
    }

    # Docs
    if name in {"readme", "readme.md", "readme.rst"} or ext in {".md", ".rst", ".adoc"}:
        return "docs"
    if "/docs/" in lower or "/doc/" in lower:
        return "docs"

    # Tests
    if any(part in lower for part in ("/test/", "/tests/", "/__tests__/", "/spec/", "/specs/")):
        return "tests"
    if name.startswith("test_") or name.endswith("_test" + ext):
        return "tests"

    # Config / CI / build metadata
    config_like = {
        "cmakelists.txt",
        "makefile",
        "dockerfile",
        "docker-compose.yml",
        "package.json",
        "cargo.toml",
        "pyproject.toml",
        "setup.py",
        "tree-sitter.json",
        "requirements.txt",
    }
    if ext in {".yml", ".yaml", ".toml", ".ini", ".json"} or name in config_like:
        return "config"

    # Code
    if ext in code_exts:
        return "code"

    return "other"


def _compute_semantic_coverage_summary(
    project_root: Path,
    scope_path: Path,
    scope_label: str,
    debug_dir: Path,
) -> dict[str, Any]:
    """Compute simple semantic coverage metrics for diagnostic purposes.

    This reads semantic_results_merged.json and compares it against the actual
    files present under the scope on disk. It does not influence retrieval.
    """
    merged_path = debug_dir / "semantic_results_merged.json"
    if not merged_path.exists():
        return {}

    try:
        merged_results = json.loads(merged_path.read_text(encoding="utf-8"))
    except Exception:
        return {}

    # Discover all files in the scoped folder, relative to project_root.
    ignore_dirs = {".git", ".chunkhound", ".venv", "venv", "__pycache__", ".mypy_cache"}
    scoped_files: set[str] = set()
    for path in scope_path.rglob("*"):
        if not path.is_file():
            continue
        if any(part in ignore_dirs for part in path.parts):
            continue
        try:
            rel = path.relative_to(project_root)
        except ValueError:
            continue
        scoped_files.add(str(rel).replace("\\", "/"))

    total_files = len(scoped_files)

    # Aggregate coverage from merged semantic results.
    files_with_hits: set[str] = set()
    category_snippets: dict[str, int] = {}
    category_files: dict[str, set[str]] = {}

    total_snippets = 0
    for result in merged_results:
        path = (result.get("path") or result.get("file_path") or "").replace("\\", "/")
        if not path:
            continue
        total_snippets += 1
        files_with_hits.add(path)
        category = _categorize_file_path(path)
        category_snippets[category] = category_snippets.get(category, 0) + 1
        bucket = category_files.setdefault(category, set())
        bucket.add(path)

    scoped_files_with_hits = scoped_files & files_with_hits
    files_with_semantic_hits = len(scoped_files_with_hits)
    coverage_ratio = float(files_with_semantic_hits) / float(total_files) if total_files else 0.0

    category_summary: dict[str, Any] = {}
    for category, snippet_count in category_snippets.items():
        paths = category_files.get(category, set())
        # Intersection with scoped_files keeps the summary aligned to the scope.
        in_scope_paths = paths & scoped_files
        category_summary[category] = {
            "snippets": snippet_count,
            "files": len(in_scope_paths),
        }

    return {
        "scope_label": scope_label,
        "project_root": str(project_root),
        "scope_path": str(scope_path),
        "total_files": total_files,
        "files_with_semantic_hits": files_with_semantic_hits,
        "coverage_ratio": coverage_ratio,
        "coverage_percent": coverage_ratio * 100.0,
        "total_snippets": total_snippets,
        "category_summary": category_summary,
    }


def _collect_scope_files(scope_path: Path, project_root: Path) -> list[str]:
    """Collect a list of file paths within the scope, relative to project_root.

    This is intentionally lightweight: we only gather paths, not full contents,
    and we skip typical noise directories.
    """
    ignore_dirs = {".git", ".chunkhound", ".venv", "venv", "__pycache__", ".mypy_cache"}
    file_paths: list[str] = []
    for path in scope_path.rglob("*"):
        if not path.is_file():
            continue
        # Skip ignored directories anywhere in the path
        if any(part in ignore_dirs for part in path.parts):
            continue
        try:
            rel = path.relative_to(project_root)
        except ValueError:
            # Should not happen because scope_path is under project_root,
            # but guard just in case.
            continue
        file_paths.append(str(rel))

    # Keep a stable, deterministic order and cap to a reasonable count
    file_paths = sorted(file_paths)
    max_files = 200
    if len(file_paths) > max_files:
        file_paths = file_paths[:max_files]
    return file_paths


def _count_scope_files_for_coverage(scope_path: Path, project_root: Path) -> int:
    """Count the number of files within the scope, relative to project_root.

    This is used for coverage statistics in generation_stats and may traverse
    more than the capped list used for HyDE prompts. It remains lightweight by
    skipping typical noise directories.
    """
    ignore_dirs = {".git", ".chunkhound", ".venv", "venv", "__pycache__", ".mypy_cache"}
    total_files = 0
    for path in scope_path.rglob("*"):
        if not path.is_file():
            continue
        if any(part in ignore_dirs for part in path.parts):
            continue
        try:
            path.relative_to(project_root)
        except ValueError:
            continue
        total_files += 1
    return total_files


def _load_hyde_prompt_template() -> str:
    """Load the HyDE-only scope prompt template from disk.

    This keeps the HyDE CTA prompt in a separate text file for easier editing.
    The template uses str.format(...) placeholders such as {scope_display},
    {created}, and {files_block}.
    """
    template_path = Path(__file__).parent / "agent_doc_hyde_prompt.txt"
    try:
        return template_path.read_text(encoding="utf-8").strip()
    except Exception:
        return ""


def _build_hyde_scope_prompt(
    meta: AgentDocMetadata,
    scope_label: str,
    file_paths: list[str],
) -> str:
    """Construct a HyDE-only prompt using only file layout within the scope.

    NOTE: This prompt is intentionally project-agnostic. It should not mention
    ChunkHound or any product-specific semantics; the goal is to hallucinate a
    generic architectural plan purely from file naming and layout.
    """
    scope_display = "/" if scope_label == "/" else f"./{scope_label}"

    files_block = "\n".join(f"- {p}" for p in file_paths) if file_paths else "- (no files discovered)"

    # Build a lightweight code context block from a small subset of files in
    # the scope. This remains filesystem-only (no DB) and is intended purely
    # to give HyDE a better feel for the project while staying cheap.
    max_files_for_snippets = 12
    max_chars_per_file = 800
    code_snippets: list[str] = []

    project_root = Path.cwd()

    for rel_path in file_paths[:max_files_for_snippets]:
        try:
            path = project_root / rel_path
            if not path.is_file():
                continue
            text = path.read_text(encoding="utf-8", errors="replace")
            if not text.strip():
                continue
        except Exception:
            continue

        snippet = text[:max_chars_per_file]
        # Try to preserve whole lines
        if len(text) > max_chars_per_file:
            last_nl = snippet.rfind("\n")
            if last_nl > 0:
                snippet = snippet[:last_nl]

        # Infer language hint for fenced block from extension
        ext = path.suffix.lstrip(".").lower()
        lang = ext if ext in {"py", "rs", "ts", "tsx", "js", "jsx", "go", "rb", "java", "kt", "c", "h", "cpp", "md"} else ""
        fence = f"```{lang}" if lang else "```"

        code_snippets.append(
            f"File: {rel_path}\n{fence}\n{snippet}\n```"
        )

    if code_snippets:
        code_context_block = "\n\n".join(code_snippets)
    else:
        code_context_block = "(no sample code snippets available)"

    created = meta.created_from_sha
    previous = meta.previous_target_sha
    target = meta.target_sha

    template = _load_hyde_prompt_template()
    if not template:
        # Fallback to a compact inline HyDE research-plan prompt if the template
        # file is unavailable. This keeps HyDE-only mode functional.
        return (
            "You are generating a hypothetical research plan for a scoped folder "
            "based only on file layout and a few sample code snippets.\n\n"
            f"In-scope folder: {scope_display}\n"
            f"Files:\n{files_block}\n\n"
            f"Sample code snippets:\n{code_context_block}\n\n"
            "Output a markdown '# HyDE Research Plan for {scope_display}' with "
            "a '## Global Hooks' section and a '## Subsystem Hooks' section, "
            "where each '-' bullet is a focused research question for future "
            "code analysis."
        )

    return template.format(
        created=created,
        previous=previous,
        target=target,
        scope_display=scope_display,
        files_block=files_block,
        code_context_block=code_context_block,
    ).strip()


async def _run_hyde_only_query(
    llm_manager: LLMManager | None,
    prompt: str,
) -> str:
    """Run a HyDE-only query using the synthesis LLM provider.

    This bypasses deep_research_impl and does not hit the database or embeddings;
    it simply feeds the prompt directly to the synthesis model.
    """
    if not llm_manager or not llm_manager.is_configured():
        return "LLM not configured for HyDE-only mode."

    try:
        provider = llm_manager.get_synthesis_provider()
    except Exception:
        return "Synthesis provider unavailable for HyDE-only mode."

    # Strong system message to keep HyDE project-agnostic and focused on planning,
    # while encouraging creative, slightly flamboyant hypotheses and rich hook sets.
    system_msg = (
        "You are generating a hypothetical research plan for an unknown software "
        "project based only on file and directory names. You have no prior knowledge "
        "of its real name, domain, or tech stack. Do not mention specific product, "
        "repository, or technology names. Speak only in generic, project-agnostic "
        "terms such as 'database', 'search index', 'configuration module', "
        "'API layer', etc. Your output must be a plan (hooks/questions), not a "
        "final architecture document. Err on the side of over-generating hooks: "
        "for larger scopes, produce many imaginative, richly worded research "
        "questions, as long as each '-' bullet is a clear, self-contained hook "
        "that could drive a deep code-research pass."
    )

    try:
        response = await provider.complete(
            prompt=prompt,
            system=system_msg,
            max_completion_tokens=2048,
        )
        if not response or not getattr(response, "content", None):
            return "HyDE-only synthesis returned no content."
        return response.content
    except Exception as exc:
        return f"HyDE-only synthesis failed: {exc}"


async def _trim_doc_for_scope(
    llm_manager: LLMManager | None,
    body: str,
    scope_label: str,
) -> str:
    """Optionally run a final LLM pass to trim the doc to the selected scope.

    This uses the synthesis provider directly and does not perform any additional
    database or embedding work. If no LLM manager is configured or any error
    occurs, the original body is returned unchanged.
    """
    # No trimming for full-workspace scope
    if scope_label == "/":
        return body

    if not llm_manager or not llm_manager.is_configured():
        return body

    try:
        provider = llm_manager.get_synthesis_provider()
    except Exception:
        return body

    scope_display = f"./{scope_label}" if scope_label != "/" else "/"

    prompt = f"""
You are refining an existing, long markdown architecture and operations
document for a codebase.

The documentation was generated for the entire workspace, but we now only care
about content that is directly relevant to the in-scope folder:

- in-scope folder: {scope_display}

The current document is provided between <<<DOC>>> and <<<ENDDOC>>> below.

Your task:
- Produce a trimmed version of the SAME document that:
  - Keeps sections and paragraphs that primarily describe code, tests, configs,
    and behaviors inside the in-scope folder ({scope_display}).
  - Keeps global constraints and invariants ONLY as needed to understand the
    in-scope folder (e.g., SerialDatabaseProvider, batching rules).
  - Removes or sharply compresses sections that are mostly about unrelated
    subsystems outside the in-scope folder.
- Preserve:
  - The top-level heading: '# Agent Doc'
  - The numbered '##' sections (1..6 and 7 if present), but you may
    aggressively shorten their content if it is out-of-scope.
  - Any clearly in-scope subsystem '###' sections.
- It is acceptable to leave short references to out-of-scope modules if they
  are necessary to understand the in-scope behavior, but avoid long digressions.
- DO NOT add new headings that change the overall structure; just trim and
  lightly rewrite existing content.
- DO NOT include the metadata comment header at the top; only return the
  markdown body starting from '# Agent Doc'.

<<<DOC>>>
{body}
<<<ENDDOC>>>
""".strip()

    try:
        response = await provider.complete(
            prompt=prompt,
            system=None,
            max_completion_tokens=4096,
        )
        if not response or not getattr(response, "content", None):
            return body
        trimmed = response.content.strip()
        # Basic sanity: ensure we still have the root heading; otherwise keep original
        if "# Agent Doc" not in trimmed:
            return body
        return trimmed
    except Exception:
        # On any failure, fall back to the untrimmed body
        return body


async def _run_hyde_map_passes(
    services: Any,
    embedding_manager: EmbeddingManager | None,
    llm_manager: LLMManager | None,
    scope_label: str,
    hyde_plan: str | None,
) -> tuple[list[dict[str, Any]], int]:
    """Run HyDE-guided deep-research passes and return map-level findings.

    Each finding contains:
    - \"bullet\": original HyDE bullet text
    - \"analysis\": cleaned deep-research answer without any Sources footer
    - \"sources\": metadata[\"sources\"] from deep_research (if present)
    """
    if not hyde_plan or not hyde_plan.strip():
        return [], 0

    if services is None or embedding_manager is None:
        return [], 0

    # Respect workspace/full-scope semantics from the main run.
    scope_display = "/" if scope_label == "/" else f"./{scope_label}"

    # Decide whether to drive map passes from individual bullets or grouped
    # sections (title + body). This is controlled via an environment variable to
    # allow experimentation without breaking existing behavior.
    group_mode = os.getenv("CH_AGENT_DOC_CODE_RESEARCH_HYDE_GROUP_MODE", "bullet")

    if group_mode == "section":
        units = _extract_hyde_sections(hyde_plan, max_sections=10_000)
    else:
        bullets = _extract_hyde_bullets(hyde_plan, max_bullets=10_000)
        units = [{"title": b, "body": b} for b in bullets]

    limit_env = os.getenv("CH_AGENT_DOC_CODE_RESEARCH_HYDE_MAP_LIMIT")
    if limit_env:
        try:
            n = int(limit_env)
            if n >= 0:
                units = units[:n]
        except ValueError:
            # Ignore invalid limits and fall back to default behavior.
            pass
    if not units:
        return [], 0

    findings: list[dict[str, Any]] = []
    calls = 0

    for unit in units:
        title = unit["title"]
        body = unit["body"]

        # Build a focused, scoped query for this HyDE aspect or section.
        query = f"""
Within the scoped folder {scope_display}, perform a focused deep research pass
for the following HyDE section from the agent-doc plan:

<<<SECTION_TITLE>>>
{title}
<<<END_SECTION_TITLE>>>

<<<SECTION_BODY>>>
{body}
<<<END_SECTION_BODY>>>

Explain which concrete code, tests, and configs implement this aspect, describe
their responsibilities and interactions, and include inline [N] citations that
refer to the Sources table in the final agent doc.

Produce a concise section (1–3 short paragraphs). Do NOT output a top-level
'# Agent Doc' heading or a '## Sources' footer.
""".strip()

        answer, meta = await _run_research_query_with_metadata(
            services=services,
            embedding_manager=embedding_manager,
            llm_manager=llm_manager,
            prompt=query,
            scope_label=scope_label,
        )
        calls += 1

        # Strip any per-section sources footer if the deep-research pipeline
        # decides to add one despite the prompt.
        cleaned, _ = _split_sources_footer(answer)
        cleaned = cleaned.strip()
        if not cleaned:
            continue

        findings.append(
            {
                "bullet": title,
                "analysis": cleaned,
                "sources": meta.get("sources") or {},
            }
        )

    return findings, calls


async def _run_hyde_map_deep_research(
    services: Any,
    embedding_manager: EmbeddingManager | None,
    llm_manager: LLMManager | None,
    scope_label: str,
    hyde_plan: str | None,
) -> tuple[str, list[dict[str, Any]], int]:
    """Run optional HyDE-guided deep-research passes per bullet.

    This mirrors the semantic HyDE loop but uses deep_research_impl for each
    selected HyDE bullet. Calls are sequential to preserve single-threaded DB
    access and are always scoped via the same path_filter as the main run.
    """
    findings, calls = await _run_hyde_map_passes(
        services=services,
        embedding_manager=embedding_manager,
        llm_manager=llm_manager,
        scope_label=scope_label,
        hyde_plan=hyde_plan,
    )
    if not findings:
        return "", [], 0

    sections: list[str] = []
    sections.append("## 7. HyDE-Guided Deep Dives")
    sources_metadata: list[dict[str, Any]] = []

    for idx, finding in enumerate(findings, 1):
        bullet = finding["bullet"]
        cleaned = finding["analysis"]
        sources_meta = finding.get("sources") or {}
        if sources_meta:
            sources_metadata.append(sources_meta)

        # Shorten the bullet for the heading to keep things compact.
        heading_text = bullet.strip()
        max_heading_len = 80
        if len(heading_text) > max_heading_len:
            heading_text = heading_text[: max_heading_len - 1].rstrip() + "…"

        sections.append("")
        sections.append(f"### {idx}. {heading_text}")
        sections.append("")
        sections.append(cleaned)

    return "\n".join(sections).strip(), sources_metadata, calls


async def _run_map_only_merge(
    llm_manager: LLMManager | None,
    project_root: Path,
    scope_label: str,
    meta: AgentDocMetadata,
    reference_table: str,
    findings: list[str],
) -> str:
    """Merge multiple map-level findings into a single Agent Doc.

    This uses the synthesis provider directly and does not perform any
    additional database or embedding work. If the LLM manager is not
    configured, a simple concatenation fallback is returned.
    """
    # Degenerate but safe fallback: ensure we always return a doc body.
    cleaned_findings = [f.strip() for f in findings if f.strip()]
    if not cleaned_findings:
        return (
            "# Agent Doc\n\n"
            "_No map-level analyses were available for this agent-doc run._"
        )

    if not llm_manager or not llm_manager.is_configured():
        merged = "\n\n".join(cleaned_findings)
        return f"# Agent Doc\n\n{merged}"

    try:
        provider = llm_manager.get_synthesis_provider()
    except Exception:
        merged = "\n\n".join(cleaned_findings)
        return f"# Agent Doc\n\n{merged}"

    scope_display = "/" if scope_label == "/" else f"./{scope_label}"

    created = meta.created_from_sha
    previous = meta.previous_target_sha
    target = meta.target_sha

    findings_blocks: list[str] = []
    for idx, content in enumerate(cleaned_findings, 1):
        findings_blocks.append(f"## Finding {idx}\n\n{content}")
    findings_block = "\n\n".join(findings_blocks)

    reference_section = reference_table.strip()
    if not reference_section:
        reference_section = (
            "## Source References\n\n"
            "No structured source metadata was available for this run. "
            "If the findings contain [N] citations, treat them as best-effort hints."
        )

    # Load the map-only merge CTA prompt from an external template for easier
    # iteration. Fall back to the previous inline prompt if the template is
    # unavailable.
    merge_template_path = Path(__file__).parent / "agent_doc_map_merge_prompt.txt"
    try:
        merge_template = merge_template_path.read_text(encoding="utf-8").strip()
    except Exception:
        merge_template = ""

    if merge_template:
        prompt = merge_template.format(
            project_root=str(project_root),
            scope_display=scope_display,
            created=created,
            previous=previous,
            target=target,
            reference_section=reference_section,
            findings_block=findings_block,
        ).strip()
    else:
        prompt = (
            "You are synthesizing multiple prior analyses into a single architecture "
            "and operations document for an AI agent.\n\n"
            f"Workspace root: {project_root}\n"
            f"In-scope folder: {scope_display}\n"
            f"created_from_sha={created}, previous_target_sha={previous}, target_sha={target}\n\n"
            f"{reference_section}\n\n"
            f"{findings_block}\n\n"
            "Return a markdown '# Agent Doc' with the standard 1–6 sections, "
            "preserving all existing [N] citations without changing their numbers."
        )

    try:
        response = await provider.complete(
            prompt=prompt,
            system=(
                "You are synthesizing multiple prior map-level analyses into a single, precise Agent Doc for another AI agent. "
                "Focus on clarity, architectural depth, and preserving existing citations without changing their [N] numbers."
            ),
            max_completion_tokens=8000,
        )
    except Exception:
        merged = "\n\n".join(cleaned_findings)
        return f"# Agent Doc\n\n{merged}"

    if not response or not getattr(response, "content", None):
        merged = "\n\n".join(cleaned_findings)
        return f"# Agent Doc\n\n{merged}"

    merged_text = response.content.strip()
    if "# Agent Doc" not in merged_text:
        merged_text = "# Agent Doc\n\n" + merged_text

    return merged_text


async def _generate_agent_doc(
    project_root: Path,
    output_path: Path,
    scope: Optional[str] = None,
    skip_scope_trim: bool = False,
    hyde_only: bool = False,
    hyde_bootstrap: bool = False,
    generator_mode: str = "code_research",
    out_dir: Optional[Path] = None,
    debug_dump: bool = False,
    semantic_lexical_sidecar: bool = True,
    semantic_gap_fill: bool = False,
) -> None:
    """Main implementation that coordinates config, research, and writing."""
    # Ensure we resolve paths early to avoid surprises
    project_root = project_root.resolve()
    output_path = output_path.resolve()

    # Resolve scope folder within project_root
    if not scope or scope in {".", "/"}:
        scope_path = project_root
        scope_label = "/"
    else:
        candidate = (project_root / scope).resolve()
        try:
            # Ensure the scope stays inside the project root
            candidate.relative_to(project_root)
        except ValueError:
            raise SystemExit(
                f"Scope '{scope}' resolves outside project root: {candidate}"
            )
        if not candidate.exists() or not candidate.is_dir():
            raise SystemExit(f"Scope folder not found or not a directory: {candidate}")
        scope_path = candidate
        scope_label = str(candidate.relative_to(project_root))

    # Determine commit metadata (baseline, previous, current)
    existing_meta = _parse_existing_metadata(output_path)
    current_sha = _get_head_sha(project_root)

    if existing_meta is None:
        created_from_sha = current_sha
        previous_target_sha = current_sha
    else:
        created_from_sha = existing_meta.created_from_sha
        previous_target_sha = existing_meta.target_sha

    # Prepare ChunkHound config and (optionally) services.
    # Force Config to treat project_root as the target directory so that
    # .chunkhound.json and database paths are resolved relative to the
    # workspace we are documenting (not necessarily the current CWD).
    config_args = argparse.Namespace(path=str(project_root), config=None)
    config = Config(args=config_args)

    # LLM manager is required for all modes; DB/services are only required when
    # not running in HyDE-only mode.
    llm_manager: LLMManager | None = None
    try:
        if config.llm:
            utility_config, synthesis_config = config.llm.get_provider_configs()
            llm_manager = LLMManager(utility_config, synthesis_config)
    except ValueError as exc:
        raise SystemExit(
            f"LLM provider setup failed: {exc}\n"
            "Configure an LLM provider in .chunkhound.json or via CHUNKHOUND_LLM_API_KEY."
        ) from exc
    except Exception as exc:  # pragma: no cover - defensive
        raise SystemExit(f"Unexpected error setting up LLM provider: {exc}") from exc

    services: Any | None = None
    embedding_manager: EmbeddingManager | None = None

    if not hyde_only:
        db_path = config.database.path
        if not db_path:
            raise SystemExit("Database path not configured in ChunkHound config.")

        if not db_path.exists():
            raise SystemExit(
                f"Database not found at {db_path}. Run 'chunkhound index .' in the project root first."
            )

        embedding_manager = EmbeddingManager()

        try:
            if config.embedding:
                provider = EmbeddingProviderFactory.create_provider(config.embedding)
                embedding_manager.register_provider(provider, set_default=True)
        except ValueError as exc:
            raise SystemExit(
                f"Embedding provider setup failed: {exc}\n"
                "Configure embeddings in .chunkhound.json or via CHUNKHOUND_EMBEDDING__API_KEY."
            ) from exc
        except Exception as exc:  # pragma: no cover - defensive
            raise SystemExit(
                f"Unexpected error setting up embedding provider: {exc}"
            ) from exc

        services = create_services(
            db_path=db_path,
            config=config,
            embedding_manager=embedding_manager,
        )

        # Ensure the database actually contains indexed code before invoking research.
        try:
            stats = services.provider.get_stats()
        except Exception as exc:  # pragma: no cover - defensive
            raise SystemExit(f"Failed to read database stats: {exc}") from exc

        if not stats or stats.get("files", 0) == 0:
            raise SystemExit(
                "ChunkHound database contains no indexed files.\n"
                "Run 'chunkhound index .' in the project root before generating the agent doc."
            )

        if stats.get("embeddings", 0) == 0:
            raise SystemExit(
                "ChunkHound database contains no embeddings.\n"
                "Run 'chunkhound index .' without --no-embeddings so deep research can use semantic search."
            )

    # Capture LLM configuration snapshot for metadata (excluding secrets)
    llm_meta: dict[str, str] = {}
    if config.llm:
        llm = config.llm
        llm_meta["provider"] = llm.provider
        if llm.synthesis_provider:
            llm_meta["synthesis_provider"] = llm.synthesis_provider
        if llm.synthesis_model:
            llm_meta["synthesis_model"] = llm.synthesis_model
        if llm.utility_model:
            llm_meta["utility_model"] = llm.utility_model
        if llm.codex_reasoning_effort_synthesis:
            llm_meta["codex_reasoning_effort_synthesis"] = (
                llm.codex_reasoning_effort_synthesis
            )
        if llm.codex_reasoning_effort_utility:
            llm_meta["codex_reasoning_effort_utility"] = (
                llm.codex_reasoning_effort_utility
            )

    meta = AgentDocMetadata(
        created_from_sha=created_from_sha,
        previous_target_sha=previous_target_sha,
        target_sha=current_sha,
        generated_at=datetime.now(timezone.utc).isoformat(),
        llm_config=llm_meta,
        generation_stats={},
    )

    # Compute diffs for the prompt (workspace-wide), then scope-filter them
    raw_diff_since_created = _get_name_status_diff(
        meta.created_from_sha,
        meta.target_sha,
        project_root,
    )
    raw_diff_since_previous = _get_name_status_diff(
        meta.previous_target_sha,
        meta.target_sha,
        project_root,
    )

    def _filter_diff_for_scope(diff_lines: list[str]) -> list[str]:
        if scope_label == "/":
            return diff_lines
        prefix = scope_label.rstrip("/") + "/"
        filtered: list[str] = []
        for line in diff_lines:
            parts = line.split("\t", 1)
            path = parts[1] if len(parts) == 2 else ""
            if path.startswith(prefix):
                filtered.append(line)
        return filtered

    diff_since_created = _filter_diff_for_scope(raw_diff_since_created)
    diff_since_previous = _filter_diff_for_scope(raw_diff_since_previous)

    # Optional HyDE bootstrap: generate a heuristic plan from file layout and
    # feed it into deep research as planning context. This is disabled when
    # running in HyDE-only mode (which already uses HyDE output directly).
    hyde_plan: str | None = None
    if hyde_bootstrap and not hyde_only:
        file_paths = _collect_scope_files(scope_path, project_root)
        hyde_prompt = _build_hyde_scope_prompt(
            meta=meta,
            scope_label=scope_label,
            file_paths=file_paths,
        )
        hyde_plan = await _run_hyde_only_query(
            llm_manager=llm_manager,
            prompt=hyde_prompt,
        )
        # Persist HyDE plan for debugging/inspection if output directory provided
        if out_dir is not None and hyde_plan and hyde_plan.strip():
            safe_scope = scope_label.replace("/", "_") or "root"
            hyde_path = out_dir / f"hyde_plan_{safe_scope}.md"
            hyde_path.parent.mkdir(parents=True, exist_ok=True)
            hyde_path.write_text(hyde_plan, encoding="utf-8")

    debug_dir = _get_debug_dir(out_dir=out_dir, debug_dump=debug_dump)
    if debug_dir is not None:
        debug_dir.mkdir(parents=True, exist_ok=True)

    sources_footer: str | None = None

    # For code_research mode we track deep-research calls and sources across
    # all passes (overview, HyDE map, or map-only).
    unified_source_files: dict[str, str] = {}
    unified_source_chunks: list[dict[str, Any]] = []

    def _merge_sources_from_metadata(meta: dict[str, Any]) -> None:
        sources = meta.get("sources") or {}
        for fp in sources.get("files", []):
            if fp:
                unified_source_files.setdefault(fp, "")
        for c in sources.get("chunks", []):
            fp = c.get("file_path")
            if not fp:
                continue
            unified_source_chunks.append(
                {
                    "file_path": fp,
                    "start_line": c.get("start_line"),
                    "end_line": c.get("end_line"),
                }
            )

    # Map-only code_research is opt-in via environment to avoid breaking
    # existing flows.
    map_only_env = os.getenv("CH_AGENT_DOC_CODE_RESEARCH_MAP_ONLY", "0")
    code_research_map_only = (
        not hyde_only and generator_mode == "code_research" and map_only_env == "1"
    )

    body: str = ""
    total_research_calls = 0
    enable_hyde_map = False
    map_findings: list[dict[str, Any]] = []

    # Run overview research (or HyDE-only / semantic-only synthesis) for the
    # main document body, or collect map-only findings for later merge.
    if hyde_only:
        # HyDE-only mode: build prompt purely from scope file layout and LLM plan
        file_paths = _collect_scope_files(scope_path, project_root)
        overview_prompt = _build_hyde_scope_prompt(
            meta=meta,
            scope_label=scope_label,
            file_paths=file_paths,
        )
        overview_answer = await _run_hyde_only_query(
            llm_manager=llm_manager,
            prompt=overview_prompt,
        )
        body = overview_answer
    else:
        assert services is not None
        if generator_mode == "semantic":
            overview_answer = await _run_semantic_overview_query(
                services=services,
                llm_manager=llm_manager,
                meta=meta,
                scope_label=scope_label,
                hyde_plan=hyde_plan,
                debug_dir=debug_dir,
                lexical_sidecar=semantic_lexical_sidecar,
                gap_fill=semantic_gap_fill,
            )
            body = overview_answer
        elif code_research_map_only:
            assert embedding_manager is not None
            # Prefer HyDE bullets for map-only when a HyDE plan is available.
            if hyde_plan and hyde_plan.strip():
                map_findings, total_research_calls = await _run_hyde_map_passes(
                    services=services,
                    embedding_manager=embedding_manager,
                    llm_manager=llm_manager,
                    scope_label=scope_label,
                    hyde_plan=hyde_plan,
                )
            else:
                # Fallback: single overview-style map pass using the standard
                # research prompt, treated as one of the map findings.
                overview_prompt = _build_research_prompt(
                    meta,
                    diff_since_created,
                    diff_since_previous,
                    scope_label=scope_label,
                    hyde_plan=None,
                )
                answer, overview_meta = await _run_research_query_with_metadata(
                    services=services,
                    embedding_manager=embedding_manager,
                    llm_manager=llm_manager,
                    prompt=overview_prompt,
                    scope_label=scope_label,
                )
                total_research_calls = 1
                cleaned, _ = _split_sources_footer(answer)
                cleaned = cleaned.strip()
                if cleaned:
                    map_findings.append(
                        {
                            "bullet": "Overview",
                            "analysis": cleaned,
                            "sources": overview_meta.get("sources") or {},
                        }
                    )

            # Merge sources from all map-level passes.
            for finding in map_findings:
                sources_meta = finding.get("sources") or {}
                _merge_sources_from_metadata({"sources": sources_meta})
        else:
            # Standard code_research overview with optional HyDE-guided deep dives.
            overview_prompt = _build_research_prompt(
                meta,
                diff_since_created,
                diff_since_previous,
                scope_label=scope_label,
                hyde_plan=hyde_plan,
            )
            assert embedding_manager is not None
            overview_answer, overview_meta = await _run_research_query_with_metadata(
                services=services,
                embedding_manager=embedding_manager,
                llm_manager=llm_manager,
                prompt=overview_prompt,
                scope_label=scope_label,
            )

            # Deep research synthesis already appends a Sources footer. Strip it
            # so we can build a unified footer later from merged sources.
            body, _ = _split_sources_footer(overview_answer)
            _merge_sources_from_metadata(overview_meta)
            total_research_calls = 1

            # Optional HyDE-guided map loop for code_research: run follow-up
            # deep research passes per HyDE bullet and append a dedicated
            # section with the resulting deep dives. Track iteration counts for
            # generation_stats.
            hyde_map_env = os.getenv("CH_AGENT_DOC_CODE_RESEARCH_HYDE_MAP", "0")
            enable_hyde_map = (
                hyde_map_env == "1"
                and hyde_plan is not None
                and hyde_plan.strip() != ""
            )
            if enable_hyde_map:
                assert services is not None
                (
                    hyde_map_section,
                    hyde_map_sources_list,
                    hyde_calls,
                ) = await _run_hyde_map_deep_research(
                    services=services,
                    embedding_manager=embedding_manager,
                    llm_manager=llm_manager,
                    scope_label=scope_label,
                    hyde_plan=hyde_plan,
                )
                if hyde_map_section:
                    body = body.rstrip() + "\n\n" + hyde_map_section
                total_research_calls += hyde_calls
                for src_meta in hyde_map_sources_list:
                    _merge_sources_from_metadata(src_meta)

    # For any code_research mode, deduplicate the collected sources and build a
    # global file reference map that can be reused for both map-only merging
    # and the final Sources footer.
    unified_chunks_dedup: list[dict[str, Any]] = []
    ref_map: dict[str, int] = {}
    cm_for_sources: CitationManager | None = None

    if not hyde_only and generator_mode == "code_research":
        # Deduplicate chunks by (file_path, start_line, end_line)
        dedup: dict[tuple[str, int | None, int | None], dict[str, Any]] = {}
        for chunk in unified_source_chunks:
            key = (
                chunk.get("file_path"),
                chunk.get("start_line"),
                chunk.get("end_line"),
            )
            if not key[0]:
                continue
            if key not in dedup:
                dedup[key] = chunk

        unified_chunks_dedup = list(dedup.values())

        if unified_source_files and unified_chunks_dedup:
            cm_for_sources = CitationManager()
            ref_map = cm_for_sources.build_file_reference_map(
                unified_chunks_dedup, unified_source_files
            )

    # In map-only mode, perform a single agentic merge over the collected
    # map-level findings using only the synthesis provider (no DB access).
    if code_research_map_only:
        cm = cm_for_sources or CitationManager()
        reference_table = cm.format_reference_table(ref_map) if ref_map else ""
        remapped_findings: list[str] = []
        for finding in map_findings:
            analysis = finding["analysis"]
            sources_meta = finding.get("sources") or {}
            source_files = sources_meta.get("files") or []
            source_chunks = sources_meta.get("chunks") or []
            if ref_map and source_files:
                cluster_files_dict = {fp: "" for fp in source_files}
                cluster_ref_map = cm.build_file_reference_map(
                    source_chunks, cluster_files_dict
                )
                analysis = cm.remap_cluster_citations(
                    analysis, cluster_ref_map, ref_map
                )
            remapped_findings.append(analysis)

        body = await _run_map_only_merge(
            llm_manager=llm_manager,
            project_root=project_root,
            scope_label=scope_label,
            meta=meta,
            reference_table=reference_table,
            findings=remapped_findings,
        )

    # Run a final trimming pass to focus on the selected scope (HyDE
    # refinement), unless explicitly disabled. In HyDE-only mode we skip
    # trimming to expose the raw HyDE backbone. For semantic-only mode we also
    # skip trimming, because the semantic prompt is already scoped and an
    # additional ChunkHound-specific refinement pass can over-collapse
    # non-ChunkHound docs. In map-only mode, the merge prompt is already
    # scope-aware, so we skip a second trimming pass.
    if hyde_only or skip_scope_trim or generator_mode == "semantic" or code_research_map_only:
        body_to_use = body
    else:
        trimmed_body = await _trim_doc_for_scope(llm_manager, body, scope_label)
        body_to_use = trimmed_body or body

    # Build a unified Sources footer for code_research mode using the merged
    # sources metadata from overview and/or map passes. This replaces any
    # internal deep-research footer so the final tree reflects all passes.
    if not hyde_only and generator_mode == "code_research":
        if unified_source_files and unified_chunks_dedup:
            try:
                cm = cm_for_sources or CitationManager()
                sources_footer = cm.build_sources_footer(
                    unified_chunks_dedup, unified_source_files, ref_map or None
                )
                # Remove any residual Sources footer that might have slipped in
                # from earlier flows, then append the unified one.
                if "## Sources" in body_to_use:
                    cleaned_body, _ = _split_sources_footer(body_to_use)
                    body_to_use = cleaned_body
                body_to_use = body_to_use.rstrip() + "\n\n" + sources_footer
            except Exception:
                # Footer is best-effort; never break main generation.
                pass

    # Populate generation_stats for visibility into the generation process.
    gen_stats: dict[str, str] = {}
    gen_stats["generator_mode"] = generator_mode
    gen_stats["hyde_bootstrap"] = "1" if hyde_bootstrap else "0"
    gen_stats["hyde_map_enabled"] = "1" if enable_hyde_map else "0"
    if code_research_map_only:
        gen_stats["code_research_map_only"] = "1"
    if not hyde_only and generator_mode == "code_research":
        gen_stats["code_research_total_calls"] = str(total_research_calls)
        gen_stats["code_research_sources_files"] = str(len(unified_source_files))
        gen_stats["code_research_sources_chunks"] = str(len(unified_chunks_dedup))
        # Compute coverage stats for the scoped folder using the database as
        # ground truth for which files and chunks are actually indexed in ChunkHound.
        scope_total_files = 0
        scope_total_chunks = 0
        try:
            provider = getattr(services, "provider", None)
            if provider is not None:
                chunks_meta = provider.get_all_chunks_with_metadata()
                scoped_files: set[str] = set()
                prefix = None if scope_label == "/" else scope_label.rstrip("/") + "/"
                for chunk in chunks_meta:
                    path = (chunk.get("file_path") or "").replace("\\", "/")
                    if not path:
                        continue
                    if prefix and not path.startswith(prefix):
                        continue
                    scoped_files.add(path)
                    scope_total_chunks += 1
                scope_total_files = len(scoped_files)
        except Exception:
            scope_total_files = 0
            scope_total_chunks = 0

        if scope_total_files > 0:
            gen_stats["scope_total_files_indexed"] = str(scope_total_files)
            coverage_ratio = float(len(unified_source_files)) / float(scope_total_files)
            gen_stats["scope_coverage_percent_indexed"] = f"{coverage_ratio * 100.0:.2f}"
        if scope_total_chunks > 0:
            gen_stats["scope_total_chunks_indexed"] = str(scope_total_chunks)
            chunk_cov = float(len(unified_chunks_dedup)) / float(scope_total_chunks)
            gen_stats["scope_chunk_coverage_percent_indexed"] = f"{chunk_cov * 100.0:.2f}"
        limit_env = os.getenv("CH_AGENT_DOC_CODE_RESEARCH_HYDE_MAP_LIMIT")
        if limit_env:
            gen_stats["hyde_map_limit_env"] = limit_env

    # Merge into metadata for header rendering (existing values win if set).
    meta.generation_stats.update({k: v for k, v in gen_stats.items() if v != ""})

    # For semantic mode with debug enabled, compute a coverage summary using the
    # merged semantic results and actual scope files. This is purely diagnostic.
    if generator_mode == "semantic" and debug_dir is not None:
        try:
            summary = _compute_semantic_coverage_summary(
                project_root=project_root,
                scope_path=scope_path,
                scope_label=scope_label,
                debug_dir=debug_dir,
            )
            if summary:
                (debug_dir / "semantic_coverage_summary.json").write_text(
                    json.dumps(summary, indent=2),
                    encoding="utf-8",
                )
        except Exception:
            # Metrics are best-effort; never break main generation.
            pass

    # Write monolithic doc with metadata header
    metadata_block = _format_metadata_block(meta)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    final_doc = metadata_block + body_to_use
    output_path.write_text(final_doc, encoding="utf-8")

    # Also emit split per-subsystem docs for agentic search
    meta_dict: dict[str, Any] = {
        "created_from_sha": meta.created_from_sha,
        "previous_target_sha": meta.previous_target_sha,
        "target_sha": meta.target_sha,
        "generated_at": meta.generated_at,
    }
    if meta.llm_config:
        meta_dict["llm_config"] = meta.llm_config
    # Also emit split per-subsystem docs for agentic search if out_dir provided
    if out_dir is not None:
        split_agent_doc_text(body_to_use, meta_dict, base_output_dir=out_dir)


def main() -> None:
    """Entry point for CLI usage."""
    parser = argparse.ArgumentParser(
        description=(
            "Generate or update a single agent-facing architecture/operations document "
            "for a scoped folder using ChunkHound's deep research pipeline."
        )
    )
    parser.add_argument(
        "--path",
        type=Path,
        default=Path("."),
        help="Project root path (defaults to current directory).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_RELATIVE,
        help=(
            "Output markdown file path for the agent doc. "
            "Defaults to operations/chunkhound_agent_doc.md."
        ),
    )
    parser.add_argument(
        "--scope",
        type=str,
        default=None,
        help=(
            "Relative folder within the project to document "
            "(e.g. 'chunkhound', '.' or '/' for entire workspace). "
            "If omitted, you will be prompted interactively."
        ),
    )
    parser.add_argument(
        "--no-scope-trim",
        action="store_true",
        help=(
            "Disable the final scope-aware trimming pass and output the raw "
            "deep research / HyDE-generated document instead."
        ),
    )
    parser.add_argument(
        "--hyde-only",
        action="store_true",
        help=(
            "Use HyDE-only mode: call the synthesis LLM directly with HyDE "
            "prompts, without deep_research or database/embedding access."
        ),
    )
    parser.add_argument(
        "--hyde-bootstrap",
        action="store_true",
        help=(
            "Run a HyDE planning pass over the scoped folder and feed the "
            "result into deep_research as planning context (do not copy "
            "phrases from the plan; re-derive everything from real code)."
        ),
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["code_research", "semantic"],
        default="code_research",
        help=(
            "Generation mode for the final document. "
            "'code_research' (default) uses deep research; "
            "'semantic' uses a faster semantic-search-only overview."
        ),
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("operations"),
        help=(
            "Base directory for generated documentation and HyDE planning files "
            "(relative to workspace root unless absolute). The main doc will be "
            "written as '<out-dir>/chunkhound_agent_doc.md' unless --output is "
            "explicitly provided."
        ),
    )
    parser.add_argument(
        "--debug-dump",
        action="store_true",
        help=(
            "Enable debug dumps for agent-doc runs. When set (or when "
            "CH_AGENT_DOC_DEBUG=1), semantic mode writes intermediate "
            "artifacts under '<out-dir>/debug/'."
        ),
    )
    parser.add_argument(
        "--no-semantic-lexical-sidecar",
        action="store_true",
        help=(
            "Disable the optional lexical sidecar for semantic mode. By default, "
            "semantic mode may run a lightweight regex search alongside dense "
            "semantic search to broaden coverage within the scoped path."
        ),
    )
    parser.add_argument(
        "--semantic-gap-fill",
        action="store_true",
        help=(
            "Enable an experimental gap-filling pass for semantic mode. When enabled, "
            "semantic search may issue additional small regex searches based on file "
            "names discovered in the initial semantic landscape to pull in snippets "
            "from additional files within the scoped folder."
        ),
    )

    args = parser.parse_args()

    try:
        scope = args.scope
        if scope is None:
            try:
                user_input = input(
                    "Folder to document (relative to project root, '.' or '/' for entire workspace): "
                ).strip()
                scope = user_input or "."
            except EOFError:
                scope = "."

        # Resolve project root and output directory
        project_root = args.path.resolve()

        # Determine output directory relative to project root
        if args.output != DEFAULT_OUTPUT_RELATIVE:
            # Explicit output path wins; out-dir is derived from it
            output_path = (
                args.output
                if args.output.is_absolute()
                else (project_root / args.output).resolve()
            )
            out_dir = output_path.parent
        else:
            if args.out_dir.is_absolute():
                out_dir = args.out_dir.resolve()
            else:
                out_dir = (project_root / args.out_dir).resolve()
            output_path = out_dir / DEFAULT_OUTPUT_RELATIVE.name

        debug_dump = bool(
            args.debug_dump or os.getenv("CH_AGENT_DOC_DEBUG") == "1"
        )

        # Lexical sidecar is disabled by default and can be enabled via env.
        # CLI flag only disables it when explicitly enabled in the environment.
        sidecar_env = os.getenv("CH_AGENT_DOC_SEMANTIC_LEXICAL_SIDECAR", "0")
        sidecar_from_env = sidecar_env == "1"
        semantic_lexical_sidecar = sidecar_from_env and not args.no_semantic_lexical_sidecar

        # Gap-filling is experimental and opt-in. It can be enabled via env or CLI.
        gap_fill_env = os.getenv("CH_AGENT_DOC_SEMANTIC_GAP_FILL", "0")
        semantic_gap_fill = gap_fill_env == "1" or args.semantic_gap_fill

        asyncio.run(
            _generate_agent_doc(
                project_root=project_root,
                output_path=output_path,
                scope=scope,
                skip_scope_trim=args.no_scope_trim,
                hyde_only=args.hyde_only,
                hyde_bootstrap=args.hyde_bootstrap,
                generator_mode=args.mode,
                out_dir=out_dir,
                debug_dump=debug_dump,
                semantic_lexical_sidecar=semantic_lexical_sidecar,
                semantic_gap_fill=semantic_gap_fill,
            )
        )
    except KeyboardInterrupt:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
