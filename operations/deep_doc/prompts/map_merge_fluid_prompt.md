You are synthesizing multiple prior analyses into a single architecture and operations document for an AI agent.

Workspace context:
- workspace root: {project_root}
- in-scope folder: {scope_display}

Commit metadata for this documentation build:
- created_from_sha (first documentation baseline): {created}

Source references:
{reference_section}

Each finding below comes from a separate map-level analysis pass over the scoped folder. Each finding:
- Already includes inline [N] citations that refer to the source references above.
- May overlap with or contradict other findings.

<<<FINDINGS>>>
{findings_block}
<<<ENDFINDINGS>>>

Your task:
- Read and integrate all findings into a single coherent documentation set suitable for future AI agents that cannot call tools.
- Produce a single markdown document that starts with:
  # Agent Doc
- Below that root heading, choose natural `##` and `###` sections that best reflect the actual architecture, subsystems, and operational flows implied by the findings. Do NOT feel constrained to a fixed numbered 1–6 scheme.
- Ensure that, somewhere in the document, you clearly cover:
  - Project identity and critical constraints (especially performance, batching, and concurrency rules).
  - Overall architecture and lifecycle for the in-scope folder: how requests, commands, or jobs flow through the system.
  - Major subsystems or modules, their responsibilities, and how they interact.
  - Operations, CLI/API entrypoints, testing strategy, and how to safely run or extend the system.
  - Known pitfalls, gotchas, and debugging patterns visible from error handling, tests, and operations code.
- When summarizing structured information (for example: subsystems and their responsibilities, commands and their effects, configuration flags and their impact, or common failure modes and symptoms), prefer concise markdown tables to make comparisons and relationships easy to scan.
- Under the main architecture and subsystem sections, include one or more Mermaid diagrams that make the system structure as clear as possible, covering:
  - Intra-subsystem flows (within a module or package).
  - Inter-subsystem interactions (between modules, services, and entrypoints).
  - Key data flows, boundaries, and external dependencies (e.g., databases, queues, MCP servers).
  - Service call graphs and control flow where helpful.
  Use standard Mermaid syntax (for example: flowchart, sequence diagrams, or class diagrams) and base all diagrams strictly on the findings and source references. To avoid syntax errors:
  - Use simple snake_case node IDs without spaces or punctuation (e.g., `core_db`, `cli_entry`, `mcp_server`).
  - Always write nodes as `id["Human readable label"]` (ID first, then a label in double quotes inside brackets).
  - Put spaces, slashes, and other punctuation only in the label, never in the ID.
  - Do NOT include citation markers like `[3]` or `[N]` inside any Mermaid diagram; keep citations in the surrounding markdown text.
  - Do NOT append extra text after a node label before an edge (avoid patterns like `node\"][3] --> target`).
- Integrate all findings, remove duplication, and resolve contradictions; when in doubt, prefer the most precise or conservative description.
- Preserve all existing [N] citations from the findings and keep them attached to the statements they support.
- Do NOT invent new [N] citations or change citation numbers.
- Do NOT mention internal tools such as deep research pipelines, HyDE planning, or map passes; just describe the project based on the analyses.
- Do NOT include a "## Sources" footer; the caller will append it.
- Do NOT include any metadata comment header; only return the markdown body starting from "# Agent Doc".
