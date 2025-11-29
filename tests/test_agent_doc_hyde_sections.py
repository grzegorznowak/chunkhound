import textwrap

from operations.deep_doc.deep_doc import _extract_hyde_sections


def test_extract_hyde_sections_groups_global_and_subsystem_hooks():
    plan = textwrap.dedent(
        """
        # HyDE Research Plan for ./scope

        ## Scope Summary
        This folder appears to encapsulate a generic subsystem with commands,
        services, configuration, and presentation assets.

        ## Global Hooks
        - Sketch the likely end-to-end lifecycle: which entrypoints initiate runs,
          how orchestration components assemble data, and how final artifacts are produced.
        - Determine how configuration likely influences controllers, jobs, or services,
          and where environment-specific behavior might live.

        ## Subsystem Hooks

        ### Commands / Jobs
        - Analyze each console command's likely responsibilities and how they orchestrate
          background jobs or workers.
        - Understand how job queues might coordinate retries, state tracking, and error handling.

        ### Services / Orchestration
        - Explore how core services probably compose external APIs, repositories, and
          content processors into a cohesive pipeline.

        ### Views / Templates
        - Catalog the different template directories to infer how multiple themes or
          layouts might be supported for the same logical output.
        """
    ).strip()

    sections = _extract_hyde_sections(plan, max_sections=10_000)

    # Expect one section for Global Hooks plus one per "###" subsection.
    titles = [s["title"] for s in sections]

    # Global Hooks title is synthesized from the first bullet and may be truncated.
    assert any("Sketch the likely end-to-end lifecycle" in t for t in titles)
    assert "Commands / Jobs" in titles
    assert "Services / Orchestration" in titles
    assert "Views / Templates" in titles

    # Global Hooks body should contain both bullets.
    global_index = next(i for i, t in enumerate(titles) if "Sketch the likely end-to-end lifecycle" in t)
    global_section = sections[global_index]
    assert "- Sketch the likely end-to-end lifecycle" in global_section["body"]
    assert "- Determine how configuration likely influences controllers" in global_section["body"]
