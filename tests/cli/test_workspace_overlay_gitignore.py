from __future__ import annotations

import subprocess
from pathlib import Path


def _run(cmd: list[str], cwd: Path | None = None, timeout: int = 25) -> subprocess.CompletedProcess[str]:
    return subprocess.run(["uv", "run", *cmd], cwd=str(cwd) if cwd else None, text=True, capture_output=True, timeout=timeout)


def test_workspace_overlay_applies_to_child_repos(tmp_path: Path) -> None:
    root = tmp_path

    # Workspace-level .gitignore excludes 'vendor/'
    (root / ".gitignore").write_text("vendor/\n")

    # Two child repos
    (root / "repoA" / ".git").mkdir(parents=True, exist_ok=True)
    (root / "repoA" / "vendor").mkdir(parents=True, exist_ok=True)
    (root / "repoA" / "vendor" / "file.txt").write_text("x")

    (root / "repoB" / ".git").mkdir(parents=True, exist_ok=True)
    (root / "repoB" / "src").mkdir(parents=True, exist_ok=True)
    (root / "repoB" / "src" / "main.txt").write_text("x")

    # CH config: use .gitignore sentinel and enable workspace overlay
    (root / ".chunkhound.json").write_text('{"indexing": {"exclude": ".gitignore", "workspace_gitignore_overlay": true}}\n')

    proc = _run(["chunkhound", "index", "--simulate", str(root)])
    assert proc.returncode == 0, proc.stderr
    out = set(proc.stdout.strip().splitlines())

    # Workspace vendor rule excludes vendor/ under repoA
    assert "repoA/vendor/file.txt" not in out
    # Non-excluded repoB/src/main.txt remains
    assert "repoB/src/main.txt" in out

