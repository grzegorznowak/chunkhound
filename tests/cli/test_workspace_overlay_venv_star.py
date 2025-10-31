from __future__ import annotations

import subprocess
from pathlib import Path


def _run(cmd: list[str], cwd: Path | None = None, timeout: int = 25) -> subprocess.CompletedProcess[str]:
    return subprocess.run(["uv", "run", *cmd], cwd=str(cwd) if cwd else None, text=True, capture_output=True, timeout=timeout)


def test_workspace_overlay_venv_star_excludes_nested(tmp_path: Path) -> None:
    root = tmp_path
    (root / ".gitignore").write_text("""
.venv/
.venv*/
""".strip()+"\n")

    (root / "repo" / ".git").mkdir(parents=True, exist_ok=True)
    (root / "repo" / ".venv-docling" / "lib").mkdir(parents=True, exist_ok=True)
    (root / "repo" / ".venv-docling" / "lib" / "x.txt").write_text("x")
    (root / "repo" / "src").mkdir(parents=True, exist_ok=True)
    (root / "repo" / "src" / "ok.txt").write_text("x")

    (root / ".chunkhound.json").write_text('{"indexing": {"exclude": ".gitignore", "workspace_gitignore_overlay": true}}\n')

    proc = _run(["chunkhound", "index", "--simulate", str(root)])
    assert proc.returncode == 0, proc.stderr
    out = set(proc.stdout.strip().splitlines())
    assert "repo/.venv-docling/lib/x.txt" not in out
    assert "repo/src/ok.txt" in out
