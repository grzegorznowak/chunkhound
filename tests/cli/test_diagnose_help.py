from __future__ import annotations

import subprocess


def _run(cmd: list[str], timeout: int = 20) -> subprocess.CompletedProcess[str]:
    return subprocess.run(["uv", "run", *cmd], text=True, capture_output=True, timeout=timeout)


def test_diagnose_help() -> None:
    proc = _run(["chunkhound", "diagnose", "--help"]) 
    assert proc.returncode == 0, proc.stderr
    assert "diagnose" in proc.stdout.lower()

