"""IgnoreEngine: central exclusion logic with gitwildmatch semantics.

Initial implementation supports root-level .gitignore and .chignore files
via the `pathspec` library using gitwildmatch patterns. This is sufficient to
make the initial tests pass; we will extend it to per-directory inheritance
and richer rule origins in follow-up steps.
"""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
from typing import Iterable, Optional, Dict, List, Tuple

try:
    from pathspec import PathSpec
    from pathspec.patterns.gitwildmatch import GitWildMatchPattern
except Exception as e:  # pragma: no cover - import error surfaced at runtime
    PathSpec = None  # type: ignore
    GitWildMatchPattern = None  # type: ignore


@dataclass
class MatchInfo:
    matched: bool
    source: Optional[Path] = None
    pattern: Optional[str] = None


class IgnoreEngine:
    def __init__(self, root: Path, compiled_specs: list[tuple[Path, "PathSpec"]]):
        self.root = root.resolve()
        self._compiled_specs = compiled_specs

    def matches(self, path: Path, is_dir: bool) -> Optional[MatchInfo]:
        # Normalize to root-relative POSIX path
        try:
            rel = path.resolve().relative_to(self.root)
        except Exception:
            rel = path.resolve()
        rel_str = rel.as_posix()

        # Evaluate specs in precedence order; first match wins
        for src, spec in self._compiled_specs:
            if spec.match_file(rel_str) or (is_dir and spec.match_file(rel_str + "/")):
                return MatchInfo(matched=True, source=src, pattern=None)
        return None


def _compile_gitwildmatch(patterns: Iterable[str]) -> "PathSpec":
    if PathSpec is None or GitWildMatchPattern is None:
        raise RuntimeError(
            "pathspec is required for IgnoreEngine; please add dependency 'pathspec'"
        )
    return PathSpec.from_lines(GitWildMatchPattern, patterns)


def build_ignore_engine(
    root: Path,
    sources: list[str],
    chignore_file: str = ".chignore",
    config_exclude: Optional[Iterable[str]] = None,
) -> IgnoreEngine:
    """Build an IgnoreEngine for the given root and sources.

    Currently supports:
    - gitignore: uses only the root-level .gitignore file
    - chignore: uses a root-level .chignore file
    - config: uses provided glob-like patterns (gitwildmatch semantics)
    """
    compiled: list[tuple[Path, PathSpec]] = []
    root = root.resolve()

    # Always enforce config_exclude (default excludes) regardless of sources
    if config_exclude:
        compiled.append((root, _compile_gitwildmatch(config_exclude)))

    for src in sources:
        if src == "gitignore":
            # Collect and transform .gitignore rules across the tree to root-relative patterns
            pre_spec = None
            if config_exclude:
                pre_spec = _compile_gitwildmatch(config_exclude)
            pats = _collect_gitignore_patterns(root, pre_spec)
            if pats:
                compiled.append((root / ".gitignore", _compile_gitwildmatch(pats)))
        elif src == "chignore":
            ci = root / chignore_file
            if ci.exists():
                compiled.append((ci, _compile_gitwildmatch(ci.read_text().splitlines())))
        elif src == "config":
            pats = list(config_exclude or [])
            if pats:
                compiled.append((root, _compile_gitwildmatch(pats)))

    return IgnoreEngine(root, compiled)


def _collect_gitignore_patterns(root: Path, pre_exclude_spec: Optional["PathSpec"] = None) -> list[str]:
    """Return root-relative gitwildmatch patterns transformed from .gitignore files.

    We walk the directory tree top-down so that root patterns appear before
    child directory patterns; last match still wins in PathSpec.
    """
    out: list[str] = []
    root = root.resolve()
    for dirpath, dirnames, filenames in os.walk(root, topdown=True):
        dpath = Path(dirpath)
        # Prune excluded subtrees early based on config excludes (e.g., node_modules)
        if pre_exclude_spec is not None:
            rel_base = "." if dpath == root else dpath.relative_to(root).as_posix()
            # Mutate dirnames in-place to prevent descending
            to_remove = []
            for dn in dirnames:
                child = dn if rel_base == "." else f"{rel_base}/{dn}"
                if pre_exclude_spec.match_file(child) or pre_exclude_spec.match_file(child + "/"):
                    to_remove.append(dn)
            for dn in to_remove:
                dirnames.remove(dn)
        gi = dpath / ".gitignore"
        if not gi.exists():
            continue
        try:
            lines = gi.read_text(encoding="utf-8", errors="ignore").splitlines()
        except Exception:
            continue
        rel_from_root = dpath.relative_to(root)
        dir_rel = "." if str(rel_from_root) == "." else rel_from_root.as_posix()
        for raw in lines:
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            out.extend(_transform_gitignore_line(dir_rel, line))
    return out


def _detect_repo_roots(root: Path, pre_exclude_spec: Optional["PathSpec"] = None) -> list[Path]:
    """Detect Git repository roots under root by looking for .git dir or file.

    Prunes excluded subtrees using pre_exclude_spec (e.g., node_modules) to
    avoid unnecessary traversal.
    """
    roots: list[Path] = []
    root = root.resolve()
    for dirpath, dirnames, filenames in os.walk(root, topdown=True):
        dpath = Path(dirpath)

        # Prune excluded dirs
        if pre_exclude_spec is not None:
            rel_base = "." if dpath == root else dpath.relative_to(root).as_posix()
            to_remove = []
            for dn in dirnames:
                child = dn if rel_base == "." else f"{rel_base}/{dn}"
                if pre_exclude_spec.match_file(child) or pre_exclude_spec.match_file(child + "/"):
                    to_remove.append(dn)
            for dn in to_remove:
                dirnames.remove(dn)

        # Repo root if .git dir exists or .git file exists (submodule)
        if (dpath / ".git").is_dir() or (dpath / ".git").is_file():
            roots.append(dpath)
    # Sort deepest first for nearest-ancestor selection convenience later
    roots.sort(key=lambda p: len(p.as_posix()))
    return roots


class RepoAwareIgnoreEvaluator:
    """Repo-boundary aware evaluator that selects per-repo engines by path.

    - For a path inside a detected repo root, use only that repo's engine.
    - For paths outside any repo, use a workspace-scoped engine.
    - Config excludes are compiled into each engine and used to prune during
      .gitignore collection.
    """

    def __init__(
        self,
        workspace_root: Path,
        repo_roots: list[Path],
        sources: list[str],
        chignore_file: str,
        config_exclude: Optional[Iterable[str]] = None,
        overlay_patterns: Optional[Iterable[str]] = None,
    ) -> None:
        self.root = workspace_root.resolve()
        self.repo_roots = sorted([p.resolve() for p in repo_roots], key=lambda p: len(p.as_posix()), reverse=True)
        self.sources = sources
        self.chignore_file = chignore_file
        self.config_exclude = list(config_exclude or [])

        # Build per-repo engines
        self._per_repo: Dict[Path, IgnoreEngine] = {}
        for rr in self.repo_roots:
            self._per_repo[rr] = build_ignore_engine(rr, sources, chignore_file, self.config_exclude)
        # Workspace engine for non-repo areas (may include workspace-level .gitignore)
        self._workspace_engine = build_ignore_engine(self.root, sources, chignore_file, self.config_exclude)

        # Optional workspace overlay: a global PathSpec compiled from the CH root .gitignore
        self._overlay_spec = None
        if overlay_patterns:
            try:
                self._overlay_spec = _compile_gitwildmatch(list(overlay_patterns))
            except Exception:
                self._overlay_spec = None

    def _nearest_repo(self, path: Path) -> Optional[Path]:
        p = path.resolve()
        for rr in self.repo_roots:
            try:
                p.relative_to(rr)
                return rr
            except Exception:
                continue
        return None

    def matches(self, path: Path, is_dir: bool) -> Optional[MatchInfo]:
        # Global overlay first (non-negatable by repo rules)
        try:
            if self._overlay_spec is not None:
                rel = path.resolve().relative_to(self.root).as_posix()
                if self._overlay_spec.match_file(rel) or (is_dir and self._overlay_spec.match_file(rel + "/")):
                    return MatchInfo(matched=True, source=self.root / ".gitignore", pattern=None)
        except Exception:
            pass

        rr = self._nearest_repo(path)
        if rr is not None:
            return self._per_repo[rr].matches(path, is_dir)
        return self._workspace_engine.matches(path, is_dir)


def build_repo_aware_ignore_engine(
    root: Path,
    sources: list[str],
    chignore_file: str = ".chignore",
    config_exclude: Optional[Iterable[str]] = None,
    workspace_overlay: bool = False,
) -> RepoAwareIgnoreEvaluator:
    pre_spec = _compile_gitwildmatch(config_exclude or []) if (config_exclude) else None
    repo_roots = _detect_repo_roots(root, pre_spec)
    overlay_patterns = None
    if workspace_overlay:
        gi = root.resolve() / ".gitignore"
        if gi.exists():
            try:
                overlay_patterns = gi.read_text(encoding="utf-8", errors="ignore").splitlines()
            except Exception:
                overlay_patterns = None
    return RepoAwareIgnoreEvaluator(root, repo_roots, sources, chignore_file, config_exclude, overlay_patterns)



def _transform_gitignore_line(dir_rel: str, line: str) -> list[str]:
    """Transform a .gitignore pattern from a directory into root-relative patterns.

    Handles negation (!), anchored (/), and directory-only (trailing /) forms by
    emitting patterns that constrain the match to the originating subtree.
    """
    neg = False
    if line.startswith("!"):
        neg = True
        line = line[1:]

    # Directory-only patterns (ending with '/')
    is_dir_pat = line.endswith("/")
    if is_dir_pat:
        line = line[:-1]

    # Build base (relative to root) for anchored vs unanchored
    parts: list[str] = []

    def add(p: str) -> None:
        if is_dir_pat:
            p = f"{p}/**"
        if neg:
            p = "!" + p
        parts.append(p)

    # Resolve directory prefix
    if dir_rel == ".":
        # root-level
        if line.startswith("/"):
            # Keep leading slash to enforce root anchoring semantics
            add(line)
        else:
            # both direct and nested under root
            add(line)
            add(f"**/{line}")
    else:
        # subdirectory scope
        if line.startswith("/"):
            # anchored to this directory
            add(f"{dir_rel}/{line[1:]}")
        else:
            # both direct and nested within this subtree
            add(f"{dir_rel}/{line}")
            add(f"{dir_rel}/**/{line}")

    return parts


__all__ = ["IgnoreEngine", "MatchInfo", "build_ignore_engine"]
