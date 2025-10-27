
"""Utilities for normalizing Helm-style templated YAML before parsing."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

_PLACEHOLDER_BLOCK = "__CH_TPL_BLOCK__"
_PLACEHOLDER_ITEM = "__CH_TPL_ITEM__"
_PLACEHOLDER_MAP_KEY = "__CH_TPL_MAP__"
_BLOCK_TEMPLATE_HINTS = (
    "nindent",
    "toyaml",
    "toYaml",
    " indent",
    "indent(",
    "tplvalues.render",
)

# Non-YAML fragments where RapidYAML brings little value and often fails
_NON_YAML_MARKERS = (
    "<?xml",
    "server {",
    "%%httpGet",
)


_CTRL_KEYWORDS = (
    "if",
    "else if",
    "else",
    "with",
    "range",
    "end",
    "block",
    "define",
)

# Non-YAML fragments where RapidYAML brings little value and often fails
_NON_YAML_MARKERS = (
    "<?xml",
    "server {",
    "%%httpGet",
)

# Threshold of templated lines inside a single block scalar beyond which
# we pre-skip trying RapidYAML entirely.
_BLOCK_SCALAR_TPL_THRESHOLD = 12


@dataclass(frozen=True)
class TemplateRewrite:
    """Metadata describing a sanitizer substitution."""

    line: int
    indent: int
    kind: str
    snippet: str


@dataclass(frozen=True)
class SanitizedYaml:
    """Result of sanitizing a templated YAML document."""

    text: str
    rewrites: list[TemplateRewrite] = field(default_factory=list)
    pre_skip: bool = False
    pre_skip_reason: str | None = None
    pre_skip: bool = False
    pre_skip_reason: str | None = None

    @property
    def changed(self) -> bool:
        return bool(self.rewrites)


def sanitize_helm_templates(content: str) -> SanitizedYaml:
    """Best-effort normalization for Helm templated YAML."""

    if not content:
        return SanitizedYaml(text=content, rewrites=[])

    lines = content.splitlines(keepends=True)
    sanitized: list[str] = []
    rewrites: list[TemplateRewrite] = []

    in_block_scalar = False
    block_scalar_indent = 0
    in_template_comment = False

    # Pre-skip for obvious non-YAML fragments
    if any(marker in content for marker in _NON_YAML_MARKERS):
        return SanitizedYaml(
            text=content,
            rewrites=[],
            pre_skip=True,
            pre_skip_reason="non_yaml_fragment",
        )

    # Pre-skip: templated key (e.g., "{{ template ... }}": value)
    for _ln in content.splitlines():
        s = _ln.strip()
        # Find colon boundary and inspect the key portion
        if ":" in s:
            key_part = s.split(":", 1)[0].strip()
            if (key_part.startswith(("'", '"')) and key_part.endswith(("'", '"')) and "{{" in key_part and "}}" in key_part):
                return SanitizedYaml(
                    text=content,
                    rewrites=[],
                    pre_skip=True,
                    pre_skip_reason="templated_key",
                )

    # Pre-skip for files that clearly embed non-YAML fragments
    if any(marker in content for marker in _NON_YAML_MARKERS):
        return SanitizedYaml(
            text=content,
            rewrites=[],
            pre_skip=True,
            pre_skip_reason="non_yaml_fragment",
        )

    templated_lines_in_block = 0

    for idx, original in enumerate(lines, start=1):
        core, newline = _split_newline(original)
        indent_chars, stripped = _split_indent(core)

        if in_template_comment and not stripped.startswith("{{"):
            if stripped:
                rewrites.append(
                    TemplateRewrite(
                        line=idx,
                        indent=len(indent_chars),
                        kind="comment_block",
                        snippet=_shorten(stripped),
                    )
                )
                sanitized.append(f"{indent_chars}# {stripped}{newline}")
            else:
                sanitized.append(indent_chars + newline if indent_chars else newline)
            if "*/" in stripped:
                in_template_comment = False
            continue

        if in_block_scalar:
            if stripped.startswith("{{"):
                body = _extract_template_body(stripped)
                comment = f"{indent_chars}  # CH_TPL_BLOCK: {body}{newline}"
                rewrites.append(
                    TemplateRewrite(
                        line=idx,
                        indent=len(indent_chars) + 2,
                        kind="block_directive",
                        snippet=_shorten(body),
                    )
                )
                sanitized.append(comment)
                templated_lines_in_block += 1
                continue
            if stripped and len(indent_chars) > block_scalar_indent:
                sanitized.append(original)
                continue
            in_block_scalar = False

        template_body = (
            _extract_template_body(stripped) if stripped.startswith("{{") else None
        )
        start_comment_block = False
        if template_body and template_body.startswith("/*"):
            start_comment_block = True

        rewritten = _rewrite_line(
            stripped=stripped,
            indent=indent_chars,
            newline=newline,
            record=lambda kind, snippet, _idx=idx, _indent=len(indent_chars): rewrites.append(
                TemplateRewrite(
                    line=_idx,
                    indent=_indent,
                    kind=kind,
                    snippet=_shorten(snippet),
                )
            ),
        )

        sanitized.append(rewritten)

        if start_comment_block and template_body and "*/" not in template_body:
            in_template_comment = True

        if _starts_block_scalar(rewritten, indent_chars):
            in_block_scalar = True
            block_scalar_indent = len(indent_chars)

    # If we scrubbed too many templated lines inside literals, pre-skip ryml
    if templated_lines_in_block > _BLOCK_SCALAR_TPL_THRESHOLD:
        return SanitizedYaml(
            text="".join(sanitized),
            rewrites=rewrites,
            pre_skip=True,
            pre_skip_reason="block_scalar_template_heavy",
        )

    sanitized_text = "".join(sanitized)
    return SanitizedYaml(text=sanitized_text, rewrites=rewrites)


def _rewrite_line(
    *,
    stripped: str,
    indent: str,
    newline: str,
    record: Callable[[str, str], None],
) -> str:
    """Rewrite a single YAML line if it contains templating constructs."""

    if not stripped:
        return indent + newline

    if stripped.startswith("{{"):
        return indent + _rewrite_template_only_line(stripped, newline, record)

    if ":" in stripped:
        before, after = stripped.split(":", 1)
        if "{{" in after:
            record("inline_map", stripped)
            key = before.rstrip()
            template_note = after.strip()
            if any(token in template_note for token in _BLOCK_TEMPLATE_HINTS):
                child_indent = indent + "  "
                nl = newline or "\n"
                lines = [f"{indent}{key}:"]
                lines.append(f"{child_indent}{_PLACEHOLDER_MAP_KEY}: \"{_PLACEHOLDER_BLOCK}\"")
                if template_note:
                    lines.append(f"{child_indent}# CH_TPL_INLINE: {template_note}")
                return nl.join(lines) + newline
            spacing = "" if key.endswith(" ") else " "
            result = f'{indent}{key}:{spacing}"{_PLACEHOLDER_BLOCK}"{newline}'
            if template_note:
                comment_indent = indent + "  "
                result += f"{comment_indent}# CH_TPL_INLINE: {template_note}{newline}"
            return result

    stripped_l = stripped.lstrip()
    if stripped_l.startswith("-"):
        remainder = stripped_l[1:].lstrip()
        if remainder.startswith("{{"):
            record("seq_item", stripped)
            return f'{indent}- "{_PLACEHOLDER_ITEM}"{newline}'

    return indent + stripped + newline


def _rewrite_template_only_line(
    stripped: str, newline: str, record: Callable[[str, str], None]
) -> str:
    body = _extract_template_body(stripped)

    lowered = body.lower()
    if any(lowered.startswith(keyword) for keyword in _CTRL_KEYWORDS):
        record("control", stripped)
        return f"# CH_TPL_CTRL: {body}{newline}"

    if ":=" in body:
        record("assignment", stripped)
        return f"# CH_TPL_SET: {body}{newline}"

    record("template", stripped)
    return f"# CH_TPL_INCLUDE: {body}{newline}"


def _extract_template_body(stripped: str) -> str:
    body = stripped[2:]
    if body.startswith("-"):
        body = body[1:]
    body = body.strip()
    if body.endswith("}}"):  # {{ ... }}
        body = body[:-2].rstrip()
    elif body.endswith("-}}"):  # {{- ... -}}
        body = body[:-3].rstrip()
    return body


def _shorten(snippet: str, limit: int = 60) -> str:
    snippet = snippet.strip()
    if len(snippet) <= limit:
        return snippet
    return snippet[: limit - 3] + "..."


def _split_indent(text: str) -> tuple[str, str]:
    idx = 0
    length = len(text)
    while idx < length and text[idx] in (" ", "	"):
        idx += 1
    return text[:idx], text[idx:]


def _split_newline(line: str) -> tuple[str, str]:
    if line.endswith("\r\n"):
        return line[:-2], "\r\n"
    if line.endswith("\n"):
        return line[:-1], "\n"
    if line.endswith("\r"):
        return line[:-1], "\r"
    return line, ""


def _starts_block_scalar(line: str, indent: str) -> bool:
    stripped = line[len(indent) :]
    colon_idx = stripped.find(":")
    if colon_idx == -1:
        return False
    suffix = stripped[colon_idx + 1 :].strip()
    return suffix.startswith("|") or suffix.startswith(">")


__all__ = [
    "SanitizedYaml",
    "TemplateRewrite",
    "sanitize_helm_templates",
]
