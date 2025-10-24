"""Codex CLI LLM provider for ChunkHound.

Wraps `codex exec` to run local-agent synthesis using the user's Codex
credentials and configuration. Designed for the final synthesis step in
code_research; keeps MCP stdio clean by never printing to stdout.
"""

from __future__ import annotations

import asyncio
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any

from loguru import logger

from chunkhound.interfaces.llm_provider import LLMProvider, LLMResponse
from chunkhound.utils.json_extraction import extract_json_from_response


class CodexCLIProvider(LLMProvider):
    """Provider that shells out to `codex exec`.

    Notes
    - Uses stdin to avoid argv length limits for large prompts.
    - Defaults to an isolated, read-only, non-interactive run (no history).
    - Never writes to stdout; only returns captured content to caller.
    """

    # Token estimate (chars per token) — align with CLI providers
    TOKEN_CHARS_RATIO = 4

    # Timeouts used in health checks (seconds)
    VERSION_CHECK_TIMEOUT = 5
    HEALTH_CHECK_TIMEOUT = 30

    def __init__(
        self,
        api_key: str | None = None,  # Unused (CLI handles auth)
        model: str = "codex",
        base_url: str | None = None,  # Unused
        timeout: int = 60,
        max_retries: int = 3,
    ) -> None:
        self._model = model
        self._timeout = timeout
        self._max_retries = max_retries

        # Usage accounting (estimates)
        self._requests_made = 0
        self._estimated_tokens_used = 0
        self._estimated_prompt_tokens = 0
        self._estimated_completion_tokens = 0

        if not self._codex_available():
            logger.warning("Codex CLI not found in PATH (codex)")

    # ----- Internals -----

    def _get_base_codex_home(self) -> Path | None:
        base = os.getenv("CODEX_HOME")
        if base:
            p = Path(base).expanduser()
            return p if p.exists() else None
        # Default location
        default = Path.home() / ".codex"
        return default if default.exists() else None

    def _build_overlay_home(self) -> str:
        """Create an overlay CODEX_HOME inheriting auth but overriding config.

        - Copies the base CODEX_HOME (if it exists) to a temp dir
        - Replaces config.toml with a minimal one (no MCP, no history persistence)
        - Sets fast model defaults (best effort)
        - Copies only a minimal subset by default to reduce exposure
        """
        overlay = Path(tempfile.mkdtemp(prefix="chunkhound-codex-overlay-"))
        base = self._get_base_codex_home()
        try:
            if base and base.exists():
                copy_all = os.getenv("CHUNKHOUND_CODEX_COPY_ALL", "0") == "1"
                max_bytes = int(os.getenv("CHUNKHOUND_CODEX_MAX_COPY_BYTES", "1000000"))

                def _should_copy_dir(name: str) -> bool:
                    n = name.lower()
                    if copy_all:
                        return True
                    # Likely auth/session state we may need
                    return n in {"sessions", "session", "auth", "profiles", "state"}

                def _should_copy_file(p: Path) -> bool:
                    if copy_all:
                        return True
                    if p.name.lower() == "config.toml":
                        return False  # we write our own config below
                    if p.suffix.lower() in {".json", ".toml", ".ini"}:
                        try:
                            return p.stat().st_size <= max_bytes
                        except Exception:
                            return False
                    return False

                # Minimal selective copy
                for item in base.iterdir():
                    dest = overlay / item.name
                    try:
                        if item.is_dir():
                            if _should_copy_dir(item.name):
                                shutil.copytree(item, dest, dirs_exist_ok=False)
                        else:
                            if _should_copy_file(item):
                                shutil.copy2(item, dest)
                    except Exception:
                        # Best-effort copy; skip items we cannot read/copy
                        pass

            # Write our minimal config.toml ensuring no MCP and no history
            config_path = overlay / "config.toml"
            # Many Codex builds expect top-level `model` keys (not a [model] table)
            cfg = (
                "[history]\n"
                "persistence = \"none\"\n\n"
                "model = \"gpt-5-codex\"\n"
                "model_reasoning_effort = \"low\"\n"
            )
            config_path.write_text(cfg, encoding="utf-8")
        except Exception as e:
            logger.warning(f"Failed to build Codex overlay home: {e}")
        return str(overlay)

    def _codex_available(self) -> bool:
        """Return True if `codex` binary looks available (sync check)."""
        import subprocess

        try:
            res = subprocess.run(
                ["codex", "--version"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                timeout=self.VERSION_CHECK_TIMEOUT,
                check=False,
            )
            # Any exit status implies the binary exists; only ENOENT means absent
            return res.returncode is not None
        except (subprocess.SubprocessError, FileNotFoundError):
            return False

    async def _run_exec(
        self,
        content: str,
        *,
        cwd: str | None = None,
        max_tokens: int,
        timeout: int | None,
        model: str | None,
    ) -> str:
        """Run `codex exec` and capture stdout with robust fallbacks."""
        binary = os.getenv("CHUNKHOUND_CODEX_BIN", "codex")
        # Build overlay CODEX_HOME inheriting auth but with safe config
        overlay_home = self._build_overlay_home()

        if model:
            # Model preference is already in overlay config; keep CLI override minimal for compatibility
            pass

        env = os.environ.copy()
        env["CODEX_HOME"] = overlay_home

        request_timeout = timeout if timeout is not None else self._timeout

        # Privacy-first strategy: use stdin by default to avoid argv leaking prompt in process list.
        # If the CLI rejects stdin, we fallback to argv.
        # Legacy behavior can be restored by setting CHUNKHOUND_CODEX_STDIN_FIRST=0, which will
        # use argv for small prompts and switch to stdin only for very large inputs.
        MAX_ARG_CHARS = int(os.getenv("CHUNKHOUND_CODEX_ARG_LIMIT", "200000"))
        stdin_first = os.getenv("CHUNKHOUND_CODEX_STDIN_FIRST", "1") != "0"
        use_stdin = True if stdin_first else (len(content) > MAX_ARG_CHARS)
        add_skip_git = False

        last_error: Exception | None = None
        try:
            for attempt in range(self._max_retries):
                proc: asyncio.subprocess.Process | None = None
                try:
                    if use_stdin:
                        proc = await asyncio.create_subprocess_exec(
                            binary,
                            "exec",
                            "-",
                            *( ["--skip-git-repo-check"] if add_skip_git else [] ),
                            cwd=cwd,
                            stdin=asyncio.subprocess.PIPE,
                            stdout=asyncio.subprocess.PIPE,
                            stderr=asyncio.subprocess.PIPE,
                            env=env,
                        )
                        assert proc.stdin is not None
                        proc.stdin.write(content.encode("utf-8"))
                        await proc.stdin.drain()
                        proc.stdin.close()
                        stdout, stderr = await asyncio.wait_for(
                            proc.communicate(), timeout=request_timeout
                        )
                    else:
                        # argv mode
                        proc = await asyncio.create_subprocess_exec(
                            binary,
                            "exec",
                            content,
                            *( ["--skip-git-repo-check"] if add_skip_git else [] ),
                            cwd=cwd,
                            stdout=asyncio.subprocess.PIPE,
                            stderr=asyncio.subprocess.PIPE,
                            env=env,
                        )
                        stdout, stderr = await asyncio.wait_for(
                            proc.communicate(), timeout=request_timeout
                        )

                    if proc.returncode != 0:
                        raw_err = stderr.decode("utf-8", errors="ignore")
                        err = self._sanitize_text(raw_err)
                        # Skip-git repo check negotiation for newer Codex builds
                        if "skip-git-repo-check" in err and not add_skip_git:
                            add_skip_git = True
                            logger.warning("codex exec requires --skip-git-repo-check; retrying with flag")
                            continue

                        # If stdin failed (e.g., BrokenPipe or codex not reading stdin), fall back to argv with truncation.
                        if use_stdin and ("broken pipe" in err.lower() or "stdin" in err.lower()):
                            use_stdin = False
                            logger.warning("codex exec stdin not supported; retrying with argv mode")
                            continue
                        last_error = RuntimeError(
                            f"codex exec failed (exit {proc.returncode}): {err}"
                        )
                        if attempt < self._max_retries - 1:
                            logger.warning(
                                f"codex exec attempt {attempt + 1} failed: {err}; retrying"
                            )
                            continue
                        raise last_error

                    return stdout.decode("utf-8", errors="ignore").strip()

                except asyncio.TimeoutError as e:
                    if proc and proc.returncode is None:
                        proc.kill()
                        await proc.wait()
                    last_error = RuntimeError(
                        f"codex exec timed out after {request_timeout}s"
                    )
                    if attempt < self._max_retries - 1:
                        logger.warning(
                            f"codex exec attempt {attempt + 1} timed out; retrying"
                        )
                        continue
                    raise last_error from e
                except OSError as e:
                    # Handle OS-level argv length errors by switching to stdin mode
                    if e.errno == 7:  # Argument list too long
                        if not use_stdin:
                            use_stdin = True
                            logger.warning("codex exec argv too long; retrying with stdin mode")
                            continue
                    raise
                except BrokenPipeError as e:
                    # Switch to argv mode on BrokenPipe
                    use_stdin = False
                    if attempt < self._max_retries - 1:
                        logger.warning("codex exec BrokenPipe on stdin; retrying with argv mode")
                        continue
                    raise RuntimeError("codex exec failed: BrokenPipe on stdin and no retries left") from e
                # Let unexpected exceptions propagate; overlay cleanup happens in the outer finally
                finally:
                    # No per-attempt cleanup of overlay; cleanup performed in outer finally
                    pass

        finally:
            # Cleanup overlay home regardless of success or failure
            try:
                if overlay_home and Path(overlay_home).exists():
                    shutil.rmtree(overlay_home, ignore_errors=True)
            except Exception:
                pass

        raise last_error or RuntimeError("codex exec failed after retries")

    def _sanitize_text(self, s: str, max_len: int | None = None) -> str:
        """Truncate and redact potential secrets in log/error text.

        - Truncates to CHUNKHOUND_CODEX_LOG_MAX_ERR (default 800 chars)
        - Redacts common token patterns and Authorization headers
        """
        try:
            import re

            limit = max_len or int(os.getenv("CHUNKHOUND_CODEX_LOG_MAX_ERR", "800"))
            text = s if len(s) <= limit else (s[:limit] + "…[truncated]")
            # Redact bearer tokens, API keys, and cookie-like secrets
            patterns = [
                r"(?i)(authorization\s*:\s*bearer\s+)[A-Za-z0-9._-]+",
                r"(?i)(api[_-]?key\s*[=:]\s*)([A-Za-z0-9-_]{10,})",
                r"(?i)(secret|token)[\s=:]+([A-Za-z0-9._-]{10,})",
                r"(?i)(set-cookie\s*:\s*)([^;\n]+)",
            ]
            redacted = text
            for pat in patterns:
                redacted = re.sub(pat, r"\1[REDACTED]", redacted)
            return redacted
        except Exception:
            return s[: max_len or 800]

    def _merge_prompts(self, prompt: str, system: str | None) -> str:
        if system and system.strip():
            return f"System Instructions:\n{system.strip()}\n\nUser Request:\n{prompt}"
        return prompt

    # ----- LLMProvider interface -----

    @property
    def name(self) -> str:  # pragma: no cover - trivial
        return "codex-cli"

    @property
    def model(self) -> str:  # pragma: no cover - trivial
        return self._model

    async def complete(
        self,
        prompt: str,
        system: str | None = None,
        max_completion_tokens: int = 4096,
        timeout: int | None = None,
    ) -> LLMResponse:
        text = self._merge_prompts(prompt, system)
        output = await self._run_exec(
            text,
            cwd=None,
            max_tokens=max_completion_tokens,
            timeout=timeout,
            model=self._model,
        )

        if not output or not output.strip():
            raise RuntimeError("Codex CLI returned empty output")

        self._requests_made += 1
        p_tokens = self.estimate_tokens(text)
        c_tokens = self.estimate_tokens(output)
        total = p_tokens + c_tokens
        self._estimated_prompt_tokens += p_tokens
        self._estimated_completion_tokens += c_tokens
        self._estimated_tokens_used += total

        return LLMResponse(
            content=output,
            tokens_used=total,
            model=self._model,
            finish_reason="stop",
        )

    async def complete_structured(
        self,
        prompt: str,
        json_schema: dict[str, Any],
        system: str | None = None,
        max_completion_tokens: int = 4096,
        timeout: int | None = None,
    ) -> dict[str, Any]:
        import json

        schema_block = json.dumps(json_schema, indent=2)
        structured_prompt = (
            "Please respond with ONLY valid JSON that conforms to this schema:\n\n"
            + schema_block
            + "\n\nUser request: "
            + prompt
            + "\n\nRespond with JSON only, no additional text."
        )

        text = self._merge_prompts(structured_prompt, system)
        output = await self._run_exec(
            text,
            cwd=None,
            max_tokens=max_completion_tokens,
            timeout=timeout,
            model=self._model,
        )

        if not output or not output.strip():
            raise RuntimeError("Codex CLI structured completion returned empty output")

        self._requests_made += 1
        p_tokens = self.estimate_tokens(text)
        c_tokens = self.estimate_tokens(output)
        total = p_tokens + c_tokens
        self._estimated_prompt_tokens += p_tokens
        self._estimated_completion_tokens += c_tokens
        self._estimated_tokens_used += total

        try:
            json_str = extract_json_from_response(output)
            parsed = json.loads(json_str)
            if not isinstance(parsed, dict):
                raise ValueError("Expected top-level JSON object")
            # Minimal required-field check if provided
            if "required" in json_schema:
                missing = [f for f in json_schema["required"] if f not in parsed]
                if missing:
                    raise ValueError(f"Missing required fields: {missing}")
            return parsed
        except Exception as e:  # noqa: BLE001
            logger.error(f"Codex CLI structured output parse failed: {e}")
            raise RuntimeError(f"Invalid JSON in structured output: {e}") from e

    async def batch_complete(
        self,
        prompts: list[str],
        system: str | None = None,
        max_completion_tokens: int = 4096,
    ) -> list[LLMResponse]:
        results: list[LLMResponse] = []
        for p in prompts:
            results.append(
                await self.complete(p, system=system, max_completion_tokens=max_completion_tokens)
            )
        return results

    def estimate_tokens(self, text: str) -> int:
        return len(text) // self.TOKEN_CHARS_RATIO

    async def health_check(self) -> dict[str, Any]:
        if not self._codex_available():
            return {"status": "unhealthy", "provider": self.name, "error": "codex not found"}
        try:
            sample = await self.complete("Say 'OK'", max_completion_tokens=10, timeout=self.HEALTH_CHECK_TIMEOUT)
            return {
                "status": "healthy",
                "provider": self.name,
                "model": self._model,
                "test_response": sample.content[:50],
            }
        except Exception as e:  # noqa: BLE001
            return {"status": "unhealthy", "provider": self.name, "error": str(e)}

    def get_usage_stats(self) -> dict[str, Any]:
        return {
            "requests_made": self._requests_made,
            "total_tokens_estimated": self._estimated_tokens_used,
            "prompt_tokens_estimated": self._estimated_prompt_tokens,
            "completion_tokens_estimated": self._estimated_completion_tokens,
        }
