"""Claude Code CLI LLM provider implementation for ChunkHound deep research.

This provider wraps the Claude Code CLI (claude --print) to enable deep research
using the user's existing Claude subscription instead of API credits.

Note: The CLI includes Claude Code's agentic system prompt and cannot provide
"raw" API access. Use the Anthropic SDK provider if you need clean API access.
"""

import asyncio
import json
import os
import re
import subprocess
from typing import Any

from loguru import logger

from chunkhound.interfaces.llm_provider import LLMProvider, LLMResponse


class ClaudeCodeCLIProvider(LLMProvider):
    """Claude Code CLI provider using subprocess calls to claude --print."""

    # Constants for timeouts and estimation
    TOKEN_CHARS_RATIO = 4  # Approximate characters per token for Claude models
    VERSION_CHECK_TIMEOUT = 5  # Seconds to wait for CLI version check
    HEALTH_CHECK_TIMEOUT = 30  # Seconds to wait for health check completion

    def __init__(
        self,
        api_key: str | None = None,
        model: str = "claude-sonnet-4-5-20250929",
        base_url: str | None = None,
        timeout: int = 60,
        max_retries: int = 3,
    ):
        """Initialize Claude Code CLI provider.

        Args:
            api_key: Not used (subscription-based authentication)
            model: Model name to use (e.g., "claude-sonnet-4-5-20250929")
            base_url: Not used (CLI uses default endpoints)
            timeout: Request timeout in seconds
            max_retries: Number of retry attempts for failed requests
        """
        self._model = model
        self._timeout = timeout
        self._max_retries = max_retries

        # Usage tracking
        self._requests_made = 0
        self._estimated_tokens_used = 0
        self._estimated_prompt_tokens = 0
        self._estimated_completion_tokens = 0

        # Verify CLI is available and compatible
        if not self._is_cli_available():
            logger.warning(
                "Claude Code CLI not found. Install from: https://claude.com/claude-code"
            )
        else:
            # Check compatibility with required features
            is_compatible, error_msg = self._check_cli_compatibility()
            if not is_compatible:
                logger.warning(
                    f"Claude Code CLI compatibility issue: {error_msg}. "
                    "Update CLI from: https://claude.com/claude-code"
                )

    def _is_cli_available(self) -> bool:
        """Check if claude CLI is available in PATH."""
        try:
            subprocess.run(
                ["claude", "--version"],
                capture_output=True,
                timeout=self.VERSION_CHECK_TIMEOUT,
                check=False,
            )
            return True
        except (subprocess.SubprocessError, FileNotFoundError):
            return False

    def _check_cli_compatibility(self) -> tuple[bool, str]:
        """Check if CLI supports required features.

        Returns:
            Tuple of (is_compatible, error_message)
            If compatible, error_message is empty string.
        """
        try:
            # Test that --print flag works
            result = subprocess.run(
                ["claude", "--print", "test"],
                capture_output=True,
                timeout=self.VERSION_CHECK_TIMEOUT,
                check=False,
            )

            # If exit code is 0 or 1 (model error), CLI is available
            # Exit code 2 usually means unknown flag
            if result.returncode == 2:
                return False, "CLI does not support --print flag (upgrade required)"

            # Check for specific error messages indicating missing features
            stderr = result.stderr.decode("utf-8").lower()
            if "unknown option" in stderr or "unrecognized" in stderr:
                if "--print" in stderr:
                    return False, "CLI does not support --print flag (upgrade required)"
                if "--model" in stderr:
                    return False, "CLI does not support --model flag (upgrade required)"
                if "--output-format" in stderr:
                    return (
                        False,
                        "CLI does not support --output-format flag (upgrade required)",
                    )

            return True, ""

        except subprocess.TimeoutExpired:
            # Timeout during compatibility check - CLI might be too old or hanging
            return False, "CLI compatibility check timed out"
        except (subprocess.SubprocessError, FileNotFoundError):
            return False, "CLI not found in PATH"

    def _map_model_to_cli_arg(self, model: str) -> str:
        """Map full model name to CLI model argument.

        The Claude Code CLI accepts full model names directly (e.g., "claude-sonnet-4-5-20250929").
        Short names like "sonnet-4-5" are NOT accepted and result in exit code 1.

        Args:
            model: Full model name (e.g., "claude-sonnet-4-5-20250929")

        Returns:
            CLI model argument (same as input - full model name)
        """
        # Pass through model name as-is - CLI requires full names
        return model

    def _extract_json_from_response(self, content: str) -> str:
        """Extract JSON from response, handling markdown code blocks.

        Handles multiple patterns:
        - Raw JSON (no code blocks)
        - JSON in ```json code block
        - JSON in generic ``` code block
        - Nested code blocks (takes the first valid one)

        Args:
            content: Response content potentially containing JSON

        Returns:
            Extracted JSON string

        Raises:
            ValueError: If no valid JSON content can be extracted
        """
        # Try to find JSON in markdown code blocks using regex
        # Pattern matches ```json or ``` followed by content until closing ```
        code_block_pattern = r"```(?:json)?\s*\n?(.*?)\n?```"
        matches = re.findall(code_block_pattern, content, re.DOTALL)

        if matches:
            # Return the first non-empty match
            for match in matches:
                if match.strip():
                    return match.strip()

        # No code blocks found, try to use content as-is
        # Strip any leading/trailing whitespace
        json_content = content.strip()

        # If content is empty, raise error
        if not json_content:
            raise ValueError("No JSON content found in response")

        return json_content

    async def _run_cli_command(
        self,
        prompt: str,
        system: str | None = None,
        max_completion_tokens: int | None = None,
        timeout: int | None = None,
    ) -> str:
        """Run claude CLI command and return output.

        Args:
            prompt: User prompt
            system: Optional system prompt (appended to default)
            max_completion_tokens: Maximum tokens to generate
            timeout: Optional timeout override

        Returns:
            CLI output text

        Raises:
            RuntimeError: If CLI command fails
        """
        # Build CLI command
        model_arg = self._map_model_to_cli_arg(self._model)
        cmd = ["claude", "--print", "--model", model_arg, "--output-format", "text"]

        # Add system prompt if provided (appends to default)
        if system:
            cmd.extend(["--append-system-prompt", system])

        # Add the user prompt
        cmd.append(prompt)

        # Set environment for subscription-based auth
        env = os.environ.copy()
        env["CLAUDE_USE_SUBSCRIPTION"] = "true"

        # Remove ANTHROPIC_API_KEY if present to force subscription auth
        env.pop("ANTHROPIC_API_KEY", None)

        # Use provided timeout or default
        request_timeout = timeout if timeout is not None else self._timeout

        # Run command with retry logic
        last_error = None
        for attempt in range(self._max_retries):
            process = None
            try:
                # Create subprocess (without timeout - this is usually instant)
                process = await asyncio.create_subprocess_exec(
                    *cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    env=env,
                )

                # Wrap communicate() with timeout (this is the long-running part)
                stdout, stderr = await asyncio.wait_for(
                    process.communicate(),
                    timeout=request_timeout,
                )

                if process.returncode != 0:
                    error_msg = stderr.decode("utf-8") if stderr else "Unknown error"
                    last_error = RuntimeError(
                        f"CLI command failed (exit {process.returncode}): {error_msg}"
                    )
                    if attempt < self._max_retries - 1:
                        logger.warning(
                            f"CLI attempt {attempt + 1} failed, retrying: {error_msg}"
                        )
                        continue
                    raise last_error

                return stdout.decode("utf-8").strip()

            except asyncio.TimeoutError as e:
                # Kill the subprocess if it's still running
                if process and process.returncode is None:
                    process.kill()
                    await process.wait()

                last_error = RuntimeError(f"CLI command timed out after {request_timeout}s")
                if attempt < self._max_retries - 1:
                    logger.warning(f"CLI attempt {attempt + 1} timed out, retrying")
                    continue
                raise last_error from e

            except Exception as e:
                # Kill the subprocess if it's still running on unexpected errors
                if process and process.returncode is None:
                    process.kill()
                    await process.wait()

                last_error = RuntimeError(f"CLI command failed: {e}")
                if attempt < self._max_retries - 1:
                    logger.warning(f"CLI attempt {attempt + 1} failed: {e}")
                    continue
                raise last_error from e

        # Should not reach here, but just in case
        raise last_error or RuntimeError("CLI command failed after retries")

    @property
    def name(self) -> str:
        """Provider name."""
        return "claude-code-cli"

    @property
    def model(self) -> str:
        """Model name."""
        return self._model

    async def complete(
        self,
        prompt: str,
        system: str | None = None,
        max_completion_tokens: int = 4096,
        timeout: int | None = None,
    ) -> LLMResponse:
        """Generate a completion for the given prompt.

        Args:
            prompt: The user prompt
            system: Optional system prompt (appended to CLI default)
            max_completion_tokens: Maximum tokens to generate
            timeout: Optional timeout in seconds (overrides default)

        Returns:
            LLMResponse with content and estimated token usage
        """
        try:
            content = await self._run_cli_command(prompt, system, max_completion_tokens, timeout)

            # Track usage (estimates since CLI doesn't return token counts)
            self._requests_made += 1
            prompt_tokens = self.estimate_tokens(prompt)
            if system:
                prompt_tokens += self.estimate_tokens(system)
            completion_tokens = self.estimate_tokens(content)
            total_tokens = prompt_tokens + completion_tokens

            self._estimated_prompt_tokens += prompt_tokens
            self._estimated_completion_tokens += completion_tokens
            self._estimated_tokens_used += total_tokens

            return LLMResponse(
                content=content,
                tokens_used=total_tokens,
                model=self._model,
                finish_reason="stop",  # CLI doesn't provide this
            )

        except Exception as e:
            logger.error(f"Claude Code CLI completion failed: {e}")
            raise RuntimeError(f"LLM completion failed: {e}") from e

    async def complete_structured(
        self,
        prompt: str,
        json_schema: dict[str, Any],
        system: str | None = None,
        max_completion_tokens: int = 4096,
        timeout: int | None = None,
    ) -> dict[str, Any]:
        """Generate a structured JSON completion conforming to the given schema.

        Since Claude Code CLI doesn't support native JSON schema validation,
        we include the schema in the prompt and request JSON output.

        Args:
            prompt: The user prompt
            json_schema: JSON Schema definition for structured output
            system: Optional system prompt (appended to CLI default)
            max_completion_tokens: Maximum tokens to generate
            timeout: Optional timeout in seconds (overrides default)

        Returns:
            Parsed JSON object

        Raises:
            RuntimeError: If output is not valid JSON or doesn't match schema
        """
        # Build structured prompt with schema
        structured_prompt = f"""Please respond with ONLY valid JSON that conforms to this schema:

{json.dumps(json_schema, indent=2)}

User request: {prompt}

Respond with JSON only, no additional text."""

        try:
            content = await self._run_cli_command(
                structured_prompt, system, max_completion_tokens, timeout
            )

            # Track usage
            self._requests_made += 1
            prompt_tokens = self.estimate_tokens(structured_prompt)
            if system:
                prompt_tokens += self.estimate_tokens(system)
            completion_tokens = self.estimate_tokens(content)
            total_tokens = prompt_tokens + completion_tokens

            self._estimated_prompt_tokens += prompt_tokens
            self._estimated_completion_tokens += completion_tokens
            self._estimated_tokens_used += total_tokens

            # Extract JSON from response (handle markdown code blocks)
            json_content = self._extract_json_from_response(content)

            # Parse JSON
            parsed = json.loads(json_content)

            # Ensure parsed is a dict
            if not isinstance(parsed, dict):
                raise ValueError(f"Expected JSON object, got {type(parsed).__name__}")

            # Basic schema validation (check required fields if specified)
            if "required" in json_schema:
                missing = [
                    field for field in json_schema["required"] if field not in parsed
                ]
                if missing:
                    raise ValueError(f"Missing required fields: {missing}")

            return parsed

        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse structured output as JSON: {e}")
            logger.debug(f"Raw output: {content}")
            raise RuntimeError(f"Invalid JSON in structured output: {e}") from e
        except Exception as e:
            logger.error(f"Claude Code CLI structured completion failed: {e}")
            raise RuntimeError(f"LLM structured completion failed: {e}") from e

    async def batch_complete(
        self,
        prompts: list[str],
        system: str | None = None,
        max_completion_tokens: int = 4096,
    ) -> list[LLMResponse]:
        """Generate completions for multiple prompts concurrently.

        Note: CLI doesn't support true batch API, so we run sequentially
        to avoid overwhelming the CLI or subscription rate limits.
        """
        results = []
        for prompt in prompts:
            result = await self.complete(prompt, system, max_completion_tokens)
            results.append(result)
        return results

    def estimate_tokens(self, text: str) -> int:
        """Estimate token count for text.

        Uses rough approximation since we don't have direct tokenizer access.
        Claude typically uses ~4 characters per token.
        """
        return len(text) // self.TOKEN_CHARS_RATIO

    async def health_check(self) -> dict[str, Any]:
        """Perform health check by testing CLI availability and basic completion."""
        # First check if CLI is available
        if not self._is_cli_available():
            return {
                "status": "unhealthy",
                "provider": "claude-code-cli",
                "error": "Claude Code CLI not found in PATH",
            }

        # Try a simple completion
        try:
            response = await self.complete(
                "Say 'OK'", max_completion_tokens=10, timeout=self.HEALTH_CHECK_TIMEOUT
            )
            return {
                "status": "healthy",
                "provider": "claude-code-cli",
                "model": self._model,
                "test_response": response.content[:50],
            }
        except Exception as e:
            return {
                "status": "unhealthy",
                "provider": "claude-code-cli",
                "error": str(e),
            }

    def get_usage_stats(self) -> dict[str, Any]:
        """Get usage statistics (estimates since CLI doesn't return actual counts)."""
        return {
            "requests_made": self._requests_made,
            "total_tokens_estimated": self._estimated_tokens_used,
            "prompt_tokens_estimated": self._estimated_prompt_tokens,
            "completion_tokens_estimated": self._estimated_completion_tokens,
        }
