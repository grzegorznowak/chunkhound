"""OpenAI LLM provider implementation for ChunkHound deep research."""

import asyncio
import json
from typing import Any

from loguru import logger

from chunkhound.interfaces.llm_provider import LLMProvider, LLMResponse

try:
    from openai import AsyncOpenAI

    OPENAI_AVAILABLE = True
except ImportError:
    AsyncOpenAI = None  # type: ignore
    OPENAI_AVAILABLE = False
    logger.warning("OpenAI not available - install with: uv pip install openai")


class OpenAILLMProvider(LLMProvider):
    """OpenAI LLM provider using GPT models."""

    def __init__(
        self,
        api_key: str | None = None,
        model: str = "gpt-5-nano-mini",
        base_url: str | None = None,
        timeout: int = 60,
        max_retries: int = 3,
    ):
        """Initialize OpenAI LLM provider.

        Args:
            api_key: OpenAI API key (defaults to OPENAI_API_KEY env var)
            model: Model name to use
            base_url: Base URL for OpenAI API (optional for custom endpoints)
            timeout: Request timeout in seconds
            max_retries: Number of retry attempts for failed requests
        """
        if not OPENAI_AVAILABLE:
            raise ImportError("OpenAI not available - install with: uv pip install openai")

        self._model = model
        self._timeout = timeout
        self._max_retries = max_retries

        # Initialize client
        client_kwargs: dict[str, Any] = {
            "api_key": api_key,
            "timeout": timeout,
            "max_retries": max_retries,
        }
        if base_url:
            client_kwargs["base_url"] = base_url

        self._client = AsyncOpenAI(**client_kwargs)

        # Usage tracking
        self._requests_made = 0
        self._tokens_used = 0
        self._prompt_tokens = 0
        self._completion_tokens = 0

    @property
    def name(self) -> str:
        """Provider name."""
        return "openai"

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
            system: Optional system prompt
            max_completion_tokens: Maximum tokens to generate
            timeout: Optional timeout in seconds (overrides default)
        """
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        # Use provided timeout or fall back to default
        request_timeout = timeout if timeout is not None else self._timeout

        try:
            response = await self._client.chat.completions.create(
                model=self._model,
                messages=messages,
                max_completion_tokens=max_completion_tokens,
                timeout=request_timeout,
            )

            self._requests_made += 1
            if response.usage:
                self._prompt_tokens += response.usage.prompt_tokens
                self._completion_tokens += response.usage.completion_tokens
                self._tokens_used += response.usage.total_tokens

            # Extract response metadata
            content = response.choices[0].message.content
            tokens = response.usage.total_tokens if response.usage else 0
            finish_reason = response.choices[0].finish_reason

            # Validate content is not None or empty
            if content is None:
                logger.error(
                    f"OpenAI returned None content (finish_reason={finish_reason}, "
                    f"tokens={tokens})"
                )
                raise RuntimeError(
                    f"LLM returned empty response (finish_reason={finish_reason}). "
                    "This may indicate a content filter, API error, or model refusal."
                )

            if not content.strip():
                logger.warning(
                    f"OpenAI returned empty content (finish_reason={finish_reason}, "
                    f"tokens={tokens})"
                )
                raise RuntimeError(
                    f"LLM returned empty response (finish_reason={finish_reason}). "
                    "This may indicate a content filter, API error, or model refusal."
                )

            # Reject truncated responses (finish_reason="length")
            if finish_reason == "length":
                usage_info = ""
                if response.usage:
                    usage_info = (
                        f" (prompt={response.usage.prompt_tokens:,}, "
                        f"completion={response.usage.completion_tokens:,})"
                    )

                raise RuntimeError(
                    f"LLM response truncated - token limit exceeded{usage_info}. "
                    f"For reasoning models (GPT-5, Gemini 2.5), this indicates the query requires "
                    f"extensive reasoning that exhausted the output budget. "
                    f"The output budget is fixed at {max_completion_tokens:,} tokens for all queries "
                    f"to accommodate internal 'thinking' tokens (OUTPUT_TOKENS_WITH_REASONING). "
                    f"Try breaking your query into smaller, more focused questions."
                )

            # Warn on other unexpected finish_reason
            if finish_reason not in ("stop",):
                logger.warning(
                    f"Unexpected finish_reason: {finish_reason} "
                    f"(content_length={len(content)})"
                )
                if finish_reason == "content_filter":
                    raise RuntimeError(
                        "LLM response blocked by content filter. "
                        "Try rephrasing your query or adjusting the prompt."
                    )

            return LLMResponse(
                content=content,
                tokens_used=tokens,
                model=self._model,
                finish_reason=finish_reason,
            )

        except Exception as e:
            logger.error(f"OpenAI completion failed: {e}")
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

        Uses OpenAI's structured outputs with strict JSON Schema validation.
        Best practice for GPT-5-Nano: Guarantees valid, parseable JSON output.

        Args:
            prompt: The user prompt
            json_schema: JSON Schema definition for structured output
            system: Optional system prompt
            max_completion_tokens: Maximum tokens to generate
            timeout: Optional timeout in seconds (overrides default)

        Returns:
            Parsed JSON object conforming to schema
        """
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        # Use provided timeout or fall back to default
        request_timeout = timeout if timeout is not None else self._timeout

        try:
            response = await self._client.chat.completions.create(
                model=self._model,
                messages=messages,
                max_completion_tokens=max_completion_tokens,
                timeout=request_timeout,
                response_format={
                    "type": "json_schema",
                    "json_schema": {
                        "name": "structured_response",
                        "strict": True,
                        "schema": json_schema,
                    },
                },
            )

            self._requests_made += 1
            if response.usage:
                self._prompt_tokens += response.usage.prompt_tokens
                self._completion_tokens += response.usage.completion_tokens
                self._tokens_used += response.usage.total_tokens

            content = response.choices[0].message.content
            finish_reason = response.choices[0].finish_reason

            # Reject truncated responses (finish_reason="length")
            if finish_reason == "length":
                usage_info = ""
                if response.usage:
                    usage_info = (
                        f" (prompt={response.usage.prompt_tokens:,}, "
                        f"completion={response.usage.completion_tokens:,})"
                    )

                raise RuntimeError(
                    f"LLM structured completion truncated - token limit exceeded{usage_info}. "
                    f"This indicates insufficient max_completion_tokens for the structured output. "
                    f"Consider increasing the token limit or reducing input context."
                )

            # Validate content is not None or empty
            if content is None or not content.strip():
                logger.error(
                    f"OpenAI structured completion returned empty content "
                    f"(finish_reason={finish_reason})"
                )
                raise RuntimeError(
                    f"LLM structured completion returned empty response "
                    f"(finish_reason={finish_reason})"
                )

            parsed = json.loads(content)

            return parsed

        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse structured output as JSON: {e}")
            raise RuntimeError(f"Invalid JSON in structured output: {e}") from e
        except Exception as e:
            logger.error(f"OpenAI structured completion failed: {e}")
            raise RuntimeError(f"LLM structured completion failed: {e}") from e

    async def batch_complete(
        self,
        prompts: list[str],
        system: str | None = None,
        max_completion_tokens: int = 4096,
    ) -> list[LLMResponse]:
        """Generate completions for multiple prompts concurrently."""
        tasks = [
            self.complete(prompt, system, max_completion_tokens) for prompt in prompts
        ]
        return await asyncio.gather(*tasks)

    def estimate_tokens(self, text: str) -> int:
        """Estimate token count for text (rough approximation)."""
        # Rough estimation: ~4 chars per token for GPT models
        return len(text) // 4

    async def health_check(self) -> dict[str, Any]:
        """Perform health check."""
        try:
            response = await self.complete("Say 'OK'", max_completion_tokens=10)
            return {
                "status": "healthy",
                "provider": "openai",
                "model": self._model,
                "test_response": response.content[:50],
            }
        except Exception as e:
            return {
                "status": "unhealthy",
                "provider": "openai",
                "error": str(e),
            }

    def get_usage_stats(self) -> dict[str, Any]:
        """Get usage statistics."""
        return {
            "requests_made": self._requests_made,
            "total_tokens": self._tokens_used,
            "prompt_tokens": self._prompt_tokens,
            "completion_tokens": self._completion_tokens,
        }

    def get_synthesis_concurrency(self) -> int:
        """Get recommended concurrency for parallel synthesis operations.

        Returns:
            3 for OpenAI (conservative default based on tier limits)
        """
        return 3
