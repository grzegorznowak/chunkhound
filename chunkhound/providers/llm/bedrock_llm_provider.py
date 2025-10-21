"""AWS Bedrock LLM provider implementation for ChunkHound deep research."""

import asyncio
import json
import random
import re
import time
from typing import Any

from loguru import logger

from chunkhound.interfaces.llm_provider import LLMProvider, LLMResponse

try:
    import boto3  # type: ignore
    from botocore.config import Config  # type: ignore
    from botocore.exceptions import ClientError  # type: ignore

    BEDROCK_AVAILABLE = True
except ImportError:
    boto3 = None
    Config = None  # type: ignore
    ClientError = Exception  # type: ignore
    BEDROCK_AVAILABLE = False
    logger.warning("boto3 not available - install with: uv pip install boto3")


class BedrockLLMProvider(LLMProvider):
    """AWS Bedrock LLM provider using Anthropic Claude models."""

    def __init__(
        self,
        model: str = "us.anthropic.claude-sonnet-4-5-20250929-v1:0",
        region: str | None = None,
        timeout: int = 60,
        max_retries: int = 3,
    ):
        """Initialize AWS Bedrock LLM provider.

        Args:
            model: Model ID to use (inference profile, e.g., 'us.anthropic.claude-sonnet-4-5-20250929-v1:0')
            region: AWS region (defaults to AWS_REGION or AWS_DEFAULT_REGION env var)
            timeout: Request timeout in seconds (60s for Claude 4 recommended)
            max_retries: Number of retry attempts for failed requests
        """
        if not BEDROCK_AVAILABLE:
            raise ImportError(
                "boto3 not available - install with: uv pip install boto3"
            )

        self._model = model
        self._timeout = timeout
        self._max_retries = max_retries
        self._region = region

        # Initialize bedrock-runtime client with timeout configuration
        # Uses AWS credential chain: env vars, ~/.aws/credentials, IAM roles
        # Timeout: 60s recommended for Claude 4 models (can take up to 60 minutes)
        boto_config = Config(
            read_timeout=timeout,
            connect_timeout=10,
            retries={"max_attempts": 0},  # We handle retries ourselves
        )

        client_kwargs: dict[str, Any] = {
            "service_name": "bedrock-runtime",
            "config": boto_config,
        }
        if region:
            client_kwargs["region_name"] = region

        self._client = boto3.client(**client_kwargs)

        # Usage tracking
        self._requests_made = 0
        self._tokens_used = 0
        self._prompt_tokens = 0
        self._completion_tokens = 0

    @property
    def name(self) -> str:
        """Provider name."""
        return "bedrock"

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
        # Build Anthropic Messages API format
        messages = [{"role": "user", "content": prompt}]

        body: dict[str, Any] = {
            "anthropic_version": "bedrock-2023-05-31",
            "messages": messages,
            "max_tokens": max_completion_tokens,
        }

        if system:
            body["system"] = system

        # Use provided timeout or fall back to default
        request_timeout = timeout if timeout is not None else self._timeout

        try:
            # boto3 is synchronous, run in thread pool for async
            response = await asyncio.to_thread(
                self._invoke_model_with_retry,
                body=body,
                timeout=request_timeout,
            )

            # Parse response
            response_body = json.loads(response["body"].read())

            # Extract content and usage
            content = response_body.get("content", [{}])[0].get("text", "")
            usage = response_body.get("usage", {})
            input_tokens = usage.get("input_tokens", 0)
            output_tokens = usage.get("output_tokens", 0)
            total_tokens = input_tokens + output_tokens
            finish_reason = response_body.get("stop_reason")

            # Track usage
            self._requests_made += 1
            self._prompt_tokens += input_tokens
            self._completion_tokens += output_tokens
            self._tokens_used += total_tokens

            return LLMResponse(
                content=content,
                tokens_used=total_tokens,
                model=self._model,
                finish_reason=finish_reason,
            )

        except Exception as e:
            logger.error(f"Bedrock completion failed: {e}")
            raise RuntimeError(f"LLM completion failed: {e}") from e

    def _invoke_model_with_retry(
        self,
        body: dict[str, Any],
        timeout: int,
    ) -> dict[str, Any]:
        """Invoke model with retry logic and exponential backoff (synchronous).

        Args:
            body: Request body
            timeout: Request timeout in seconds

        Returns:
            Response dict from bedrock-runtime

        Raises:
            RuntimeError: For non-retryable errors or after max retries
        """
        last_error = None

        for attempt in range(self._max_retries):
            try:
                response = self._client.invoke_model(
                    modelId=self._model,
                    body=json.dumps(body),
                    contentType="application/json",
                    accept="application/json",
                )
                return response  # type: ignore[no-any-return]

            except ClientError as e:
                error_code = e.response.get("Error", {}).get("Code", "")
                last_error = e

                # Retryable errors: throttling, service unavailable
                if error_code in ("ThrottlingException", "ServiceUnavailableException"):
                    if attempt < self._max_retries - 1:
                        # Exponential backoff with jitter
                        delay = min(60, (2**attempt) + random.uniform(0, 1))
                        logger.warning(
                            f"Bedrock {error_code} on attempt {attempt + 1}, "
                            f"retrying in {delay:.2f}s"
                        )
                        time.sleep(delay)
                        continue
                    else:
                        logger.error(
                            f"Bedrock {error_code} after {self._max_retries} attempts"
                        )
                        raise RuntimeError(
                            f"Bedrock throttled after {self._max_retries} retries"
                        ) from e

                # Non-retryable errors: validation, access denied, model not found
                elif error_code in (
                    "ValidationException",
                    "AccessDeniedException",
                    "ResourceNotFoundException",
                    "ModelNotReadyException",
                ):
                    logger.error(f"Bedrock {error_code}: {e}")
                    raise RuntimeError(f"Bedrock error: {error_code}") from e

                # Unknown error - retry
                else:
                    if attempt < self._max_retries - 1:
                        delay = min(60, (2**attempt) + random.uniform(0, 1))
                        logger.warning(
                            f"Bedrock error on attempt {attempt + 1}, "
                            f"retrying in {delay:.2f}s: {e}"
                        )
                        time.sleep(delay)
                        continue

            except Exception as e:
                # Non-AWS errors (network, etc.) - retry
                last_error = e
                if attempt < self._max_retries - 1:
                    delay = min(60, (2**attempt) + random.uniform(0, 1))
                    logger.warning(
                        f"Bedrock attempt {attempt + 1} failed, "
                        f"retrying in {delay:.2f}s: {e}"
                    )
                    time.sleep(delay)
                    continue

        if last_error:
            raise RuntimeError(f"Bedrock invocation failed after retries") from last_error
        raise RuntimeError("Bedrock invocation failed with no error")

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

    async def complete_structured(
        self,
        prompt: str,
        json_schema: dict[str, Any],
        system: str | None = None,
        max_completion_tokens: int = 4096,
        timeout: int | None = None,
    ) -> dict[str, Any]:
        """Generate a structured JSON completion conforming to the given schema.

        Since Bedrock doesn't natively support JSON schema validation in the same way
        as OpenAI, we include the schema in the prompt and request JSON output.

        Args:
            prompt: The user prompt
            json_schema: JSON Schema definition for structured output
            system: Optional system prompt
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
            # Use complete() to generate response
            response = await self.complete(
                structured_prompt, system, max_completion_tokens, timeout
            )

            # Extract JSON from response (handle markdown code blocks)
            json_content = self._extract_json_from_response(response.content)

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
            raise RuntimeError(f"Invalid JSON in structured output: {e}") from e
        except Exception as e:
            logger.error(f"Bedrock structured completion failed: {e}")
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
        # Rough estimation: ~4 chars per token for Claude models
        return len(text) // 4

    async def health_check(self) -> dict[str, Any]:
        """Perform health check."""
        try:
            response = await self.complete("Say 'OK'", max_completion_tokens=10)
            return {
                "status": "healthy",
                "provider": "bedrock",
                "model": self._model,
                "region": self._region or "default",
                "test_response": response.content[:50],
            }
        except Exception as e:
            return {
                "status": "unhealthy",
                "provider": "bedrock",
                "model": self._model,
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
