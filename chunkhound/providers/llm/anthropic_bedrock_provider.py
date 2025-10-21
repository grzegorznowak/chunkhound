"""Anthropic Bedrock LLM provider implementation for ChunkHound deep research."""

import asyncio
import json
import os
from typing import Any

from loguru import logger

from chunkhound.interfaces.llm_provider import LLMProvider, LLMResponse
from chunkhound.utils.json_extraction import extract_json_from_response

try:
    import boto3
    from botocore.config import Config
    from botocore.exceptions import (
        BotoCoreError,
        ClientError,
        NoCredentialsError,
        PartialCredentialsError,
    )

    BEDROCK_AVAILABLE = True
except ImportError:
    boto3 = None  # type: ignore
    Config = None  # type: ignore
    BotoCoreError = None  # type: ignore
    ClientError = None  # type: ignore
    NoCredentialsError = None  # type: ignore
    PartialCredentialsError = None  # type: ignore
    BEDROCK_AVAILABLE = False
    logger.warning("boto3 not available - install with: uv pip install boto3")


class AnthropicBedrockProvider(LLMProvider):
    """Anthropic Bedrock LLM provider using AWS Bedrock."""

    def __init__(
        self,
        api_key: str | None = None,  # Not used - kept for interface compatibility
        model: str = "claude-3-5-sonnet-20241022",
        base_url: str | None = None,  # Not used - kept for interface compatibility
        timeout: int = 60,
        max_retries: int = 3,
        bedrock_region: str | None = None,
    ):
        """Initialize Anthropic Bedrock LLM provider.

        Args:
            api_key: Not used (boto3 uses AWS credentials). Present for interface compatibility.
            model: Model name to use (e.g., "claude-3-5-sonnet-20241022")
            base_url: Not used (Bedrock uses AWS endpoints). Present for interface compatibility.
            timeout: Request timeout in seconds
            max_retries: Number of retry attempts for failed requests
            bedrock_region: AWS region for Bedrock (defaults to AWS_REGION env var)
        """
        if not BEDROCK_AVAILABLE:
            raise ImportError("boto3 not available - install with: uv pip install boto3")

        # Explicitly mark unused parameters (kept for LLMProvider interface compatibility)
        _ = api_key
        _ = base_url

        self._model = model
        self._timeout = timeout
        self._max_retries = max_retries

        # Initialize Bedrock client
        # AWS credential resolution (boto3 handles automatically):
        #   1. AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY environment variables
        #   2. ~/.aws/credentials file
        #   3. IAM role (when running on AWS infrastructure)
        #
        # Region resolution order (first non-None wins):
        #   1. bedrock_region parameter (from config or constructor)
        #   2. AWS_REGION environment variable
        #   3. AWS_DEFAULT_REGION environment variable
        #   4. Hardcoded fallback: "us-east-1"
        region = (
            bedrock_region
            or os.getenv("AWS_REGION")
            or os.getenv("AWS_DEFAULT_REGION")
            or "us-east-1"
        )

        try:
            self._client = boto3.client(
                "bedrock-runtime",
                region_name=region,
                config=Config(
                    retries={"max_attempts": max_retries, "mode": "adaptive"},
                    read_timeout=timeout,
                ),
            )
        except (NoCredentialsError, PartialCredentialsError) as e:
            raise ValueError(
                f"AWS credentials not configured: {e}\n\n"
                "Please configure AWS credentials using one of these methods:\n"
                "1. Environment variables: AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY\n"
                "2. AWS credentials file: ~/.aws/credentials\n"
                "3. IAM role (when running on AWS infrastructure)\n\n"
                f"Also ensure AWS_REGION is set (current: {region})"
            ) from e

        # Validate credentials by attempting a lightweight operation
        try:
            # Get caller identity to verify credentials work
            sts_client = boto3.client("sts", region_name=region)
            sts_client.get_caller_identity()
        except NoCredentialsError as e:
            raise ValueError(
                "AWS credentials not configured. "
                "Set AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY, "
                "or configure ~/.aws/credentials"
            ) from e
        except ClientError as e:
            error_code = e.response.get("Error", {}).get("Code", "Unknown")
            if error_code == "InvalidClientTokenId":
                raise ValueError(
                    "Invalid AWS credentials. Please check your AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY"
                ) from e
            elif error_code == "SignatureDoesNotMatch":
                raise ValueError(
                    "AWS signature mismatch. Please check your AWS_SECRET_ACCESS_KEY"
                ) from e
            else:
                logger.warning(f"AWS credential validation failed with {error_code}: {e}")
        except Exception as e:
            logger.warning(f"Could not validate AWS credentials: {e}")

        # Usage tracking
        self._requests_made = 0
        self._tokens_used = 0
        self._prompt_tokens = 0
        self._completion_tokens = 0

    @property
    def name(self) -> str:
        """Provider name."""
        return "anthropic-bedrock"

    @property
    def model(self) -> str:
        """Model name."""
        return self._model

    def _get_bedrock_model_id(self, model: str) -> str:
        """Map friendly model name to Bedrock model ID.

        Args:
            model: Friendly model name (e.g., "claude-3-5-sonnet-20241022")

        Returns:
            Bedrock model ID (e.g., "anthropic.claude-3-5-sonnet-20241022-v2:0")
        """
        # Allow direct Bedrock model IDs to pass through
        if model.startswith("anthropic."):
            return model

        # Map friendly names to Bedrock IDs
        mapping = {
            "claude-3-5-sonnet-20241022": "anthropic.claude-3-5-sonnet-20241022-v2:0",
            "claude-3-5-haiku-20241022": "anthropic.claude-3-5-haiku-20241022-v1:0",
            "claude-3-opus-20240229": "anthropic.claude-3-opus-20240229-v1:0",
            "claude-3-sonnet-20240229": "anthropic.claude-3-sonnet-20240229-v1:0",
            "claude-3-haiku-20240307": "anthropic.claude-3-haiku-20240307-v1:0",
        }

        return mapping.get(model, model)

    async def _invoke_bedrock_async(
        self,
        messages: list[dict[str, str]],
        max_tokens: int,
        system: str | None = None,
    ) -> dict[str, Any]:
        """Async wrapper around synchronous boto3 Bedrock call.

        Args:
            messages: List of message dicts with role and content
            max_tokens: Maximum tokens to generate
            system: Optional system prompt

        Returns:
            Bedrock response body as dict
        """

        def _sync_invoke():
            body = {
                "anthropic_version": "bedrock-2023-05-31",
                "messages": messages,
                "max_tokens": max_tokens,
            }

            # Add system prompt if provided (native Bedrock parameter)
            if system:
                body["system"] = system

            response = self._client.invoke_model(
                modelId=self._get_bedrock_model_id(self._model),
                body=json.dumps(body),
            )

            return json.loads(response["body"].read())

        # Run synchronous boto3 call in thread pool
        return await asyncio.to_thread(_sync_invoke)

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
            timeout: Optional timeout in seconds (not used - boto3 handles timeouts)

        Returns:
            LLMResponse with content and metadata
        """
        # Build messages array
        messages = [{"role": "user", "content": prompt}]

        try:
            response_body = await self._invoke_bedrock_async(
                messages=messages,
                max_tokens=max_completion_tokens,
                system=system,
            )

            # Update usage stats
            self._requests_made += 1
            usage = response_body.get("usage", {})
            input_tokens = usage.get("input_tokens", 0)
            output_tokens = usage.get("output_tokens", 0)

            self._prompt_tokens += input_tokens
            self._completion_tokens += output_tokens
            self._tokens_used += input_tokens + output_tokens

            # Extract content from response
            content = response_body["content"][0]["text"]
            stop_reason = response_body.get("stop_reason")

            return LLMResponse(
                content=content,
                tokens_used=input_tokens + output_tokens,
                model=self._model,
                finish_reason=stop_reason,
            )

        except NoCredentialsError as e:
            logger.error("AWS credentials not found")
            raise RuntimeError(
                "AWS credentials not configured. "
                "Set AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY environment variables"
            ) from e
        except ClientError as e:
            error_code = e.response.get("Error", {}).get("Code", "Unknown")
            error_message = e.response.get("Error", {}).get("Message", str(e))

            if error_code == "ThrottlingException":
                logger.error(f"Bedrock throttling: {error_message}")
                raise RuntimeError(
                    f"AWS Bedrock rate limit exceeded: {error_message}. "
                    "Consider reducing concurrent requests or increasing quota"
                ) from e
            elif error_code == "AccessDeniedException":
                logger.error(f"Bedrock access denied: {error_message}")
                raise RuntimeError(
                    f"AWS Bedrock access denied: {error_message}. "
                    f"Ensure your AWS credentials have bedrock:InvokeModel permission for {self._model}"
                ) from e
            elif error_code == "ResourceNotFoundException":
                logger.error(f"Bedrock model not found: {error_message}")
                raise RuntimeError(
                    f"Model '{self._model}' not found in AWS Bedrock. "
                    "Check model availability in your region or verify model ID"
                ) from e
            elif error_code == "ValidationException":
                logger.error(f"Bedrock validation error: {error_message}")
                raise RuntimeError(
                    f"Invalid request to Bedrock: {error_message}"
                ) from e
            else:
                logger.error(f"Bedrock API error ({error_code}): {error_message}")
                raise RuntimeError(
                    f"AWS Bedrock API error ({error_code}): {error_message}"
                ) from e
        except BotoCoreError as e:
            logger.error(f"Bedrock connection error: {e}")
            raise RuntimeError(
                f"AWS Bedrock connection failed: {e}. "
                "Check network connectivity and region configuration"
            ) from e
        except Exception as e:
            logger.error(f"Bedrock completion failed: {e}")
            raise RuntimeError(f"LLM completion failed: {e}") from e

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

    async def complete_structured(
        self,
        prompt: str,
        json_schema: dict[str, Any],
        system: str | None = None,
        max_completion_tokens: int = 4096,
        timeout: int | None = None,
    ) -> dict[str, Any]:
        """Generate a structured JSON completion conforming to the given schema.

        Since AWS Bedrock doesn't natively support JSON schema validation,
        we include the schema in the prompt and request JSON output.

        Args:
            prompt: The user prompt
            json_schema: JSON Schema definition for structured output
            system: Optional system prompt
            max_completion_tokens: Maximum tokens to generate
            timeout: Optional timeout in seconds (not used - boto3 handles timeouts)

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
            response = await self.complete(
                structured_prompt, system, max_completion_tokens, timeout
            )

            # Extract JSON from response (handle markdown code blocks)
            json_content = extract_json_from_response(response.content)

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
        except ValueError as e:
            logger.error(f"Schema validation failed: {e}")
            raise RuntimeError(f"Structured output validation failed: {e}") from e
        except Exception as e:
            logger.error(f"Bedrock structured completion failed: {e}")
            raise RuntimeError(f"LLM structured completion failed: {e}") from e

    def estimate_tokens(self, text: str) -> int:
        """Estimate token count for text.

        Note: This is a rough approximation using ~4 characters per token.
        For production use, consider using a proper tokenizer like tiktoken.
        """
        # Rough estimation: ~4 chars per token for Claude models
        return len(text) // 4

    async def health_check(self) -> dict[str, Any]:
        """Perform health check."""
        try:
            response = await self.complete("Say 'OK'", max_completion_tokens=10)
            return {
                "status": "healthy",
                "provider": "anthropic-bedrock",
                "model": self._model,
                "test_response": response.content[:50],
            }
        except Exception as e:
            return {
                "status": "unhealthy",
                "provider": "anthropic-bedrock",
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
