"""Tests for AWS Bedrock LLM provider."""

import asyncio
import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from chunkhound.interfaces.llm_provider import LLMResponse
from chunkhound.providers.llm.bedrock_llm_provider import BedrockLLMProvider


@pytest.fixture
def mock_boto3_client():
    """Mock boto3 client for Bedrock."""
    with patch("boto3.client") as mock:
        client = MagicMock()
        mock.return_value = client
        yield client


@pytest.fixture
def provider(mock_boto3_client):
    """Create a BedrockLLMProvider instance for testing."""
    return BedrockLLMProvider(
        model="us.anthropic.claude-sonnet-4-5-20250929-v1:0",
        region="us-east-1",
        timeout=60,
        max_retries=3,
    )


class TestBedrockLLMProvider:
    """Test suite for BedrockLLMProvider."""

    def test_provider_name(self, provider):
        """Test that provider name is correct."""
        assert provider.name == "bedrock"

    def test_provider_model(self, provider):
        """Test that model name is stored correctly."""
        assert provider.model == "us.anthropic.claude-sonnet-4-5-20250929-v1:0"

    def test_estimate_tokens(self, provider):
        """Test token estimation (rough approximation)."""
        text = "a" * 400  # 400 characters
        tokens = provider.estimate_tokens(text)
        assert tokens == 100  # 400 / 4 = 100 tokens

        empty_text = ""
        assert provider.estimate_tokens(empty_text) == 0

    @pytest.mark.asyncio
    async def test_complete_success(self, provider, mock_boto3_client):
        """Test successful completion."""
        # Mock Bedrock response
        mock_response = {
            "body": MagicMock(
                read=MagicMock(
                    return_value=json.dumps(
                        {
                            "content": [{"text": "Test response"}],
                            "usage": {
                                "input_tokens": 10,
                                "output_tokens": 20,
                            },
                            "stop_reason": "end_turn",
                        }
                    ).encode("utf-8")
                )
            )
        }
        mock_boto3_client.invoke_model.return_value = mock_response

        response = await provider.complete("Test prompt")

        assert isinstance(response, LLMResponse)
        assert response.content == "Test response"
        assert response.model == "us.anthropic.claude-sonnet-4-5-20250929-v1:0"
        assert response.tokens_used == 30  # 10 + 20
        assert response.finish_reason == "end_turn"

        # Verify usage tracking
        assert provider._requests_made == 1
        assert provider._tokens_used == 30
        assert provider._prompt_tokens == 10
        assert provider._completion_tokens == 20

    @pytest.mark.asyncio
    async def test_complete_with_system_prompt(self, provider, mock_boto3_client):
        """Test completion with system prompt."""
        mock_response = {
            "body": MagicMock(
                read=MagicMock(
                    return_value=json.dumps(
                        {
                            "content": [{"text": "Response with system"}],
                            "usage": {"input_tokens": 15, "output_tokens": 25},
                            "stop_reason": "end_turn",
                        }
                    ).encode("utf-8")
                )
            )
        }
        mock_boto3_client.invoke_model.return_value = mock_response

        response = await provider.complete("User prompt", system="System instructions")

        assert response.content == "Response with system"

        # Verify invoke_model was called with system prompt
        call_args = mock_boto3_client.invoke_model.call_args
        body = json.loads(call_args[1]["body"])
        assert "system" in body
        assert body["system"] == "System instructions"

    @pytest.mark.asyncio
    async def test_complete_throttling_with_retry(self, provider, mock_boto3_client):
        """Test retry logic for throttling errors."""
        from botocore.exceptions import ClientError

        # First call throttles, second succeeds
        throttle_error = ClientError(
            {"Error": {"Code": "ThrottlingException", "Message": "Rate exceeded"}},
            "InvokeModel",
        )

        success_response = {
            "body": MagicMock(
                read=MagicMock(
                    return_value=json.dumps(
                        {
                            "content": [{"text": "Success after retry"}],
                            "usage": {"input_tokens": 10, "output_tokens": 20},
                            "stop_reason": "end_turn",
                        }
                    ).encode("utf-8")
                )
            )
        }

        mock_boto3_client.invoke_model.side_effect = [throttle_error, success_response]

        # Patch time.sleep to avoid actual delays in tests
        with patch("time.sleep"):
            response = await provider.complete("Test prompt")

        assert response.content == "Success after retry"
        assert mock_boto3_client.invoke_model.call_count == 2

    @pytest.mark.asyncio
    async def test_complete_validation_error_no_retry(self, provider, mock_boto3_client):
        """Test that validation errors are not retried."""
        from botocore.exceptions import ClientError

        validation_error = ClientError(
            {
                "Error": {
                    "Code": "ValidationException",
                    "Message": "Invalid model ID",
                }
            },
            "InvokeModel",
        )

        mock_boto3_client.invoke_model.side_effect = validation_error

        with pytest.raises(RuntimeError, match="Bedrock error: ValidationException"):
            await provider.complete("Test prompt")

        # Should NOT retry validation errors
        assert mock_boto3_client.invoke_model.call_count == 1

    @pytest.mark.asyncio
    async def test_complete_max_retries_exceeded(self, provider, mock_boto3_client):
        """Test that error is raised after max retries."""
        from botocore.exceptions import ClientError

        throttle_error = ClientError(
            {"Error": {"Code": "ThrottlingException", "Message": "Rate exceeded"}},
            "InvokeModel",
        )

        mock_boto3_client.invoke_model.side_effect = throttle_error

        with patch("time.sleep"):
            with pytest.raises(RuntimeError, match="Bedrock throttled after 3 retries"):
                await provider.complete("Test prompt")

        # Should have tried max_retries (3) times
        assert mock_boto3_client.invoke_model.call_count == 3

    @pytest.mark.asyncio
    async def test_complete_structured_success(self, provider, mock_boto3_client):
        """Test successful structured completion with JSON."""
        json_response_data = {"result": "success", "data": [1, 2, 3]}

        mock_response = {
            "body": MagicMock(
                read=MagicMock(
                    return_value=json.dumps(
                        {
                            "content": [{"text": json.dumps(json_response_data)}],
                            "usage": {"input_tokens": 50, "output_tokens": 30},
                            "stop_reason": "end_turn",
                        }
                    ).encode("utf-8")
                )
            )
        }
        mock_boto3_client.invoke_model.return_value = mock_response

        schema = {
            "type": "object",
            "properties": {
                "result": {"type": "string"},
                "data": {"type": "array"},
            },
            "required": ["result", "data"],
        }

        response = await provider.complete_structured("Test prompt", schema)

        assert isinstance(response, dict)
        assert response["result"] == "success"
        assert response["data"] == [1, 2, 3]

    @pytest.mark.asyncio
    async def test_complete_structured_with_markdown_code_block(
        self, provider, mock_boto3_client
    ):
        """Test structured completion with JSON in markdown code block."""
        json_data = {"result": "success"}
        markdown_response = f"```json\n{json.dumps(json_data)}\n```"

        mock_response = {
            "body": MagicMock(
                read=MagicMock(
                    return_value=json.dumps(
                        {
                            "content": [{"text": markdown_response}],
                            "usage": {"input_tokens": 50, "output_tokens": 30},
                            "stop_reason": "end_turn",
                        }
                    ).encode("utf-8")
                )
            )
        }
        mock_boto3_client.invoke_model.return_value = mock_response

        schema = {"type": "object", "properties": {"result": {"type": "string"}}}

        response = await provider.complete_structured("Test prompt", schema)

        assert isinstance(response, dict)
        assert response["result"] == "success"

    @pytest.mark.asyncio
    async def test_complete_structured_invalid_json(self, provider, mock_boto3_client):
        """Test handling of invalid JSON in structured completion."""
        mock_response = {
            "body": MagicMock(
                read=MagicMock(
                    return_value=json.dumps(
                        {
                            "content": [{"text": "Not valid JSON!"}],
                            "usage": {"input_tokens": 10, "output_tokens": 20},
                            "stop_reason": "end_turn",
                        }
                    ).encode("utf-8")
                )
            )
        }
        mock_boto3_client.invoke_model.return_value = mock_response

        schema = {"type": "object"}

        with pytest.raises(RuntimeError, match="Invalid JSON"):
            await provider.complete_structured("Test prompt", schema)

    @pytest.mark.asyncio
    async def test_complete_structured_missing_required_field(
        self, provider, mock_boto3_client
    ):
        """Test validation of required fields in structured output."""
        json_data = {"data": [1, 2, 3]}  # Missing 'result'

        mock_response = {
            "body": MagicMock(
                read=MagicMock(
                    return_value=json.dumps(
                        {
                            "content": [{"text": json.dumps(json_data)}],
                            "usage": {"input_tokens": 50, "output_tokens": 30},
                            "stop_reason": "end_turn",
                        }
                    ).encode("utf-8")
                )
            )
        }
        mock_boto3_client.invoke_model.return_value = mock_response

        schema = {
            "type": "object",
            "properties": {"result": {"type": "string"}, "data": {"type": "array"}},
            "required": ["result", "data"],
        }

        with pytest.raises(RuntimeError, match="Missing required fields"):
            await provider.complete_structured("Test prompt", schema)

    @pytest.mark.asyncio
    async def test_batch_complete(self, provider, mock_boto3_client):
        """Test batch completion (concurrent calls)."""
        responses = [
            {
                "body": MagicMock(
                    read=MagicMock(
                        return_value=json.dumps(
                            {
                                "content": [{"text": f"Response {i}"}],
                                "usage": {"input_tokens": 10, "output_tokens": 20},
                                "stop_reason": "end_turn",
                            }
                        ).encode("utf-8")
                    )
                )
            }
            for i in range(3)
        ]

        mock_boto3_client.invoke_model.side_effect = responses

        prompts = ["Prompt 1", "Prompt 2", "Prompt 3"]
        results = await provider.batch_complete(prompts)

        assert len(results) == 3
        assert all(isinstance(r, LLMResponse) for r in results)

    @pytest.mark.asyncio
    async def test_health_check_healthy(self, provider, mock_boto3_client):
        """Test health check when Bedrock is available and working."""
        mock_response = {
            "body": MagicMock(
                read=MagicMock(
                    return_value=json.dumps(
                        {
                            "content": [{"text": "OK"}],
                            "usage": {"input_tokens": 5, "output_tokens": 2},
                            "stop_reason": "end_turn",
                        }
                    ).encode("utf-8")
                )
            )
        }
        mock_boto3_client.invoke_model.return_value = mock_response

        result = await provider.health_check()

        assert result["status"] == "healthy"
        assert result["provider"] == "bedrock"
        assert result["model"] == "us.anthropic.claude-sonnet-4-5-20250929-v1:0"
        assert result["region"] == "us-east-1"
        assert "test_response" in result

    @pytest.mark.asyncio
    async def test_health_check_unhealthy(self, provider, mock_boto3_client):
        """Test health check when Bedrock call fails."""
        from botocore.exceptions import ClientError

        error = ClientError(
            {"Error": {"Code": "AccessDeniedException", "Message": "No access"}},
            "InvokeModel",
        )

        mock_boto3_client.invoke_model.side_effect = error

        result = await provider.health_check()

        assert result["status"] == "unhealthy"
        assert result["provider"] == "bedrock"
        assert "error" in result

    def test_get_usage_stats(self, provider):
        """Test usage statistics retrieval."""
        # Initially zero
        stats = provider.get_usage_stats()
        assert stats["requests_made"] == 0
        assert stats["total_tokens"] == 0

        # Manually increment (normally done by complete methods)
        provider._requests_made = 5
        provider._tokens_used = 1000
        provider._prompt_tokens = 600
        provider._completion_tokens = 400

        stats = provider.get_usage_stats()
        assert stats["requests_made"] == 5
        assert stats["total_tokens"] == 1000
        assert stats["prompt_tokens"] == 600
        assert stats["completion_tokens"] == 400

    def test_extract_json_from_response_raw_json(self, provider):
        """Test JSON extraction from raw JSON."""
        raw_json = '{"key": "value"}'
        extracted = provider._extract_json_from_response(raw_json)
        assert extracted == '{"key": "value"}'

    def test_extract_json_from_response_code_block(self, provider):
        """Test JSON extraction from markdown code block."""
        markdown = '```json\n{"key": "value"}\n```'
        extracted = provider._extract_json_from_response(markdown)
        assert extracted == '{"key": "value"}'

    def test_extract_json_from_response_generic_code_block(self, provider):
        """Test JSON extraction from generic code block."""
        markdown = '```\n{"key": "value"}\n```'
        extracted = provider._extract_json_from_response(markdown)
        assert extracted == '{"key": "value"}'

    def test_extract_json_from_response_empty_content(self, provider):
        """Test that empty content raises ValueError."""
        with pytest.raises(ValueError, match="No JSON content found"):
            provider._extract_json_from_response("")

    @pytest.mark.asyncio
    async def test_exponential_backoff_delays(self, provider, mock_boto3_client):
        """Test that exponential backoff increases delay between retries."""
        from botocore.exceptions import ClientError

        throttle_error = ClientError(
            {"Error": {"Code": "ThrottlingException", "Message": "Rate exceeded"}},
            "InvokeModel",
        )

        mock_boto3_client.invoke_model.side_effect = throttle_error

        delays = []

        def track_sleep(seconds):
            delays.append(seconds)

        with patch("time.sleep", side_effect=track_sleep):
            with pytest.raises(RuntimeError):
                await provider.complete("Test prompt")

        # Should have 2 delays (before retry 2 and retry 3)
        assert len(delays) == 2
        # Delays should be increasing (exponential backoff)
        assert delays[0] < delays[1]
        # First delay should be around 2^0 + jitter = 1-2 seconds
        assert 1 <= delays[0] <= 3
        # Second delay should be around 2^1 + jitter = 2-3 seconds
        assert 2 <= delays[1] <= 4
