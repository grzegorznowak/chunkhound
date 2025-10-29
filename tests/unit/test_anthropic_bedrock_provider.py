"""Tests for Anthropic Bedrock LLM provider."""

import json
from unittest.mock import MagicMock, Mock, patch

import pytest

from chunkhound.interfaces.llm_provider import LLMResponse


@pytest.fixture
def mock_boto3():
    """Mock boto3 to avoid real AWS calls."""
    with patch("chunkhound.providers.llm.anthropic_bedrock_provider.boto3") as mock:
        # Mock bedrock-runtime client
        mock_client = MagicMock()
        mock.client.return_value = mock_client

        # Mock STS client for credential validation
        mock_sts = MagicMock()
        mock_sts.get_caller_identity.return_value = {"Account": "123456789012"}

        def client_factory(service_name, **kwargs):
            if service_name == "sts":
                return mock_sts
            elif service_name == "bedrock-runtime":
                return mock_client
            return MagicMock()

        mock.client.side_effect = client_factory
        yield mock


@pytest.fixture
def mock_config():
    """Mock botocore Config class."""
    with patch("chunkhound.providers.llm.anthropic_bedrock_provider.Config") as mock:
        yield mock


@pytest.fixture
def provider(mock_boto3, mock_config):
    """Create an AnthropicBedrockProvider instance for testing."""
    from chunkhound.providers.llm.anthropic_bedrock_provider import (
        AnthropicBedrockProvider,
    )

    return AnthropicBedrockProvider(
        model="claude-3-5-sonnet-20241022",
        timeout=60,
        max_retries=3,
    )


class TestAnthropicBedrockProvider:
    """Test suite for AnthropicBedrockProvider."""

    def test_provider_name(self, provider):
        """Test that provider name is correct."""
        assert provider.name == "anthropic-bedrock"

    def test_provider_model(self, provider):
        """Test that model name is stored correctly."""
        assert provider.model == "claude-3-5-sonnet-20241022"

    def test_model_id_mapping(self, provider):
        """Test friendly model name to Bedrock ID mapping."""
        # Known mappings
        assert (
            provider._get_bedrock_model_id("claude-3-5-sonnet-20241022")
            == "anthropic.claude-3-5-sonnet-20241022-v2:0"
        )
        assert (
            provider._get_bedrock_model_id("claude-3-5-haiku-20241022")
            == "anthropic.claude-3-5-haiku-20241022-v1:0"
        )
        assert (
            provider._get_bedrock_model_id("claude-3-opus-20240229")
            == "anthropic.claude-3-opus-20240229-v1:0"
        )

        # Pass-through for direct Bedrock IDs
        bedrock_id = "anthropic.claude-3-5-sonnet-20241022-v2:0"
        assert provider._get_bedrock_model_id(bedrock_id) == bedrock_id

        # Fallback for unknown models
        assert provider._get_bedrock_model_id("unknown-model") == "unknown-model"

    def test_estimate_tokens(self, provider):
        """Test token estimation (rough approximation)."""
        text = "a" * 400  # 400 characters
        tokens = provider.estimate_tokens(text)
        assert tokens == 100  # 400 / 4 = 100 tokens

        empty_text = ""
        assert provider.estimate_tokens(empty_text) == 0

    def test_extract_json_from_code_block(self):
        """Test JSON extraction from markdown code blocks."""
        from chunkhound.utils.json_extraction import extract_json_from_response

        # JSON in ```json block
        response = '```json\n{"key": "value"}\n```'
        assert extract_json_from_response(response) == '{"key": "value"}'

        # JSON in generic ``` block
        response = '```\n{"key": "value"}\n```'
        assert extract_json_from_response(response) == '{"key": "value"}'

        # Raw JSON (no code blocks)
        response = '{"key": "value"}'
        assert extract_json_from_response(response) == '{"key": "value"}'

        # Multiple code blocks (takes first)
        response = '```json\n{"first": 1}\n```\ntext\n```json\n{"second": 2}\n```'
        assert extract_json_from_response(response) == '{"first": 1}'

    def test_extract_json_empty_content_raises_error(self):
        """Test that empty content raises ValueError."""
        from chunkhound.utils.json_extraction import extract_json_from_response

        with pytest.raises(ValueError, match="No JSON content found"):
            extract_json_from_response("")

        with pytest.raises(ValueError, match="No JSON content found"):
            extract_json_from_response("   ")

    @pytest.mark.asyncio
    async def test_complete_success(self, provider, mock_boto3):
        """Test successful completion."""
        # Mock Bedrock response
        mock_response = {
            "body": Mock(
                read=lambda: json.dumps(
                    {
                        "content": [{"text": "Hello, world!"}],
                        "stop_reason": "end_turn",
                        "usage": {"input_tokens": 10, "output_tokens": 5},
                    }
                ).encode()
            )
        }
        provider._client.invoke_model.return_value = mock_response

        response = await provider.complete("Say hello")

        assert isinstance(response, LLMResponse)
        assert response.content == "Hello, world!"
        assert response.tokens_used == 15
        assert response.model == "claude-3-5-sonnet-20241022"
        assert response.finish_reason == "end_turn"

        # Verify usage tracking
        assert provider._requests_made == 1
        assert provider._tokens_used == 15
        assert provider._prompt_tokens == 10
        assert provider._completion_tokens == 5

    @pytest.mark.asyncio
    async def test_complete_with_system_prompt(self, provider, mock_boto3):
        """Test completion with system prompt."""
        # Mock Bedrock response
        mock_response = {
            "body": Mock(
                read=lambda: json.dumps(
                    {
                        "content": [{"text": "Response"}],
                        "stop_reason": "end_turn",
                        "usage": {"input_tokens": 20, "output_tokens": 10},
                    }
                ).encode()
            )
        }
        provider._client.invoke_model.return_value = mock_response

        response = await provider.complete(
            "User prompt", system="You are a helpful assistant"
        )

        assert response.content == "Response"

        # Verify invoke_model was called with system parameter
        call_args = provider._client.invoke_model.call_args
        body = json.loads(call_args[1]["body"])
        assert "system" in body
        assert body["system"] == "You are a helpful assistant"
        assert body["messages"] == [{"role": "user", "content": "User prompt"}]

    @pytest.mark.asyncio
    async def test_complete_throttling_error(self, provider, mock_boto3):
        """Test handling of AWS throttling errors."""
        from botocore.exceptions import ClientError

        error_response = {
            "Error": {"Code": "ThrottlingException", "Message": "Rate exceeded"}
        }
        provider._client.invoke_model.side_effect = ClientError(
            error_response, "InvokeModel"
        )

        with pytest.raises(RuntimeError, match="rate limit exceeded"):
            await provider.complete("Test prompt")

    @pytest.mark.asyncio
    async def test_complete_access_denied_error(self, provider, mock_boto3):
        """Test handling of AWS access denied errors."""
        from botocore.exceptions import ClientError

        error_response = {
            "Error": {"Code": "AccessDeniedException", "Message": "Not authorized"}
        }
        provider._client.invoke_model.side_effect = ClientError(
            error_response, "InvokeModel"
        )

        with pytest.raises(RuntimeError, match="access denied"):
            await provider.complete("Test prompt")

    @pytest.mark.asyncio
    async def test_complete_model_not_found_error(self, provider, mock_boto3):
        """Test handling of model not found errors."""
        from botocore.exceptions import ClientError

        error_response = {
            "Error": {"Code": "ResourceNotFoundException", "Message": "Model not found"}
        }
        provider._client.invoke_model.side_effect = ClientError(
            error_response, "InvokeModel"
        )

        with pytest.raises(RuntimeError, match="not found"):
            await provider.complete("Test prompt")

    @pytest.mark.asyncio
    async def test_complete_no_credentials_error(self, provider, mock_boto3):
        """Test handling of missing AWS credentials."""
        from botocore.exceptions import NoCredentialsError

        provider._client.invoke_model.side_effect = NoCredentialsError()

        with pytest.raises(RuntimeError, match="credentials not configured"):
            await provider.complete("Test prompt")

    @pytest.mark.asyncio
    async def test_batch_complete(self, provider, mock_boto3):
        """Test batch completion."""
        # Mock Bedrock responses
        def mock_invoke(*args, **kwargs):
            return {
                "body": Mock(
                    read=lambda: json.dumps(
                        {
                            "content": [{"text": "Response"}],
                            "stop_reason": "end_turn",
                            "usage": {"input_tokens": 10, "output_tokens": 5},
                        }
                    ).encode()
                )
            }

        provider._client.invoke_model.side_effect = mock_invoke

        prompts = ["Prompt 1", "Prompt 2", "Prompt 3"]
        responses = await provider.batch_complete(prompts)

        assert len(responses) == 3
        assert all(isinstance(r, LLMResponse) for r in responses)
        assert all(r.content == "Response" for r in responses)

        # Verify invoke_model called 3 times
        assert provider._client.invoke_model.call_count == 3

    @pytest.mark.asyncio
    async def test_complete_structured_success(self, provider, mock_boto3):
        """Test structured completion with valid JSON."""
        # Mock Bedrock response with JSON in markdown
        mock_response = {
            "body": Mock(
                read=lambda: json.dumps(
                    {
                        "content": [
                            {"text": '```json\n{"name": "Alice", "age": 30}\n```'}
                        ],
                        "stop_reason": "end_turn",
                        "usage": {"input_tokens": 20, "output_tokens": 10},
                    }
                ).encode()
            )
        }
        provider._client.invoke_model.return_value = mock_response

        schema = {
            "type": "object",
            "properties": {"name": {"type": "string"}, "age": {"type": "integer"}},
            "required": ["name", "age"],
        }

        result = await provider.complete_structured("Get user info", schema)

        assert result == {"name": "Alice", "age": 30}

    @pytest.mark.asyncio
    async def test_complete_structured_missing_required_field(self, provider, mock_boto3):
        """Test structured completion with missing required fields."""
        # Mock Bedrock response with incomplete JSON
        mock_response = {
            "body": Mock(
                read=lambda: json.dumps(
                    {
                        "content": [{"text": '{"name": "Alice"}'}],
                        "stop_reason": "end_turn",
                        "usage": {"input_tokens": 20, "output_tokens": 10},
                    }
                ).encode()
            )
        }
        provider._client.invoke_model.return_value = mock_response

        schema = {
            "type": "object",
            "properties": {"name": {"type": "string"}, "age": {"type": "integer"}},
            "required": ["name", "age"],
        }

        with pytest.raises(RuntimeError, match="validation failed"):
            await provider.complete_structured("Get user info", schema)

    @pytest.mark.asyncio
    async def test_complete_structured_invalid_json(self, provider, mock_boto3):
        """Test structured completion with invalid JSON."""
        # Mock Bedrock response with invalid JSON
        mock_response = {
            "body": Mock(
                read=lambda: json.dumps(
                    {
                        "content": [{"text": "Not valid JSON"}],
                        "stop_reason": "end_turn",
                        "usage": {"input_tokens": 20, "output_tokens": 10},
                    }
                ).encode()
            )
        }
        provider._client.invoke_model.return_value = mock_response

        schema = {"type": "object", "properties": {}}

        with pytest.raises(RuntimeError, match="Invalid JSON"):
            await provider.complete_structured("Get info", schema)

    @pytest.mark.asyncio
    async def test_health_check_healthy(self, provider, mock_boto3):
        """Test health check when provider is healthy."""
        # Mock successful response
        mock_response = {
            "body": Mock(
                read=lambda: json.dumps(
                    {
                        "content": [{"text": "OK"}],
                        "stop_reason": "end_turn",
                        "usage": {"input_tokens": 5, "output_tokens": 2},
                    }
                ).encode()
            )
        }
        provider._client.invoke_model.return_value = mock_response

        result = await provider.health_check()

        assert result["status"] == "healthy"
        assert result["provider"] == "anthropic-bedrock"
        assert result["model"] == "claude-3-5-sonnet-20241022"
        assert "OK" in result["test_response"]

    @pytest.mark.asyncio
    async def test_health_check_unhealthy(self, provider, mock_boto3):
        """Test health check when provider is unhealthy."""
        from botocore.exceptions import ClientError

        error_response = {"Error": {"Code": "ServiceError", "Message": "Service down"}}
        provider._client.invoke_model.side_effect = ClientError(
            error_response, "InvokeModel"
        )

        result = await provider.health_check()

        assert result["status"] == "unhealthy"
        assert result["provider"] == "anthropic-bedrock"
        assert "error" in result

    def test_get_usage_stats(self, provider):
        """Test usage statistics tracking."""
        # Initially zero
        stats = provider.get_usage_stats()
        assert stats["requests_made"] == 0
        assert stats["total_tokens"] == 0
        assert stats["prompt_tokens"] == 0
        assert stats["completion_tokens"] == 0

        # Manually update counters (simulating requests)
        provider._requests_made = 5
        provider._tokens_used = 1000
        provider._prompt_tokens = 600
        provider._completion_tokens = 400

        stats = provider.get_usage_stats()
        assert stats["requests_made"] == 5
        assert stats["total_tokens"] == 1000
        assert stats["prompt_tokens"] == 600
        assert stats["completion_tokens"] == 400

    def test_initialization_without_boto3_raises_error(self):
        """Test that initialization fails gracefully when boto3 not available."""
        with patch(
            "chunkhound.providers.llm.anthropic_bedrock_provider.BEDROCK_AVAILABLE",
            False,
        ):
            from chunkhound.providers.llm.anthropic_bedrock_provider import (
                AnthropicBedrockProvider,
            )

            with pytest.raises(ImportError, match="boto3 not available"):
                AnthropicBedrockProvider()

    def test_credential_validation_on_init(self, mock_boto3, mock_config):
        """Test that AWS credentials are validated during initialization."""
        from chunkhound.providers.llm.anthropic_bedrock_provider import (
            AnthropicBedrockProvider,
        )

        # Mock STS call to succeed
        mock_sts = MagicMock()
        mock_sts.get_caller_identity.return_value = {"Account": "123456789012"}

        def client_factory(service_name, **kwargs):
            if service_name == "sts":
                return mock_sts
            return MagicMock()

        mock_boto3.client.side_effect = client_factory

        # Should not raise
        provider = AnthropicBedrockProvider()
        assert provider is not None

    def test_invalid_credentials_on_init_raises_error(self, mock_boto3, mock_config):
        """Test that invalid AWS credentials raise helpful error."""
        from botocore.exceptions import NoCredentialsError

        from chunkhound.providers.llm.anthropic_bedrock_provider import (
            AnthropicBedrockProvider,
        )

        # Mock Session
        mock_session = MagicMock()
        mock_boto3.Session.return_value = mock_session

        # Mock STS to raise NoCredentialsError
        mock_sts = MagicMock()
        mock_sts.get_caller_identity.side_effect = NoCredentialsError()

        def session_client_factory(service_name, **kwargs):
            if service_name == "sts":
                return mock_sts
            return MagicMock()

        mock_session.client.side_effect = session_client_factory

        with pytest.raises(ValueError, match="credentials not configured"):
            AnthropicBedrockProvider()

    def test_profile_name_creates_session(self, mock_boto3, mock_config):
        """Test that profile_name parameter creates a boto3 Session."""
        from chunkhound.providers.llm.anthropic_bedrock_provider import (
            AnthropicBedrockProvider,
        )

        # Mock Session
        mock_session = MagicMock()
        mock_boto3.Session.return_value = mock_session

        # Mock client creation from session
        mock_client = MagicMock()
        mock_session.client.return_value = mock_client

        # Mock STS for validation
        mock_sts = MagicMock()
        mock_sts.get_caller_identity.return_value = {
            "Account": "123456789012",
            "Arn": "arn:aws:iam::123456789012:user/test-user",
        }

        def session_client_factory(service_name, **kwargs):
            if service_name == "sts":
                return mock_sts
            elif service_name == "bedrock-runtime":
                return mock_client
            return MagicMock()

        mock_session.client.side_effect = session_client_factory

        # Create provider with profile_name
        provider = AnthropicBedrockProvider(profile_name="my-profile")

        # Verify Session was created with profile_name
        mock_boto3.Session.assert_called_once_with(profile_name="my-profile")

        # Verify client was created from session
        assert mock_session.client.call_count >= 1
        bedrock_call = [
            call
            for call in mock_session.client.call_args_list
            if call[0][0] == "bedrock-runtime"
        ][0]
        assert bedrock_call[1]["region_name"] == "us-east-1"
        assert bedrock_call[1]["endpoint_url"] is None

    def test_profile_name_overrides_env_var(self, mock_boto3, mock_config, monkeypatch):
        """Test that profile_name parameter overrides AWS_PROFILE env var."""
        from chunkhound.providers.llm.anthropic_bedrock_provider import (
            AnthropicBedrockProvider,
        )

        # Set AWS_PROFILE env var
        monkeypatch.setenv("AWS_PROFILE", "env-profile")

        # Mock Session
        mock_session = MagicMock()
        mock_boto3.Session.return_value = mock_session

        # Mock clients
        mock_client = MagicMock()
        mock_sts = MagicMock()
        mock_sts.get_caller_identity.return_value = {"Account": "123456789012"}

        def session_client_factory(service_name, **kwargs):
            if service_name == "sts":
                return mock_sts
            return mock_client

        mock_session.client.side_effect = session_client_factory

        # Create provider with explicit profile_name (should override env var)
        provider = AnthropicBedrockProvider(profile_name="param-profile")

        # Verify Session was created with parameter value, not env var
        mock_boto3.Session.assert_called_once_with(profile_name="param-profile")

    def test_custom_endpoint_url_via_base_url(self, mock_boto3, mock_config):
        """Test that base_url parameter sets custom Bedrock endpoint URL."""
        from chunkhound.providers.llm.anthropic_bedrock_provider import (
            AnthropicBedrockProvider,
        )

        # Mock Session
        mock_session = MagicMock()
        mock_boto3.Session.return_value = mock_session

        # Mock client
        mock_client = MagicMock()
        mock_session.client.return_value = mock_client

        custom_endpoint = "https://vpce-123.bedrock-runtime.us-west-2.vpce.amazonaws.com"

        # Create provider with custom endpoint
        provider = AnthropicBedrockProvider(base_url=custom_endpoint)

        # Verify client was created with custom endpoint_url
        bedrock_call = [
            call
            for call in mock_session.client.call_args_list
            if call[0][0] == "bedrock-runtime"
        ][0]
        assert bedrock_call[1]["endpoint_url"] == custom_endpoint

    def test_custom_endpoint_skips_sts_validation(self, mock_boto3, mock_config):
        """Test that STS validation is skipped when custom endpoint_url is provided."""
        from chunkhound.providers.llm.anthropic_bedrock_provider import (
            AnthropicBedrockProvider,
        )

        # Mock Session
        mock_session = MagicMock()
        mock_boto3.Session.return_value = mock_session

        # Mock bedrock client
        mock_client = MagicMock()
        mock_session.client.return_value = mock_client

        custom_endpoint = "https://vpce-123.bedrock-runtime.us-west-2.vpce.amazonaws.com"

        # Create provider with custom endpoint (should NOT call STS)
        provider = AnthropicBedrockProvider(base_url=custom_endpoint)

        # Verify STS client was NOT created
        sts_calls = [
            call for call in mock_session.client.call_args_list if call[0][0] == "sts"
        ]
        assert len(sts_calls) == 0

    def test_profile_and_endpoint_together(self, mock_boto3, mock_config):
        """Test that profile_name and base_url can be used together."""
        from chunkhound.providers.llm.anthropic_bedrock_provider import (
            AnthropicBedrockProvider,
        )

        # Mock Session
        mock_session = MagicMock()
        mock_boto3.Session.return_value = mock_session

        # Mock client
        mock_client = MagicMock()
        mock_session.client.return_value = mock_client

        custom_endpoint = "https://vpce-123.bedrock-runtime.us-west-2.vpce.amazonaws.com"

        # Create provider with both profile and custom endpoint
        provider = AnthropicBedrockProvider(
            profile_name="vpc-profile", base_url=custom_endpoint
        )

        # Verify Session was created with profile
        mock_boto3.Session.assert_called_once_with(profile_name="vpc-profile")

        # Verify client was created with custom endpoint
        bedrock_call = [
            call
            for call in mock_session.client.call_args_list
            if call[0][0] == "bedrock-runtime"
        ][0]
        assert bedrock_call[1]["endpoint_url"] == custom_endpoint

        # Verify STS was NOT called (custom endpoint skips validation)
        sts_calls = [
            call for call in mock_session.client.call_args_list if call[0][0] == "sts"
        ]
        assert len(sts_calls) == 0
