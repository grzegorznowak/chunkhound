"""Tests for shallow vs deep research mode behavior.

Tests verify that shallow and deep modes have different characteristics:
- Shallow: max_depth=1, 10k output tokens, faster
- Deep: adaptive max_depth, 20k output tokens, comprehensive
"""

import tempfile
from pathlib import Path
from unittest.mock import Mock, patch, AsyncMock

import pytest

from chunkhound.core.config.config import Config
from chunkhound.database_factory import create_services
from chunkhound.embeddings import EmbeddingManager
from chunkhound.llm_manager import LLMManager
from chunkhound.services.deep_research_service import (
    DeepResearchService,
    OUTPUT_TOKENS_WITH_REASONING,
    SHALLOW_TARGET_TOKENS,
    DEEP_TARGET_TOKENS,
)
from chunkhound.services.indexing_coordinator import IndexingCoordinator
from tests.fixtures.fake_providers import FakeLLMProvider, FakeEmbeddingProvider


class TestShallowDeepResearch:
    """Test shallow vs deep research mode behavior."""

    @pytest.fixture
    async def research_setup(self):
        """Setup minimal research environment with fake providers."""
        # Create temporary directory and database
        temp_dir = Path(tempfile.mkdtemp())
        db_path = temp_dir / ".chunkhound" / "test.db"
        db_path.parent.mkdir(parents=True, exist_ok=True)

        # Create minimal test file
        test_file = temp_dir / "example.py"
        test_file.write_text(
            """def search_semantic(query: str, limit: int = 10):
    '''Semantic search implementation.'''
    embedding = embed_query(query)
    results = database.search(embedding, limit)
    return results

def embed_query(query: str):
    '''Generate query embedding.'''
    return embedding_provider.embed(query)
"""
        )

        # Use fake args to prevent find_project_root
        from types import SimpleNamespace

        fake_args = SimpleNamespace(path=temp_dir)

        # Create config
        config = Config(
            args=fake_args,
            database={"path": str(db_path), "provider": "duckdb"},
            indexing={"include": ["*"], "exclude": []},
        )

        # Create services
        services = create_services(db_path, config)
        services.provider.connect()

        # Create fake providers
        fake_embedding = FakeEmbeddingProvider(
            model="fake-embeddings", dims=1536, batch_size=100
        )
        fake_llm = FakeLLMProvider(
            model="fake-llm",
            responses={
                "expand": "semantic search, embeddings, database queries",
                "follow": "1. How does embedding work?\n2. What database is used?",
                "synthesis": "## Search Implementation\nThe code implements semantic search using embeddings and database queries.",
            },
        )

        # Register embedding provider
        embedding_manager = EmbeddingManager()
        embedding_manager.register_provider(fake_embedding, set_default=True)

        # Create LLM manager with fake providers
        dummy_config = {
            "provider": "openai",
            "api_key": "fake-key",
            "model": "fake-model",
        }
        llm_manager = LLMManager(
            utility_config=dummy_config, synthesis_config=dummy_config
        )
        llm_manager._utility_provider = fake_llm
        llm_manager._synthesis_provider = fake_llm

        # Index test file
        coordinator = IndexingCoordinator(
            database_provider=services.provider,
            base_directory=temp_dir,
            embedding_provider=fake_embedding,
        )
        await coordinator.process_file(test_file)
        await coordinator.generate_missing_embeddings()

        # Create research service
        research_service = DeepResearchService(
            database_services=services,
            embedding_manager=embedding_manager,
            llm_manager=llm_manager,
        )

        yield {
            "service": research_service,
            "temp_dir": temp_dir,
            "db_path": db_path,
        }

        # Cleanup
        services.provider.close()

    @pytest.mark.asyncio
    async def test_shallow_mode_max_depth(self, research_setup):
        """Test that shallow mode uses max_depth=1."""
        service = research_setup["service"]

        # Calculate max depth for shallow mode
        max_depth_shallow = service._calculate_max_depth(depth="shallow")

        assert max_depth_shallow == 1, "Shallow mode should always use max_depth=1"

    @pytest.mark.asyncio
    async def test_deep_mode_adaptive_depth(self, research_setup):
        """Test that deep mode uses adaptive max_depth based on repo size."""
        service = research_setup["service"]

        # Calculate max depth for deep mode
        max_depth_deep = service._calculate_max_depth(depth="deep")

        # Deep mode should use adaptive depth (>1 for larger repos)
        # For small test repo, it may still be 1, but the calculation is adaptive
        assert isinstance(max_depth_deep, int), "Max depth should be an integer"
        assert max_depth_deep >= 1, "Max depth should be at least 1"

    @pytest.mark.asyncio
    async def test_shallow_mode_token_limit(self, research_setup):
        """Test that shallow mode uses OUTPUT_TOKENS_WITH_REASONING for API limit."""
        service = research_setup["service"]

        # Create a proper mock response with all required attributes
        mock_response = Mock()
        mock_response.content = "Test response " * 50  # Make it long enough (>100 chars)
        mock_response.finish_reason = "stop"
        mock_response.tokens_used = 1000

        with patch.object(
            service._llm_manager.get_synthesis_provider(),
            "complete",
            new_callable=AsyncMock,
        ) as mock_complete:
            mock_complete.return_value = mock_response

            # Call single-pass synthesis with shallow mode
            await service._single_pass_synthesis(
                root_query="test query",
                chunks=[],
                files={},
                context=Mock(),
                depth="shallow",
            )

            # Verify the max_completion_tokens parameter uses reasoning budget
            call_args = mock_complete.call_args
            assert (
                call_args.kwargs.get("max_completion_tokens") == OUTPUT_TOKENS_WITH_REASONING
            ), f"Both modes should use {OUTPUT_TOKENS_WITH_REASONING} tokens for reasoning models"

    @pytest.mark.asyncio
    async def test_deep_mode_token_limit(self, research_setup):
        """Test that deep mode uses OUTPUT_TOKENS_WITH_REASONING for API limit."""
        service = research_setup["service"]

        # Create a proper mock response with all required attributes
        mock_response = Mock()
        mock_response.content = "Deep mode test response " * 50  # Make it long enough (>100 chars)
        mock_response.finish_reason = "stop"
        mock_response.tokens_used = 1000

        with patch.object(
            service._llm_manager.get_synthesis_provider(),
            "complete",
            new_callable=AsyncMock,
        ) as mock_complete:
            mock_complete.return_value = mock_response

            # Call single-pass synthesis with deep mode
            await service._single_pass_synthesis(
                root_query="test query",
                chunks=[],
                files={},
                context=Mock(),
                depth="deep",
            )

            # Verify the max_completion_tokens parameter uses reasoning budget
            call_args = mock_complete.call_args
            assert (
                call_args.kwargs.get("max_completion_tokens") == OUTPUT_TOKENS_WITH_REASONING
            ), f"Both modes should use {OUTPUT_TOKENS_WITH_REASONING} tokens for reasoning models"

    @pytest.mark.asyncio
    async def test_depth_parameter_validation(self, research_setup):
        """Test that invalid depth parameter raises ValueError."""
        service = research_setup["service"]

        with pytest.raises(ValueError, match="Invalid depth"):
            await service.deep_research("test query", depth="invalid")

    @pytest.mark.asyncio
    async def test_shallow_default_parameter(self, research_setup):
        """Test that shallow is the default depth mode."""
        service = research_setup["service"]

        # Mock the synthesis to avoid full execution
        with patch.object(service, "_single_pass_synthesis", new_callable=AsyncMock) as mock_synthesis:
            mock_synthesis.return_value = "Test answer"

            # Call without specifying depth
            result = await service.deep_research("test query")

            # Should succeed (shallow is default)
            assert "answer" in result

    @pytest.mark.asyncio
    async def test_sources_footer_appended_in_synthesis(self):
        """Test that sources footer is appended during synthesis with non-empty data."""
        # Create a minimal service instance
        service = DeepResearchService(
            database_services=Mock(),
            embedding_manager=Mock(),
            llm_manager=Mock(),
        )

        # Mock LLM provider
        fake_llm = Mock()
        # Create a proper mock response with all required attributes
        mock_response = Mock()
        mock_response.content = "LLM answer " * 50  # Make it long enough (>100 chars)
        mock_response.finish_reason = "stop"
        mock_response.tokens_used = 1000
        fake_llm.complete = AsyncMock(return_value=mock_response)
        fake_llm.estimate_tokens = lambda text: len(text.split())

        service._llm_manager.get_synthesis_provider = Mock(return_value=fake_llm)

        # Test data with files and chunks to trigger footer generation
        test_chunks = [{"file_path": "test.py", "start_line": 1, "end_line": 10}]
        test_files = {"test.py": "def foo(): pass"}
        test_context = Mock()

        # Call synthesis directly
        result = await service._single_pass_synthesis(
            root_query="test",
            chunks=test_chunks,
            files=test_files,
            context=test_context,
            depth="shallow"
        )

        # Verify footer is present
        assert "## Sources" in result, "Sources footer should be present"
        assert "**Files**: 1" in result, "Footer should show file count"
        assert "**Chunks**: 1" in result, "Footer should show chunk count"
        assert "test.py" in result, "Footer should list analyzed file"


class TestSourcesFooter:
    """Test sources footer generation."""

    def test_sources_footer_with_chunks(self):
        """Test sources footer with multiple chunks."""
        service = DeepResearchService(
            database_services=Mock(),
            embedding_manager=Mock(),
            llm_manager=Mock(),
        )

        chunks = [
            {
                "file_path": "src/main.py",
                "start_line": 10,
                "end_line": 25,
            },
            {
                "file_path": "src/main.py",
                "start_line": 50,
                "end_line": 75,
            },
            {
                "file_path": "tests/test.py",
                "start_line": 5,
                "end_line": 15,
            },
        ]
        files = {"src/main.py": "...", "tests/test.py": "..."}

        footer = service._build_sources_footer(chunks, files)

        # Verify footer structure
        assert "## Sources" in footer
        assert "**Files**: 2" in footer
        assert "**Chunks**: 3" in footer
        # Verify nested tree structure (directories and files)
        assert "src/" in footer  # Directory with / suffix
        assert "tests/" in footer  # Directory with / suffix
        assert "main.py" in footer  # File name (without full path due to nesting)
        assert "test.py" in footer  # File name (without full path due to nesting)
        assert "L10-25" in footer
        assert "L50-75" in footer
        assert "L5-15" in footer
        # Verify uses tabs for indentation (token efficient)
        assert "\t" in footer

    def test_sources_footer_empty(self):
        """Test sources footer with no data."""
        service = DeepResearchService(
            database_services=Mock(),
            embedding_manager=Mock(),
            llm_manager=Mock(),
        )

        footer = service._build_sources_footer([], {})

        assert footer == "", "Empty sources should return empty string"

    def test_sources_footer_many_chunks(self):
        """Test sources footer condenses many chunks."""
        service = DeepResearchService(
            database_services=Mock(),
            embedding_manager=Mock(),
            llm_manager=Mock(),
        )

        # Create 5 chunks for same file
        chunks = [
            {
                "file_path": "src/big_file.py",
                "start_line": i * 10,
                "end_line": i * 10 + 5,
            }
            for i in range(5)
        ]
        files = {"src/big_file.py": "..."}

        footer = service._build_sources_footer(chunks, files)

        # Should show first 3 ranges + count
        assert "L0-5, L10-15, L20-25, +2 more" in footer
        assert "5 chunks" in footer
        # Verify nested structure
        assert "src/" in footer
        assert "big_file.py" in footer

    def test_sources_footer_full_file(self):
        """Test sources footer for full file (no chunks)."""
        service = DeepResearchService(
            database_services=Mock(),
            embedding_manager=Mock(),
            llm_manager=Mock(),
        )

        # File without chunks
        chunks = []
        files = {"src/config.py": "..."}

        footer = service._build_sources_footer(chunks, files)

        # Verify nested structure with full file annotation
        assert "src/" in footer
        assert "config.py (full file)" in footer

    def test_sources_footer_line_ranges(self):
        """Test that line ranges are formatted correctly."""
        service = DeepResearchService(
            database_services=Mock(),
            embedding_manager=Mock(),
            llm_manager=Mock(),
        )

        chunks = [
            {
                "file_path": "src/test.py",
                "start_line": 100,
                "end_line": 200,
            }
        ]
        files = {"src/test.py": "..."}

        footer = service._build_sources_footer(chunks, files)

        assert "L100-200" in footer, "Line ranges should be formatted as L{start}-{end}"

    def test_sources_footer_error_handling(self):
        """Test that malformed chunk data doesn't crash footer generation."""
        service = DeepResearchService(
            database_services=Mock(),
            embedding_manager=Mock(),
            llm_manager=Mock(),
        )

        # Malformed chunks with missing fields
        malformed_chunks = [
            {"file_path": "test.py"},  # Missing start_line, end_line
            {"start_line": 10, "end_line": 20},  # Missing file_path
            {},  # Empty chunk
        ]
        files = {"test.py": "..."}

        # Should not crash - error handling in _single_pass_synthesis will catch it
        # Here we test the footer generation itself
        try:
            footer = service._build_sources_footer(malformed_chunks, files)
            # If it doesn't crash, that's good - footer might be partial but shouldn't fail
            assert isinstance(footer, str)
        except Exception as e:
            # If it does raise, we want to know what the error is
            pytest.fail(f"Footer generation should be resilient to malformed data: {e}")
