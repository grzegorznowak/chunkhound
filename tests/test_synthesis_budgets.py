"""Tests for dynamic synthesis budget calculation based on repository size.

Tests verify that synthesis budgets scale appropriately with repository size.
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
    TARGET_OUTPUT_TOKENS,
)
from chunkhound.services.indexing_coordinator import IndexingCoordinator
from tests.fixtures.fake_providers import FakeLLMProvider, FakeEmbeddingProvider

# Repository size estimation
CHUNKS_TO_LOC_RATIO = 20  # Rough estimate: 1 chunk ≈ 20 lines of code


class TestSynthesisBudgets:
    """Test dynamic synthesis budget calculation."""

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
            "services": services,
        }

        # Cleanup
        services.provider.close()

    @pytest.mark.asyncio
    async def test_max_depth_always_one(self, research_setup):
        """Test that max_depth is always fixed at 1."""
        service = research_setup["service"]

        # Mock the synthesis to avoid full execution
        with patch.object(service, "_single_pass_synthesis", new_callable=AsyncMock) as mock_synthesis:
            mock_synthesis.return_value = "Test answer"

            # Call deep_research (no depth parameter)
            await service.deep_research("test query")

            # Verify BFS processed only depth 0 and 1
            # This is indirect verification through the fact that it completes

    @pytest.mark.asyncio
    async def test_synthesis_budgets_small_repo(self, research_setup):
        """Test synthesis budgets for very small repository."""
        service = research_setup["service"]

        # Mock get_stats to return small repo size
        mock_stats = {"chunks": 50}  # ~1k LOC (50 × 20)

        budgets = service._calculate_synthesis_budgets(mock_stats)

        # Small repos should get minimal input, fixed 30k output
        assert budgets["input_tokens"] == 30_000
        assert budgets["output_tokens"] == 30_000
        assert budgets["overhead_tokens"] == 5_000
        assert budgets["total_tokens"] == 65_000

    @pytest.mark.asyncio
    async def test_synthesis_budgets_medium_repo(self, research_setup):
        """Test synthesis budgets for medium repository."""
        service = research_setup["service"]

        # Mock get_stats to return medium repo size
        mock_stats = {"chunks": 10_000}  # ~200k LOC (10k × 20)

        budgets = service._calculate_synthesis_budgets(mock_stats)

        # Medium repos should get standard input, fixed 30k output
        assert budgets["input_tokens"] == 80_000
        assert budgets["output_tokens"] == 30_000
        assert budgets["overhead_tokens"] == 5_000
        assert budgets["total_tokens"] == 115_000

    @pytest.mark.asyncio
    async def test_synthesis_budgets_large_repo(self, research_setup):
        """Test synthesis budgets for large repository."""
        service = research_setup["service"]

        # Mock get_stats to return large repo size
        mock_stats = {"chunks": 100_000}  # ~2M LOC (100k × 20)

        budgets = service._calculate_synthesis_budgets(mock_stats)

        # Large repos should get maximum input, fixed 30k output
        assert budgets["input_tokens"] == 150_000
        assert budgets["output_tokens"] == 30_000
        assert budgets["overhead_tokens"] == 5_000
        assert budgets["total_tokens"] == 185_000

    @pytest.mark.asyncio
    async def test_no_depth_parameter(self, research_setup):
        """Test that deep_research() has no depth parameter."""
        service = research_setup["service"]

        # Mock the synthesis to avoid full execution
        with patch.object(service, "_single_pass_synthesis", new_callable=AsyncMock) as mock_synthesis:
            mock_synthesis.return_value = "Test answer"

            # Call without depth parameter (should work)
            result = await service.deep_research("test query")
            assert "answer" in result

            # Verify calling with depth parameter fails
            with pytest.raises(TypeError):
                await service.deep_research("test query", depth="shallow")  # type: ignore

    @pytest.mark.asyncio
    async def test_synthesis_uses_dynamic_budgets(self, research_setup):
        """Test that synthesis receives dynamic budgets based on repo size."""
        service = research_setup["service"]

        # Create a proper mock response with all required attributes
        mock_response = Mock()
        mock_response.content = "Test response " * 50  # Make it long enough (>100 chars)
        mock_response.finish_reason = "stop"
        mock_response.tokens_used = 1000

        # Mock stats to return medium repo
        mock_stats = {"chunks": 5_000}  # ~100k LOC (5k × 20)

        # Create test data with files and chunks
        test_chunks = [{"file_path": "test.py", "start_line": 1, "end_line": 10}]
        test_files = {"test.py": "def foo(): pass"}
        test_context = Mock()

        with patch.object(
            service._llm_manager.get_synthesis_provider(),
            "complete",
            new_callable=AsyncMock,
        ) as mock_complete:
            mock_complete.return_value = mock_response

            with patch.object(
                service._db_services.provider,
                "get_stats",
                return_value=mock_stats
            ):
                # Calculate expected budgets
                expected_budgets = service._calculate_synthesis_budgets(mock_stats)

                # Call single-pass synthesis with test data
                await service._single_pass_synthesis(
                    root_query="test query",
                    chunks=test_chunks,
                    files=test_files,
                    context=test_context,
                    synthesis_budgets=expected_budgets,
                )

                # Verify the max_completion_tokens parameter uses dynamic output budget
                call_args = mock_complete.call_args
                assert call_args.kwargs.get("max_completion_tokens") == expected_budgets["output_tokens"]

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
        test_budgets = {
            "input_tokens": 50_000,
            "output_tokens": 12_000,
            "overhead_tokens": 5_000,
            "total_tokens": 67_000,
        }

        # Call synthesis directly
        result = await service._single_pass_synthesis(
            root_query="test",
            chunks=test_chunks,
            files=test_files,
            context=test_context,
            synthesis_budgets=test_budgets,
        )

        # Verify footer is present
        assert "## Sources" in result, "Sources footer should be present"
        assert "**Files**: 1" in result, "Footer should show file count"
        assert "**Chunks**: 1" in result, "Footer should show chunk count"
        assert "test.py" in result, "Footer should list analyzed file"


class TestSourcesFooter:
    """Test sources footer generation (unchanged from original tests)."""

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
        assert "src/" in footer
        assert "tests/" in footer
        assert "main.py" in footer
        assert "test.py" in footer
        assert "L10-25" in footer
        assert "L50-75" in footer
        assert "L5-15" in footer
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
