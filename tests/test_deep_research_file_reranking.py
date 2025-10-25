"""Tests for file-level reranking in deep research synthesis budget allocation."""

import pytest
from unittest.mock import AsyncMock, Mock, patch
from dataclasses import dataclass

from chunkhound.services.deep_research_service import DeepResearchService


@dataclass
class MockRerankResult:
    """Mock rerank result with index and score."""
    index: int
    score: float


class TestFileReranking:
    """Test file-level reranking for synthesis budget allocation."""

    @pytest.fixture
    def mock_services(self):
        """Create mock services for deep research."""
        # Mock database services
        database_services = Mock()
        database_services.search_service = Mock()

        # Mock embedding manager with provider
        embedding_manager = Mock()
        embedding_provider = Mock()
        embedding_provider.name = "test-provider"
        embedding_provider.get_max_rerank_batch_size = Mock(return_value=100)
        embedding_manager.get_provider = Mock(return_value=embedding_provider)

        # Mock LLM manager with utility provider
        llm_manager = Mock()
        llm_utility = Mock()
        llm_utility.estimate_tokens = Mock(side_effect=lambda text: len(text) // 4)
        llm_manager.get_utility_provider = Mock(return_value=llm_utility)
        llm_manager.get_synthesis_provider = Mock()

        return database_services, embedding_manager, llm_manager, embedding_provider

    @pytest.fixture
    def research_service(self, mock_services):
        """Create DeepResearchService with mocked dependencies."""
        database_services, embedding_manager, llm_manager, _ = mock_services

        service = DeepResearchService(
            database_services=database_services,
            embedding_manager=embedding_manager,
            llm_manager=llm_manager,
        )

        # Mock emit event to avoid complexity
        service._emit_event = AsyncMock()

        return service

    @pytest.fixture
    def sample_chunks(self):
        """Create sample chunks for testing."""
        return [
            {
                "chunk_id": "1",
                "file_path": "file_a.py",
                "content": "def function_a(): pass",
                "start_line": 1,
                "end_line": 1,
                "score": 0.9,
            },
            {
                "chunk_id": "2",
                "file_path": "file_a.py",
                "content": "def function_b(): pass",
                "start_line": 3,
                "end_line": 3,
                "score": 0.8,
            },
            {
                "chunk_id": "3",
                "file_path": "file_b.py",
                "content": "class ClassB: pass",
                "start_line": 1,
                "end_line": 1,
                "score": 0.7,
            },
            {
                "chunk_id": "4",
                "file_path": "file_c.py",
                "content": "import os",
                "start_line": 1,
                "end_line": 1,
                "score": 0.6,
            },
        ]

    @pytest.fixture
    def sample_files(self):
        """Create sample file contents."""
        return {
            "file_a.py": "def function_a(): pass\n\ndef function_b(): pass\n",
            "file_b.py": "class ClassB:\n    pass\n",
            "file_c.py": "import os\nimport sys\n",
        }

    @pytest.mark.asyncio
    async def test_successful_file_reranking(
        self, research_service, mock_services, sample_chunks, sample_files
    ):
        """Test that successful reranking prioritizes files by rerank scores."""
        _, _, _, embedding_provider = mock_services

        # Mock rerank results (file_b ranked highest, file_c second, file_a third)
        rerank_results = [
            MockRerankResult(index=1, score=0.95),  # file_b.py
            MockRerankResult(index=2, score=0.85),  # file_c.py
            MockRerankResult(index=0, score=0.75),  # file_a.py
        ]
        embedding_provider.rerank = AsyncMock(return_value=rerank_results)

        # Call budget management
        prioritized_chunks, budgeted_files, budget_info = (
            await research_service._manage_token_budget_for_synthesis(
                chunks=sample_chunks,
                files=sample_files,
                root_query="test query",
                depth="shallow",
            )
        )

        # Verify rerank was called
        embedding_provider.rerank.assert_called_once()
        call_args = embedding_provider.rerank.call_args
        assert call_args[1]["query"] == "test query"
        assert call_args[1]["top_k"] is None
        assert len(call_args[1]["documents"]) == 3  # 3 files

        # Verify files are prioritized by rerank scores, not chunk scores
        # file_b has highest rerank score (0.95) despite file_a having higher chunk scores
        file_list = list(budgeted_files.keys())
        assert file_list[0] == "file_b.py", "file_b.py should be first (highest rerank score)"

    @pytest.mark.asyncio
    async def test_fallback_to_chunk_scores_on_rerank_failure(
        self, research_service, mock_services, sample_chunks, sample_files
    ):
        """Test fallback to accumulated chunk scores when reranking fails."""
        _, _, _, embedding_provider = mock_services

        # Mock rerank to raise exception
        embedding_provider.rerank = AsyncMock(side_effect=Exception("API error"))

        # Call budget management
        prioritized_chunks, budgeted_files, budget_info = (
            await research_service._manage_token_budget_for_synthesis(
                chunks=sample_chunks,
                files=sample_files,
                root_query="test query",
                depth="shallow",
            )
        )

        # Verify rerank was attempted
        embedding_provider.rerank.assert_called_once()

        # Verify fallback: files prioritized by accumulated chunk scores
        # file_a has 2 chunks with scores 0.9 + 0.8 = 1.7 (highest)
        # file_b has 1 chunk with score 0.7
        # file_c has 1 chunk with score 0.6
        file_list = list(budgeted_files.keys())
        assert file_list[0] == "file_a.py", "file_a.py should be first (highest accumulated score)"

    @pytest.mark.asyncio
    async def test_empty_file_list_handled_gracefully(
        self, research_service, mock_services
    ):
        """Test that empty file list is handled without errors."""
        _, _, _, embedding_provider = mock_services

        # Mock rerank to return empty list for empty input
        embedding_provider.rerank = AsyncMock(return_value=[])

        # Call with empty chunks and files
        prioritized_chunks, budgeted_files, budget_info = (
            await research_service._manage_token_budget_for_synthesis(
                chunks=[],
                files={},
                root_query="test query",
                depth="shallow",
            )
        )

        # Verify no crash and empty results
        assert len(prioritized_chunks) == 0
        assert len(budgeted_files) == 0
        assert budget_info["total_files"] == 0

    @pytest.mark.asyncio
    async def test_single_file_handled_correctly(
        self, research_service, mock_services
    ):
        """Test that single file case works correctly."""
        _, _, _, embedding_provider = mock_services

        single_chunk = [
            {
                "chunk_id": "1",
                "file_path": "single.py",
                "content": "def func(): pass",
                "start_line": 1,
                "end_line": 1,
                "score": 0.9,
            }
        ]
        single_file = {"single.py": "def func(): pass\n"}

        # Mock rerank for single file
        rerank_results = [MockRerankResult(index=0, score=0.95)]
        embedding_provider.rerank = AsyncMock(return_value=rerank_results)

        # Call budget management
        prioritized_chunks, budgeted_files, budget_info = (
            await research_service._manage_token_budget_for_synthesis(
                chunks=single_chunk,
                files=single_file,
                root_query="test query",
                depth="shallow",
            )
        )

        # Verify single file included
        assert len(budgeted_files) == 1
        assert "single.py" in budgeted_files
        assert budget_info["files_included_fully"] == 1

    @pytest.mark.asyncio
    async def test_representative_document_construction(
        self, research_service, mock_services, sample_chunks, sample_files
    ):
        """Test that file representative documents are constructed correctly."""
        _, _, _, embedding_provider = mock_services

        # Mock rerank to capture documents
        captured_documents = []

        async def capture_rerank(query, documents, top_k):
            captured_documents.extend(documents)
            return [MockRerankResult(index=i, score=0.9 - i * 0.1) for i in range(len(documents))]

        embedding_provider.rerank = AsyncMock(side_effect=capture_rerank)

        # Call budget management
        await research_service._manage_token_budget_for_synthesis(
            chunks=sample_chunks,
            files=sample_files,
            root_query="test query",
            depth="shallow",
        )

        # Verify representative documents were created
        assert len(captured_documents) == 3  # 3 files

        # Verify file_a representative includes file path and top chunks
        file_a_doc = captured_documents[0]
        assert "file_a.py" in file_a_doc
        assert "Lines 1-1:" in file_a_doc or "Lines 3-3:" in file_a_doc

    @pytest.mark.asyncio
    async def test_batch_size_logging(
        self, research_service, mock_services, sample_chunks, sample_files
    ):
        """Test that batch size limits are logged when exceeded."""
        _, _, _, embedding_provider = mock_services

        # Set small batch size
        embedding_provider.get_max_rerank_batch_size = Mock(return_value=2)

        # Mock rerank
        rerank_results = [
            MockRerankResult(index=i, score=0.9 - i * 0.1)
            for i in range(3)
        ]
        embedding_provider.rerank = AsyncMock(return_value=rerank_results)

        # Call budget management
        await research_service._manage_token_budget_for_synthesis(
            chunks=sample_chunks,
            files=sample_files,
            root_query="test query",
            depth="shallow",
        )

        # Verify batch size check was called
        embedding_provider.get_max_rerank_batch_size.assert_called_once()

        # Verify rerank was still called (provider handles splitting)
        embedding_provider.rerank.assert_called_once()

    @pytest.mark.asyncio
    async def test_file_diversity_improvement(
        self, research_service, mock_services
    ):
        """Test that reranking improves file diversity vs accumulated scores."""
        _, _, _, embedding_provider = mock_services

        # Create chunks where one file has many low-scoring chunks
        # and another has few high-scoring chunks
        chunks = [
            {"chunk_id": f"a{i}", "file_path": "common.py", "content": f"line{i}",
             "start_line": i, "end_line": i, "score": 0.6}
            for i in range(10)  # 10 chunks, score sum = 6.0
        ] + [
            {"chunk_id": "b1", "file_path": "rare.py", "content": "important",
             "start_line": 1, "end_line": 1, "score": 0.95},  # 1 chunk, score sum = 0.95
        ]

        files = {
            "common.py": "\n".join([f"line{i}" for i in range(10)]),
            "rare.py": "important code",
        }

        # Mock rerank to prioritize rare.py (despite lower accumulated score)
        # Note: file_paths list is built in order of first appearance in sorted chunks
        # rare.py has score 0.95, common.py chunks have score 0.6
        # So rare.py appears first (index 0), common.py second (index 1)
        rerank_results = [
            MockRerankResult(index=0, score=0.98),  # rare.py (higher rerank score)
            MockRerankResult(index=1, score=0.75),  # common.py (lower rerank score)
        ]
        embedding_provider.rerank = AsyncMock(return_value=rerank_results)

        # Call budget management
        prioritized_chunks, budgeted_files, budget_info = (
            await research_service._manage_token_budget_for_synthesis(
                chunks=chunks,
                files=files,
                root_query="find important algorithm",
                depth="shallow",
            )
        )

        # Verify rare.py is prioritized despite common.py having higher accumulated scores
        # (common.py: 6.0 vs rare.py: 0.95)
        file_list = list(budgeted_files.keys())
        assert file_list[0] == "rare.py", (
            "Reranking should prioritize rare.py (high relevance) over "
            "common.py (high accumulated score but lower relevance)"
        )


class TestFileRerankingExceptionTypes:
    """Test exception handling in file reranking."""

    @pytest.mark.asyncio
    async def test_exception_handling_and_fallback(self):
        """Test that exceptions are handled and fallback works correctly."""
        from unittest.mock import AsyncMock, Mock

        # Create minimal service
        database_services = Mock()
        database_services.search_service = Mock()

        embedding_manager = Mock()
        embedding_provider = Mock()
        embedding_provider.name = "test"
        embedding_provider.get_max_rerank_batch_size = Mock(return_value=100)
        # Rerank raises ValueError
        embedding_provider.rerank = AsyncMock(side_effect=ValueError("Test error"))
        embedding_manager.get_provider = Mock(return_value=embedding_provider)

        llm_manager = Mock()
        llm_utility = Mock()
        llm_utility.estimate_tokens = Mock(side_effect=lambda text: len(text) // 4)
        llm_manager.get_utility_provider = Mock(return_value=llm_utility)

        service = DeepResearchService(
            database_services=database_services,
            embedding_manager=embedding_manager,
            llm_manager=llm_manager,
        )
        service._emit_event = AsyncMock()

        # Call with sample data
        chunks = [{"chunk_id": "1", "file_path": "a.py", "content": "x",
                   "start_line": 1, "end_line": 1, "score": 0.9}]
        files = {"a.py": "x"}

        # Should not raise - fallback should work
        prioritized_chunks, budgeted_files, budget_info = (
            await service._manage_token_budget_for_synthesis(
                chunks=chunks, files=files, root_query="test", depth="shallow"
            )
        )

        # Verify fallback succeeded - file should still be included
        assert len(budgeted_files) == 1
        assert "a.py" in budgeted_files
        # Verify rerank was attempted (which raised the exception)
        embedding_provider.rerank.assert_called_once()
