"""Tests for citation sequence sorting in deep research service."""

import pytest
from unittest.mock import Mock

from chunkhound.services.deep_research_service import DeepResearchService


class TestCitationSorting:
    """Test citation sequence sorting functionality."""

    @pytest.fixture
    def research_service(self):
        """Create DeepResearchService with minimal mocked dependencies."""
        # Mock database services
        database_services = Mock()
        database_services.search_service = Mock()

        # Mock embedding manager
        embedding_manager = Mock()
        embedding_provider = Mock()
        embedding_provider.name = "test-provider"
        embedding_manager.get_provider = Mock(return_value=embedding_provider)

        # Mock LLM manager
        llm_manager = Mock()
        llm_utility = Mock()
        llm_utility.estimate_tokens = Mock(side_effect=lambda text: len(text) // 4)
        llm_manager.get_utility_provider = Mock(return_value=llm_utility)

        service = DeepResearchService(
            database_services=database_services,
            embedding_manager=embedding_manager,
            llm_manager=llm_manager,
            tool_name="test",
        )

        return service

    def test_basic_sorting(self, research_service):
        """Test basic citation sequence sorting."""
        input_text = "Algorithm [11][2][1] uses BFS"
        expected = "Algorithm [1][2][11] uses BFS"
        result = research_service._sort_citation_sequences(input_text)
        assert result == expected

    def test_already_sorted_idempotent(self, research_service):
        """Test that already-sorted sequences remain unchanged (idempotent)."""
        input_text = "Function [1][2][3] implements logic"
        result = research_service._sort_citation_sequences(input_text)
        assert result == input_text

    def test_single_citation_unchanged(self, research_service):
        """Test that single citations remain unchanged."""
        input_text = "Constant [5] defines timeout"
        result = research_service._sort_citation_sequences(input_text)
        assert result == input_text

    def test_separated_citations_unchanged(self, research_service):
        """Test that citations separated by text remain in original positions."""
        input_text = "Timeout [5] is used in module [3] for rate limiting"
        result = research_service._sort_citation_sequences(input_text)
        # Each citation stays where it is, no reordering across text
        assert result == input_text

    def test_multiple_sequences(self, research_service):
        """Test multiple separate citation sequences are each sorted independently."""
        input_text = "Function [10][5] calls helper [3][1][2] for processing"
        expected = "Function [5][10] calls helper [1][2][3] for processing"
        result = research_service._sort_citation_sequences(input_text)
        assert result == expected

    def test_no_citations(self, research_service):
        """Test that text without citations passes through unchanged."""
        input_text = "This is plain text without any citations."
        result = research_service._sort_citation_sequences(input_text)
        assert result == input_text

    def test_numerical_not_lexical_sorting(self, research_service):
        """Test that sorting is numerical, not lexical."""
        input_text = "Data [2][11][1][20] shows pattern"
        expected = "Data [1][2][11][20] shows pattern"
        result = research_service._sort_citation_sequences(input_text)
        assert result == expected
        # Verify lexical sorting would give wrong order
        assert result != "Data [1][11][2][20] shows pattern"

    def test_large_numbers(self, research_service):
        """Test sorting with large reference numbers."""
        input_text = "References [100][20][3][50] are relevant"
        expected = "References [3][20][50][100] are relevant"
        result = research_service._sort_citation_sequences(input_text)
        assert result == expected

    def test_multiline_text(self, research_service):
        """Test citation sorting across multiple lines."""
        input_text = """First line [5][3][1] explains the concept.
Second line [10][2] provides details.
Third line [7][4][6] concludes."""
        expected = """First line [1][3][5] explains the concept.
Second line [2][10] provides details.
Third line [4][6][7] concludes."""
        result = research_service._sort_citation_sequences(input_text)
        assert result == expected

    def test_citations_at_line_boundaries(self, research_service):
        """Test citations at start and end of lines."""
        input_text = "[3][1][2] Start of line\nEnd of line [6][4][5]"
        expected = "[1][2][3] Start of line\nEnd of line [4][5][6]"
        result = research_service._sort_citation_sequences(input_text)
        assert result == expected

    def test_many_citations_in_sequence(self, research_service):
        """Test sorting a long sequence of citations."""
        input_text = "Complex [15][3][8][1][12][5][20][2] analysis"
        expected = "Complex [1][2][3][5][8][12][15][20] analysis"
        result = research_service._sort_citation_sequences(input_text)
        assert result == expected

    def test_duplicate_numbers_in_sequence(self, research_service):
        """Test handling of duplicate citation numbers (edge case)."""
        input_text = "Function [3][1][3][2] has duplicate refs"
        # Should preserve duplicates and sort them
        expected = "Function [1][2][3][3] has duplicate refs"
        result = research_service._sort_citation_sequences(input_text)
        assert result == expected

    def test_empty_string(self, research_service):
        """Test that empty string is handled gracefully."""
        input_text = ""
        result = research_service._sort_citation_sequences(input_text)
        assert result == ""

    def test_markdown_formatted_text(self, research_service):
        """Test sorting in markdown-formatted research output."""
        input_text = """## Algorithm Analysis [10][5][1]

The timeout mechanism [3][2] uses exponential backoff [7][4][6].

**Constants**: See references [12][8][15]."""
        expected = """## Algorithm Analysis [1][5][10]

The timeout mechanism [2][3] uses exponential backoff [4][6][7].

**Constants**: See references [8][12][15]."""
        result = research_service._sort_citation_sequences(input_text)
        assert result == expected

    def test_citations_in_code_blocks(self, research_service):
        """Test that citations in text (including near code-like syntax) are sorted."""
        input_text = "Variable `timeout` [5][3] is set to 5.0 [1]"
        expected = "Variable `timeout` [3][5] is set to 5.0 [1]"
        result = research_service._sort_citation_sequences(input_text)
        assert result == expected
