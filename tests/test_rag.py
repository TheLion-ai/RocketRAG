"""Tests for RAG class."""

import pytest
from unittest.mock import Mock, patch

from rocketrag.rag import RAG
from rocketrag.data_models import SearchResult
from rocketrag.db import MilvusLiteDB
from rocketrag.llm import LLamaLLM


class TestRAG:
    """Test cases for RAG class."""

    @pytest.fixture
    def mock_db(self):
        """Create a mock database."""
        db = Mock(spec=MilvusLiteDB)
        db.search.return_value = [
            SearchResult(
                chunk="This is about artificial intelligence.",
                filename="doc1.txt",
                score=0.1,
            ),
            SearchResult(
                chunk="Machine learning is a subset of AI.",
                filename="doc2.txt",
                score=0.2,
            ),
            SearchResult(
                chunk="Neural networks are used in deep learning.",
                filename="doc3.txt",
                score=0.3,
            ),
        ]
        return db

    @pytest.fixture
    def mock_llm(self):
        """Create a mock LLM."""
        llm = Mock(spec=LLamaLLM)
        llm.n_ctx = 2048
        llm.context_message_template = (
            "Question: {question}\nContext: {context}\nAnswer:"
        )
        llm.system_message = "You are a helpful assistant."

        # Mock the tokenize method
        llm.llm = Mock()
        llm.llm.tokenize.side_effect = lambda text: list(
            range(len(text.decode("utf-8")) // 4)
        )  # Rough approximation

        llm.run.return_value = "This is a test response."
        llm.stream.return_value = "This is a streamed response."

        return llm

    @pytest.fixture
    def rag_instance(self, mock_db, mock_llm):
        """Create a RAG instance with mocked dependencies."""
        return RAG(db=mock_db, llm=mock_llm)

    def test_initialization(self, mock_db, mock_llm):
        """Test RAG initialization."""
        rag = RAG(db=mock_db, llm=mock_llm)

        assert rag.db == mock_db
        assert rag.llm == mock_llm
        assert rag.context_window == mock_llm.n_ctx
        assert rag.response_buffer == 100

    def test_construct_context_message_basic(self, rag_instance):
        """Test basic context message construction."""
        question = "What is AI?"
        docs = ["AI is artificial intelligence.", "Machine learning is part of AI."]

        with patch("builtins.print"):  # Suppress debug print
            context_message = rag_instance.construct_context_message(question, docs)

        assert question in context_message
        assert "DOCUMENT0" in context_message
        assert "DOCUMENT1" in context_message
        assert "AI is artificial intelligence." in context_message
        assert "Machine learning is part of AI." in context_message

    def test_construct_context_message_empty_docs(self, rag_instance):
        """Test context message construction with empty documents."""
        question = "What is AI?"
        docs = []

        with patch("builtins.print"):  # Suppress debug print
            context_message = rag_instance.construct_context_message(question, docs)

        assert question in context_message
        # Should still contain the template structure
        assert "Question:" in context_message
        assert "Context:" in context_message

    def test_construct_context_message_token_limit(self, rag_instance):
        """Test context message construction with token limits."""
        question = "What is AI?"
        # Create very long documents that would exceed token limit
        long_doc = "This is a very long document. " * 1000
        docs = [long_doc, long_doc, long_doc]

        with patch("builtins.print"):  # Suppress debug print
            context_message = rag_instance.construct_context_message(question, docs)

        # Should still contain the question
        assert question in context_message
        # Should have some content but be truncated
        assert len(context_message) < len(question + long_doc + long_doc + long_doc)

    def test_construct_context_message_single_doc_truncation(self, rag_instance):
        """Test context message construction with single document truncation."""
        question = "What is AI?"
        # Create a single very long document
        long_doc = "This is a very long document that needs truncation. " * 1000
        docs = [long_doc]

        with patch("builtins.print"):  # Suppress debug print
            context_message = rag_instance.construct_context_message(question, docs)

        assert question in context_message
        assert "DOCUMENT0" in context_message
        # Should be truncated but still contain some content
        assert len(context_message) < len(question + long_doc)

    def test_run_method(self, rag_instance, mock_db, mock_llm):
        """Test the run method."""
        question = "What is artificial intelligence?"

        with patch("builtins.print"):  # Suppress debug print
            response, sources = rag_instance.run(question)

        # Verify database search was called
        mock_db.search.assert_called_once_with(question, top_k=5)

        # Verify LLM run was called with proper messages
        mock_llm.run.assert_called_once()
        call_args = mock_llm.run.call_args[0][0]
        assert len(call_args) == 2
        assert call_args[0]["role"] == "system"
        assert call_args[1]["role"] == "user"
        assert question in call_args[1]["content"]

        # Verify return values
        assert response == "This is a test response."
        assert len(sources) == 3
        assert all(isinstance(source, SearchResult) for source in sources)

    def test_stream_method(self, rag_instance, mock_db, mock_llm):
        """Test the stream method."""
        question = "What is machine learning?"

        with patch("builtins.print"):  # Suppress debug print
            response, sources = rag_instance.stream(question)

        # Verify database search was called
        mock_db.search.assert_called_once_with(question, top_k=5)

        # Verify LLM stream was called with proper messages
        mock_llm.stream.assert_called_once()
        call_args = mock_llm.stream.call_args[0][0]
        assert len(call_args) == 2
        assert call_args[0]["role"] == "system"
        assert call_args[1]["role"] == "user"
        assert question in call_args[1]["content"]

        # Verify return values
        assert response == "This is a streamed response."
        assert len(sources) == 3
        assert all(isinstance(source, SearchResult) for source in sources)

    def test_run_with_no_search_results(self, rag_instance, mock_db, mock_llm):
        """Test run method when no search results are found."""
        question = "What is quantum computing?"
        mock_db.search.return_value = []  # No search results

        with patch("builtins.print"):  # Suppress debug print
            response, sources = rag_instance.run(question)

        # Should still call LLM even with no sources
        mock_llm.run.assert_called_once()
        assert response == "This is a test response."
        assert sources == []

    def test_context_message_includes_all_search_results(self, rag_instance, mock_db):
        """Test that context message includes content from all search results."""
        question = "Tell me about AI"

        with patch("builtins.print"):  # Suppress debug print
            response, sources = rag_instance.run(question)

        # Get the context message that was constructed
        call_args = rag_instance.llm.run.call_args[0][0]
        context_content = call_args[1]["content"]

        # Verify all search result chunks are included
        for source in sources:
            assert source.chunk in context_content

    def test_search_top_k_parameter(self, rag_instance, mock_db):
        """Test that search is called with correct top_k parameter."""
        question = "What is deep learning?"

        with patch("builtins.print"):  # Suppress debug print
            rag_instance.run(question)

        # Verify search was called with top_k=5
        mock_db.search.assert_called_once_with(question, top_k=5)

    def test_system_message_included(self, rag_instance, mock_llm):
        """Test that system message is included in LLM calls."""
        question = "What is neural networks?"

        with patch("builtins.print"):  # Suppress debug print
            rag_instance.run(question)

        call_args = mock_llm.run.call_args[0][0]
        assert call_args[0]["role"] == "system"
        assert call_args[0]["content"] == mock_llm.system_message

    def test_debug_print_output(self, rag_instance, capsys):
        """Test that debug information is printed."""
        question = "What is AI?"

        rag_instance.run(question)

        captured = capsys.readouterr()
        assert "Debug: Using" in captured.out
        assert "documents" in captured.out
        assert "tokens" in captured.out

    @pytest.mark.parametrize("method_name", ["run", "stream"])
    def test_both_methods_use_same_search_logic(
        self, rag_instance, mock_db, method_name
    ):
        """Test that both run and stream methods use the same search logic."""
        question = "What is machine learning?"

        with patch("builtins.print"):  # Suppress debug print
            method = getattr(rag_instance, method_name)
            response, sources = method(question)

        # Both methods should call search with same parameters
        mock_db.search.assert_called_once_with(question, top_k=5)
        assert len(sources) == 3

    def test_context_window_calculation(self, mock_db, mock_llm):
        """Test that context window is properly calculated from LLM."""
        mock_llm.n_ctx = 4096  # Different context size
        rag = RAG(db=mock_db, llm=mock_llm)

        assert rag.context_window == 4096
        assert rag.response_buffer == 100

    def test_response_buffer_usage(self, rag_instance):
        """Test that response buffer is considered in token calculations."""
        # This is tested indirectly through the context construction
        # The response buffer should reduce available tokens for context
        available_tokens = rag_instance.context_window - rag_instance.response_buffer
        expected_available = 2048 - 100  # n_ctx - response_buffer

        assert available_tokens == expected_available
