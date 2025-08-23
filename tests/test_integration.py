"""Integration tests for RocketRAG system."""

import os
import pytest

from rocketrag import RocketRAG


class TestRocketRAGIntegration:
    """Integration tests for complete RocketRAG workflow."""

    @pytest.fixture
    def integration_rocketrag(self, test_models_config, temp_dir, sample_pdf_path):
        """Create RocketRAG instance for integration testing."""
        # Create a test data directory with the sample PDF
        test_data_dir = os.path.join(temp_dir, "integration_data")
        os.makedirs(test_data_dir, exist_ok=True)

        # Copy sample PDF to test data directory
        import shutil

        test_pdf_path = os.path.join(test_data_dir, "sample-report.pdf")
        shutil.copy2(sample_pdf_path, test_pdf_path)

        db_path = os.path.join(temp_dir, "integration.db")

        # Create components with test models
        from rocketrag.vectors import SentenceTransformersVectorizer
        from rocketrag.chonk import ChonkieChunker
        from rocketrag.llm import LLamaLLM
        from rocketrag.loaders import KreuzbergLoader

        vectorizer = SentenceTransformersVectorizer(
            model_name=test_models_config["embedding_model"]
        )
        chunker = ChonkieChunker(
            method="semantic",
            embedding_model=test_models_config["embedding_model"],
            chunk_size=256,  # Smaller chunks for faster testing
        )
        llm = LLamaLLM(
            repo_id=test_models_config["llm_repo_id"],
            filename=test_models_config["llm_filename"],
            n_ctx=test_models_config["n_ctx"],
        )
        loader = KreuzbergLoader()

        rag = RocketRAG(
            data_dir=test_data_dir,
            db_path=db_path,
            collection_name="integration_test",
            vectorizer=vectorizer,
            chunker=chunker,
            llm=llm,
            loader=loader,
        )

        return rag

    def test_complete_rag_workflow(self, integration_rocketrag):
        """Test complete RAG workflow from document loading to question answering."""
        # Step 1: Prepare the database with documents
        integration_rocketrag.prepare()

        # Verify documents were loaded
        db = integration_rocketrag.db
        record_count = db.get_total_count()
        assert record_count > 0, "No documents were loaded into the database"

        # Verify we have the expected filename
        filenames = db.get_unique_filenames()
        assert "sample-report.pdf" in filenames

        # Step 2: Test search functionality
        search_results = db.search("report", top_k=5)
        assert len(search_results) > 0, "Search returned no results"

        # Verify search results have expected structure
        for result in search_results:
            assert hasattr(result, "chunk")
            assert hasattr(result, "filename")
            assert hasattr(result, "score")
            assert result.filename == "sample-report.pdf"
            assert isinstance(result.chunk, str)
            assert len(result.chunk) > 0

    def test_ask_question_with_context(self, integration_rocketrag):
        """Test asking a question that should find relevant context."""
        # Prepare the database
        integration_rocketrag.prepare()

        # Ask a question that should find relevant content
        question = "What is this document about?"
        response, sources = integration_rocketrag.ask(question)

        # Verify response
        assert isinstance(response, str)
        assert len(response) > 0

        # Verify sources
        assert isinstance(sources, list)
        assert len(sources) > 0

        # All sources should be from our sample PDF
        for source in sources:
            assert source.filename == "sample-report.pdf"
            assert isinstance(source.chunk, str)
            assert len(source.chunk) > 0
            assert isinstance(source.score, float)

    def test_stream_ask_question(self, integration_rocketrag):
        """Test streaming question answering."""
        # Prepare the database
        integration_rocketrag.prepare()

        # Ask a question using streaming
        question = "What are the main topics covered?"
        response, sources = integration_rocketrag.stream_ask(question)

        # Verify response (streaming should still return complete response)
        assert isinstance(response, str)
        assert len(response) > 0

        # Verify sources
        assert isinstance(sources, list)
        assert len(sources) > 0

    def test_question_with_no_relevant_context(self, integration_rocketrag):
        """Test asking a question that might not have relevant context."""
        # Prepare the database
        integration_rocketrag.prepare()

        # Ask a very specific question unlikely to be in the document
        question = "What is the molecular structure of caffeine?"
        response, sources = integration_rocketrag.ask(question)

        # Should still get a response (LLM should handle gracefully)
        assert isinstance(response, str)
        assert len(response) > 0

        # Sources might be empty or have low relevance scores
        assert isinstance(sources, list)
        # Don't assert on sources length as it depends on search threshold

    def test_multiple_questions_same_session(self, integration_rocketrag):
        """Test asking multiple questions in the same session."""
        # Prepare the database once
        integration_rocketrag.prepare()

        questions = [
            "What is this document about?",
            "What are the key findings?",
            "Who is the author?",
        ]

        responses = []
        all_sources = []

        for question in questions:
            response, sources = integration_rocketrag.ask(question)
            responses.append(response)
            all_sources.extend(sources)

            # Each response should be valid
            assert isinstance(response, str)
            assert len(response) > 0

        # Should have gotten responses for all questions
        assert len(responses) == len(questions)

        # Should have found some sources across all questions
        assert len(all_sources) > 0

    def test_database_persistence(self, integration_rocketrag, temp_dir):
        """Test that database persists across RocketRAG instances."""
        # Prepare the database
        integration_rocketrag.prepare()
        initial_count = integration_rocketrag.db.get_total_count()
        assert initial_count > 0

        # Create a new RocketRAG instance with the same database
        from rocketrag.vectors import SentenceTransformersVectorizer

        vectorizer = SentenceTransformersVectorizer(
            model_name=integration_rocketrag.vectorizer.config["model_name"]
        )

        new_rag = RocketRAG(
            data_dir=integration_rocketrag.data_dir,
            db_path=integration_rocketrag.db_path,
            collection_name=integration_rocketrag.collection_name,
            vectorizer=vectorizer,
        )

        # Should be able to search without preparing again
        search_results = new_rag.db.search("document", top_k=3)
        assert len(search_results) > 0

        # Record count should be the same
        new_count = new_rag.db.get_total_count()
        assert new_count == initial_count

    def test_recreate_database(self, integration_rocketrag):
        """Test recreating the database."""
        # Prepare the database
        integration_rocketrag.prepare()
        initial_count = integration_rocketrag.db.count_records()
        assert initial_count > 0

        # Recreate the database
        integration_rocketrag.prepare(recreate=True)
        new_count = integration_rocketrag.db.get_total_count()

        # Should have the same number of records (same documents)
        assert new_count == initial_count

        # Should still be able to search
        search_results = integration_rocketrag.db.search("report", top_k=3)
        assert len(search_results) > 0

    def test_empty_data_directory(self, test_models_config, temp_dir):
        """Test behavior with empty data directory."""
        # Create empty data directory
        empty_data_dir = os.path.join(temp_dir, "empty_data")
        os.makedirs(empty_data_dir, exist_ok=True)

        db_path = os.path.join(temp_dir, "empty.db")

        from rocketrag.vectors import SentenceTransformersVectorizer

        vectorizer = SentenceTransformersVectorizer(
            model_name=test_models_config["embedding_model"]
        )

        rag = RocketRAG(
            data_dir=empty_data_dir,
            db_path=db_path,
            collection_name="empty_test",
            vectorizer=vectorizer,
        )

        # Prepare should work but result in empty database
        rag.prepare()

        # Database should exist but be empty
        assert rag.db.get_total_count() == 0

        # Search should return empty results
        search_results = rag.db.search("anything", top_k=5)
        assert len(search_results) == 0

        # Asking questions should still work (LLM without context)
        response, sources = rag.ask("What is AI?")
        assert isinstance(response, str)
        assert len(response) > 0
        assert len(sources) == 0

    def test_database_metadata_consistency(self, integration_rocketrag):
        """Test that database metadata is consistent."""
        # Prepare the database
        integration_rocketrag.prepare()

        # Get metadata from database
        db_metadata = integration_rocketrag.db.get_collection_metadata()

        # Should contain expected metadata fields
        expected_fields = [
            "data_dir",
            "chonker",
            "vectorizer",
            "db_path",
            "collection_name",
        ]

        for field in expected_fields:
            assert field in db_metadata

        # Metadata should match RocketRAG configuration
        assert db_metadata["data_dir"] == integration_rocketrag.data_dir
        assert db_metadata["db_path"] == integration_rocketrag.db_path
        assert db_metadata["collection_name"] == integration_rocketrag.collection_name

    def test_chunking_and_vectorization_integration(self, integration_rocketrag):
        """Test that chunking and vectorization work together properly."""
        # Prepare the database
        integration_rocketrag.prepare()

        # Get some records to verify chunking worked
        search_results = integration_rocketrag.db.search("document", top_k=10)
        assert len(search_results) > 0

        # Verify chunks have reasonable sizes
        for result in search_results:
            chunk_length = len(result.chunk)
            # Chunks should be within reasonable bounds (allowing for very small chunks)
            assert (
                10 <= chunk_length <= 1000
            )  # Allowing more flexibility for small chunks

        # Verify vectors exist for chunks
        vectors = integration_rocketrag.db.get_vectors_by_filename("sample-report.pdf")
        assert len(vectors) > 0

        # Each vector should have the expected dimensionality
        vector_dim = len(vectors[0])
        assert vector_dim > 0

        # All vectors should have the same dimensionality
        for vector in vectors:
            assert len(vector) == vector_dim

    @pytest.mark.parametrize("top_k", [1, 3, 5, 10])
    def test_different_top_k_values(self, integration_rocketrag, top_k):
        """Test RAG with different top_k values."""
        # Prepare the database
        integration_rocketrag.prepare()

        # Ask question (top_k is hardcoded in RAG class)
        question = "What is this document about?"
        response, sources = integration_rocketrag.ask(question)

        # Response should always be valid
        assert isinstance(response, str)
        assert len(response) > 0

        # Sources should respect top_k limit (hardcoded to 5 in RAG class)
        assert len(sources) <= 5

        # If we have documents, we should get some sources
        total_records = integration_rocketrag.db.get_total_count()
        if total_records > 0:
            assert len(sources) > 0

    def test_llm_loading_efficiency(self, integration_rocketrag):
        """Test that LLM is loaded efficiently (only once)."""
        # Prepare the database
        integration_rocketrag.prepare()

        # LLM should not be loaded initially
        assert not integration_rocketrag.llm_loaded

        # First question should load LLM
        integration_rocketrag.ask("First question?")
        assert integration_rocketrag.llm_loaded

        # Subsequent questions should not reload LLM
        # (This is tested by checking the llm_loaded flag remains True)
        integration_rocketrag.ask("Second question?")
        assert integration_rocketrag.llm_loaded

        integration_rocketrag.stream_ask("Third question?")
        assert integration_rocketrag.llm_loaded
