"""Tests for RocketRAG main class."""

import os
import tempfile
import pytest
from unittest.mock import patch

from rocketrag import RocketRAG
from rocketrag.data_models import SearchResult, Document
from rocketrag.vectors import SentenceTransformersVectorizer
from rocketrag.chonk import ChonkieChunker
from rocketrag.llm import LLamaLLM
from rocketrag.loaders import KreuzbergLoader


class TestRocketRAG:
    """Test cases for RocketRAG main class."""

    def test_initialization_with_defaults(self):
        """Test RocketRAG initialization with default parameters."""
        with tempfile.TemporaryDirectory() as temp_dir:
            db_path = os.path.join(temp_dir, "test.db")

            rag = RocketRAG(
                data_dir="test_data", db_path=db_path, collection_name="test_collection"
            )

            assert rag.data_dir == "test_data"
            assert rag.db_path == db_path
            assert rag.collection_name == "test_collection"
            assert not rag.llm_loaded
            assert rag.vectorizer is not None
            assert rag.chonker is not None
            assert rag.llm is not None
            assert rag.loader is not None
            assert rag.db is not None
            assert rag.rag is not None

    def test_initialization_with_custom_components(self, test_models_config, temp_dir):
        """Test RocketRAG initialization with custom components."""
        db_path = os.path.join(temp_dir, "custom.db")

        vectorizer = SentenceTransformersVectorizer(
            model_name=test_models_config["embedding_model"]
        )
        chunker = ChonkieChunker(
            method="semantic",
            embedding_model=test_models_config["embedding_model"],
            chunk_size=256,
        )
        llm = LLamaLLM(
            repo_id=test_models_config["llm_repo_id"],
            filename=test_models_config["llm_filename"],
            n_ctx=test_models_config["n_ctx"],
        )
        loader = KreuzbergLoader()

        rag = RocketRAG(
            data_dir="custom_data",
            db_path=db_path,
            collection_name="custom_collection",
            vectorizer=vectorizer,
            chunker=chunker,
            llm=llm,
            loader=loader,
        )

        assert rag.vectorizer == vectorizer
        assert rag.chonker == chunker
        assert rag.llm == llm
        assert rag.loader == loader

    def test_initialization_with_none_components(self, temp_dir):
        """Test RocketRAG initialization with None components."""
        db_path = os.path.join(temp_dir, "none_test.db")

        # RocketRAG constructor has default values, so passing None will use defaults
        rag = RocketRAG(
            data_dir="test_data",
            db_path=db_path,
        )

        # Should create default components
        assert rag.vectorizer is not None
        assert rag.chonker is not None
        assert rag.llm is not None
        assert rag.loader is not None

    def test_config_attributes(self, test_rocketrag):
        """Test that config attributes are properly set."""
        assert hasattr(test_rocketrag, "vectorizer_config")
        assert hasattr(test_rocketrag, "chonker_config")
        assert hasattr(test_rocketrag, "llm_config")
        assert hasattr(test_rocketrag, "loader_config")
        assert hasattr(test_rocketrag, "metadata")

        # Configs should be dictionaries
        assert isinstance(test_rocketrag.vectorizer_config, dict)
        assert isinstance(test_rocketrag.chonker_config, dict)
        assert isinstance(test_rocketrag.llm_config, dict)
        assert isinstance(test_rocketrag.loader_config, dict)
        assert isinstance(test_rocketrag.metadata, dict)

    def test_prepare_method_success(self, test_rocketrag, test_data_dir):
        """Test successful prepare method execution."""
        # Mock the loader to return sample documents
        mock_documents = [
            Document(content="Test content 1", filename="test1.pdf"),
            Document(content="Test content 2", filename="test2.pdf"),
        ]

        with patch.object(
            test_rocketrag.loader, "load_files_from_dir", return_value=mock_documents
        ):
            with patch.object(test_rocketrag.db, "create_collection_if_not_exists"):
                with patch.object(test_rocketrag.db, "add_documents"):
                    test_rocketrag.prepare()

                    # Verify methods were called
                    test_rocketrag.loader.load_files_from_dir.assert_called_once_with(
                        test_rocketrag.data_dir
                    )
                    test_rocketrag.db.create_collection_if_not_exists.assert_called_once_with(
                        False
                    )
                    test_rocketrag.db.add_documents.assert_called_once_with(
                        mock_documents
                    )

    def test_prepare_method_with_recreate(self, test_rocketrag):
        """Test prepare method with recreate=True."""
        mock_documents = [Document(content="Test content", filename="test.pdf")]

        with patch.object(
            test_rocketrag.loader, "load_files_from_dir", return_value=mock_documents
        ):
            with patch.object(test_rocketrag.db, "create_collection_if_not_exists"):
                with patch.object(test_rocketrag.db, "add_documents"):
                    test_rocketrag.prepare(recreate=True)

                    test_rocketrag.db.create_collection_if_not_exists.assert_called_once_with(
                        True
                    )

    def test_prepare_method_no_loader(self, test_rocketrag):
        """Test prepare method when loader is None."""
        test_rocketrag.loader = None

        with pytest.raises(ValueError, match="Loader is not defined"):
            test_rocketrag.prepare()

    def test_prepare_method_no_data_dir(self, test_rocketrag):
        """Test prepare method when data_dir is None."""
        test_rocketrag.data_dir = None

        with pytest.raises(ValueError, match="Data directory is not defined"):
            test_rocketrag.prepare()

    def test_ensure_llm_loaded_decorator(self, test_rocketrag):
        """Test that LLM is loaded when calling LLM methods."""
        # Initially LLM should not be loaded
        assert not test_rocketrag.llm_loaded

        with patch.object(test_rocketrag.llm, "load") as mock_load:
            with patch.object(test_rocketrag.llm, "run", return_value="test response"):
                # Call a method that requires LLM
                test_rocketrag.run_llm([{"role": "user", "content": "test"}])

                # LLM should be loaded
                mock_load.assert_called_once()
                assert test_rocketrag.llm_loaded

    def test_ensure_llm_loaded_only_once(self, test_rocketrag):
        """Test that LLM is only loaded once."""
        with patch.object(test_rocketrag.llm, "load") as mock_load:
            with patch.object(test_rocketrag.llm, "run", return_value="test response"):
                # Call LLM method twice
                test_rocketrag.run_llm([{"role": "user", "content": "test1"}])
                test_rocketrag.run_llm([{"role": "user", "content": "test2"}])

                # LLM should only be loaded once
                mock_load.assert_called_once()

    def test_run_llm_method(self, test_rocketrag):
        """Test run_llm method."""
        messages = [{"role": "user", "content": "What is AI?"}]
        expected_response = "AI is artificial intelligence."

        with patch.object(test_rocketrag.llm, "load"):
            with patch.object(
                test_rocketrag.llm, "run", return_value=expected_response
            ) as mock_run:
                response = test_rocketrag.run_llm(messages)

                mock_run.assert_called_once_with(messages)
                assert response == expected_response

    def test_stream_llm_method(self, test_rocketrag):
        """Test stream_llm method."""
        messages = [{"role": "user", "content": "What is ML?"}]
        expected_response = "ML is machine learning."

        with patch.object(test_rocketrag.llm, "load"):
            with patch.object(
                test_rocketrag.llm, "stream", return_value=expected_response
            ) as mock_stream:
                response = test_rocketrag.stream_llm(messages)

                mock_stream.assert_called_once_with(messages)
                assert response == expected_response

    def test_ask_method(self, test_rocketrag):
        """Test ask method."""
        question = "What is artificial intelligence?"
        expected_response = "AI is a field of computer science."
        expected_sources = [
            SearchResult(chunk="AI definition", filename="ai.pdf", score=0.1)
        ]

        with patch.object(test_rocketrag.llm, "load"):
            with patch.object(
                test_rocketrag.rag,
                "run",
                return_value=(expected_response, expected_sources),
            ) as mock_run:
                response, sources = test_rocketrag.ask(question)

                mock_run.assert_called_once_with(question)
                assert response == expected_response
                assert sources == expected_sources

    def test_stream_ask_method(self, test_rocketrag):
        """Test stream_ask method."""
        question = "What is machine learning?"
        expected_response = "ML is a subset of AI."
        expected_sources = [
            SearchResult(chunk="ML definition", filename="ml.pdf", score=0.2)
        ]

        with patch.object(test_rocketrag.llm, "load"):
            with patch.object(
                test_rocketrag.rag,
                "stream",
                return_value=(expected_response, expected_sources),
            ) as mock_stream:
                response, sources = test_rocketrag.stream_ask(question)

                mock_stream.assert_called_once_with(question)
                assert response == expected_response
                assert sources == expected_sources

    def test_ask_methods_load_llm(self, test_rocketrag):
        """Test that ask methods properly load LLM."""
        question = "Test question"

        with patch.object(test_rocketrag.llm, "load") as mock_load:
            with patch.object(test_rocketrag.rag, "run", return_value=("response", [])):
                test_rocketrag.ask(question)
                mock_load.assert_called_once()

            # Reset for second test
            mock_load.reset_mock()
            test_rocketrag.llm_loaded = False

            with patch.object(
                test_rocketrag.rag, "stream", return_value=("response", [])
            ):
                test_rocketrag.stream_ask(question)
                mock_load.assert_called_once()

    def test_metadata_construction(self, test_rocketrag):
        """Test that metadata is properly constructed."""
        metadata = test_rocketrag.metadata

        # Check that metadata contains expected keys
        expected_keys = [
            "data_dir",
            "chonker",
            "chonker_args",
            "vectorizer",
            "vectorizer_args",
            "loader",
            "loader_args",
            "db_path",
            "collection_name",
        ]

        for key in expected_keys:
            assert key in metadata

        # Check specific values
        assert metadata["data_dir"] == test_rocketrag.data_dir
        assert metadata["db_path"] == test_rocketrag.db_path
        assert metadata["collection_name"] == test_rocketrag.collection_name

    def test_db_initialization(self, test_rocketrag):
        """Test that database is properly initialized."""
        db = test_rocketrag.db

        assert db.db_path == test_rocketrag.db_path
        assert db.collection_name == test_rocketrag.collection_name
        assert db.vectorizer == test_rocketrag.vectorizer
        assert db.chunker == test_rocketrag.chonker
        assert db.metadata == test_rocketrag.metadata

    def test_rag_initialization(self, test_rocketrag):
        """Test that RAG is properly initialized."""
        rag = test_rocketrag.rag

        assert rag.db == test_rocketrag.db
        assert rag.llm == test_rocketrag.llm

    def test_component_configs_are_accessible(self, test_rocketrag):
        """Test that component configurations are accessible."""
        # All components should have config attributes
        assert hasattr(test_rocketrag.vectorizer, "config")
        assert hasattr(test_rocketrag.chonker, "config")
        assert hasattr(test_rocketrag.llm, "config")
        assert hasattr(test_rocketrag.loader, "config")

        # Configs should be stored in RocketRAG instance
        assert test_rocketrag.vectorizer_config == test_rocketrag.vectorizer.config
        assert test_rocketrag.chonker_config == test_rocketrag.chonker.config
        assert test_rocketrag.llm_config == test_rocketrag.llm.config
        assert test_rocketrag.loader_config == test_rocketrag.loader.config

    @pytest.mark.parametrize("method_name", ["ask", "stream_ask"])
    def test_ask_methods_with_empty_question(self, test_rocketrag, method_name):
        """Test ask methods with empty question."""
        with patch.object(test_rocketrag.llm, "load"):
            with patch.object(
                test_rocketrag.rag,
                "run" if method_name == "ask" else "stream",
                return_value=("response", []),
            ) as mock_method:
                method = getattr(test_rocketrag, method_name)
                response, sources = method("")

                mock_method.assert_called_once_with("")
                assert response == "response"
                assert sources == []

    def test_llm_loaded_state_persistence(self, test_rocketrag):
        """Test that LLM loaded state persists across method calls."""
        with patch.object(test_rocketrag.llm, "load") as mock_load:
            with patch.object(test_rocketrag.llm, "run", return_value="response"):
                with patch.object(
                    test_rocketrag.rag, "run", return_value=("response", [])
                ):
                    # First call should load LLM
                    test_rocketrag.ask("question1")
                    assert test_rocketrag.llm_loaded

                    # Second call should not load LLM again
                    test_rocketrag.ask("question2")
                    mock_load.assert_called_once()  # Only called once
