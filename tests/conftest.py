"""Test configuration and fixtures for RocketRAG tests."""

import os
import tempfile
import shutil
from pathlib import Path
import pytest

from rocketrag import RocketRAG
from rocketrag.vectors import SentenceTransformersVectorizer
from rocketrag.chonk import ChonkieChunker
from rocketrag.llm import LLamaLLM
from rocketrag.loaders import KreuzbergLoader
from rocketrag.db import MilvusLiteDB
from rocketrag.data_models import Document


@pytest.fixture(scope="session")
def test_models_config():
    """Configuration for small test models."""
    return {
        "llm_repo_id": "unsloth/gemma-3-270m-it-GGUF",
        "llm_filename": "gemma-3-270m-it-Q4_0.gguf",
        "embedding_model": "minishlab/potion-base-8M",
        "n_ctx": 2048,  # Smaller context for faster testing
    }


@pytest.fixture
def temp_dir():
    """Create a temporary directory for test files."""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    shutil.rmtree(temp_dir)


@pytest.fixture
def sample_pdf_path():
    """Path to the sample PDF file."""
    return Path(__file__).parent / "sample-report.pdf"


@pytest.fixture
def test_data_dir(temp_dir, sample_pdf_path):
    """Create a test data directory with sample PDF."""
    data_dir = Path(temp_dir) / "test_data"
    data_dir.mkdir()

    # Copy sample PDF to test data directory
    shutil.copy(sample_pdf_path, data_dir / "sample-report.pdf")

    return str(data_dir)


@pytest.fixture
def test_vectorizer(test_models_config):
    """Create a test vectorizer with small model."""
    return SentenceTransformersVectorizer(
        model_name=test_models_config["embedding_model"]
    )


@pytest.fixture
def test_chunker(test_models_config):
    """Create a test chunker with small model."""
    return ChonkieChunker(
        method="semantic",
        embedding_model=test_models_config["embedding_model"],
        chunk_size=256,  # Smaller chunks for faster testing
    )


@pytest.fixture
def test_llm(test_models_config):
    """Create a test LLM with small model."""
    return LLamaLLM(
        repo_id=test_models_config["llm_repo_id"],
        filename=test_models_config["llm_filename"],
        n_ctx=test_models_config["n_ctx"],
        verbose=False,
    )


@pytest.fixture
def test_loader():
    """Create a test loader."""
    return KreuzbergLoader()


@pytest.fixture
def test_db(temp_dir, test_vectorizer, test_chunker):
    """Create a test database instance."""
    db_path = os.path.join(temp_dir, "test.db")
    collection_name = "test_collection"

    metadata = {
        "data_dir": "test_data",
        "vectorizer": "SentenceTransformersVectorizer",
        "vectorizer_args": {"model_name": "minishlab/potion-base-8M"},
        "chunker": "ChonkieChunker",
        "chunker_args": {"method": "semantic", "chunk_size": 256},
    }

    return MilvusLiteDB(
        db_path=db_path,
        collection_name=collection_name,
        vectorizer=test_vectorizer,
        chunker=test_chunker,
        metadata=metadata,
    )


@pytest.fixture
def test_rocketrag(test_data_dir, temp_dir, test_models_config):
    """Create a test RocketRAG instance."""
    db_path = os.path.join(temp_dir, "test_rocket.db")

    return RocketRAG(
        data_dir=test_data_dir,
        db_path=db_path,
        collection_name="test_rocket",
        vectorizer=SentenceTransformersVectorizer(
            model_name=test_models_config["embedding_model"]
        ),
        chunker=ChonkieChunker(
            method="semantic",
            embedding_model=test_models_config["embedding_model"],
            chunk_size=256,
        ),
        llm=LLamaLLM(
            repo_id=test_models_config["llm_repo_id"],
            filename=test_models_config["llm_filename"],
            n_ctx=test_models_config["n_ctx"],
            verbose=False,
        ),
        loader=KreuzbergLoader(),
    )


@pytest.fixture
def sample_documents():
    """Create sample documents for testing."""
    return [
        Document(
            content="This is a test document about artificial intelligence and machine learning.",
            filename="test1.txt",
        ),
        Document(
            content="This document discusses natural language processing and text analysis.",
            filename="test2.txt",
        ),
        Document(
            content="Here we explore vector databases and similarity search techniques.",
            filename="test3.txt",
        ),
    ]


@pytest.fixture(scope="session", autouse=True)
def setup_test_environment():
    """Set up test environment variables."""
    # Disable MPS fallback warnings during tests
    os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
    # Set minimal logging for tests
    os.environ["TRANSFORMERS_VERBOSITY"] = "error"
    yield
    # Cleanup is handled by fixture teardown
