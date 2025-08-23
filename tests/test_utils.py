"""Test utilities and helper functions."""

import os
import shutil
from typing import List, Dict, Any

from rocketrag.data_models import Document, SearchResult


def create_test_documents(num_docs: int = 3) -> List[Document]:
    """Create a list of test documents for testing.

    Args:
        num_docs: Number of test documents to create

    Returns:
        List of Document objects with test content
    """
    documents = []

    for i in range(num_docs):
        content = f"""This is test document {i + 1}.
        
It contains information about topic {i + 1}.
This document discusses various aspects of the subject matter.
The content is designed to be searchable and relevant for testing purposes.
        
Key points in this document:
- Point A: Important information about aspect A
- Point B: Critical details regarding aspect B  
- Point C: Essential facts about aspect C
        
Conclusion: This document provides comprehensive coverage of topic {i + 1}.
        """

        documents.append(Document(content=content, filename=f"test_doc_{i + 1}.txt"))

    return documents


def create_test_search_results(num_results: int = 3) -> List[SearchResult]:
    """Create a list of test search results.

    Args:
        num_results: Number of search results to create

    Returns:
        List of SearchResult objects
    """
    results = []

    for i in range(num_results):
        chunk = f"This is a relevant chunk {i + 1} that matches the search query. It contains important information."

        results.append(
            SearchResult(
                chunk=chunk,
                filename=f"source_{i + 1}.pdf",
                score=0.1 * (i + 1),  # Increasing scores
            )
        )

    return results


def create_temp_data_directory(
    temp_dir: str, include_pdf: bool = False, pdf_path: str = None
) -> str:
    """Create a temporary data directory with test files.

    Args:
        temp_dir: Base temporary directory
        include_pdf: Whether to include a PDF file
        pdf_path: Path to PDF file to copy (if include_pdf is True)

    Returns:
        Path to the created data directory
    """
    data_dir = os.path.join(temp_dir, "test_data")
    os.makedirs(data_dir, exist_ok=True)

    # Create some test text files
    test_files = [
        ("doc1.txt", "This is the first test document with important information."),
        ("doc2.txt", "This is the second test document discussing different topics."),
        ("doc3.txt", "This is the third test document with additional content."),
    ]

    for filename, content in test_files:
        file_path = os.path.join(data_dir, filename)
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(content)

    # Copy PDF if requested
    if include_pdf and pdf_path and os.path.exists(pdf_path):
        pdf_dest = os.path.join(data_dir, "sample.pdf")
        shutil.copy2(pdf_path, pdf_dest)

    return data_dir


def assert_valid_search_results(results: List[SearchResult], min_results: int = 1):
    """Assert that search results are valid.

    Args:
        results: List of search results to validate
        min_results: Minimum number of expected results
    """
    assert isinstance(results, list), "Results should be a list"
    assert len(results) >= min_results, (
        f"Expected at least {min_results} results, got {len(results)}"
    )

    for i, result in enumerate(results):
        assert isinstance(result, SearchResult), (
            f"Result {i} should be a SearchResult object"
        )
        assert isinstance(result.chunk, str), f"Result {i} chunk should be a string"
        assert len(result.chunk) > 0, f"Result {i} chunk should not be empty"
        assert isinstance(result.filename, str), (
            f"Result {i} filename should be a string"
        )
        assert len(result.filename) > 0, f"Result {i} filename should not be empty"
        assert isinstance(result.score, (int, float)), (
            f"Result {i} score should be numeric"
        )
        assert result.score >= 0, f"Result {i} score should be non-negative"


def assert_valid_documents(documents: List[Document], min_docs: int = 1):
    """Assert that documents are valid.

    Args:
        documents: List of documents to validate
        min_docs: Minimum number of expected documents
    """
    assert isinstance(documents, list), "Documents should be a list"
    assert len(documents) >= min_docs, (
        f"Expected at least {min_docs} documents, got {len(documents)}"
    )

    for i, doc in enumerate(documents):
        assert isinstance(doc, Document), f"Document {i} should be a Document object"
        assert isinstance(doc.content, str), f"Document {i} content should be a string"
        assert len(doc.content) > 0, f"Document {i} content should not be empty"
        assert isinstance(doc.filename, str), (
            f"Document {i} filename should be a string"
        )
        assert len(doc.filename) > 0, f"Document {i} filename should not be empty"


def assert_valid_llm_response(response: str, min_length: int = 1):
    """Assert that LLM response is valid.

    Args:
        response: LLM response to validate
        min_length: Minimum expected response length
    """
    assert isinstance(response, str), "Response should be a string"
    assert len(response) >= min_length, (
        f"Response should be at least {min_length} characters long"
    )
    assert response.strip(), "Response should not be empty or just whitespace"


def assert_valid_metadata(metadata: Dict[str, Any], required_keys: List[str] = None):
    """Assert that metadata dictionary is valid.

    Args:
        metadata: Metadata dictionary to validate
        required_keys: List of required keys in metadata
    """
    assert isinstance(metadata, dict), "Metadata should be a dictionary"

    if required_keys:
        for key in required_keys:
            assert key in metadata, f"Required key '{key}' missing from metadata"


def create_mock_llm_messages(num_messages: int = 2) -> List[Dict[str, str]]:
    """Create mock LLM messages for testing.

    Args:
        num_messages: Number of messages to create

    Returns:
        List of message dictionaries
    """
    messages = []

    # Always start with a system message
    messages.append({"role": "system", "content": "You are a helpful AI assistant."})

    # Add user messages
    for i in range(num_messages - 1):
        messages.append(
            {
                "role": "user",
                "content": f"This is test message {i + 1}. Please respond appropriately.",
            }
        )

    return messages


def get_test_file_size(file_path: str) -> int:
    """Get the size of a test file in bytes.

    Args:
        file_path: Path to the file

    Returns:
        File size in bytes
    """
    if os.path.exists(file_path):
        return os.path.getsize(file_path)
    return 0


def cleanup_test_files(*file_paths: str):
    """Clean up test files and directories.

    Args:
        *file_paths: Variable number of file/directory paths to clean up
    """
    for path in file_paths:
        if os.path.exists(path):
            if os.path.isfile(path):
                os.remove(path)
            elif os.path.isdir(path):
                shutil.rmtree(path)


def assert_database_state(
    db, expected_count: int = None, expected_filenames: List[str] = None
):
    """Assert database is in expected state.

    Args:
        db: Database instance to check
        expected_count: Expected number of records
        expected_filenames: Expected list of filenames in database
    """
    if expected_count is not None:
        actual_count = db.count_records()
        assert actual_count == expected_count, (
            f"Expected {expected_count} records, got {actual_count}"
        )

    if expected_filenames is not None:
        actual_filenames = db.get_unique_filenames()
        for filename in expected_filenames:
            assert filename in actual_filenames, (
                f"Expected filename '{filename}' not found in database"
            )


def create_test_config(temp_dir: str) -> Dict[str, Any]:
    """Create a test configuration dictionary.

    Args:
        temp_dir: Temporary directory for test files

    Returns:
        Test configuration dictionary
    """
    return {
        "data_dir": os.path.join(temp_dir, "data"),
        "db_path": os.path.join(temp_dir, "test.db"),
        "collection_name": "test_collection",
        "embedding_model": "minishlab/potion-base-8M",
        "llm_repo_id": "unsloth/gemma-3-270m-it-GGUF",
        "llm_filename": "gemma-3-270m-it-Q4_0.gguf",
        "n_ctx": 512,
        "chunk_size": 256,
        "chunk_overlap": 50,
        "top_k": 5,
    }


def verify_test_environment():
    """Verify that the test environment is properly set up.

    Returns:
        True if environment is ready, False otherwise
    """
    try:
        # Check if required packages are available
        import tempfile
        import os

        # Check if we can create temporary directories
        with tempfile.TemporaryDirectory() as temp_dir:
            test_file = os.path.join(temp_dir, "test.txt")
            with open(test_file, "w") as f:
                f.write("test")
            assert os.path.exists(test_file)

        return True
    except Exception:
        return False


class TestDataManager:
    """Helper class for managing test data."""

    def __init__(self, temp_dir: str):
        self.temp_dir = temp_dir
        self.created_files = []
        self.created_dirs = []

    def create_file(self, filename: str, content: str) -> str:
        """Create a test file with given content.

        Args:
            filename: Name of the file to create
            content: Content to write to the file

        Returns:
            Full path to the created file
        """
        file_path = os.path.join(self.temp_dir, filename)

        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(file_path), exist_ok=True)

        with open(file_path, "w", encoding="utf-8") as f:
            f.write(content)

        self.created_files.append(file_path)
        return file_path

    def create_directory(self, dirname: str) -> str:
        """Create a test directory.

        Args:
            dirname: Name of the directory to create

        Returns:
            Full path to the created directory
        """
        dir_path = os.path.join(self.temp_dir, dirname)
        os.makedirs(dir_path, exist_ok=True)
        self.created_dirs.append(dir_path)
        return dir_path

    def cleanup(self):
        """Clean up all created files and directories."""
        cleanup_test_files(*self.created_files, *self.created_dirs)
        self.created_files.clear()
        self.created_dirs.clear()
