"""Tests for loader classes and file format validation."""

import os
import tempfile
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock

from rocketrag.loaders import KreuzbergLoader, init_loader
from rocketrag.base import BaseLoader
from rocketrag.data_models import Document


class TestBaseLoader:
    """Tests for BaseLoader file format validation methods."""

    def test_validate_file_format_with_supported_formats(self):
        """Test file format validation when supported formats are defined."""
        class TestLoader(BaseLoader):
            name = "test"
            supported_formats = {"pdf", "txt", "docx"}
            
            def load_files_from_dir(self, path: str):
                return []
        
        loader = TestLoader()
        
        # Test supported formats
        assert loader._validate_file_format(Path("test.pdf")) is True
        assert loader._validate_file_format(Path("test.txt")) is True
        assert loader._validate_file_format(Path("test.docx")) is True
        
        # Test unsupported formats
        assert loader._validate_file_format(Path("test.jpg")) is False
        assert loader._validate_file_format(Path("test.mp4")) is False
        assert loader._validate_file_format(Path("test.unknown")) is False

    def test_validate_file_format_case_insensitive(self):
        """Test that file format validation is case insensitive."""
        class TestLoader(BaseLoader):
            name = "test"
            supported_formats = {"pdf", "txt"}
            
            def load_files_from_dir(self, path: str):
                return []
        
        loader = TestLoader()
        
        # Test different cases
        assert loader._validate_file_format(Path("test.PDF")) is True
        assert loader._validate_file_format(Path("test.Txt")) is True
        assert loader._validate_file_format(Path("test.TXT")) is True

    def test_validate_file_format_no_supported_formats(self):
        """Test file format validation when no supported formats are defined."""
        class TestLoader(BaseLoader):
            name = "test"
            # No supported_formats defined, should default to empty set
            
            def load_files_from_dir(self, path: str):
                return []
        
        loader = TestLoader()
        
        # Should return True for any format when no formats are specified
        assert loader._validate_file_format(Path("test.pdf")) is True
        assert loader._validate_file_format(Path("test.unknown")) is True

    def test_raise_unsupported_format_error(self):
        """Test that unsupported format error is raised correctly."""
        class TestLoader(BaseLoader):
            name = "test"
            supported_formats = {"pdf", "txt"}
            
            def load_files_from_dir(self, path: str):
                return []
        
        loader = TestLoader()
        
        with pytest.raises(ValueError) as exc_info:
            loader._raise_unsupported_format_error(Path("test.jpg"))
        
        error_message = str(exc_info.value)
        assert "Unsupported file format 'jpg' for test loader" in error_message
        assert "Supported formats: pdf, txt" in error_message

    def test_raise_unsupported_format_error_no_extension(self):
        """Test error handling for files without extensions."""
        class TestLoader(BaseLoader):
            name = "test"
            supported_formats = {"pdf", "txt"}
            
            def load_files_from_dir(self, path: str):
                return []
        
        loader = TestLoader()
        
        with pytest.raises(ValueError) as exc_info:
            loader._raise_unsupported_format_error(Path("test_file"))
        
        error_message = str(exc_info.value)
        assert "Unsupported file format '' for test loader" in error_message


class TestKreuzbergLoader:
    """Tests for KreuzbergLoader class."""

    def test_supported_formats_defined(self):
        """Test that KreuzbergLoader has supported formats defined."""
        loader = KreuzbergLoader()
        
        # Check that supported formats are defined
        assert hasattr(loader, 'supported_formats')
        assert isinstance(loader.supported_formats, set)
        assert len(loader.supported_formats) > 0
        
        # Check some expected formats
        expected_formats = {"pdf", "docx", "txt", "jpg", "png", "xlsx", "pptx", "html"}
        assert expected_formats.issubset(loader.supported_formats)

    def test_validate_supported_file_formats(self):
        """Test validation of supported file formats."""
        loader = KreuzbergLoader()
        
        # Test document formats
        assert loader._validate_file_format(Path("test.pdf")) is True
        assert loader._validate_file_format(Path("test.docx")) is True
        assert loader._validate_file_format(Path("test.txt")) is True
        
        # Test image formats
        assert loader._validate_file_format(Path("test.jpg")) is True
        assert loader._validate_file_format(Path("test.png")) is True
        
        # Test spreadsheet formats
        assert loader._validate_file_format(Path("test.xlsx")) is True
        assert loader._validate_file_format(Path("test.csv")) is True
        
        # Test presentation formats
        assert loader._validate_file_format(Path("test.pptx")) is True
        
        # Test web formats
        assert loader._validate_file_format(Path("test.html")) is True

    def test_validate_unsupported_file_formats(self):
        """Test validation of unsupported file formats."""
        loader = KreuzbergLoader()
        
        # Test unsupported formats
        assert loader._validate_file_format(Path("test.mp4")) is False
        assert loader._validate_file_format(Path("test.mp3")) is False
        assert loader._validate_file_format(Path("test.exe")) is False
        assert loader._validate_file_format(Path("test.unknown")) is False

    @patch('rocketrag.loaders.extract_file_sync')
    def test_load_files_from_dir_success(self, mock_extract):
        """Test successful file loading from directory."""
        # Mock extract_file_sync to return content
        mock_result = MagicMock()
        mock_result.content = "Test content"
        mock_extract.return_value = mock_result
        
        loader = KreuzbergLoader()
        
        # Create temporary directory with test files
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create test files with supported formats
            test_files = ["test1.pdf", "test2.txt", "test3.docx"]
            for filename in test_files:
                Path(temp_dir, filename).touch()
            
            documents = loader.load_files_from_dir(temp_dir)
            
            # Verify results
            assert len(documents) == 3
            assert all(isinstance(doc, Document) for doc in documents)
            assert all(doc.content == "Test content" for doc in documents)
            assert {doc.filename for doc in documents} == set(test_files)
            
            # Verify extract_file_sync was called for each file
            assert mock_extract.call_count == 3

    def test_load_files_from_dir_unsupported_format(self):
        """Test error handling for unsupported file formats."""
        loader = KreuzbergLoader()
        
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create test file with unsupported format
            unsupported_file = Path(temp_dir, "test.mp4")
            unsupported_file.touch()
            
            with pytest.raises(ValueError) as exc_info:
                loader.load_files_from_dir(temp_dir)
            
            error_message = str(exc_info.value)
            assert "Unsupported file format 'mp4' for kreuzberg loader" in error_message

    def test_load_files_from_dir_skips_directories(self):
        """Test that directories are skipped during file loading."""
        loader = KreuzbergLoader()
        
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create a subdirectory
            subdir = Path(temp_dir, "subdir")
            subdir.mkdir()
            
            # Create a test file
            test_file = Path(temp_dir, "test.txt")
            test_file.touch()
            
            with patch('rocketrag.loaders.extract_file_sync') as mock_extract:
                mock_result = MagicMock()
                mock_result.content = "Test content"
                mock_extract.return_value = mock_result
                
                documents = loader.load_files_from_dir(temp_dir)
                
                # Should only process the file, not the directory
                assert len(documents) == 1
                assert documents[0].filename == "test.txt"
                assert mock_extract.call_count == 1

    @patch('rocketrag.loaders.extract_file_sync')
    def test_load_files_from_dir_extraction_error(self, mock_extract):
        """Test error handling when file extraction fails."""
        # Mock extract_file_sync to raise an exception
        mock_extract.side_effect = Exception("Extraction failed")
        
        loader = KreuzbergLoader()
        
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create test file
            test_file = Path(temp_dir, "test.pdf")
            test_file.touch()
            
            with pytest.raises(ValueError) as exc_info:
                loader.load_files_from_dir(temp_dir)
            
            error_message = str(exc_info.value)
            assert "Failed to process file 'test.pdf'" in error_message
            assert "Extraction failed" in error_message
            assert "unsupported file format or corrupted file" in error_message


class TestInitLoader:
    """Tests for init_loader function."""

    def test_init_kreuzberg_loader(self):
        """Test initialization of KreuzbergLoader by name."""
        loader = init_loader("kreuzberg")
        
        assert isinstance(loader, KreuzbergLoader)
        assert loader.name == "kreuzberg"

    def test_init_loader_with_kwargs(self):
        """Test loader initialization with keyword arguments."""
        test_config = {"param1": "value1", "param2": "value2"}
        loader = init_loader("kreuzberg", **test_config)
        
        assert isinstance(loader, KreuzbergLoader)
        assert loader.config == test_config

    def test_init_loader_unknown_loader(self):
        """Test error handling for unknown loader names."""
        with pytest.raises(ValueError) as exc_info:
            init_loader("unknown_loader")
        
        error_message = str(exc_info.value)
        assert "Unknown loader: unknown_loader" in error_message
        assert "Available: ['kreuzberg']" in error_message


class TestBackwardCompatibility:
    """Tests for backward compatibility with loaders without defined formats."""

    def test_loader_without_supported_formats(self):
        """Test that loaders without supported_formats still work."""
        class LegacyLoader(BaseLoader):
            name = "legacy"
            # No supported_formats defined
            
            def load_files_from_dir(self, path: str):
                return [Document("test content", "test.txt")]
        
        loader = LegacyLoader()
        
        # Should accept any file format
        assert loader._validate_file_format(Path("test.pdf")) is True
        assert loader._validate_file_format(Path("test.unknown")) is True
        
        # Should work normally
        documents = loader.load_files_from_dir("/fake/path")
        assert len(documents) == 1
        assert documents[0].content == "test content"

    def test_loader_with_empty_supported_formats(self):
        """Test loader with explicitly empty supported_formats."""
        class EmptyFormatsLoader(BaseLoader):
            name = "empty"
            supported_formats = set()  # Explicitly empty
            
            def load_files_from_dir(self, path: str):
                return []
        
        loader = EmptyFormatsLoader()
        
        # Should accept any file format when set is empty
        assert loader._validate_file_format(Path("test.pdf")) is True
        assert loader._validate_file_format(Path("test.unknown")) is True