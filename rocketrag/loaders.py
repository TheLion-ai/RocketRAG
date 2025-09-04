from pathlib import Path
from kreuzberg import extract_file_sync
from .data_models import Document
from .base import BaseLoader


class KreuzbergLoader(BaseLoader):
    name = "kreuzberg"
    supported_formats = {
        # Documents
        "pdf", "docx", "doc", "rtf", "txt", "epub",
        # Images
        "jpg", "jpeg", "png", "tiff", "bmp", "gif", "webp",
        # Spreadsheets
        "xlsx", "xls", "csv", "ods",
        # Presentations
        "pptx", "ppt", "odp",
        # Web
        "html", "xml", "mhtml"
    }

    def __init__(self, **kwargs: dict):
        super().__init__(**kwargs)

    def load_files_from_dir(self, path: str):
        documents: list[Document] = []
        for file in Path(path).iterdir():
            if file.is_file(): 
                if not self._validate_file_format(file):
                    self._raise_unsupported_format_error(file)
                
                try:
                    content = extract_file_sync(file).content
                    documents.append(Document(content, file.name))
                except Exception as e:
                    # Re-raise with more context about the file
                    raise ValueError(
                        f"Failed to process file '{file.name}': {str(e)}. "
                        f"This may be due to an unsupported file format or corrupted file."
                    ) from e
        return documents


def init_loader(loader: str, **kwargs: dict):
    """Initialize a loader by name using abstract base class discovery."""
    for cls in BaseLoader.__subclasses__():
        if hasattr(cls, "name") and cls.name == loader:
            return cls(**kwargs)
    raise ValueError(
        f"Unknown loader: {loader}. Available: {[cls.name for cls in BaseLoader.__subclasses__() if hasattr(cls, 'name')]}"
    )
