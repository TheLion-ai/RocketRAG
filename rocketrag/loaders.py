from pathlib import Path
from kreuzberg import extract_file_sync
from .data_models import Document
from .base import BaseLoader


class KreuzbergLoader(BaseLoader):
    name = "kreuzberg"

    def __init__(self, **kwargs: dict):
        super().__init__(**kwargs)

    def load_files_from_dir(self, path: str):
        documents: list[Document] = []
        for file in Path(path).iterdir():
            documents.append(Document(extract_file_sync(file).content, file.name))
        return documents


def init_loader(loader: str, **kwargs: dict):
    """Initialize a loader by name using abstract base class discovery."""
    for cls in BaseLoader.__subclasses__():
        if hasattr(cls, "name") and cls.name == loader:
            return cls(**kwargs)
    raise ValueError(
        f"Unknown loader: {loader}. Available: {[cls.name for cls in BaseLoader.__subclasses__() if hasattr(cls, 'name')]}"
    )
