from dataclasses import dataclass


@dataclass
class Document:
    content: str
    filename: str
    chunks: list[str] = None


@dataclass
class SearchResult:
    chunk: str
    filename: str
    score: float
