"""rocketrag - Fast, efficient, minimal, extendible and elegant RAG system."""

import os

os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

from .base import BaseVectorizer, BaseChunker, BaseLLM, BaseLoader
from .vectors import init_vectorizer
from .db import MilvusLiteDB
from .chonk import init_chonker
from .llm import init_llm
from .rag import RAG
from .loaders import init_loader
from .data_models import Document
from .rocketrag import RocketRAG
from .display_utils import display_streaming_answer
from .webserver import start_server

__version__ = "0.1.0"

__all__ = [
    "RocketRAG",
    # Core classes and abstract base classes
    "BaseVectorizer",
    "BaseChunker",
    "BaseLLM",
    "BaseLoader",
    "init_vectorizer",
    "MilvusLiteDB",
    "init_chonker",
    "init_llm",
    "RAG",
    "init_loader",
    "Document",
    "display_streaming_answer",
    "start_server",
]
