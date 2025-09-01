"""rocketrag - Fast, efficient, minimal, extendible and elegant RAG system."""

import os

# Lazy imports to improve startup performance
def __getattr__(name):
    """Lazy import mechanism for package-level imports."""
    if name == "BaseVectorizer":
        from .base import BaseVectorizer
        return BaseVectorizer
    elif name == "BaseChunker":
        from .base import BaseChunker
        return BaseChunker
    elif name == "BaseLLM":
        from .base import BaseLLM
        return BaseLLM
    elif name == "BaseLoader":
        from .base import BaseLoader
        return BaseLoader
    elif name == "init_vectorizer":
        from .vectors import init_vectorizer
        return init_vectorizer
    elif name == "MilvusLiteDB":
        from .db import MilvusLiteDB
        return MilvusLiteDB
    elif name == "init_chonker":
        from .chonk import init_chonker
        return init_chonker
    elif name == "init_llm":
        from .llm import init_llm
        return init_llm
    elif name == "RAG":
        from .rag import RAG
        return RAG
    elif name == "init_loader":
        from .loaders import init_loader
        return init_loader
    elif name == "Document":
        from .data_models import Document
        return Document
    elif name == "RocketRAG":
        from .rocketrag import RocketRAG
        return RocketRAG
    elif name == "display_streaming_answer":
        from .display_utils import display_streaming_answer
        return display_streaming_answer
    elif name == "start_server":
        from .webserver import start_server
        return start_server
    else:
        raise AttributeError(f"module '{__name__}' has no attribute '{name}'")

# Set environment variable only when heavy dependencies are actually used
def _set_pytorch_env():
    os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

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
