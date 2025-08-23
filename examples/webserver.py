#!/usr/bin/env python3
"""
RocketRAG Web Server Example

This example demonstrates:
- Setting up a RocketRAG web server with custom configuration
- Configuring high-performance components
- Starting the server with OpenAI-compatible API endpoints
- Accessing the interactive web interface

The server provides:
- REST API endpoints for question answering
- Interactive web interface at http://localhost:8000
- Document browsing and visualization
- Streaming responses
- OpenAI-compatible chat completions API
"""

from rich.console import Console

from rocketrag import RocketRAG, start_server
from rocketrag.vectors import SentenceTransformersVectorizer
from rocketrag.chonk import ChonkieChunker
from rocketrag.llm import LLamaLLM
from rocketrag.loaders import KreuzbergLoader

console = Console()


def main():
    """Main function to start the RocketRAG web server."""

    # Server configuration
    host = "127.0.0.1"
    port = 8000
    data_dir = "pdf"
    db_path = "webserver_example.db"
    collection_name = "webserver_demo"

    vectorizer = SentenceTransformersVectorizer(
        model_name="minishlab/potion-multilingual-128M"  # Fast multilingual embeddings
    )

    chunker = ChonkieChunker(
        method="semantic",  # Semantic chunking for better context
        embedding_model="minishlab/potion-multilingual-128M",
        chunk_size=512,  # Optimal chunk size for retrieval
        overlap=50,  # Small overlap for context continuity
    )

    llm = LLamaLLM(
        repo_id="unsloth/gemma-3n-E2B-it-GGUF",
        filename="*Q8_0.gguf",  # Quantized model for speed
        n_ctx=4096,  # Context window size
        n_threads=4,  # Optimize for your CPU
        verbose=False,  # Reduce logging noise
    )

    loader = KreuzbergLoader()

    rag = RocketRAG(
        data_dir=data_dir,
        db_path=db_path,
        collection_name=collection_name,
        vectorizer=vectorizer,
        chunker=chunker,
        llm=llm,
        loader=loader,
    )

    rag.prepare()

    start_server(rag, port=port, host=host)
