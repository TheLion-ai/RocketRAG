#!/usr/bin/env python3
"""
Basic RocketRAG Usage Example

This example demonstrates:
- Basic RocketRAG initialization with default settings
- Document preparation and indexing
- Simple question answering
- Streaming responses with rich display
- Error handling and configuration options
"""

from rich.console import Console
from rich.panel import Panel

from rocketrag import RocketRAG

console = Console()

data_dir = "pdf"  # Directory containing documents
db_path = "basic_example.db"
collection_name = "basic_demo"


rag = RocketRAG(
    data_dir=data_dir,
    db_path=db_path,
    collection_name=collection_name,
    # Using default components for optimal performance:
    # - SentenceTransformersVectorizer with multilingual model
    # - ChonkieChunker with semantic chunking
    # - LLamaLLM with quantized Gemma model
    # - KreuzbergLoader for fast document processing
)

# Prepare documents (index them into the vector database)
rag.prepare(recreate=True)  # recreate=True ensures fresh indexing

questions = [
    "What is the main topic of the documents?",
    "Can you summarize the key findings?",
    "What are the most important points mentioned?",
]


for i, question in enumerate(questions, 1):
    console.print("\n[dim]Getting answer...[/dim]")
    answer, sources = rag.ask(question)

    console.print(
        Panel(answer, title="[bold green]Answer[/bold green]", border_style="green")
    )

    # Display sources
    if sources:
        console.print("\n[bold]📚 Sources:[/bold]")
        for j, source in enumerate(sources[:3], 1):  # Show top 3 sources
            console.print(f"  {j}. {source.filename} (score: {source.score:.3f})")
            console.print(f"     {source.chunk[:100]}...")
