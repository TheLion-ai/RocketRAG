import os

os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

import typer
from rich import print
from rich.console import Console
from rich.panel import Panel
import json
import warnings
import os
from .vectors import init_vectorizer
from .db import MilvusLiteDB
from .chonk import init_chonker
from .llm import LLamaLLM
from .loaders import init_loader
from .visualization import VectorVisualizer
from .display_utils import (
    display_streaming_answer,
)
from .rocketrag import RocketRAG

# Suppress pkg_resources deprecation warnings from milvus-lite
warnings.filterwarnings("ignore", message=".*pkg_resources is deprecated.*")

# Set environment variables to suppress gRPC warnings
os.environ["GRPC_VERBOSITY"] = "ERROR"
os.environ["GRPC_TRACE"] = ""


app = typer.Typer()


@app.command()
def prepare(
    data_dir: str = typer.Option(
        "pdf", help="Directory containing documents to process"
    ),
    chonker: str = typer.Option(
        "chonkie", help="Chunking strategy to use (e.g., 'chonkie')"
    ),
    chonker_args: str = typer.Option(
        '{"method": "recursive", "chunk_size": 500}',
        help="JSON string with chunker configuration arguments",
    ),
    vectorizer_args: str = typer.Option(
        '{"model_name": "minishlab/potion-multilingual-128M"}',
        help="JSON string with vectorizer configuration arguments",
    ),
    loader: str = typer.Option(
        "kreuzberg", help="Document loader to use (e.g., 'kreuzberg')"
    ),
    loader_args: str = typer.Option(
        "{}", help="JSON string with loader configuration arguments"
    ),
    db_path: str = typer.Option("rag.db", help="Path to the database file"),
    collection_name: str = typer.Option(
        "rag", help="Name of the collection in the database"
    ),
    recreate: bool = typer.Option(False, help="Recreate the collection if it exists"),
):
    # Parse JSON arguments
    chonker_config = json.loads(chonker_args)
    vectorizer_config = json.loads(vectorizer_args)
    loader_config = json.loads(loader_args)

    vectorizer = init_vectorizer("sentence_transformers", **vectorizer_config)
    chunker = init_chonker(chonker, **chonker_config)
    loader = init_loader(loader, **loader_config)

    rag = RocketRAG(
        data_dir,
        db_path,
        collection_name,
        vectorizer,
        chunker,
        loader,
    )
    rag.prepare(recreate)


@app.command()
def llm(
    question: str = typer.Argument(..., help="Question to ask the LLM directly"),
    repo_id: str = typer.Option(
        "unsloth/gemma-3n-E2B-it-GGUF",
        help="Hugging Face repository ID for the LLM model",
    ),
    filename: str = typer.Option(
        "*Q8_0.gguf", help="Model filename pattern to load from the repository"
    ),
):
    llm = LLamaLLM(repo_id, filename)
    stream = llm.stream([{"role": "user", "content": question}])
    display_streaming_answer(question, stream)


@app.command()
def ask(
    question: str = typer.Argument(..., help="Question to ask the RAG system"),
    repo_id: str = typer.Option(
        "unsloth/gemma-3n-E2B-it-GGUF",
        help="Hugging Face repository ID for the LLM model",
    ),
    filename: str = typer.Option(
        "*Q8_0.gguf", help="Model filename pattern to load from the repository"
    ),
    vectorizer_args: str = typer.Option(
        '{"model_name": "all-MiniLM-L6-v2"}',
        help="JSON string with vectorizer configuration arguments",
    ),
    db_path: str = typer.Option("rag.db", help="Path to the database file"),
    collection_name: str = typer.Option(
        "rag", help="Name of the collection in the database"
    ),
):
    vectorizer_config = json.loads(vectorizer_args)
    vectorizer = init_vectorizer("sentence_transformers", **vectorizer_config)

    llm = LLamaLLM(**{"repo_id": repo_id, "filename": filename})

    rag = RocketRAG(
        db_path=db_path,
        collection_name=collection_name,
        vectorizer=vectorizer,
        llm=llm,
    )
    stream, sources = rag.stream_ask(question)
    display_streaming_answer(question, stream, sources)


@app.command()
def server(
    repo_id: str = typer.Option(
        "unsloth/gemma-3n-E2B-it-GGUF",
        help="Hugging Face repository ID for the LLM model",
    ),
    filename: str = typer.Option(
        "*Q8_0.gguf", help="Model filename pattern to load from the repository"
    ),
    port: int = typer.Option(8000, help="Port number to run the server on"),
    host: str = typer.Option("127.0.0.1", help="Host address to bind the server to"),
    chonker: str = typer.Option(
        "chonkie", help="Chunking strategy to use (e.g., 'chonkie')"
    ),
    chonker_args: str = typer.Option(
        '{"method": "recursive", "chunk_size": 500}',
        help="JSON string with chunker configuration arguments",
    ),
    vectorizer_args: str = typer.Option(
        '{"model_name": "minishlab/potion-multilingual-128M"}',
        help="JSON string with vectorizer configuration arguments",
    ),
    loader: str = typer.Option(
        "kreuzberg", help="Document loader to use (e.g., 'kreuzberg')"
    ),
    loader_args: str = typer.Option(
        "{}", help="JSON string with loader configuration arguments"
    ),
    db_path: str = typer.Option("rag.db", help="Path to the database file"),
    collection_name: str = typer.Option(
        "rag", help="Name of the collection in the database"
    ),
):
    """Start the RAG web server with OpenAI compatible API endpoints."""
    from .webserver import start_server

    print(f"Starting rocketrag server on {host}:{port}")

    # Parse JSON arguments for server configuration
    chonker_config = json.loads(chonker_args)
    vectorizer_config = json.loads(vectorizer_args)
    loader_config = json.loads(loader_args)
    vectorizer = init_vectorizer("sentence_transformers", **vectorizer_config)
    chunker = init_chonker(chonker, **chonker_config)
    loader = init_loader(loader, **loader_config)
    rag = RocketRAG(
        db_path=db_path,
        collection_name=collection_name,
        vectorizer=vectorizer,
        chunker=chunker,
        loader=loader,
    )
    start_server(rag)


@app.command()
def visualize(
    db_path: str = typer.Option("rag.db", help="Path to the database file"),
    collection_name: str = typer.Option(
        "rag", help="Name of the collection in the database"
    ),
    vectorizer_args: str = typer.Option(
        '{"model_name": "minishlab/potion-multilingual-128M"}',
        help="JSON string with vectorizer configuration arguments",
    ),
    limit: int = typer.Option(
        500, help="Maximum number of vectors to visualize (for performance)"
    ),
    filename_filter: str = typer.Option(
        None, help="Filter vectors by specific filename"
    ),
    question: str = typer.Option(
        None, help="Question to encode and visualize on the map with similar chunks"
    ),
):
    console = Console()

    try:
        vectorizer_config = json.loads(vectorizer_args)
        vectorizer = init_vectorizer("sentence_transformers", **vectorizer_config)

        db = MilvusLiteDB(
            db_path=db_path, collection_name=collection_name, vectorizer=vectorizer
        )

        if collection_name not in db.client.list_collections():
            console.print(
                Panel(
                    f"Collection '{collection_name}' not found. Run 'prepare' command first.",
                    title="Error",
                    border_style="red",
                )
            )
            return

        if filename_filter:
            records = db.get_vectors_by_filename(filename_filter)
            title_suffix = f" - {filename_filter}"
        else:
            records = db.get_vectors_with_metadata(limit=limit)
            title_suffix = (
                f" (showing {min(limit, len(records))} vectors)"
                if len(records) == limit
                else ""
            )

        if not records:
            console.print(
                Panel(
                    "No vectors found in the database.",
                    title="No Data",
                    border_style="yellow",
                )
            )
            return

        vectors = [record["vector"] for record in records]
        metadata = [
            {"id": record["id"], "text": record["text"], "filename": record["filename"]}
            for record in records
        ]

        visualizer = VectorVisualizer(console=console)

        with console.status("[bold green]Analyzing vectors..."):
            analysis_result = visualizer.analyze_vectors(vectors, metadata)

        if "error" in analysis_result:
            console.print(
                Panel(
                    f"Analysis error: {analysis_result['error']}",
                    title="Error",
                    border_style="red",
                )
            )
            return

        question_point = None
        similar_chunks = None

        if question:
            with console.status("[bold green]Processing question..."):
                question_result = visualizer.process_question(
                    question, vectors, metadata, analysis_result, vectorizer
                )
                if "error" not in question_result:
                    question_point = question_result.get("question_point")
                    similar_chunks = question_result.get("similar_chunks")

        console.print("\n")
        console.print(
            visualizer.create_visualization_panel(
                analysis_result,
                title=f"Vector Space Visualization{title_suffix}",
                color_by="filename",
                question_point=question_point,
                question_text=question,
            )
        )

        if similar_chunks:
            console.print("\n")
            console.print(
                visualizer.create_similar_chunks_panel(similar_chunks, question)
            )

        console.print("\n[dim]Visualization complete![/dim]")

    except json.JSONDecodeError as e:
        console.print(
            Panel(
                f"Invalid JSON in arguments: {e}",
                title="Configuration Error",
                border_style="red",
            )
        )
    except Exception as e:
        console.print(
            Panel(f"Visualization error: {e}", title="Error", border_style="red")
        )


def main():
    app()


if __name__ == "__main__":
    app()
