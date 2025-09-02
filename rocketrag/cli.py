import os
import typer
from rich import print
from rich.console import Console
from rich.panel import Panel
import json
import warnings


# Lazy imports - only import heavy dependencies when needed
def _lazy_imports():
    """Import heavy dependencies only when actually needed."""
    from rich.progress import Progress, SpinnerColumn, TextColumn
    
    # Set PyTorch environment variable
    os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

    # Suppress pkg_resources deprecation warnings from milvus-lite
    warnings.filterwarnings("ignore", message=".*pkg_resources is deprecated.*")

    # Set environment variables to suppress gRPC warnings
    os.environ["GRPC_VERBOSITY"] = "ERROR"
    os.environ["GRPC_TRACE"] = ""

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        transient=True,
    ) as progress:
        task_id = progress.add_task("Loading dependencies...", total=None)
        
        progress.update(task_id, description="Loading vector processing...")
        from .vectors import init_vectorizer
        
        progress.update(task_id, description="Loading database components...")
        from .db import MilvusLiteDB
        
        progress.update(task_id, description="Loading chunking components...")
        from .chonk import init_chonker
        
        progress.update(task_id, description="Loading language models...")
        from .llm import LLamaLLM
        
        progress.update(task_id, description="Loading document loaders...")
        from .loaders import init_loader
        
        progress.update(task_id, description="Loading visualization components...")
        from .visualization import VectorVisualizer
        
        progress.update(task_id, description="Loading display utilities...")
        from .display_utils import display_streaming_answer
        
        progress.update(task_id, description="Loading RAG framework...")
        from .rocketrag import RocketRAG
        
        progress.update(task_id, description="Dependencies loaded ✓")

    return {
        "init_vectorizer": init_vectorizer,
        "MilvusLiteDB": MilvusLiteDB,
        "init_chonker": init_chonker,
        "LLamaLLM": LLamaLLM,
        "init_loader": init_loader,
        "VectorVisualizer": VectorVisualizer,
        "display_streaming_answer": display_streaming_answer,
        "RocketRAG": RocketRAG,
    }


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
    """Prepare the RAG system by processing documents and creating embeddings."""
    # Import heavy dependencies only when needed
    imports = _lazy_imports()

    print(Panel("[bold blue]Preparing RAG system...[/bold blue]"))

    # Parse JSON arguments
    chonker_args_dict = json.loads(chonker_args)
    vectorizer_args_dict = json.loads(vectorizer_args)
    loader_args_dict = json.loads(loader_args)

    # Initialize components
    vectorizer = imports["init_vectorizer"](
        "sentence_transformers", **vectorizer_args_dict
    )
    chunker = imports["init_chonker"](chonker, **chonker_args_dict)
    loader = imports["init_loader"](loader, **loader_args_dict)

    # Create RAG system
    rag = imports["RocketRAG"](
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
    """Query the LLM directly"""
    imports = _lazy_imports()
    llm = imports["LLamaLLM"](repo_id, filename)
    stream = llm.stream([{"role": "user", "content": question}])
    imports["display_streaming_answer"](question, stream)


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
        '{"model_name": "minishlab/potion-multilingual-128M"}',
        help="JSON string with vectorizer configuration arguments",
    ),
    db_path: str = typer.Option("rag.db", help="Path to the database file"),
    collection_name: str = typer.Option(
        "rag", help="Name of the collection in the database"
    ),
):
    """Ask a question to the RAG system"""
    imports = _lazy_imports()

    vectorizer_config = json.loads(vectorizer_args)
    vectorizer = imports["init_vectorizer"](
        "sentence_transformers", **vectorizer_config
    )

    llm = imports["LLamaLLM"](**{"repo_id": repo_id, "filename": filename})

    rag = imports["RocketRAG"](
        db_path=db_path,
        collection_name=collection_name,
        vectorizer=vectorizer,
        llm=llm,
    )
    stream, sources = rag.stream_ask(question)
    imports["display_streaming_answer"](question, stream, sources)


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
    """Start the RAG web server"""
    os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

    from .webserver import start_server

    print(f"Starting rocketrag server on {host}:{port}")

    imports = _lazy_imports()

    # Parse JSON arguments for server configuration
    chonker_config = json.loads(chonker_args)
    vectorizer_config = json.loads(vectorizer_args)
    loader_config = json.loads(loader_args)
    vectorizer = imports["init_vectorizer"](
        "sentence_transformers", **vectorizer_config
    )
    chunker = imports["init_chonker"](chonker, **chonker_config)
    loader = imports["init_loader"](loader, **loader_config)
    rag = imports["RocketRAG"](
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
    """Visualize the vector space in the database"""
    console = Console()

    imports = _lazy_imports()

    try:
        vectorizer_config = json.loads(vectorizer_args)
        vectorizer = imports["init_vectorizer"](
            "sentence_transformers", **vectorizer_config
        )

        db = imports["MilvusLiteDB"](
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

        visualizer = imports["VectorVisualizer"](console=console)

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
