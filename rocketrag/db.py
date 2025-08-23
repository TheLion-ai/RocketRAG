import hashlib
import json
import os
from pathlib import Path
from pymilvus import MilvusClient
from rich.console import Console
from rich.panel import Panel
from rich.text import Text
from rich.progress import track
from .base import BaseVectorizer, BaseChunker
from .data_models import Document, SearchResult
from .utils import compare_medatata_dicts


class MilvusLiteDB:
    def __init__(
        self,
        db_path: str,
        collection_name: str,
        vectorizer: BaseVectorizer | None = None,
        chunker: BaseChunker | None = None,
        metadata: dict | None = None,
    ):
        self.client = MilvusClient(db_path)
        self.collection_name = collection_name
        self.vectorizer = vectorizer
        self.chunker = chunker
        self.metadata = metadata
        self.db_path = db_path
        self.dimension = self.vectorizer.dimension
        self._metadata_file_path = self._get_metadata_file_path()

    def _get_metadata_file_path(self) -> str:
        """Get the path for the metadata sidecar file."""
        db_dir = Path(self.db_path).parent
        return str(db_dir / f".{self.collection_name}.meta.json")

    def _save_metadata(self, metadata: dict) -> None:
        """Save metadata to sidecar file."""
        try:
            with open(self._metadata_file_path, "w", encoding="utf-8") as f:
                json.dump(metadata, f, indent=2, ensure_ascii=False)
        except Exception as e:
            print(
                f"Warning: Failed to save metadata to {self._metadata_file_path}: {e}"
            )

    def _load_metadata(self) -> dict:
        """Load metadata from sidecar file."""
        try:
            if os.path.exists(self._metadata_file_path):
                with open(self._metadata_file_path, "r", encoding="utf-8") as f:
                    return json.load(f)
        except Exception as e:
            print(
                f"Warning: Failed to load metadata from {self._metadata_file_path}: {e}"
            )
        return {}

    def create_collection_if_not_exists(self, recreate: bool = False):
        if not self.client.has_collection(self.collection_name):
            print("Creating collection")
            self.client.create_collection(
                collection_name=self.collection_name,
                dimension=self.dimension,
                id_type="string",
                max_length=64,  # SHA-256 hash is 64 characters
            )
            self._save_metadata(self.metadata)
        elif self.metadata:
            # Load existing metadata from sidecar file
            existing_metadata = self._load_metadata()
            diff = compare_medatata_dicts(existing_metadata, self.metadata)
            if diff == {}:
                print(
                    "Collection already exists with the same metadata. Skipping initialization"
                )
            else:
                if not recreate:
                    diff_message = ""
                    for key, value in diff.items():
                        diff_message += f"Key: {key}\n  Existing DB value: {value['db']}\n  Passed value: {value['new']}\n"
                        diff_message += "Rerun the command with --recreate to recreate the collection.\n"

                    # Display diff in rich panel
                    console = Console()
                    console.print(
                        Panel(
                            diff_message,
                            title="Database config mismatch",
                            border_style="red",
                        )
                    )

                    raise ValueError("DB config missmach")
                else:
                    print("Recreating collection")
                    self.client.drop_collection(self.collection_name)
                    self.create_collection_if_not_exists()

    def _get_text_hash(self, text: str) -> str:
        """Generate SHA-256 hash of the text content."""
        return hashlib.sha256(text.encode("utf-8")).hexdigest()

    def _document_exists(self, text_hash: str) -> bool:
        """Check if a document with the given hash already exists."""
        try:
            results = self.client.query(
                collection_name=self.collection_name,
                filter=f'id == "{text_hash}"',
                output_fields=["id"],
                limit=1,
            )
            return len(results) > 0
        except Exception:
            # If query fails (e.g., collection doesn't exist), assume document doesn't exist
            return False

    def add_documents(self, documents: list[Document]):
        if self.chunker:
            documents = self.chunker.chunk_batch(documents)

        for doc in track(documents, description="Processing documents"):
            new_chunks = []
            for chunk in doc.chunks:
                text_hash = self._get_text_hash(chunk)
                if not self._document_exists(text_hash):
                    new_chunks.append(chunk)

            if not new_chunks:
                print(
                    f"All {len(doc.chunks)} chunks from {doc.filename} already exist in the collection."
                )
                continue

            print(
                f"Adding {len(new_chunks)} new chunks from {doc.filename} (skipped {len(doc.chunks) - len(new_chunks)} duplicates)."
            )
            if self.vectorizer is None:
                raise ValueError("Vectorizer is not set.")
            vectors = self.vectorizer.vectorize_batch(new_chunks)

            data = [
                {
                    "id": self._get_text_hash(new_chunks[i]),
                    "vector": vectors[i],
                    "text": new_chunks[i],
                    "filename": doc.filename,
                }
                for i in range(len(new_chunks))
            ]

            self.client.insert(collection_name=self.collection_name, data=data)
            print(f"Successfully added {len(new_chunks)} chunks to the collection.")

    def search(self, query: str, top_k: int = 5) -> list[SearchResult]:
        query_vector = self.vectorizer.vectorize(query)
        results = self.client.search(
            collection_name=self.collection_name,
            data=[query_vector],
            limit=top_k,
            output_fields=["text", "filename"],
            anns_field="vector",  # Explicitly specify the vector field
        )
        results = [
            SearchResult(
                chunk=result["entity"]["text"],
                filename=result["entity"]["filename"],
                score=result["distance"],
            )
            for result in results[0]
        ]
        return results

    def show_stats(self):
        """Display database statistics using rich formatting."""
        console = Console()

        collections = self.client.list_collections()
        if self.collection_name not in collections:
            console.print(
                Panel(
                    Text(
                        f"Collection '{self.collection_name}' does not exist.",
                        style="bold red",
                    ),
                    title="Database Status",
                    border_style="red",
                )
            )
            return

        stats = self.client.get_collection_stats(collection_name=self.collection_name)

        if "row_count" in stats and stats["row_count"] > 0:
            results = self.client.query(
                collection_name=self.collection_name,
                filter="",
                output_fields=["filename"],
                limit=16384,
            )
            unique_files = len({result["filename"] for result in results})
            if len(results) == 16384:
                unique_files = f"{unique_files}+"

            info_panel = Panel(
                f"✅ Database is ready with {stats['row_count']} chunks from {unique_files} files\n"
                f"📊 Vector space: {self.vectorizer.dimension}D\n"
                f"Vectorizer model: {self.vectorizer.model_name}\n",
                title="Database Status",
                border_style="green",
            )
        else:
            info_panel = Panel(
                "📭 Database is empty\n", title="Status", border_style="yellow"
            )

        console.print(info_panel)

    def get_all_records(self, limit: int = None, offset: int = 0):
        """Get all records from the collection with optional pagination."""
        try:
            results = self.client.query(
                collection_name=self.collection_name,
                filter="",
                output_fields=["text", "filename"],
                limit=limit,
                offset=offset,
            )
            return results
        except Exception as e:
            print(f"Error retrieving records: {e}")
            return []

    def get_total_count(self):
        """Get total number of records in the collection."""
        try:
            stats = self.client.get_collection_stats(
                collection_name=self.collection_name
            )
            return stats.get("row_count", 0)
        except Exception:
            return 0

    def get_collection_metadata(self) -> dict:
        """Get collection metadata from sidecar file."""
        return self._load_metadata()

    def get_vectors_with_metadata(self, limit: int = None, offset: int = 0):
        """Get vectors along with their metadata for visualization.

        Returns:
            list: List of dictionaries containing 'vector', 'text', 'filename', and 'id'
        """
        try:
            results = self.client.query(
                collection_name=self.collection_name,
                filter="",
                output_fields=["id", "vector", "text", "filename"],
                limit=limit,
                offset=offset,
            )
            return results
        except Exception as e:
            print(f"Error retrieving vectors with metadata: {e}")
            return []

    def get_vectors_by_filename(self, filename: str):
        """Get all vectors for a specific filename.

        Args:
            filename: The filename to filter by

        Returns:
            list: List of dictionaries containing vector data for the specified file
        """
        try:
            results = self.client.query(
                collection_name=self.collection_name,
                filter=f'filename == "{filename}"',
                output_fields=["id", "vector", "text", "filename"],
            )
            return results
        except Exception as e:
            print(f"Error retrieving vectors for filename {filename}: {e}")
            return []

    def get_unique_filenames(self):
        """Get list of unique filenames in the database.

        Returns:
            list: List of unique filenames
        """
        try:
            results = self.client.query(
                collection_name=self.collection_name,
                filter="",
                output_fields=["filename"],
                limit=16384,  # Set a reasonable limit for Milvus
            )
            filenames = list(set(record["filename"] for record in results))
            return sorted(filenames)
        except Exception as e:
            print(f"Error retrieving unique filenames: {e}")
            return []
