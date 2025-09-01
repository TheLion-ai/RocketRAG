from functools import wraps
from .base import BaseChunker, BaseLLM, BaseLoader, BaseVectorizer
from .db import MilvusLiteDB
from .rag import RAG
from .utils import construct_metadata_dict
from .vectors import SentenceTransformersVectorizer
from .chonk import ChonkieChunker
from .llm import LLamaLLM
from .loaders import KreuzbergLoader
from .data_models import SearchResult


def ensure_llm_loaded(func):
    """Decorator to ensure LLM is loaded before calling LLM methods."""

    @wraps(func)
    def wrapper(self, *args, **kwargs):
        if not self.llm_loaded:
            self.llm.load()
            self.llm_loaded = True
        return func(self, *args, **kwargs)

    return wrapper


class RocketRAG:
    def __init__(
        self,
        data_dir: str | None = "pdf",
        db_path: str = "rag.db",
        collection_name: str = "rag",
        vectorizer: BaseVectorizer | None = SentenceTransformersVectorizer(
            model_name="minishlab/potion-multilingual-128M"
        ),
        chunker: BaseChunker | None = ChonkieChunker(
            method="semantic",
            embedding_model="minishlab/potion-multilingual-128M",
            chunk_size=512,
        ),
        llm: BaseLLM | None = LLamaLLM(
            repo_id="unsloth/gemma-3n-E2B-it-GGUF", filename="*Q8_0.gguf"
        ),
        loader: BaseLoader | None = KreuzbergLoader(),
    ):
        self.data_dir = data_dir
        self.db_path = db_path
        self.collection_name = collection_name
        self.llm_loaded = False

        self.vectorizer = vectorizer
        self.vectorizer_config = vectorizer.config

        self.chonker = chunker
        self.chonker_config = chunker.config

        self.llm = llm
        self.llm_config = llm.config

        self.loader = loader
        self.loader_config = loader.config

        self.metadata = construct_metadata_dict(
            self.data_dir,
            self.chonker,
            self.chonker_config,
            self.vectorizer,
            self.vectorizer_config,
            self.loader,
            self.loader_config,
            self.db_path,
            self.collection_name,
        )

        self.db = MilvusLiteDB(
            self.db_path,
            self.collection_name,
            self.vectorizer,
            self.chonker,
            self.metadata,
        )

    def prepare(self, recreate: bool = False):
        if self.loader is None:
            raise ValueError("Loader is not defined.")
        if self.data_dir is None:
            raise ValueError("Data directory is not defined.")
        documents = self.loader.load_files_from_dir(self.data_dir)

        self.db.create_collection_if_not_exists(recreate)
        self.db.add_documents(documents)

    @ensure_llm_loaded
    def run_llm(self, messages: list[dict]) -> str:
        return self.llm.run(messages)

    @ensure_llm_loaded
    def stream_llm(self, messages: list[dict]) -> str:
        return self.llm.stream(messages)

    @ensure_llm_loaded
    def ask(self, question: str) -> tuple[str, list[SearchResult]]:
        # TODO: Should print the error on vectorizer mismach
        rag = RAG(self.db, self.llm)
        stream, sources = rag.run(question)
        return stream, sources

    @ensure_llm_loaded
    def stream_ask(self, question: str) -> tuple[str, list[SearchResult]]:
        # TODO: Should print the error on vectorizer mismach
        rag = RAG(self.db, self.llm)
        stream, sources = rag.stream(question)
        return stream, sources
