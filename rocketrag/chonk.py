from chonkie import (
    TokenChunker,
    SentenceChunker,
    RecursiveChunker,
    SemanticChunker,
    SlumberChunker,
    LateChunker,
    CodeChunker,
    NeuralChunker,
)
from .data_models import Document
from .base import BaseChunker


class ChonkieChunker(BaseChunker):
    """A unified interface for all Chonkie chunking methods.

    This chunker provides access to 8 different chunking strategies, each optimized
    for different use cases and document types.

    Args:
        method (str): The chunking method to use. Defaults to "recursive".
            Available methods:
            - "token": Fixed-size token chunks
            - "sentence": Sentence boundary splitting
            - "recursive": Recursive document chunking
            - "semantic": Semantic similarity-based grouping
            - "sdpm": Semantic Double-Pass Merging
            - "late": Late-bound token count with document embeddings
            - "code": AST-based code structure splitting
            - "neural": BERT-based semantic shift detection
        **kwargs: Method-specific parameters passed to the underlying chunker.

    Method-specific parameters:

    TokenChunker:
        - tokenizer: Tokenizer to use (default: "character")
        - chunk_size: Maximum tokens per chunk (default: 2048)
        - chunk_overlap: Overlap between chunks (default: 0)

    SentenceChunker:
        - tokenizer_or_token_counter: Tokenizer to use (default: "character")
        - chunk_size: Maximum tokens per chunk (default: 2048)
        - chunk_overlap: Overlap between chunks (default: 0)
        - min_sentences_per_chunk: Minimum sentences per chunk (default: 1)
        - min_characters_per_sentence: Minimum characters per sentence (default: 12)
        - delim: Delimiters to split sentences (default: ['.', '!', '?', '\n'])
        - include_delim: Include delimiters in chunk text (default: "prev")

    RecursiveChunker:
        - tokenizer_or_token_counter: Tokenizer to use (default: "character")
        - chunk_size: Maximum tokens per chunk (default: 2048)
        - rules: Recursive rules for chunking (default: RecursiveRules())
        - min_characters_per_chunk: Minimum characters per chunk (default: 24)

    SemanticChunker:
        - embedding_model: Model for embeddings (default: "minishlab/potion-base-8M")
        - threshold: Similarity threshold (default: "auto")
        - chunk_size: Maximum tokens per chunk (default: 2048)
        - min_sentences: Minimum sentences per chunk (default: 1)
        - similarity_window: Number of sentences for similarity calculation (default: 1)
        - delim: Delimiters to split sentences (default: ['.', '!', '?', '\n'])

    SlumberChunker:
        - embedding_model: Model for embeddings (default: "minishlab/potion-base-8M")
        - threshold: Similarity threshold (default: "auto")
        - chunk_size: Maximum tokens per chunk (default: 2048)
        - min_sentences: Minimum sentences per chunk (default: 1)
        - skip_window: Number of chunks to skip when looking for similarities (default: 1)

    LateChunker:
        - embedding_model: SentenceTransformer model (default: "all-MiniLM-L6-v2")
        - chunk_size: Maximum tokens per chunk (default: 2048)
        - rules: Recursive rules for chunking (default: RecursiveRules())
        - min_characters_per_chunk: Minimum characters per chunk (default: 24)

    CodeChunker:
        - language: Programming language (required)
        - tokenizer_or_token_counter: Tokenizer to use (default: "character")
        - chunk_size: Maximum tokens per chunk (default: 2048)
        - include_nodes: Include AST nodes in output (default: False)

    NeuralChunker:
        - model: BERT model for semantic shift detection (default: "mirth/chonky_modernbert_base_1")
        - device_map: Device to run the model (default: "cpu")
        - min_characters_per_chunk: Minimum characters per chunk (default: 10)

    Examples:
        >>> # Token-based chunking
        >>> chunker = ChonkieChunker(method="token", chunk_size=512, chunk_overlap=50)

        >>> # Semantic chunking with custom embedding model
        >>> chunker = ChonkieChunker(
        ...     method="semantic",
        ...     embedding_model="all-MiniLM-L6-v2",
        ...     threshold=0.7
        ... )

        >>> # Code chunking for Python files
        >>> chunker = ChonkieChunker(method="code", language="python", chunk_size=1024)
    """

    name = "chonkie"

    def __init__(self, **kwargs: dict):
        method = kwargs.get("method", "recursive")
        kwargs.pop("method", None)

        # Initialize the appropriate chunker based on method
        if method == "token":
            self.chunker = TokenChunker(**kwargs)
        elif method == "sentence":
            self.chunker = SentenceChunker(**kwargs)
        elif method == "recursive":
            self.chunker = RecursiveChunker(**kwargs)
        elif method == "semantic":
            self.chunker = SemanticChunker(**kwargs)
        elif method == "sdpm":
            self.chunker = SlumberChunker(**kwargs)
        elif method == "late":
            self.chunker = LateChunker(**kwargs)
        elif method == "code":
            self.chunker = CodeChunker(**kwargs)
        elif method == "neural":
            self.chunker = NeuralChunker(**kwargs)
        else:
            raise ValueError(
                f"Unknown chonker method: {method}. "
                f"Available methods: token, sentence, recursive, semantic, sdpm, late, code, neural"
            )
        super().__init__(**kwargs)

    def chunk(self, document: Document) -> Document:
        document.chunks = [chunk.text for chunk in self.chunker.chunk(document.content)]
        return document

    def chunk_batch(self, documents: list[Document]) -> list[Document]:
        for doc in documents:
            doc.chunks = [chunk.text for chunk in self.chunker.chunk(doc.content)]
        return documents


def init_chonker(chonker: str, **kwargs: dict):
    """Initialize a chonker by name using abstract base class discovery."""
    for cls in BaseChunker.__subclasses__():
        if hasattr(cls, "name") and cls.name == chonker:
            return cls(**kwargs)
    raise ValueError(
        f"Unknown chonker: {chonker}. Available: {[cls.name for cls in BaseChunker.__subclasses__() if hasattr(cls, 'name')]}"
    )
