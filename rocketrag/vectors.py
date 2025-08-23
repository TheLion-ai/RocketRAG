from sentence_transformers import SentenceTransformer
from .base import BaseVectorizer


class SentenceTransformersVectorizer(BaseVectorizer):
    name = "sentence_transformers"

    def __init__(self, **kwargs):
        self.model_name = kwargs.get("model_name")
        self.model = SentenceTransformer(self.model_name)
        self.batch_size = kwargs.get("batch_size", 16)
        self.dimension = self.model.get_sentence_embedding_dimension()
        super().__init__(**kwargs)

    def vectorize(self, text: str):
        return self.model.encode(text)

    def vectorize_batch(self, texts: list[str]):
        return self.model.encode(texts, batch_size=self.batch_size)


def init_vectorizer(vectorizer: str, **kwargs: dict):
    """Initialize a vectorizer by name using abstract base class discovery."""
    for cls in BaseVectorizer.__subclasses__():
        if hasattr(cls, "name") and cls.name == vectorizer:
            return cls(**kwargs)
    raise ValueError(
        f"Unknown vectorizer: {vectorizer}. Available: {[cls.name for cls in BaseVectorizer.__subclasses__() if hasattr(cls, 'name')]}"
    )
