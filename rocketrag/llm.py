from contextlib import redirect_stderr
from io import StringIO
from llama_cpp import Llama
from rich.console import Console
from .base import BaseLLM

console = Console()


context_message_template = """
Question: {question}

Here are some context documents that might be relevant
{context}
"""

system_message = """
You are a helpful assistant. Answer the user question.
"""


class LLamaLLM(BaseLLM):
    def __init__(self, **kwargs):
        self.repo_id = kwargs.get("repo_id")
        self.filename = kwargs.get("filename")
        self.n_ctx = kwargs.get("n_ctx", 4096)
        self.verbose = kwargs.get("verbose", False)
        self.context_message_template = context_message_template
        self.system_message = system_message
        super().__init__(**kwargs)

    def load(self):
        with console.status("Loading model...") as status:
            if self.verbose:
                self.llm = Llama.from_pretrained(
                    repo_id=self.repo_id,
                    filename=self.filename,
                    n_ctx=self.n_ctx,
                    verbose=True,
                )
            else:
                stderr_buffer = StringIO()
                with redirect_stderr(stderr_buffer):
                    self.llm = Llama.from_pretrained(
                        repo_id=self.repo_id,
                        filename=self.filename,
                        n_ctx=self.n_ctx,
                        verbose=False,
                    )
        status.update("Model loaded")

    def run(self, messages: list[dict]) -> str:
        """Run the model and return the complete response."""
        response = self.llm.create_chat_completion(
            messages=messages,
            stream=False,
        )
        return response["choices"][0]["message"]["content"]

    def stream(self, messages: list[dict]):
        """Stream the model response. Returns a generator of response chunks."""
        return self.llm.create_chat_completion(
            messages=messages,
            stream=True,
        )


def init_llm(llm_type: str, **kwargs: dict):
    """Initialize an LLM by type using abstract base class discovery."""
    # For now, we only have LLamaLLM, but this allows for future extensions
    if llm_type == "llama":
        return LLamaLLM(**kwargs)

    for cls in BaseLLM.__subclasses__():
        if hasattr(cls, "name") and cls.name == llm_type:
            return cls(**kwargs)
    raise ValueError(
        f"Unknown LLM type: {llm_type}. Available: ['llama'] + {[cls.name for cls in BaseLLM.__subclasses__() if hasattr(cls, 'name')]}"
    )
