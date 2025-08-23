from .db import MilvusLiteDB
from .llm import LLamaLLM
from rich import print
from .data_models import SearchResult


class RAG:
    def __init__(self, db: MilvusLiteDB, llm: LLamaLLM):
        self.db = db
        self.llm = llm
        self.context_window = self.llm.n_ctx
        self.response_buffer = 100  # TODO: get form llm config

    def construct_context_message(self, question: str, docs: list[str]) -> str:
        base_template = self.llm.context_message_template.format(
            question=question, context=""
        )
        base_tokens = len(self.llm.llm.tokenize(base_template.encode("utf-8")))

        available_tokens = self.context_window - base_tokens - self.response_buffer

        context_parts = []
        current_tokens = 0

        for i, doc in enumerate(docs):
            doc_text = f"DOCUMENT{i}\n{doc}"
            doc_tokens = len(self.llm.llm.tokenize(doc_text.encode("utf-8")))

            # Check if adding this document would exceed the token limit
            if current_tokens + doc_tokens <= available_tokens:
                context_parts.append(doc_text)
                current_tokens += doc_tokens
            else:
                if not context_parts:
                    words = doc.split()
                    truncated_doc = ""
                    for word in words:
                        test_doc = f"DOCUMENT{i}\n{truncated_doc} {word}".strip()
                        test_tokens = len(
                            self.llm.llm.tokenize(test_doc.encode("utf-8"))
                        )
                        if test_tokens <= available_tokens:
                            truncated_doc = f"{truncated_doc} {word}".strip()
                        else:
                            break
                    if truncated_doc:
                        context_parts.append(f"DOCUMENT{i}\n{truncated_doc}")
                break

        context = "\n".join(context_parts)
        final_message = self.llm.context_message_template.format(
            question=question, context=context
        )

        # Optional: Print debug information about token usage
        final_tokens = len(self.llm.llm.tokenize(final_message.encode("utf-8")))
        print(
            f"Debug: Using {len(context_parts)} documents, {final_tokens}/{self.context_window} tokens"
        )

        return final_message

    def run(self, question: str) -> tuple[str, list[SearchResult]]:
        """Run RAG system and return complete answer."""
        search_results = self.db.search(question, top_k=5)
        docs = [doc.chunk for doc in search_results]
        context_message = self.construct_context_message(question, docs)
        messages = [
            {"role": "system", "content": self.llm.system_message},
            {"role": "user", "content": context_message},
        ]
        return self.llm.run(messages), search_results

    def stream(self, question: str) -> tuple[str, list[SearchResult]]:
        """Stream RAG system response. Returns a generator of response chunks."""
        search_results = self.db.search(question, top_k=5)
        docs = [doc.chunk for doc in search_results]
        context_message = self.construct_context_message(question, docs)
        messages = [
            {"role": "system", "content": self.llm.system_message},
            {"role": "user", "content": context_message},
        ]
        return self.llm.stream(messages), search_results

    def run_with_display(self, question: str):
        """Run RAG system with CLI display (legacy method for CLI compatibility)."""
        context_message = self.construct_context_message(question)
        messages = [
            {"role": "system", "content": self.llm.system_message},
            {"role": "user", "content": context_message},
        ]
        return self.llm.stream_answer(messages)
