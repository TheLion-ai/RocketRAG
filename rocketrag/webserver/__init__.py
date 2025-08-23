from fastapi import FastAPI
from .routes import setup_routes
from ..rocketrag import RocketRAG
from .models import (
    QuestionRequest,
    QuestionResponse,
    ChatMessage,
    ChatCompletionRequest,
    ChatCompletionChoice,
    ChatCompletionUsage,
    ChatCompletionResponse,
    ChatCompletionStreamChoice,
    ChatCompletionStreamResponse,
)
import uvicorn

# Export all public symbols
__all__ = [
    "create_app",
    "start_server",
    "QuestionRequest",
    "QuestionResponse",
    "ChatMessage",
    "ChatCompletionRequest",
    "ChatCompletionChoice",
    "ChatCompletionUsage",
    "ChatCompletionResponse",
    "ChatCompletionStreamChoice",
    "ChatCompletionStreamResponse",
]


def create_app(rocket_rag: RocketRAG):
    """Create and configure the FastAPI application"""
    rocket_rag.llm.load()
    app = FastAPI(title="rocketrag API", version="1.0.0")

    # Setup all routes
    setup_routes(app, rocket_rag)

    return app


def start_server(
    rocket_rag: RocketRAG,
    port: int = 8000,
    host: str = "127.0.0.1",
):
    """Start the webserver with the given RocketRAG instance"""
    app = create_app(rocket_rag)
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    start_server()
