from rich.console import Console
from rich.markdown import Markdown
from rich.live import Live
from rich.panel import Panel
from rich.layout import Layout

from .data_models import SearchResult

console = Console()


def display_streaming_answer(question: str, stream, search_results: list[SearchResult]):
    """Display streaming answer with rich layout for CLI."""
    # Create layout with question and answer panels

    sources_markdown = "\n".join(
        [f"- {doc.filename}: {doc.chunk[:150]}..." for doc in search_results]
    )
    layout = Layout()
    layout.split_column(
        Layout(
            Panel(Markdown(question), title="Question", border_style="blue"), size=3
        ),
        Layout(
            Panel(Markdown(""), title="Answer", border_style="green"), name="answer"
        ),
        Layout(
            Panel(Markdown(sources_markdown), title="Sources", border_style="yellow"),
            name="sources",
        ),
    )

    full_response = ""
    with Live(layout, console=console, refresh_per_second=20):
        for output in stream:
            delta = output["choices"][0]["delta"]
            if "content" in delta:
                content = delta["content"]
                full_response += content
                layout["answer"].update(
                    Panel(
                        Markdown(full_response),
                        title="Answer",
                        border_style="green",
                    )
                )

    return full_response


def extract_question_from_context(content: str) -> str:
    """Extract question from context message template."""
    if "Question: " in content:
        question_start = content.find("Question: ") + len("Question: ")
        question_end = content.find("\n", question_start)
        if question_end != -1:
            return content[question_start:question_end].strip()
        else:
            return content[question_start:].strip()
    return ""
