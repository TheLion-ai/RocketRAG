import sys
import tty
import termios
from .base import BaseVectorizer, BaseChunker, BaseLoader


def get_key():
    """Get a single keypress from the user."""
    fd = sys.stdin.fileno()
    old_settings = termios.tcgetattr(fd)
    try:
        tty.setraw(sys.stdin.fileno())
        key = sys.stdin.read(1)
        if key == "\x1b":  # ESC sequence
            key += sys.stdin.read(2)
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
    return key


def construct_metadata_dict(
    data_dir: str,
    chonker: BaseChunker,
    chonker_args: dict,
    vectorizer: BaseVectorizer,
    vectorizer_args: dict,
    loader: BaseLoader,
    loader_args: dict,
    db_path: str = "rag.db",
    collection_name: str = "rag",
):
    return {
        "data_dir": data_dir,
        "chonker": chonker.__class__.__name__,
        "chonker_args": chonker_args,
        "vectorizer": vectorizer.__class__.__name__,
        "vectorizer_args": vectorizer_args,
        "loader": loader.__class__.__name__,
        "loader_args": loader_args,
        "db_path": db_path,
        "collection_name": collection_name,
    }


def compare_medatata_dicts(db: dict, new: dict) -> dict:
    """Check if the two dictionaries have the same values and finds where they differ"""
    diff = {}
    for key in db.keys():
        if db[key] != new[key]:
            diff[key] = {
                "db": db[key],
                "new": new[key],
            }
    return diff
