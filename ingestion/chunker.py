from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from core.config import Settings, get_settings


def get_text_splitter(settings: Settings | None = None) -> RecursiveCharacterTextSplitter:
    """Return the default recursive text splitter."""
    settings = settings or get_settings()
    return RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )


def split_documents(
    documents: list[Document], settings: Settings | None = None
) -> list[Document]:
    """Split documents into chunks while preserving metadata."""
    settings = settings or get_settings()
    splitter = get_text_splitter(settings)
    return splitter.split_documents(documents)
