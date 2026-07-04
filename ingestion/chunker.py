 

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


def split_documents_with_parent(
    documents: list[Document], settings: Settings | None = None
) -> tuple[list[Document], dict[str, Document]]:
    """
    Split documents into chunks and track their parent documents.
    Returns (chunks, parent_map).
    """
    settings = settings or get_settings()
    splitter = get_text_splitter(settings)
    parent_map: dict[str, Document] = {}
    chunks: list[Document] = []

    for parent_idx, parent in enumerate(documents):
        parent_id = f"parent_{parent_idx}"
        parent_map[parent_id] = parent
        local_chunks = splitter.split_documents([parent])
        for chunk in local_chunks:
            chunk.metadata["parent_id"] = parent_id
        chunks.extend(local_chunks)

    return chunks, parent_map
