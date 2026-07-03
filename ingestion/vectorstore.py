from pathlib import Path
from typing import List

from langchain_chroma import Chroma
from langchain_core.documents import Document

from core.config import Settings, get_settings
from core.logger import get_logger
from generation.llm import get_embeddings

logger = get_logger(__name__)


def get_vector_store(
    collection_name: str, settings: Settings | None = None
) -> Chroma:
    """Return a persistent Chroma vector store for a given collection."""
    settings = settings or get_settings()
    embeddings = get_embeddings(settings.embedding_provider, settings.embedding_model)
    logger.info(f"Loading vector store collection: {collection_name}")
    return Chroma(
        collection_name=collection_name,
        persist_directory=str(settings.chroma_persist_dir),
        embedding_function=embeddings,
    )


def create_collection(
    documents: List[Document],
    collection_name: str,
    settings: Settings | None = None,
) -> Chroma:
    """Create a new Chroma collection from documents."""
    settings = settings or get_settings()
    embeddings = get_embeddings(settings.embedding_provider, settings.embedding_model)
    logger.info(f"Creating collection '{collection_name}' with {len(documents)} documents")
    return Chroma.from_documents(
        documents=documents,
        embedding=embeddings,
        persist_directory=str(settings.chroma_persist_dir),
        collection_name=collection_name,
    )


def delete_collection(collection_name: str, settings: Settings | None = None) -> None:
    """Delete a Chroma collection."""
    settings = settings or get_settings()
    logger.info(f"Deleting collection: {collection_name}")
    store = Chroma(
        collection_name=collection_name,
        persist_directory=str(settings.chroma_persist_dir),
        embedding_function=get_embeddings(settings.embedding_provider, settings.embedding_model),
    )
    store.delete_collection()


def list_collections(settings: Settings | None = None) -> List[str]:
    """List all existing collection names."""
    settings = settings or get_settings()
    client = Chroma(
        persist_directory=str(settings.chroma_persist_dir),
        embedding_function=get_embeddings(settings.embedding_provider, settings.embedding_model),
    )
    return client._client.list_collections()
