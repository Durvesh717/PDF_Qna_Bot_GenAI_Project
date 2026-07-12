from langchain_chroma import Chroma
from langchain_core.documents import Document

from core.config import Settings, get_settings
from core.logger import get_logger
from generation.llm import get_embeddings

logger = get_logger(__name__)


def _get_client(settings: Settings):
    import chromadb

    return chromadb.PersistentClient(path=str(settings.chroma_persist_dir))


def get_vector_store(
    collection_name: str, settings: Settings | None = None
) -> Chroma:
    """Return a persistent Chroma vector store for a given collection."""
    settings = settings or get_settings()

    # Prefer the embedding model the collection was created with; embedding
    # with a different model would produce a dimension mismatch at query time.
    provider = settings.embedding_provider
    model = settings.embedding_model
    try:
        metadata = _get_client(settings).get_collection(collection_name).metadata or {}
        provider = metadata.get("embedding_provider", provider)
        model = metadata.get("embedding_model", model)
        if (provider, model) != (settings.embedding_provider, settings.embedding_model):
            logger.info(
                f"Collection '{collection_name}' was created with {provider}/{model}; "
                "using that embedding model instead of the configured one"
            )
    except Exception:
        pass

    embeddings = get_embeddings(provider, model)
    logger.info(f"Loading vector store collection: {collection_name}")
    return Chroma(
        collection_name=collection_name,
        persist_directory=str(settings.chroma_persist_dir),
        embedding_function=embeddings,
    )


def create_collection(
    documents: list[Document],
    collection_name: str,
    settings: Settings | None = None,
) -> Chroma:
    """Create a new Chroma collection from documents, replacing any existing one."""
    settings = settings or get_settings()
    embeddings = get_embeddings(settings.embedding_provider, settings.embedding_model)

    # from_documents appends into an existing collection, which would duplicate
    # every chunk on re-processing — drop the old collection first.
    try:
        _get_client(settings).delete_collection(name=collection_name)
        logger.info(f"Replaced existing collection: {collection_name}")
    except Exception:
        pass

    logger.info(f"Creating collection '{collection_name}' with {len(documents)} documents")
    return Chroma.from_documents(
        documents=documents,
        embedding=embeddings,
        persist_directory=str(settings.chroma_persist_dir),
        collection_name=collection_name,
        collection_metadata={
            "embedding_provider": settings.embedding_provider,
            "embedding_model": settings.embedding_model,
        },
    )


def delete_collection(collection_name: str, settings: Settings | None = None) -> None:
    """Delete a Chroma collection without requiring embeddings."""
    settings = settings or get_settings()
    logger.info(f"Deleting collection: {collection_name}")
    _get_client(settings).delete_collection(name=collection_name)


def list_collections(settings: Settings | None = None) -> list[str]:
    """List all existing collection names without requiring embeddings."""
    settings = settings or get_settings()
    return [c.name for c in _get_client(settings).list_collections()]
