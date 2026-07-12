from functools import lru_cache

from langchain_core.documents import Document
from sentence_transformers import CrossEncoder

from core.config import Settings, get_settings
from core.logger import get_logger

logger = get_logger(__name__)


@lru_cache
def _load_cross_encoder(model_name: str) -> CrossEncoder:
    logger.info(f"Loading cross-encoder reranker: {model_name}")
    return CrossEncoder(model_name)


class CrossEncoderReranker:
    """Rerank retrieved documents using a cross-encoder model."""

    def __init__(
        self,
        model_name: str | None = None,
        top_k: int | None = None,
        settings: Settings | None = None,
    ):
        self.settings = settings or get_settings()
        self.model_name = model_name or self.settings.reranker_model
        self.top_k = top_k or self.settings.rerank_top_k

    @property
    def model(self) -> CrossEncoder:
        return _load_cross_encoder(self.model_name)

    def rerank(self, query: str, documents: list[Document]) -> list[Document]:
        """Score and return the top-k most relevant documents."""
        if not documents:
            return []

        pairs = [(query, doc.page_content) for doc in documents]
        scores = self.model.predict(pairs)

        scored_docs = sorted(
            zip(documents, scores, strict=True), key=lambda x: x[1], reverse=True
        )
        logger.info(
            f"Reranked {len(documents)} documents; top score: {scored_docs[0][1]:.4f}"
        )
        return [doc for doc, _ in scored_docs[: self.top_k]]
