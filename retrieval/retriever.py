from langchain_chroma import Chroma
from langchain_core.documents import Document
from rank_bm25 import BM25Okapi

from core.config import Settings, get_settings
from core.logger import get_logger
from retrieval.query_transform import generate_multi_queries

logger = get_logger(__name__)


def _tokenize(text: str) -> list[str]:
    """Simple whitespace tokenizer for BM25."""
    return text.lower().split()


class HybridRetriever:
    """Combines dense vector search with BM25 keyword search using Reciprocal Rank Fusion."""

    def __init__(
        self,
        vector_store: Chroma,
        top_k: int | None = None,
        settings: Settings | None = None,
    ):
        self.settings = settings or get_settings()
        self.vector_store = vector_store
        self.top_k = top_k or self.settings.top_k
        self._documents: list[Document] = []
        self._bm25: BM25Okapi | None = None
        self._build_bm25()

    def _build_bm25(self) -> None:
        """Build BM25 index from all documents in the vector store."""
        try:
            # Fetch the raw corpus directly; unlike similarity_search this
            # needs no embedding call (embedding "" fails on some providers).
            data = self.vector_store.get()
            self._documents = [
                Document(page_content=text, metadata=metadata or {})
                for text, metadata in zip(data["documents"], data["metadatas"], strict=True)
            ]
        except Exception:
            logger.exception("Failed to load corpus for BM25 index")
            self._documents = []

        if not self._documents:
            self._bm25 = None
            return

        logger.info(f"Building BM25 index over {len(self._documents)} documents")
        tokenized = [_tokenize(doc.page_content) for doc in self._documents]
        self._bm25 = BM25Okapi(tokenized)

    def _dense_search(self, query: str, k: int) -> list[Document]:
        return self.vector_store.similarity_search(query, k=k)

    def _sparse_search(self, query: str, k: int) -> list[Document]:
        if not self._bm25 or not self._documents:
            return []
        scores = self._bm25.get_scores(_tokenize(query))
        top_indices = sorted(
            range(len(scores)), key=lambda i: scores[i], reverse=True
        )[:k]
        return [self._documents[i] for i in top_indices]

    @staticmethod
    def _reciprocal_rank_fusion(
        results_lists: list[list[Document]], k: int = 60
    ) -> list[Document]:
        """Fuse multiple ranked lists using RRF."""
        scores: dict[tuple, float] = {}
        doc_map: dict[tuple, Document] = {}

        for results in results_lists:
            for rank, doc in enumerate(results):
                doc_id = (
                    doc.metadata.get("source"),
                    doc.metadata.get("page"),
                    doc.page_content[:100],
                )
                doc_map[doc_id] = doc
                scores[doc_id] = scores.get(doc_id, 0) + 1 / (k + rank + 1)

        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return [doc_map[doc_id] for doc_id, _ in ranked]

    def retrieve(self, query: str) -> list[Document]:
        """Retrieve documents using hybrid search + RRF fusion."""
        logger.info(f"Hybrid retrieval for query: {query}")
        dense = self._dense_search(query, self.top_k)
        sparse = self._sparse_search(query, self.top_k)
        fused = self._reciprocal_rank_fusion([dense, sparse])
        logger.info(f"Retrieved {len(fused)} documents after fusion")
        return fused

    def retrieve_multi_query(
        self, query: str, n_variations: int = 3
    ) -> list[Document]:
        """Retrieve using multiple query variations and fuse results."""
        logger.info(f"Multi-query retrieval for: {query}")
        queries = [query] + generate_multi_queries(query, n=n_variations, settings=self.settings)
        all_results = []
        for q in queries:
            dense = self._dense_search(q, self.top_k)
            sparse = self._sparse_search(q, self.top_k)
            all_results.extend([dense, sparse])

        fused = self._reciprocal_rank_fusion(all_results)
        logger.info(
            f"Multi-query retrieval returned {len(fused)} documents from {len(queries)} queries"
        )
        return fused
