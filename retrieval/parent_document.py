 

from langchain_core.documents import Document

from core.logger import get_logger
from retrieval.retriever import HybridRetriever

logger = get_logger(__name__)


class ParentDocumentRetriever:
    """Retrieve child chunks but return parent documents for richer context."""

    def __init__(
        self,
        retriever: HybridRetriever,
        parent_map: dict[str, Document],
    ):
        self.retriever = retriever
        self.parent_map = parent_map

    def retrieve(self, query: str) -> list[Document]:
        """Retrieve chunks, map to parents, and return unique parent docs."""
        logger.info(f"Parent-document retrieval for: {query}")
        child_docs = self.retriever.retrieve(query)
        parent_ids = set()
        parents = []

        for doc in child_docs:
            parent_id = doc.metadata.get("parent_id")
            if parent_id and parent_id in self.parent_map:
                if parent_id not in parent_ids:
                    parent_ids.add(parent_id)
                    parents.append(self.parent_map[parent_id])
            else:
                # Fallback: return child doc if parent not tracked
                parents.append(doc)

        logger.info(f"Resolved {len(child_docs)} chunks to {len(parents)} parent docs")
        return parents

    def retrieve_multi_query(self, query: str, n_variations: int = 3) -> list[Document]:
        """Multi-query variant that returns parent documents."""
        logger.info(f"Parent-document multi-query retrieval for: {query}")
        child_docs = self.retriever.retrieve_multi_query(query, n_variations)
        parent_ids = set()
        parents = []

        for doc in child_docs:
            parent_id = doc.metadata.get("parent_id")
            if parent_id and parent_id in self.parent_map:
                if parent_id not in parent_ids:
                    parent_ids.add(parent_id)
                    parents.append(self.parent_map[parent_id])
            else:
                parents.append(doc)

        return parents
