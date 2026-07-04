 

from langchain_community.tools import DuckDuckGoSearchResults
from langchain_core.documents import Document

from core.config import Settings, get_settings
from core.logger import get_logger

logger = get_logger(__name__)


def web_search(query: str, settings: Settings | None = None) -> list[Document]:
    """Search the web and return results as Documents."""
    settings = settings or get_settings()
    logger.info(f"Performing web search fallback for: {query}")

    try:
        # Prefer Tavily if available, otherwise DuckDuckGo
        if settings.tavily_api_key:
            from langchain_community.tools import TavilySearchResults

            search = TavilySearchResults(
                tavily_api_key=settings.tavily_api_key,
                max_results=5,
            )
            results = search.invoke({"query": query})
        else:
            search = DuckDuckGoSearchResults(num_results=5)
            results = search.run(query)

        documents = []
        if isinstance(results, str):
            # DuckDuckGo sometimes returns a string; treat as one doc
            documents.append(
                Document(
                    page_content=results,
                    metadata={"source": "web_search", "url": "duckduckgo"},
                )
            )
        elif isinstance(results, list):
            for idx, item in enumerate(results):
                content = item.get("content", item.get("snippet", str(item)))
                url = item.get("url", item.get("link", "unknown"))
                documents.append(
                    Document(
                        page_content=content,
                        metadata={"source": "web_search", "url": url, "index": idx},
                    )
                )

        logger.info(f"Web search returned {len(documents)} results")
        return documents
    except Exception as e:
        logger.exception(f"Web search failed: {e}")
        return []
