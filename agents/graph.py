from typing import Annotated, list, TypedDict

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langgraph.graph import END, StateGraph

from agents.graders import grade_answer, grade_documents, grade_hallucination
from agents.tools import web_search
from core.config import Settings, get_settings
from core.logger import get_logger
from generation.llm import get_llm
from generation.prompts import get_generation_prompt
from retrieval.query_transform import rewrite_query
from retrieval.retriever import HybridRetriever
from retrieval.reranker import CrossEncoderReranker

logger = get_logger(__name__)


class AgentState(TypedDict):
    question: str
    rewritten_question: str
    generation: str
    documents: list[Document]
    web_results: list[Document]
    steps: Annotated[list[str], lambda x, y: x + y]


def route_question(state: AgentState, settings: Settings | None = None) -> str:
    """Route question to web search or RAG."""
    settings = settings or get_settings()
    question = state["question"]
    logger.info(f"Routing question: {question}")

    # Simple heuristic: if question contains "current" or "latest" or is very broad,
    # prefer web search. Otherwise retrieve from documents.
    web_keywords = ["current", "latest", "today", "news", "recent", "2024", "2025"]
    if any(kw in question.lower() for kw in web_keywords):
        logger.info("Routing to web search")
        return "web_search"
    logger.info("Routing to document retrieval")
    return "retrieve"


def retrieve(state: AgentState, vector_store: Chroma, settings: Settings | None = None) -> AgentState:
    """Retrieve documents using hybrid search + reranking."""
    settings = settings or get_settings()
    question = state["question"]
    logger.info(f"Retrieving for: {question}")

    rewritten = rewrite_query(question, settings)
    retriever = HybridRetriever(vector_store, settings=settings)
    docs = retriever.retrieve_multi_query(rewritten)

    reranker = CrossEncoderReranker(settings=settings)
    docs = reranker.rerank(rewritten, docs)

    return {
        **state,
        "rewritten_question": rewritten,
        "documents": docs,
        "steps": ["retrieve"],
    }


def grade_documents_node(state: AgentState, settings: Settings | None = None) -> str:
    """Decide if retrieved documents are relevant enough."""
    settings = settings or get_settings()
    question = state["rewritten_question"] or state["question"]
    documents = state["documents"]

    if not documents:
        logger.warning("No documents retrieved")
        return "transform_query"

    decision = grade_documents(question, documents, settings)
    if decision == "yes":
        logger.info("Retrieved documents are relevant")
        return "generate"
    logger.info("Retrieved documents not relevant; will transform query")
    return "transform_query"


def transform_query(state: AgentState, settings: Settings | None = None) -> AgentState:
    """Rewrite the query for better retrieval."""
    settings = settings or get_settings()
    question = state["question"]
    logger.info(f"Transforming query: {question}")

    prompt = """You are generating a question that will be used to retrieve documents for a RAG system.
Look at the input and try to reason about the underlying semantic intent / meaning.
Output only the improved question.

Initial question: {question}

Improved question:"""
    model = get_llm(settings.llm_model)
    response = model.invoke(prompt.format(question=question))
    improved = response.content.strip()

    return {
        **state,
        "rewritten_question": improved,
        "steps": state["steps"] + ["transform_query"],
    }


def web_search_node(state: AgentState, settings: Settings | None = None) -> AgentState:
    """Fallback to web search."""
    settings = settings or get_settings()
    question = state["rewritten_question"] or state["question"]
    results = web_search(question, settings)
    return {
        **state,
        "web_results": results,
        "documents": results,  # Use web results as context
        "steps": state["steps"] + ["web_search"],
    }


def generate(state: AgentState, settings: Settings | None = None) -> AgentState:
    """Generate an answer with citations."""
    settings = settings or get_settings()
    question = state["question"]
    documents = state["documents"] + state.get("web_results", [])

    prompt = get_generation_prompt()
    model = get_llm(settings.llm_model)
    chain = prompt | model

    context = "\n\n---\n\n".join(
        f"[Source {i+1}] {format_source(doc)}\n{doc.page_content}"
        for i, doc in enumerate(documents)
    )

    response = chain.invoke({"question": question, "context": context})
    generation = response.content

    return {
        **state,
        "generation": generation,
        "steps": state["steps"] + ["generate"],
    }


def grade_generation(state: AgentState, settings: Settings | None = None) -> str:
    """Grade the generation for hallucinations and usefulness."""
    settings = settings or get_settings()
    question = state["question"]
    documents = state["documents"]
    generation = state["generation"]

    hallucination = grade_hallucination(documents, generation, settings)
    if hallucination == "no":
        logger.warning("Hallucination detected; regenerating")
        return "not_supported"

    usefulness = grade_answer(question, generation, settings)
    if usefulness == "no":
        logger.warning("Answer not useful; transforming query")
        return "not_useful"

    logger.info("Generation passes grading")
    return "useful"


def format_source(doc: Document) -> str:
    source = doc.metadata.get("source", "unknown")
    page = doc.metadata.get("page")
    url = doc.metadata.get("url")
    if url:
        return f"Web: {url}"
    if page:
        return f"Document: {source}, Page {page}"
    return f"Document: {source}"


def build_agent(vector_store: Chroma, settings: Settings | None = None):
    """Build and compile the CRAG agent graph."""
    settings = settings or get_settings()

    workflow = StateGraph(AgentState)

    workflow.add_node("retrieve", lambda state: retrieve(state, vector_store, settings))
    workflow.add_node("transform_query", lambda state: transform_query(state, settings))
    workflow.add_node("web_search", lambda state: web_search_node(state, settings))
    workflow.add_node("generate", lambda state: generate(state, settings))

    workflow.set_conditional_entry_point(
        lambda state: route_question(state, settings),
        {
            "retrieve": "retrieve",
            "web_search": "web_search",
        },
    )

    workflow.add_conditional_edges(
        "retrieve",
        lambda state: grade_documents_node(state, settings),
        {
            "generate": "generate",
            "transform_query": "transform_query",
        },
    )

    workflow.add_edge("transform_query", "retrieve")
    workflow.add_edge("web_search", "generate")

    workflow.add_conditional_edges(
        "generate",
        lambda state: grade_generation(state, settings),
        {
            "not_supported": "generate",
            "not_useful": "transform_query",
            "useful": END,
        },
    )

    return workflow.compile()
