import operator
from typing import Annotated, TypedDict

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
from retrieval.reranker import CrossEncoderReranker
from retrieval.retriever import HybridRetriever

logger = get_logger(__name__)


class AgentState(TypedDict):
    question: str
    chat_history: list[dict]
    rewritten_question: str
    generation: str
    documents: list[Document]
    retries: int
    steps: Annotated[list[str], operator.add]


def retrieve(
    state: AgentState,
    retriever: HybridRetriever,
    reranker: CrossEncoderReranker,
    settings: Settings,
) -> dict:
    """Retrieve documents using hybrid search + reranking."""
    question = state["question"]
    logger.info(f"Retrieving for: {question}")

    # Use the query produced by transform_query if we looped back; otherwise
    # rewrite the raw question into a standalone query using chat history.
    rewritten = state.get("rewritten_question") or rewrite_query(
        question, state.get("chat_history", []), settings
    )
    docs = retriever.retrieve_multi_query(rewritten)
    docs = reranker.rerank(rewritten, docs)

    return {
        "rewritten_question": rewritten,
        "documents": docs,
        "steps": ["retrieve"],
    }


def grade_documents_node(state: AgentState, settings: Settings) -> str:
    """Decide if retrieved documents are relevant enough."""
    question = state["rewritten_question"] or state["question"]
    documents = state["documents"]

    if documents and grade_documents(question, documents, settings) == "yes":
        logger.info("Retrieved documents are relevant")
        return "generate"

    if state.get("retries", 0) >= settings.max_retries:
        if settings.enable_web_search:
            logger.warning("Query transform budget exhausted; falling back to web search")
            return "web_search"
        logger.warning("Query transform budget exhausted; generating best-effort answer from documents")
        return "generate"

    logger.info("Retrieved documents not relevant; will transform query")
    return "transform_query"


def transform_query(state: AgentState, settings: Settings) -> dict:
    """Rewrite the query for better retrieval."""
    question = state["question"]
    logger.info(f"Transforming query: {question}")

    prompt = """You are generating a question that will be used to retrieve documents for a RAG system.
Look at the input and try to reason about the underlying semantic intent / meaning.
Output only the improved question.

Initial question: {question}

Improved question:"""
    model = get_llm(settings.llm_provider, settings.llm_model)
    response = model.invoke(prompt.format(question=question))
    improved = response.content.strip()

    return {
        "rewritten_question": improved,
        "retries": state.get("retries", 0) + 1,
        "steps": ["transform_query"],
    }


def web_search_node(state: AgentState, settings: Settings) -> dict:
    """Fallback to web search."""
    question = state["rewritten_question"] or state["question"]
    results = web_search(question, settings)
    return {
        "documents": results,
        "steps": ["web_search"],
    }


def generate(state: AgentState, settings: Settings) -> dict:
    """Generate an answer with citations."""
    question = state["question"]
    chat_history = state.get("chat_history", [])
    documents = state["documents"]

    history_str = ""
    if chat_history:
        history_str = "\n".join(
            f"{'User' if msg['role'] == 'user' else 'Assistant'}: {msg['content']}"
            for msg in chat_history
        )

    context = "\n\n---\n\n".join(
        f"[Source {i+1}] {format_source(doc)}\n{doc.page_content}"
        for i, doc in enumerate(documents)
    )

    model = get_llm(settings.llm_provider, settings.llm_model)
    chain = get_generation_prompt() | model
    response = chain.invoke({
        "question": question,
        "context": context,
        "chat_history": history_str,
    })

    return {
        "generation": response.content,
        "steps": ["generate"],
    }


def grade_generation(state: AgentState, settings: Settings) -> str:
    """Grade the generation for hallucinations and usefulness."""
    question = state["question"]
    documents = state["documents"]
    generation = state["generation"]
    generations = state["steps"].count("generate")

    hallucination = grade_hallucination(documents, generation, settings)
    if hallucination == "no":
        if generations > settings.max_retries:
            logger.warning("Regeneration budget exhausted; returning best-effort answer")
            return "useful"
        logger.warning("Hallucination detected; regenerating")
        return "not_supported"

    usefulness = grade_answer(question, generation, settings)
    if usefulness == "no":
        if state.get("retries", 0) >= settings.max_retries:
            logger.warning("Query transform budget exhausted; returning best-effort answer")
            return "useful"
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
    if page is not None:
        return f"Document: {source}, Page {page}"
    return f"Document: {source}"


def build_agent(vector_store: Chroma, settings: Settings | None = None):
    """Build and compile the CRAG agent graph."""
    settings = settings or get_settings()

    # Built once per agent so the BM25 index and cross-encoder are reused
    # across questions instead of being rebuilt on every retrieval.
    retriever = HybridRetriever(vector_store, settings=settings)
    reranker = CrossEncoderReranker(settings=settings)

    workflow = StateGraph(AgentState)

    workflow.add_node("retrieve", lambda state: retrieve(state, retriever, reranker, settings))
    workflow.add_node("transform_query", lambda state: transform_query(state, settings))
    workflow.add_node("web_search", lambda state: web_search_node(state, settings))
    workflow.add_node("generate", lambda state: generate(state, settings))

    workflow.set_entry_point("retrieve")

    workflow.add_conditional_edges(
        "retrieve",
        lambda state: grade_documents_node(state, settings),
        {
            "generate": "generate",
            "transform_query": "transform_query",
            "web_search": "web_search",
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
