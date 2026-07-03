from typing import List

from langchain_core.output_parsers import CommaSeparatedListOutputParser
from langchain_core.prompts import ChatPromptTemplate

from core.config import Settings, get_settings
from core.logger import get_logger
from generation.llm import get_llm

logger = get_logger(__name__)


def rewrite_query(query: str, settings: Settings | None = None) -> str:
    """Rewrite a user query into a standalone, retrieval-friendly query."""
    logger.info(f"Rewriting query: {query}")
    prompt = ChatPromptTemplate.from_template(
        """You are an expert at converting user questions into search queries.
Given the following user question, rewrite it as a clear, standalone search query
that captures the user's intent and can be used to retrieve relevant documents.
Do not answer the question. Only output the rewritten query.

User question: {question}

Rewritten query:"""
    )
    model = get_llm(settings.llm_model if settings else None)
    chain = prompt | model
    result = chain.invoke({"question": query})
    rewritten = result.content.strip()
    logger.info(f"Rewritten query: {rewritten}")
    return rewritten


def generate_multi_queries(query: str, n: int = 3, settings: Settings | None = None) -> List[str]:
    """Generate multiple query variations for better recall."""
    logger.info(f"Generating multi-queries for: {query}")
    prompt = ChatPromptTemplate.from_template(
        """You are an expert at information retrieval.
Generate {n} different versions of the given user question to retrieve relevant documents from a vector database.
The questions should be semantically equivalent but phrased differently.
Output only the questions, separated by commas.

Original question: {question}

Generated questions:"""
    )
    model = get_llm(settings.llm_model if settings else None)
    parser = CommaSeparatedListOutputParser()
    chain = prompt | model | parser
    questions = chain.invoke({"question": query, "n": n})
    questions = [q.strip() for q in questions if q.strip()]
    logger.info(f"Generated questions: {questions}")
    return questions
