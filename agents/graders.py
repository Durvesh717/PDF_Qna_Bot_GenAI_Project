from typing import Literal

from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

from core.config import Settings, get_settings
from core.logger import get_logger
from generation.llm import get_llm

logger = get_logger(__name__)


class GradeDocuments(BaseModel):
    """Binary score for relevance of retrieved documents to a question."""

    binary_score: Literal["yes", "no"] = Field(
        description="Documents are relevant to the question, 'yes' or 'no'"
    )


class GradeHallucinations(BaseModel):
    """Binary score for whether the answer is grounded in facts."""

    binary_score: Literal["yes", "no"] = Field(
        description="Answer is grounded in the facts, 'yes' or 'no'"
    )


class GradeAnswer(BaseModel):
    """Binary score for whether the answer addresses the question."""

    binary_score: Literal["yes", "no"] = Field(
        description="Answer addresses the question, 'yes' or 'no'"
    )


def grade_documents(question: str, documents: list, settings: Settings | None = None) -> str:
    """Grade whether retrieved documents are relevant to the question."""
    logger.info("Grading document relevance")
    settings = settings or get_settings()
    model = get_llm(settings.llm_provider, settings.llm_model).with_structured_output(GradeDocuments)

    prompt = ChatPromptTemplate.from_template(
        """You are a grader assessing relevance of a retrieved document to a user question.
If the document contains keyword(s) or semantic meaning related to the user question, grade it as relevant.
It does not need to be a stringent test. The goal is to filter out erroneous retrievals.
Give a binary score 'yes' or 'no' to indicate whether the document is relevant to the question.

Retrieved document:
{document}

User question: {question}"""
    )

    chain = prompt | model
    relevant_count = 0
    for doc in documents:
        result = chain.invoke({"document": doc.page_content, "question": question})
        if result.binary_score == "yes":
            relevant_count += 1

    decision = "yes" if relevant_count >= max(1, len(documents) // 2) else "no"
    logger.info(f"Document grading decision: {decision} ({relevant_count}/{len(documents)} relevant)")
    return decision


def grade_hallucination(documents: list, generation: str, settings: Settings | None = None) -> str:
    """Grade whether the generated answer is grounded in the retrieved documents."""
    logger.info("Grading for hallucination")
    settings = settings or get_settings()
    model = get_llm(settings.llm_provider, settings.llm_model).with_structured_output(GradeHallucinations)

    prompt = ChatPromptTemplate.from_template(
        """You are a grader assessing whether an LLM generation is supported by a set of retrieved facts.
Give a binary score 'yes' or 'no'.
'yes' means that the answer is grounded in / supported by the set of facts.

Set of facts:
{documents}

LLM generation: {generation}"""
    )

    chain = prompt | model
    result = chain.invoke(
        {
            "documents": "\n\n".join(doc.page_content for doc in documents),
            "generation": generation,
        }
    )
    logger.info(f"Hallucination grade: {result.binary_score}")
    return result.binary_score


def grade_answer(question: str, generation: str, settings: Settings | None = None) -> str:
    """Grade whether the generated answer addresses the question."""
    logger.info("Grading answer usefulness")
    settings = settings or get_settings()
    model = get_llm(settings.llm_provider, settings.llm_model).with_structured_output(GradeAnswer)

    prompt = ChatPromptTemplate.from_template(
        """You are a grader assessing whether an answer addresses / resolves a question.
Give a binary score 'yes' or 'no'.
'yes' means that the answer resolves the question.

User question: {question}

LLM generation: {generation}"""
    )

    chain = prompt | model
    result = chain.invoke({"question": question, "generation": generation})
    logger.info(f"Answer usefulness grade: {result.binary_score}")
    return result.binary_score
