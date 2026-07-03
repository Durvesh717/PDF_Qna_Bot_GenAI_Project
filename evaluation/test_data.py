from typing import List

from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate

from core.config import Settings, get_settings
from core.logger import get_logger
from generation.llm import get_llm

logger = get_logger(__name__)


def generate_test_questions(
    documents: List[Document], n: int = 5, settings: Settings | None = None
) -> List[dict]:
    """Generate synthetic question-context-answer triples from documents."""
    settings = settings or get_settings()
    model = get_llm(settings.llm_model)
    prompt = ChatPromptTemplate.from_template(
        """You are creating a test set for a RAG system.
Given the following context, generate a question that can be answered using ONLY the context,
along with the correct answer.

Context:
{context}

Output exactly in this format:
Question: <question>
Answer: <answer>"""
    )

    chain = prompt | model
    questions = []

    step = max(1, len(documents) // n)
    for i in range(0, min(len(documents), n * step), step):
        doc = documents[i]
        try:
            response = chain.invoke({"context": doc.page_content})
            text = response.content.strip()
            q_part, a_part = text.split("Answer:", 1)
            question = q_part.replace("Question:", "").strip()
            answer = a_part.strip()
            questions.append(
                {
                    "question": question,
                    "answer": answer,
                    "contexts": [doc.page_content],
                }
            )
        except Exception as e:
            logger.warning(f"Failed to generate test question: {e}")

    return questions
