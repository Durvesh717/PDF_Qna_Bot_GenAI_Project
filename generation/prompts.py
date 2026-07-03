from langchain_core.prompts import ChatPromptTemplate


def get_generation_prompt() -> ChatPromptTemplate:
    """Return the prompt used for answer generation with citations."""
    return ChatPromptTemplate.from_template(
        """You are an assistant for question-answering tasks.
Use the following pieces of retrieved context to answer the question.
If you don't know the answer, just say that you don't know.
Always cite your sources using [Source X] markers in your answer.

Question: {question}

Context:
{context}

Answer:"""
    )
