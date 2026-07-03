from functools import lru_cache

from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings

from core.config import get_settings


@lru_cache
def get_llm(model_name: str | None = None) -> ChatGoogleGenerativeAI:
    """Return the main chat LLM."""
    settings = get_settings()
    return ChatGoogleGenerativeAI(
        model=model_name or settings.llm_model,
        temperature=0.2,
        convert_system_message_to_human=True,
    )


@lru_cache
def get_embeddings(model_name: str | None = None) -> GoogleGenerativeAIEmbeddings:
    """Return the embedding model."""
    settings = get_settings()
    return GoogleGenerativeAIEmbeddings(model=model_name or settings.embedding_model)


@lru_cache
def get_vision_llm(model_name: str | None = None) -> ChatGoogleGenerativeAI:
    """Return the vision-capable LLM for image analysis."""
    settings = get_settings()
    return ChatGoogleGenerativeAI(
        model=model_name or settings.vision_model,
        temperature=0.1,
        max_output_tokens=1024,
    )
