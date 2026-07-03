from functools import lru_cache
from typing import Union

from langchain_core.embeddings import Embeddings
from langchain_core.language_models.chat_models import BaseChatModel

from core.config import Settings, get_settings


def _get_google_llm(settings: Settings, model: str, temperature: float, **kwargs) -> BaseChatModel:
    from langchain_google_genai import ChatGoogleGenerativeAI

    config = {"model": model, "temperature": temperature, **kwargs}
    if settings.google_api_key:
        config["google_api_key"] = settings.google_api_key
    return ChatGoogleGenerativeAI(**config)


def _get_openai_llm(settings: Settings, model: str, temperature: float, **kwargs) -> BaseChatModel:
    from langchain_openai import ChatOpenAI

    config = {"model": model, "temperature": temperature, **kwargs}
    if settings.openai_api_key:
        config["api_key"] = settings.openai_api_key
    return ChatOpenAI(**config)


def _get_bedrock_llm(settings: Settings, model: str, temperature: float, **kwargs) -> BaseChatModel:
    from langchain_aws import ChatBedrock

    config = {
        "model_id": model,
        "temperature": temperature,
        "region_name": settings.aws_region,
        **kwargs,
    }
    if settings.aws_access_key_id:
        config["aws_access_key_id"] = settings.aws_access_key_id
    if settings.aws_secret_access_key:
        config["aws_secret_access_key"] = settings.aws_secret_access_key
    return ChatBedrock(**config)


@lru_cache
def get_llm(
    provider: str | None = None,
    model_name: str | None = None,
    temperature: float = 0.2,
) -> BaseChatModel:
    """Return the main chat LLM based on provider and model."""
    settings = get_settings()
    provider = provider or settings.llm_provider
    model = model_name or settings.llm_model

    if provider == "google":
        return _get_google_llm(settings, model, temperature)
    if provider == "openai":
        return _get_openai_llm(settings, model, temperature)
    if provider == "bedrock":
        return _get_bedrock_llm(settings, model, temperature)

    raise ValueError(f"Unsupported LLM provider: {provider}")


@lru_cache
def get_embeddings(
    provider: str | None = None,
    model_name: str | None = None,
) -> Embeddings:
    """Return the embedding model based on provider and model."""
    settings = get_settings()
    provider = provider or settings.embedding_provider
    model = model_name or settings.embedding_model

    if provider == "google":
        from langchain_google_genai import GoogleGenerativeAIEmbeddings

        config = {"model": model}
        if settings.google_api_key:
            config["google_api_key"] = settings.google_api_key
        return GoogleGenerativeAIEmbeddings(**config)
    if provider == "openai":
        from langchain_openai import OpenAIEmbeddings

        config = {"model": model}
        if settings.openai_api_key:
            config["api_key"] = settings.openai_api_key
        return OpenAIEmbeddings(**config)
    if provider == "bedrock":
        from langchain_aws import BedrockEmbeddings

        config = {"model_id": model, "region_name": settings.aws_region}
        if settings.aws_access_key_id:
            config["aws_access_key_id"] = settings.aws_access_key_id
        if settings.aws_secret_access_key:
            config["aws_secret_access_key"] = settings.aws_secret_access_key
        return BedrockEmbeddings(**config)

    raise ValueError(f"Unsupported embedding provider: {provider}")


@lru_cache
def get_vision_llm(
    provider: str | None = None,
    model_name: str | None = None,
) -> BaseChatModel:
    """Return the vision-capable LLM for image analysis."""
    settings = get_settings()
    provider = provider or settings.vision_provider
    model = model_name or settings.vision_model

    kwargs = {"max_tokens": 1024}

    if provider == "google":
        return _get_google_llm(settings, model, temperature=0.1, **kwargs)
    if provider == "openai":
        return _get_openai_llm(settings, model, temperature=0.1, **kwargs)
    if provider == "bedrock":
        return _get_bedrock_llm(settings, model, temperature=0.1, **kwargs)

    raise ValueError(f"Unsupported vision provider: {provider}")


def list_available_models() -> dict[str, list[str]]:
    """Return a mapping of providers to available models."""
    return {
        "google": [
            "gemini-2.0-flash",
            "gemini-2.0-flash-lite",
            "gemini-2.5-pro-exp-03-25",
            "gemini-1.5-pro",
        ],
        "openai": [
            "gpt-4o",
            "gpt-4o-mini",
            "gpt-4-turbo",
            "gpt-3.5-turbo",
        ],
        "bedrock": [
            "anthropic.claude-3-5-sonnet-20240620-v1:0",
            "anthropic.claude-3-sonnet-20240229-v1:0",
            "anthropic.claude-3-haiku-20240307-v1:0",
            "amazon.titan-text-express-v1",
        ],
    }


def list_available_embedding_models() -> dict[str, list[str]]:
    """Return a mapping of providers to available embedding models."""
    return {
        "google": [
            "models/text-embedding-004",
            "models/embedding-001",
        ],
        "openai": [
            "text-embedding-3-small",
            "text-embedding-3-large",
            "text-embedding-ada-002",
        ],
        "bedrock": [
            "amazon.titan-embed-text-v1",
            "amazon.titan-embed-text-v2:0",
        ],
    }
