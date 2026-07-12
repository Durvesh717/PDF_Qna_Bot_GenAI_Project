import os

from core.config import Settings, get_settings
from core.logger import get_logger

logger = get_logger(__name__)


def configure_langsmith(settings: Settings | None = None) -> None:
    """Configure LangSmith environment variables for tracing."""
    settings = settings or get_settings()

    if not settings.langsmith_api_key:
        logger.info("LangSmith API key not configured; skipping tracing setup")
        return

    os.environ.setdefault("LANGSMITH_API_KEY", settings.langsmith_api_key)
    os.environ.setdefault("LANGSMITH_TRACING", "true")
    os.environ.setdefault("LANGSMITH_PROJECT", settings.app_name.lower().replace(" ", "-"))

    logger.info("LangSmith tracing enabled")
