from langchain_chroma import Chroma

from agents.graph import AgentState, build_agent
from core.config import Settings, get_settings
from core.logger import get_logger

logger = get_logger(__name__)


class CRAGAgent:
    """Corrective RAG agent that can retrieve, grade, and fall back to web search."""

    def __init__(self, vector_store: Chroma, settings: Settings | None = None):
        self.settings = settings or get_settings()
        self.vector_store = vector_store
        self.app = build_agent(vector_store, self.settings)

    def invoke(self, question: str, chat_history: list[dict] | None = None) -> AgentState:
        """Run the agent and return the final state."""
        logger.info(f"Invoking CRAG agent for: {question}")
        initial_state: AgentState = {
            "question": question,
            "chat_history": chat_history or [],
            "rewritten_question": "",
            "generation": "",
            "documents": [],
            "retries": 0,
            "steps": [],
        }
        result = self.app.invoke(initial_state)
        logger.info(f"Agent finished with steps: {result['steps']}")
        return result
