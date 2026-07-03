from typing import List

from agents.crag_agent import CRAGAgent
from core.config import Settings, get_settings
from core.logger import get_logger
from ingestion.vectorstore import get_vector_store

logger = get_logger(__name__)


def _get_default_metrics():
    from ragas.metrics import (
        answer_relevancy,
        context_precision,
        context_recall,
        faithfulness,
    )

    return [faithfulness, answer_relevancy, context_precision, context_recall]


def evaluate_qa_pairs(
    qa_pairs: List[dict],
    collection: str = "default",
    metrics: List | None = None,
    settings: Settings | None = None,
) -> dict:
    """Evaluate a list of question-answer-context triples using RAGAS."""
    try:
        from datasets import Dataset
        from ragas import evaluate
    except Exception as e:
        logger.error(f"RAGAS not available: {e}")
        return {"error": f"RAGAS import failed: {str(e)}"}

    settings = settings or get_settings()
    metrics = metrics or _get_default_metrics()

    vector_store = get_vector_store(collection, settings)
    agent = CRAGAgent(vector_store, settings)

    predictions = []
    for pair in qa_pairs:
        result = agent.invoke(pair["question"])
        predictions.append(
            {
                "question": pair["question"],
                "answer": result.get("generation", ""),
                "contexts": [doc.page_content for doc in result.get("documents", [])],
                "ground_truth": pair["answer"],
            }
        )

    dataset = Dataset.from_list(predictions)
    logger.info("Running RAGAS evaluation")
    result = evaluate(dataset=dataset, metrics=metrics)
    return result.to_pandas().to_dict(orient="records")
