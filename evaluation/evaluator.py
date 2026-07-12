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
    qa_pairs: list[dict],
    collection: str = "default",
    metrics: list | None = None,
    settings: Settings | None = None,
) -> dict:
    """Evaluate a list of question-answer-context triples using RAGAS."""
    try:
        import sys
        try:
            # ragas imports langchain_community.chat_models.vertexai, which no
            # longer exists; alias the google package so the import resolves.
            import langchain_google_vertexai
            sys.modules['langchain_community.chat_models.vertexai'] = langchain_google_vertexai
        except ImportError:
            pass
        from ragas import EvaluationDataset, evaluate
        from ragas.embeddings import LangchainEmbeddingsWrapper
        from ragas.llms import LangchainLLMWrapper
    except Exception as e:
        logger.error(f"RAGAS not available: {e}")
        return {"error": f"RAGAS import failed: {str(e)}"}

    settings = settings or get_settings()
    metrics = metrics or _get_default_metrics()

    from generation.llm import get_embeddings, get_llm
    app_llm = get_llm(settings.llm_provider, settings.llm_model)
    app_embeddings = get_embeddings(settings.embedding_provider, settings.embedding_model)

    ragas_llm = LangchainLLMWrapper(app_llm)
    ragas_embeddings = LangchainEmbeddingsWrapper(app_embeddings)

    vector_store = get_vector_store(collection, settings)
    agent = CRAGAgent(vector_store, settings)

    predictions = []
    for pair in qa_pairs:
        result = agent.invoke(pair["question"])
        predictions.append(
            {
                "user_input": pair["question"],
                "response": result.get("generation", ""),
                "retrieved_contexts": [doc.page_content for doc in result.get("documents", [])],
                "reference": pair["answer"],
            }
        )

    dataset = EvaluationDataset.from_list(predictions)
    logger.info("Running RAGAS evaluation")
    result = evaluate(
        dataset=dataset,
        metrics=metrics,
        llm=ragas_llm,
        embeddings=ragas_embeddings,
    )
    return result.to_pandas().to_dict(orient="records")
