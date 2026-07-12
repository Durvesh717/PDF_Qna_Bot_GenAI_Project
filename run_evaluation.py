import argparse
import sys

from langchain_core.documents import Document

from evaluation.evaluator import evaluate_qa_pairs
from evaluation.test_data import generate_test_questions
from ingestion.vectorstore import get_vector_store


def main():
    parser = argparse.ArgumentParser(description="Run RAGAS evaluation on a vector store collection.")
    parser.add_argument("--collection", type=str, default="default", help="Collection name in Chroma DB")
    parser.add_argument("--num-questions", type=int, default=5, help="Number of questions to generate for evaluation")
    args = parser.parse_args()

    print(f"Loading collection '{args.collection}'...")
    try:
        vector_store = get_vector_store(args.collection)
        data = vector_store.get(limit=30)
        docs = [
            Document(page_content=text, metadata=metadata or {})
            for text, metadata in zip(data["documents"], data["metadatas"], strict=True)
        ]
        if not docs:
            print(f"Error: Collection '{args.collection}' is empty or does not exist. Please upload documents first via the Streamlit UI.")
            sys.exit(1)
    except Exception as e:
        print(f"Error loading collection: {e}")
        sys.exit(1)

    print(f"Generating {args.num_questions} synthetic Q&A pairs from documents...")
    qa_pairs = generate_test_questions(docs, n=args.num_questions)
    if not qa_pairs:
        print("Error: Failed to generate Q&A pairs. Check your API keys and configuration.")
        sys.exit(1)

    print(f"Running evaluation with RAGAS on {len(qa_pairs)} generated Q&A pairs...")
    results = evaluate_qa_pairs(qa_pairs, collection=args.collection)

    if isinstance(results, dict) and "error" in results:
        print(f"Evaluation failed: {results['error']}")
        sys.exit(1)

    print("\n--- Evaluation Results ---")
    for idx, row in enumerate(results):
        print(f"\n[Test Case {idx + 1}]")
        print(f"Question: {row.get('user_input')}")
        print(f"Answer: {row.get('response')}")
        print(f"Ground Truth: {row.get('reference')}")
        # Print RAGAS score if available
        scores = {
            k: v
            for k, v in row.items()
            if k not in ("user_input", "response", "retrieved_contexts", "reference")
        }
        if scores:
            print("Scores:")
            for metric, score in scores.items():
                print(f"  - {metric}: {score}")


if __name__ == "__main__":
    main()
