# Agentic RAG Implementation Plan

This document outlines the roadmap for transforming the PDF Q&A Bot into a production-oriented agentic RAG Streamlit application.

## Project Goals

1. Move from a single-file demo to a modular, maintainable codebase.
2. Implement advanced RAG techniques (hybrid search, reranking, query rewriting, etc.).
3. Add an agentic layer that can route, grade, self-correct, and fall back to web search.
4. Add evaluation (RAGAS) and observability (LangSmith).

## Final Project Structure

```
PDF_Qna_Bot_GenAI_Project/
├── app/              # Streamlit frontend
├── core/             # Config, logging, tracing
├── ingestion/        # PDF parsing + chunking + vector store
├── retrieval/        # Advanced retrieval
├── generation/       # LLM + prompt templates
├── agents/           # Agentic RAG
├── evaluation/       # RAGAS eval
├── tests/
├── data/chroma/      # Persistent vector store
├── requirements.txt
├── .env.example
└── README.md
```

## Commit Plan

All commits follow the format `(type): description`.

### Phase 1: Foundation & Refactor

1. `(chore): create modular project structure`
2. `(feat): add pydantic-settings and environment configuration`
3. `(feat): add structured logging utility`
4. `(refactor): extract PDF parsing to ingestion module`
5. `(refactor): extract chunking and embedding pipeline`
6. `(feat): add persistent Chroma vector store manager`
7. `(feat): centralize LLM and embedding clients`
8. `(refactor): split Streamlit app into app module using new services`

### Phase 2: Advanced RAG

9. `(feat): add hybrid dense + BM25 search`
10. `(feat): add cross-encoder reranker`
11. `(feat): add query rewriting for retrieval`
12. `(feat): add multi-query retrieval fusion`
13. `(feat): add parent document retriever`

### Phase 3: Agentic RAG

14. `(feat): add retrieval relevance grader`
15. `(feat): add hallucination and answer usefulness graders`
16. `(feat): add web search fallback tool`
17. `(feat): implement CRAG LangGraph agent`
18. `(feat): integrate agent and add source citations`

### Phase 4: Evaluation & Observability

19. `(feat): add synthetic test data generation`
20. `(feat): add RAGAS evaluation pipeline`
21. `(feat): add LangSmith tracing`

## Phase Explanations

### Phase 1: Foundation & Refactor

- **Configuration (`core/config.py`)**: Centralizes environment variables using `pydantic-settings`.
- **Logging (`core/logger.py`)**: Structured logging for production observability.
- **Ingestion (`ingestion/`)**: Separates PDF parsing, image description, chunking, and vector store creation.
- **Vector Store (`ingestion/vectorstore.py`)**: Persistent Chroma database with collection management.
- **LLM Clients (`generation/llm.py`)**: Centralized Gemini LLM and embedding model initialization.
- **Streamlit UI (`app/streamlit_app.py`)**: Thin client that calls services directly.

### Phase 2: Advanced RAG

- **Hybrid Search**: Dense vector similarity + BM25 keyword search fused with RRF.
- **Cross-Encoder Reranker**: Re-scores retrieved chunks for better precision.
- **Query Rewriting**: LLM rephrases vague questions into precise retrieval queries.
- **Multi-Query Retrieval**: Generates query variants to improve recall.
- **Parent Document Retriever**: Returns full parent sections instead of small chunks.

### Phase 3: Agentic RAG

- **Router**: Decides if retrieval or web search is needed.
- **Relevance Grader**: Scores whether retrieved chunks answer the question.
- **Hallucination Grader**: Checks if the generated answer is grounded in context.
- **Usefulness Grader**: Checks if the answer addresses the question.
- **Web Search Fallback**: DuckDuckGo/Tavily when the document lacks the answer.
- **CRAG Loop**: Rewrites query or falls back to web search when retrieval is bad.

### Phase 4: Evaluation & Observability

- **RAGAS Metrics**: Faithfulness, answer relevance, context precision, context recall.
- **Synthetic Test Data**: Generate question-answer pairs from documents.
- **LangSmith Tracing**: Trace agent steps, retrievals, and generations.
