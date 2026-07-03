# Production-Grade Agentic RAG Implementation Plan

This document outlines the full roadmap for transforming the current PDF Q&A Bot into a production-grade, agentic RAG system.

---

## Project Goals

1. Move from a single-file demo to a modular, maintainable codebase.
2. Implement advanced RAG techniques (hybrid search, reranking, query rewriting, etc.).
3. Add an agentic layer that can route, grade, self-correct, and fall back to web search.
4. Expose a production FastAPI backend with streaming and async processing.
5. Add evaluation (RAGAS) and observability (LangSmith).
6. Containerize with Docker and document thoroughly.

---

## Final Project Structure

```
PDF_Qna_Bot_GenAI_Project/
├── app/                      # Streamlit frontend
│   ├── __init__.py
│   └── streamlit_app.py
├── api/                      # FastAPI backend
│   ├── __init__.py
│   ├── main.py
│   └── routes/
│       ├── __init__.py
│       ├── upload.py
│       ├── chat.py
│       └── collections.py
├── core/                     # Config, logging, clients
│   ├── __init__.py
│   ├── config.py
│   ├── logger.py
│   └── dependencies.py
├── ingestion/                # PDF parsing + chunking
│   ├── __init__.py
│   ├── parser.py
│   ├── chunker.py
│   └── vectorstore.py
├── retrieval/                # Advanced retrieval
│   ├── __init__.py
│   ├── retriever.py
│   ├── reranker.py
│   └── query_transform.py
├── generation/               # LLM + prompt templates
│   ├── __init__.py
│   ├── llm.py
│   └── prompts.py
├── agents/                   # Agentic RAG
│   ├── __init__.py
│   ├── crag_agent.py
│   ├── graph.py
│   ├── graders.py
│   └── tools.py
├── evaluation/               # RAGAS eval
│   ├── __init__.py
│   ├── evaluator.py
│   └── test_data.py
├── tests/                    # Unit + integration tests
│   └── __init__.py
├── data/chroma/              # Persistent vector store
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
├── .env.example
└── README.md
```

---

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

### Phase 4: Production API

19. `(feat): initialize FastAPI application`
20. `(feat): add async PDF upload endpoint`
21. `(feat): add streaming chat endpoint`
22. `(feat): add collection management endpoints`

### Phase 5: Evaluation & Observability

23. `(feat): add RAGAS evaluation pipeline`
24. `(feat): add LangSmith tracing`

### Phase 6: Deployment & Docs

25. `(chore): add Dockerfile and docker-compose`
26. `(docs): update README with architecture and usage`

---

## Phase Explanations

### Phase 1: Foundation & Refactor

Before adding advanced features, the codebase must be modular. Currently everything lives in `app.py` and `utils.py`. Phase 1 separates concerns:

- **Configuration (`core/config.py`)**: Centralizes all environment variables (API keys, model names, chunk sizes) using `pydantic-settings`. This makes the app easier to configure and test.
- **Logging (`core/logger.py`)**: Replaces `print` and scattered `st.error` with structured logging so behavior is observable in production.
- **Ingestion (`ingestion/`)**: Separates PDF parsing, image description, chunking, and vector store creation.
- **Vector Store (`ingestion/vectorstore.py`)**: Uses a persistent Chroma database instead of an in-memory store that dies when the Streamlit session ends.
- **LLM Clients (`generation/llm.py`)**: Centralizes Gemini LLM and embedding model initialization.
- **Streamlit UI (`app/streamlit_app.py`)**: Becomes a thin client that calls the services.

### Phase 2: Advanced RAG

Improves retrieval quality beyond simple similarity search:

- **Hybrid Search**: Combines dense vector similarity with sparse keyword (BM25) search. Dense search catches semantic meaning; keyword search catches exact names, IDs, and rare terms.
- **Cross-Encoder Reranker**: A second-pass model re-scores the top retrieved chunks so the most relevant ones are sent to the LLM.
- **Query Rewriting**: The LLM rephrases vague or conversational user questions into precise retrieval queries.
- **Multi-Query Retrieval**: Generates multiple variants of the question, retrieves for each, and fuses the results to increase recall.
- **Parent Document Retriever**: Stores small semantic chunks for retrieval but returns the full parent section to the LLM for richer context.

### Phase 3: Agentic RAG

Adds a LangGraph agent that decides how to answer instead of blindly retrieving:

- **Router**: Decides if the question needs retrieval, direct answer, or web search.
- **Relevance Grader**: Scores whether retrieved chunks actually answer the question.
- **Hallucination Grader**: Scores whether the generated answer is grounded in the retrieved context.
- **Usefulness Grader**: Scores whether the final answer is helpful.
- **Web Search Fallback**: Uses DuckDuckGo/Tavily when the document does not contain the answer.
- **CRAG Loop**: If retrieval is bad, the agent rewrites the query or falls back to web search. If the answer is hallucinated, it retries.

### Phase 4: Production API

Wraps the system in a FastAPI backend so it can be consumed by any frontend:

- **Async Upload**: PDF processing runs in the background so the API responds immediately.
- **Streaming Chat**: The LLM response streams token-by-token to the client.
- **Collection Management**: Create, list, and delete document collections.
- **Error Handling**: Proper HTTP status codes, retries, and validation.

### Phase 5: Evaluation & Observability

Makes the system measurable and debuggable:

- **RAGAS Metrics**: Faithfulness, answer relevance, context precision, context recall.
- **Synthetic Test Data**: Generate question-answer pairs from documents for benchmarking.
- **LangSmith Tracing**: Trace every agent step, retrieval, and generation.

### Phase 6: Deployment & Docs

Makes the project runnable anywhere:

- **Dockerfile**: Containerizes the Streamlit + FastAPI services.
- **docker-compose.yml**: Spins up the app, API, and optional Redis.
- **README Update**: Architecture diagram, setup instructions, API docs.

---

## Success Criteria

- The Streamlit app still works end-to-end.
- A new FastAPI backend can be started and used via `/docs`.
- Retrieval uses hybrid search + reranking.
- The agent can detect bad retrieval and fall back to web search.
- Evaluation pipeline produces RAGAS scores.
- Everything runs inside Docker.
