# 📚 PDF Q&A Bot — Agentic RAG with Streamlit

A production-oriented Streamlit application for chatting with PDF documents. It combines multimodal document parsing, advanced retrieval techniques, and a self-correcting agentic RAG pipeline.

## 🌟 Features

- **Multimodal PDF Ingestion**: Extracts text, tables, charts, and figures using PyMuPDF4LLM + PyMuPDF image extraction.
- **Visual Content Analysis**: Gemini describes images, charts, and tables found in the PDF.
- **Advanced Retrieval**:
  - Hybrid dense + BM25 keyword search
  - Cross-encoder reranking
  - Query rewriting and multi-query retrieval
  - Parent document retrieval
- **Agentic RAG (CRAG)**: LangGraph agent that grades retrieval relevance, detects hallucinations, and falls back to web search when needed.
- **Source Citations**: Answers include references to document pages or web URLs.
- **Multi-Document Upload**: Upload multiple PDFs into one collection and chat across all of them.
- **Collection Manager**: List, load, and delete document collections from the sidebar.
- **Persistent Vector Store**: Chroma collections persist across sessions.
- **Evaluation**: RAGAS metrics for faithfulness, relevance, and context quality.
- **Observability**: LangSmith tracing support.

## 🏗️ How It Works

```
PDF Upload
    │
    ▼
┌─────────────────┐
│   Ingestion     │  PyMuPDF4LLM parses text
│     Layer       │  Vision LLM describes images/charts/tables
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│    Chunking     │  Recursive splitter with parent tracking
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Vector Store   │  Chroma + Google text-embedding-004
│   (Persistent)  │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│    CRAG Agent   │  Route → Retrieve → Grade → Generate → Grade
│   (LangGraph)   │  Falls back to web search if needed
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Streamlit UI   │  Chat interface with citations
└─────────────────┘
```

## 🤖 Supported AI Providers

The app supports switching between Google, OpenAI, and AWS Bedrock for LLM, embeddings, and vision:

| Provider | LLM Models | Embedding Models | Vision Models |
|----------|-----------|------------------|---------------|
| **Google** | gemini-2.0-flash, gemini-1.5-pro | text-embedding-004 | gemini-2.0-flash |
| **OpenAI** | gpt-4o, gpt-4o-mini, gpt-3.5-turbo | text-embedding-3-small | gpt-4o |
| **AWS Bedrock** | Claude 3.5 Sonnet, Claude 3 Haiku | Titan Embed Text | Claude 3 Sonnet |

- **BAAI/bge-reranker-base**: Cross-encoder reranker.
- **PyMuPDF4LLM + PyMuPDF**: PDF text and image extraction.

## 🛠️ Technology Stack

- **Frontend**: Streamlit
- **AI/ML**: LangChain, LangGraph, Google Generative AI, OpenAI, AWS Bedrock
- **Vector DB**: Chroma
- **Evaluation**: RAGAS
- **Observability**: LangSmith
- **Language**: Python 3.12

## 📋 Prerequisites

Get API keys for the providers you want to use:

1. **Google AI Studio**: https://aistudio.google.com/app/apikey
2. **OpenAI**: https://platform.openai.com/api-keys
3. **AWS Bedrock**: https://aws.amazon.com/console/
4. *(Optional)* **Tavily**: https://tavily.com for premium web search fallback
5. *(Optional)* **LangSmith**: https://smith.langchain.com for tracing

## 🚀 Local Setup

### 1. Clone and enter the repo

```bash
git clone https://github.com/Durvesh717/PDF_Qna_Bot_GenAI_Project
cd PDF_Qna_Bot_GenAI_Project
```

### 2. Create virtual environment

```bash
uv venv
source .venv/bin/activate
```

### 3. Install dependencies

```bash
uv pip install -r requirements.txt
```

### 4. Configure environment variables

```bash
cp .env.example .env
# Edit .env with your API keys
```

### 5. Run the app

```bash
streamlit run app.py
```

Open the UI at `http://localhost:8501`.

## 🧪 Evaluation

Generate synthetic test questions and run RAGAS evaluation:

```python
from evaluation.test_data import generate_test_questions
from evaluation.evaluator import evaluate_qa_pairs
from ingestion.vectorstore import get_vector_store

vector_store = get_vector_store("my_docs")
docs = vector_store.similarity_search("", k=10)
qa_pairs = generate_test_questions(docs)
results = evaluate_qa_pairs(qa_pairs, collection="my_docs")
print(results)
```

Metrics tracked:
- Faithfulness
- Answer Relevancy
- Context Precision
- Context Recall

## 📁 Project Structure

```
PDF_Qna_Bot_GenAI_Project/
├── app/              # Streamlit frontend
├── core/             # Config, logging, tracing
├── ingestion/        # PDF parsing, chunking, vector store
├── retrieval/        # Hybrid search, reranker, query transform
├── generation/       # LLM clients and prompts
├── agents/           # CRAG LangGraph agent
├── evaluation/       # RAGAS evaluation
├── data/             # Persistent vector store
├── requirements.txt
└── README.md
```

## 🔒 Security Notes

- API keys are loaded from environment variables (`.env`).
- No document content is sent to third parties except the configured LLM/embedding providers.
- Persistent data is stored locally in `data/`.

## 🐛 Troubleshooting

**Python version**: Use Python 3.11 or 3.12. Some dependencies do not yet support Python 3.13.

**RAGAS import errors**: RAGAS can be sensitive to LangChain versions. The evaluator uses lazy imports to avoid breaking the app.

**API key errors**: Ensure `GOOGLE_API_KEY` and `UPSTAGE_API_KEY` are set in `.env`.
