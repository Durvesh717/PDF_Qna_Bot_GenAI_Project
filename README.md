# 📚 PDF Q&A Bot with Google Gemini AI

A powerful Streamlit application that allows users to upload PDF documents and ask questions about their content using Google's Gemini AI models and advanced document processing capabilities.

## 🌟 Features

- **Multi-Modal PDF Processing**: Extracts both text and visual content (charts, tables, figures) from PDFs
- **Intelligent Document Parsing**: Uses Upstage AI for advanced document structure understanding
- **Visual Content Analysis**: Gemini AI analyzes and describes images, charts, and tables within PDFs
- **Vector-Based Search**: Implements semantic search using Chroma vector database
- **Interactive Chat Interface**: Clean, user-friendly Streamlit interface for Q&A
- **Real-time Processing**: Live document processing and question answering

## 🤖 AI Models Used

### Primary Models
- **Google Gemini 2.0 Flash**: Main language model for question answering
- **Google Gemini 1.5 Flash 8B**: Specialized model for image and visual content analysis
- **Google Text Embedding 004**: Creates vector embeddings for semantic search

### Document Processing
- **Upstage Document Parse API**: Advanced PDF parsing with structure recognition
- **PyMuPDF4LLM**: Converts PDF content to markdown format

## 🛠️ Technology Stack

- **Frontend**: Streamlit
- **AI/ML**: LangChain, Google Generative AI
- **Vector Database**: Chroma
- **Document Processing**: PyMuPDF4LLM, Upstage AI
- **Text Processing**: LangChain Text Splitters
- **Language**: Python 3.8 - 3.12 ⚠️ **NOT compatible with Python  3.13**

## 📋 Prerequisites

Before running this application, you need to obtain API keys from:

1. **Google AI Studio**: 
   - Visit: https://aistudio.google.com/app/apikey
   - Create an account and generate an API key

2. **Upstage AI**:
   - Visit: https://console.upstage.ai/docs/getting-started
   - Sign up and get your API key

## 🚀 Installation & Setup

### 1. Clone the Repository
```bash
git clone <your-repository-url>
cd PDF_Qna_GenAI_Project
```

### 2. Create Virtual Environment (Recommended)
```bash
python -m venv venv 
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Run the Application
```bash
streamlit run app.py
```

The application will open in your default web browser at `http://localhost:8501`

## 📖 How to Use

### Step 1: API Configuration
1. Launch the application
2. In the sidebar, enter your **Google AI Studio API Key**
3. Enter your **Upstage API Key**

### Step 2: Upload Document
1. Click "Choose a PDF file" in the sidebar
2. Select your PDF document
3. Click "🔄 Process PDF" to analyze the document

### Step 3: Ask Questions
1. Once processing is complete, use the text input to ask questions
2. Try sample questions provided in the sidebar
3. Get intelligent responses based on your document content

## 🔧 Key Components

### `app.py`
- Main Streamlit application
- User interface and interaction handling
- Session state management
- File upload and processing coordination

### `utils.py`
- Core document processing functions
- PDF content extraction and analysis
- Vector store creation and management
- RAG (Retrieval-Augmented Generation) chain implementation

### Key Functions:
- `extract_content_from_pdf()`: Processes uploaded PDF files
- `create_vector_store()`: Creates searchable vector embeddings
- `create_conversation_chain()`: Sets up the Q&A pipeline
- `create_image_descriptions()`: Analyzes visual content in PDFs

## 🏗️ Architecture

```
PDF Upload → Document Parsing → Content Extraction → Vector Store Creation → RAG Chain → Q&A Interface
     ↓              ↓                    ↓                     ↓              ↓           ↓
  Streamlit    Upstage AI +       Text + Image        Chroma Vector     LangChain    Streamlit
              PyMuPDF4LLM        Processing          Database         RAG Chain      Chat UI
                                (Gemini AI)
```

## 📊 Document Processing Pipeline

1. **Text Extraction**: PyMuPDF4LLM converts PDF to markdown
2. **Structure Parsing**: Upstage AI identifies document elements
3. **Image Analysis**: Gemini AI describes visual content
4. **Content Merging**: Combines text and visual descriptions
5. **Vectorization**: Creates embeddings for semantic search
6. **Storage**: Stores in Chroma vector database

## 🎯 Use Cases

- **Research**: Quickly find information in academic papers
- **Legal**: Search through legal documents and contracts
- **Business**: Analyze reports, presentations, and proposals
- **Education**: Study materials and textbook Q&A
- **Technical**: Documentation and manual queries

## 🔒 Security Notes

- API keys are handled securely through environment variables
- No document content is permanently stored
- All processing happens locally within your session

## 🐛 Troubleshooting

**Python Version Issues:**

1. **Python 3.13 Compatibility Error**: 
   - **Error**: `the configured Python interpreter version (3.13) is newer than PyO3's maximum supported version (3.12)`
   - **Solution**: Use Python 3.8-3.12. The `tokenizers` package doesn't support Python 3.13 yet
   - **Fix**: Install Python 3.11 or 3.12 and recreate your virtual environment

2. **Checking Your Python Version**:
   ```bash
   python --version  # Should show 3.8.x to 3.12.x
   ```

**Other Common Issues:**

1. **API Key Errors**: Ensure both API keys are correctly entered
2. **PDF Processing Fails**: Check if PDF is readable and not password-protected
3. **Memory Issues**: Large PDFs may require more system memory
4. **Slow Processing**: Image-heavy PDFs take longer to process
5. **Installation Failures**: Make sure you're using a compatible Python version (3.8-3.12)

## 📝 Requirements

See `requirements.txt` for complete dependency list. Key packages:
- streamlit
- langchain ecosystem (core, google-genai, chroma, text-splitters, upstage)
- google-generativeai
- pymupdf4llm
- chromadb

