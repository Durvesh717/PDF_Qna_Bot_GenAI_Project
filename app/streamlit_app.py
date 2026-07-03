import os
import tempfile

import streamlit as st

from agents.crag_agent import CRAGAgent
from core.config import get_settings
from core.logger import get_logger
from core.tracing import configure_langsmith
from generation.llm import (
    list_available_embedding_models,
    list_available_models,
)
from ingestion.chunker import split_documents
from ingestion.parser import extract_content_from_pdf
from ingestion.vectorstore import create_collection, get_vector_store

logger = get_logger(__name__)

st.set_page_config(
    page_title="PDF Q&A Bot",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        color: #1f77b4;
        margin-bottom: 2rem;
    }
    .chat-message {
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
        border: 1px solid #e0e0e0;
    }
    .user-message {
        background-color: #808080;
        border-left: 4px solid #2196f3;
    }
    .bot-message {
        background-color: #808080;
        border-left: 4px solid #9c27b0;
    }
    .sidebar-content {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 0.5rem;
    }
</style>
""",
    unsafe_allow_html=True,
)


def create_agent(vector_store):
    return CRAGAgent(vector_store)


def main():
    settings = get_settings()
    configure_langsmith(settings)

    st.markdown('<h1 class="main-header">📚 PDF Q&A Bot</h1>', unsafe_allow_html=True)
    st.markdown("Upload your PDF documents and ask questions about their content!")

    with st.sidebar:
        st.markdown('<div class="sidebar-content">', unsafe_allow_html=True)
        st.header("📁 Upload & Settings")

        st.markdown("#### 🔑 API Keys")
        google_api_key = st.text_input(
            "Google AI Studio API Key",
            type="password",
            value=settings.google_api_key or "",
            help="Required if using Google provider",
        )
        openai_api_key = st.text_input(
            "OpenAI API Key",
            type="password",
            value=settings.openai_api_key or "",
            help="Required if using OpenAI provider",
        )
        aws_access_key_id = st.text_input(
            "AWS Access Key ID",
            type="password",
            value=settings.aws_access_key_id or "",
            help="Required if using Bedrock provider",
        )
        aws_secret_access_key = st.text_input(
            "AWS Secret Access Key",
            type="password",
            value=settings.aws_secret_access_key or "",
            help="Required if using Bedrock provider",
        )

        if google_api_key:
            os.environ["GOOGLE_API_KEY"] = google_api_key
        if openai_api_key:
            os.environ["OPENAI_API_KEY"] = openai_api_key
        if aws_access_key_id:
            os.environ["AWS_ACCESS_KEY_ID"] = aws_access_key_id
        if aws_secret_access_key:
            os.environ["AWS_SECRET_ACCESS_KEY"] = aws_secret_access_key

        st.markdown("#### 🤖 Model Settings")
        llm_provider = st.selectbox(
            "LLM Provider",
            options=["google", "openai", "bedrock"],
            index=["google", "openai", "bedrock"].index(settings.llm_provider),
        )
        llm_model = st.selectbox(
            "LLM Model",
            options=list_available_models()[llm_provider],
            index=list_available_models()[llm_provider].index(settings.llm_model)
            if settings.llm_model in list_available_models()[llm_provider]
            else 0,
        )

        embedding_provider = st.selectbox(
            "Embedding Provider",
            options=["google", "openai", "bedrock"],
            index=["google", "openai", "bedrock"].index(settings.embedding_provider),
        )
        embedding_model = st.selectbox(
            "Embedding Model",
            options=list_available_embedding_models()[embedding_provider],
            index=list_available_embedding_models()[embedding_provider].index(
                settings.embedding_model
            )
            if settings.embedding_model in list_available_embedding_models()[embedding_provider]
            else 0,
        )

        vision_provider = st.selectbox(
            "Vision Provider",
            options=["google", "openai", "bedrock"],
            index=["google", "openai", "bedrock"].index(settings.vision_provider),
        )
        vision_model = st.selectbox(
            "Vision Model",
            options=list_available_models()[vision_provider],
            index=list_available_models()[vision_provider].index(settings.vision_model)
            if settings.vision_model in list_available_models()[vision_provider]
            else 0,
        )

        # Update settings in session for downstream modules
        settings.llm_provider = llm_provider
        settings.llm_model = llm_model
        settings.embedding_provider = embedding_provider
        settings.embedding_model = embedding_model
        settings.vision_provider = vision_provider
        settings.vision_model = vision_model

        uploaded_file = st.file_uploader(
            "Choose a PDF file", type="pdf", help="Upload a PDF file to analyze"
        )

        collection_name = st.text_input(
            "Collection name",
            value="default",
            help="Name for the document collection",
        )

        def _has_required_keys() -> bool:
            if llm_provider == "google" and not google_api_key:
                return False
            if embedding_provider == "google" and not google_api_key:
                return False
            if vision_provider == "google" and not google_api_key:
                return False
            if llm_provider == "openai" and not openai_api_key:
                return False
            if embedding_provider == "openai" and not openai_api_key:
                return False
            if vision_provider == "openai" and not openai_api_key:
                return False
            if llm_provider == "bedrock" and (not aws_access_key_id or not aws_secret_access_key):
                return False
            if embedding_provider == "bedrock" and (not aws_access_key_id or not aws_secret_access_key):
                return False
            if vision_provider == "bedrock" and (not aws_access_key_id or not aws_secret_access_key):
                return False
            return True

        if uploaded_file and _has_required_keys():
            st.success("✅ File uploaded successfully!")

            if st.button("🔄 Process PDF", type="primary"):
                with st.spinner("Processing PDF..."):
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                        tmp_file.write(uploaded_file.read())
                        temp_path = tmp_file.name

                    merged_docs = extract_content_from_pdf(temp_path)

                    if merged_docs:
                        chunks = split_documents(merged_docs)
                        vector_store = create_collection(chunks, collection_name)
                        st.session_state.vector_store = vector_store
                        st.session_state.agent = create_agent(vector_store)
                        st.session_state.processed = True
                        st.success(f"✅ PDF processed into collection '{collection_name}'!")
                        st.rerun()

        st.markdown("</div>", unsafe_allow_html=True)
        st.markdown("---")
        st.markdown("### 📋 Instructions:")
        st.markdown(
            """
1. Enter API keys for the providers you selected
2. Upload a PDF file
3. Click 'Process PDF'
4. Start asking questions!
"""
        )

        if st.session_state.get("processed"):
            st.markdown("---")
            st.markdown("### 💡 Sample Questions:")
            sample_questions = [
                "What is the main topic of this document?",
                "Can you summarize the key points?",
                "What are the important definitions mentioned?",
                "Explain the methodology used.",
                "What are the conclusions drawn?",
            ]
            for question in sample_questions:
                if st.button(question, key=f"sample_{question}"):
                    st.session_state.current_question = question

    if not _has_required_keys():
        st.warning("⚠️ Please enter the required API keys for your selected providers in the sidebar.")
        st.info("Google: https://aistudio.google.com/app/apikey")
        st.info("OpenAI: https://platform.openai.com/api-keys")
        st.info("AWS Bedrock: https://aws.amazon.com/console/")
    elif not uploaded_file:
        st.info("📄 Please upload a PDF file to begin asking questions.")
    elif not st.session_state.get("processed"):
        st.info("🔄 Please click 'Process PDF' in the sidebar to analyze your document.")
    else:
        if "messages" not in st.session_state:
            st.session_state.messages = []

        chat_container = st.container()
        with chat_container:
            for message in st.session_state.messages:
                css_class = (
                    "user-message" if message["role"] == "user" else "bot-message"
                )
                st.markdown(
                    f"""
                    <div class="chat-message {css_class}">
                        <strong>{'You' if message['role'] == 'user' else 'Bot'}:</strong> {message['content']}
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

        st.markdown("---")

        if "current_question" in st.session_state:
            user_question = st.session_state.current_question
            del st.session_state.current_question
        else:
            user_question = st.text_input(
                "Ask a question about your PDF:",
                placeholder="e.g., What is the main topic discussed in this document?",
            )

        if user_question and st.button("Send", type="primary"):
            st.session_state.messages.append({"role": "user", "content": user_question})

            with st.spinner("Thinking..."):
                try:
                    result = st.session_state.agent.invoke(user_question)
                    response = result.get("generation", "I could not generate an answer.")
                    st.session_state.messages.append(
                        {"role": "assistant", "content": response}
                    )
                    st.rerun()
                except Exception as e:
                    logger.exception("Error generating response")
                    st.error(f"Error generating response: {str(e)}")

        if st.session_state.messages:
            if st.button("🗑️ Clear Chat History"):
                st.session_state.messages = []
                st.rerun()


if __name__ == "__main__":
    if "processed" not in st.session_state:
        st.session_state.processed = False
    main()
