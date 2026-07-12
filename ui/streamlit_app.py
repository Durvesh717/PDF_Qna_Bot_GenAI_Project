import os
import re
from pathlib import Path

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
from ingestion.vectorstore import (
    create_collection,
    delete_collection,
    get_vector_store,
    list_collections,
)

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
        font-weight: 800;
        text-align: center;
        background: linear-gradient(45deg, #1f77b4, #9c27b0);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 1.5rem;
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


def create_agent(vector_store, settings):
    return CRAGAgent(vector_store, settings)


# Retrieved PDF/web text is untrusted and flows into the model's answer. A
# markdown image the model was tricked into emitting would make the viewer's
# browser fetch an attacker URL (data exfiltration), so strip images before
# rendering. HTML is already inert because st.markdown escapes it by default.
_MARKDOWN_IMAGE = re.compile(r"!\[[^\]]*\]\([^)]*\)")


def _sanitize_response(text: str) -> str:
    """Neutralize markdown images in model output to prevent data exfiltration."""
    return _MARKDOWN_IMAGE.sub("[image removed]", text)


def _has_required_keys(settings) -> bool:
    """Check that the selected providers have credentials available in settings."""
    providers = {settings.llm_provider, settings.embedding_provider, settings.vision_provider}
    if "google" in providers and not settings.google_api_key:
        return False
    if "openai" in providers and not settings.openai_api_key:
        return False
    if "bedrock" in providers and (not settings.aws_access_key_id or not settings.aws_secret_access_key):
        return False
    return True


def _api_key_input(label: str, current: str | None, help_text: str) -> str:
    """Render a password field that NEVER pre-fills the secret into the page.

    Pre-filling `value=` would send the key to the browser (readable via the
    reveal icon). Instead we render an empty field and only note that a key is
    already configured; a typed value is treated as an override.
    """
    typed = st.text_input(label, type="password", value="", help=help_text)
    if current and not typed:
        st.caption("✓ Loaded from environment")
    return typed


def _save_upload(file, upload_dir: Path) -> Path:
    """Persist uploaded file to local storage."""
    upload_dir.mkdir(parents=True, exist_ok=True)
    # Browser-supplied names can contain path separators; keep only the basename.
    safe_name = Path(file.name).name
    dest = upload_dir / safe_name
    content = file.getvalue()
    # A different file with the same name gets a numbered suffix instead of
    # overwriting; re-uploading the identical file reuses its path.
    counter = 1
    while dest.exists() and dest.read_bytes() != content:
        dest = upload_dir / f"{Path(safe_name).stem}_{counter}{Path(safe_name).suffix}"
        counter += 1
    dest.write_bytes(content)
    return dest


def _load_collection(collection_name: str, settings):
    """Load an existing collection into session state."""
    vector_store = get_vector_store(collection_name, settings)
    st.session_state.vector_store = vector_store
    st.session_state.agent = create_agent(vector_store, settings)
    st.session_state.processed = True
    st.session_state.current_collection = collection_name


def main():
    # Each browser session gets its own copy of the settings so sidebar
    # changes never leak into other sessions via the cached singleton.
    if "settings" not in st.session_state:
        st.session_state.settings = get_settings().model_copy(deep=True)
    settings = st.session_state.settings
    configure_langsmith(settings)

    # Initialize session state
    if "processed" not in st.session_state:
        st.session_state.processed = False
    if "current_collection" not in st.session_state:
        st.session_state.current_collection = "default"
    if "messages" not in st.session_state:
        st.session_state.messages = []

    st.markdown('<h1 class="main-header">📚 PDF Q&A Bot</h1>', unsafe_allow_html=True)
    st.markdown("Upload your PDF documents and ask questions about their content!")

    if flash := st.session_state.pop("flash", None):
        st.success(flash)

    with st.sidebar:
        st.markdown('<div class="sidebar-content">', unsafe_allow_html=True)
        st.header("📁 Upload & Settings")

        st.markdown("#### 🔑 API Keys")
        st.caption("Keys are read from your `.env`. Fields below only override them for this session.")
        google_api_key = _api_key_input(
            "Google AI Studio API Key", settings.google_api_key, "Required if using Google provider"
        )
        openai_api_key = _api_key_input(
            "OpenAI API Key", settings.openai_api_key, "Required if using OpenAI provider"
        )
        aws_access_key_id = _api_key_input(
            "AWS Access Key ID", settings.aws_access_key_id, "Required if using Bedrock provider"
        )
        aws_secret_access_key = _api_key_input(
            "AWS Secret Access Key", settings.aws_secret_access_key, "Required if using Bedrock provider"
        )

        # Apply typed overrides to this session's settings and to the process
        # env (some client libraries read the env directly). Empty fields leave
        # the .env-loaded values untouched.
        if google_api_key:
            settings.google_api_key = google_api_key
            os.environ["GOOGLE_API_KEY"] = google_api_key
        if openai_api_key:
            settings.openai_api_key = openai_api_key
            os.environ["OPENAI_API_KEY"] = openai_api_key
        if aws_access_key_id:
            settings.aws_access_key_id = aws_access_key_id
            os.environ["AWS_ACCESS_KEY_ID"] = aws_access_key_id
        if aws_secret_access_key:
            settings.aws_secret_access_key = aws_secret_access_key
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

        # Update the session's settings copy for downstream modules
        settings.llm_provider = llm_provider
        settings.llm_model = llm_model
        settings.embedding_provider = embedding_provider
        settings.embedding_model = embedding_model
        settings.vision_provider = vision_provider
        settings.vision_model = vision_model

        settings.enable_web_search = st.toggle(
            "Enable web search fallback",
            value=settings.enable_web_search,
            help=(
                "When on, questions your documents can't answer are sent to a web "
                "search engine (a third party). Off by default to keep queries local."
            ),
        )

        keys_ok = _has_required_keys(settings)

        st.markdown("---")
        st.markdown("#### 📤 Upload Documents")

        collection_name = st.text_input(
            "Collection name",
            value=st.session_state.current_collection,
            help="All uploaded files go into this collection. You can chat across them.",
        )

        uploaded_files = st.file_uploader(
            "Choose one or more PDF files",
            type="pdf",
            accept_multiple_files=True,
            help="Upload multiple PDFs to chat across all of them",
        )

        if uploaded_files and keys_ok:
            st.success(f"✅ {len(uploaded_files)} file(s) selected")

            if st.button("🔄 Process PDFs", type="primary"):
                with st.spinner("Processing PDFs..."):
                    all_chunks = []

                    for uploaded_file in uploaded_files:
                        saved_path = _save_upload(uploaded_file, settings.uploads_dir)
                        merged_docs = extract_content_from_pdf(saved_path, settings)
                        if merged_docs:
                            chunks = split_documents(merged_docs, settings)
                            all_chunks.extend(chunks)

                    if all_chunks:
                        vector_store = create_collection(all_chunks, collection_name, settings)
                        st.session_state.vector_store = vector_store
                        st.session_state.agent = create_agent(vector_store, settings)
                        st.session_state.processed = True
                        st.session_state.current_collection = collection_name
                        st.session_state.messages = []
                        st.session_state.flash = (
                            f"✅ Processed {len(uploaded_files)} file(s) into collection '{collection_name}'!"
                        )
                        st.rerun()
                    else:
                        st.error("No content could be extracted from the uploaded PDFs.")

        st.markdown("---")
        st.markdown("#### 🗂️ Collection Manager")

        try:
            existing_collections = list_collections(settings)
        except Exception:
            existing_collections = []

        if existing_collections:
            selected_collection = st.selectbox(
                "Select active collection",
                options=existing_collections,
                index=existing_collections.index(st.session_state.current_collection)
                if st.session_state.current_collection in existing_collections
                else 0,
            )

            col1, col2 = st.columns(2)
            with col1:
                if st.button("Load Collection"):
                    _load_collection(selected_collection, settings)
                    st.session_state.flash = f"Loaded collection '{selected_collection}'"
                    st.rerun()
            with col2:
                confirm_delete = st.checkbox(
                    "Confirm", key="confirm_delete", help="Deletion is permanent"
                )
                if st.button("Delete Collection", disabled=not confirm_delete):
                    try:
                        delete_collection(selected_collection, settings)
                        if st.session_state.get("current_collection") == selected_collection:
                            st.session_state.processed = False
                            st.session_state.messages = []
                        st.session_state.flash = f"Deleted collection '{selected_collection}'"
                        st.rerun()
                    except Exception:
                        logger.exception("Failed to delete collection")
                        st.error("Failed to delete collection. See server logs for details.")
        else:
            st.info("No collections yet. Upload a PDF to create one.")

        st.markdown("</div>", unsafe_allow_html=True)
        st.markdown("---")
        st.markdown("### 📋 Instructions:")
        st.markdown(
            """
1. Enter API keys for the providers you selected
2. Choose or create a collection name
3. Upload one or more PDF files
4. Click 'Process PDFs'
5. Start asking questions across all uploaded documents!
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

    if not keys_ok:
        st.warning("⚠️ Please enter the required API keys for your selected providers in the sidebar.")
        st.info("Google: https://aistudio.google.com/app/apikey")
        st.info("OpenAI: https://platform.openai.com/api-keys")
        st.info("AWS Bedrock: https://aws.amazon.com/console/")
    elif not st.session_state.get("processed"):
        st.info("📄 Please upload PDF files and click 'Process PDFs' to begin asking questions.")
    else:
        st.caption(f"💬 Chatting with collection: **{st.session_state.current_collection}**")

        # Render chat history using native Streamlit chat UI
        for message in st.session_state.messages:
            role = "user" if message["role"] == "user" else "assistant"
            with st.chat_message(role):
                st.markdown(message["content"])

        # Chat input element
        if "current_question" in st.session_state:
            user_question = st.session_state.current_question
            del st.session_state.current_question
        else:
            user_question = st.chat_input("Ask a question about your PDFs...")

        if user_question:
            # Display user message in chat container immediately
            with st.chat_message("user"):
                st.markdown(user_question)

            st.session_state.messages.append({"role": "user", "content": user_question})

            # Display assistant response
            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    try:
                        history = st.session_state.messages[:-1][-settings.max_chat_history:]
                        result = st.session_state.agent.invoke(
                            user_question,
                            chat_history=history,
                        )
                        response = _sanitize_response(
                            result.get("generation", "I could not generate an answer.")
                        )
                        st.markdown(response)
                        st.session_state.messages.append(
                            {"role": "assistant", "content": response}
                        )
                        st.rerun()
                    except Exception:
                        logger.exception("Error generating response")
                        error_msg = "Sorry, I hit an error generating a response. See server logs for details."
                        st.error(error_msg)
                        st.session_state.messages.append(
                            {"role": "assistant", "content": error_msg}
                        )

        if st.session_state.messages:
            st.markdown("---")
            if st.button("🗑️ Clear Chat History"):
                st.session_state.messages = []
                st.rerun()


if __name__ == "__main__":
    main()
