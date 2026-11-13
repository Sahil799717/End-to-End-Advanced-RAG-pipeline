
import logging
from pathlib import Path

import streamlit as st

from config import (
    PROJECT_ROOT,
    PDF_PATH,
    LLM_MODEL,
    EMBEDDING_MODEL,
    RERANKER_MODEL,
)

from rag_backend import build_agent, ask_question


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ============================================================
# 1. PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Agentic RAG Assistant",
    page_icon="📚",
    layout="wide",
)

st.title("📚 Agentic RAG Assistant")
st.caption("Ask questions about your uploaded temple heritage PDF.")


# ============================================================
# 2. SESSION STATE
# ============================================================

if "messages" not in st.session_state:
    st.session_state.messages = []

if "agent" not in st.session_state:
    st.session_state.agent = None

if "active_pdf" not in st.session_state:
    st.session_state.active_pdf = str(PDF_PATH)


# ============================================================
# 3. SIDEBAR
# ============================================================

with st.sidebar:

    st.header("Knowledge Source")

    uploaded_pdf = st.file_uploader(
        "Upload a PDF",
        type=["pdf"],
        help="Upload a PDF to create or rebuild the knowledge base.",
    )

    rebuild_clicked = st.button(
        "Build / Rebuild Knowledge Base",
        use_container_width=True,
    )

    st.divider()

    st.subheader("Pipeline Configuration")

    st.write(f"**LLM:** {LLM_MODEL}")
    st.write(f"**Embeddings:** {EMBEDDING_MODEL}")
    st.write(f"**Reranker:** {RERANKER_MODEL}")
    st.write("**Vector Store:** Chroma")
    st.write("**Orchestration:** LangGraph")

    st.divider()

    if st.button("Clear Chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()


# ============================================================
# 4. PDF REBUILD
# ============================================================

if rebuild_clicked:

    try:
        if uploaded_pdf is not None:

            if uploaded_pdf.size == 0:
                st.error("The uploaded PDF is empty.")
                st.stop()

            filename = Path(uploaded_pdf.name).name

            save_path = (
                PROJECT_ROOT
                / "Knowledge_source"
                / filename
            )

            save_path.parent.mkdir(
                parents=True,
                exist_ok=True,
            )

            with open(save_path, "wb") as file:
                file.write(uploaded_pdf.getbuffer())

            pdf_to_index = save_path

        else:
            pdf_to_index = PDF_PATH

        if not Path(pdf_to_index).exists():
            st.error(f"PDF file not found: {pdf_to_index}")
            st.stop()

        with st.spinner("Building knowledge base..."):

            st.session_state.agent = build_agent(
                pdf_path=pdf_to_index,
                rebuild=True,
            )

            st.session_state.active_pdf = str(pdf_to_index)
            st.session_state.messages = []

        st.success("Knowledge base rebuilt successfully.")

    except Exception as error:

        logger.exception("Knowledge base rebuild failed.")

        st.error(f"Unable to rebuild the knowledge base: {error}")


# ============================================================
# 5. INITIALIZE AGENT
# ============================================================

if st.session_state.agent is None:

    try:
        if not Path(st.session_state.active_pdf).exists():

            st.warning(
                "Default PDF not found. Upload a PDF and build "
                "the knowledge base from the sidebar."
            )

        else:

            with st.spinner("Initializing RAG agent..."):

                st.session_state.agent = build_agent(
                    pdf_path=st.session_state.active_pdf
                )

            st.success("RAG agent is ready.")

    except Exception as error:

        logger.exception("Agent initialization failed.")

        st.error(f"Unable to initialize the agent: {error}")


# ============================================================
# 6. DISPLAY CHAT HISTORY
# ============================================================

for message in st.session_state.messages:

    with st.chat_message(message["role"]):

        st.markdown(message["content"])

        if message["role"] == "assistant":

            sources = message.get("sources", [])

            if sources:

                with st.expander(f"Sources ({len(sources)})"):

                    for index, document in enumerate(
                        sources,
                        start=1,
                    ):

                        metadata = document.get("metadata", {})

                        page = metadata.get(
                            "page_label",
                            metadata.get("page", "Unknown"),
                        )

                        st.markdown(f"**Source {index} — Page {page}**")
                        st.write(document.get("content", ""))
                        st.caption(f"Metadata: {metadata}")

            agent_info = message.get("agent_info")

            if agent_info:

                with st.expander("Agent Execution Details"):
                    st.json(agent_info)


# ============================================================
# 7. CHAT INPUT
# ============================================================

question = st.chat_input("Ask a question about the PDF...")

if question:

    if st.session_state.agent is None:

        st.warning(
            "The RAG agent is not initialized. "
            "Please build the knowledge base first."
        )

    else:

        st.session_state.messages.append({
            "role": "user",
            "content": question,
        })

        with st.chat_message("user"):
            st.markdown(question)

        try:

            with st.chat_message("assistant"):

                with st.spinner("Thinking..."):

                    result = ask_question(
                        question,
                        
                    )

                answer = result.get(
                    "answer",
                    "No answer was generated.",
                )

                st.markdown(answer)

                source_documents = result.get(
                    "source_documents",
                    [],
                )

                if source_documents:

                    with st.expander(
                        f"Sources ({len(source_documents)})"
                    ):

                        for index, document in enumerate(
                            source_documents,
                            start=1,
                        ):

                            metadata = document.metadata or {}

                            page = metadata.get(
                                "page_label",
                                metadata.get("page", "Unknown"),
                            )

                            st.markdown(
                                f"**Source {index} — Page {page}**"
                            )

                            st.write(document.page_content)
                            st.caption(f"Metadata: {metadata}")

                # Include the actual validation error
                agent_info = {
                    "is_relevant": result.get("is_relevant"),
                    "attempts": result.get("attempts"),
                    "grade_reason": result.get("grade_reason"),
                    "search_query": result.get("search_query"),
                    "guardrail_error": result.get("guardrail_error"),
                }

                with st.expander("Agent Execution Details"):
                    st.json(agent_info)

            st.session_state.messages.append({
                "role": "assistant",
                "content": answer,
                "sources": [
                    {
                        "content": doc.page_content,
                        "metadata": doc.metadata or {},
                    }
                    for doc in source_documents
                ],
                "agent_info": agent_info,
            })

        except Exception as error:

            logger.exception("Question answering failed.")

            error_message = (
                f"An error occurred while answering: {error}"
            )

            st.error(error_message)

            st.session_state.messages.append({
                "role": "assistant",
                "content": error_message,
            })
