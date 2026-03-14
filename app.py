import streamlit as st
import os
import tempfile
from document_loader import load_document
from rag_pipeline import process_documents, get_answer, summarize_document, suggest_questions

st.set_page_config(
    page_title="Document Intelligence System",
    page_icon="🧠",
    layout="wide"
)

st.markdown("""
<style>
    .main-header {
        background: linear-gradient(90deg, #1e3a5f, #2196F3);
        padding: 20px;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin-bottom: 20px;
    }
    .source-badge {
        background-color: #e3f2fd;
        border-left: 4px solid #2196F3;
        padding: 8px 12px;
        border-radius: 4px;
        font-size: 13px;
        color: #1565c0;
        margin-top: 8px;
    }
    .summary-box {
        background-color: #f3f9f3;
        border-left: 4px solid #4CAF50;
        padding: 16px;
        border-radius: 4px;
        margin-top: 10px;
    }
    .hybrid-badge {
        background-color: #fff3e0;
        border-left: 4px solid #FF9800;
        padding: 6px 12px;
        border-radius: 4px;
        font-size: 12px;
        color: #e65100;
        margin-bottom: 10px;
        display: inline-block;
    }
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="main-header">
    <h1>🧠 Universal Document Intelligence System</h1>
    <p>Upload documents and get AI-powered answers instantly</p>
</div>
""", unsafe_allow_html=True)

if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None
if "retriever" not in st.session_state:
    st.session_state.retriever = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "doc_names" not in st.session_state:
    st.session_state.doc_names = []
if "doc_texts" not in st.session_state:
    st.session_state.doc_texts = {}

with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/artificial-intelligence.png", width=80)
    st.title("Control Panel")
    st.divider()

    st.subheader("🔑 API Settings")
    try:
        api_key = st.secrets.get("GROQ_API_KEY", "") or st.text_input("Groq API Key", type="password", placeholder="gsk_...")
    except:
        api_key = st.text_input("Groq API Key", type="password", placeholder="gsk_...")

    st.divider()
    st.subheader("📂 Upload Documents")
    uploaded_files = st.file_uploader(
        "Supported: PDF, DOCX, TXT, PPTX",
        accept_multiple_files=True,
        type=["pdf", "docx", "txt", "pptx"]
    )

    if uploaded_files:
        st.info(f"{len(uploaded_files)} file(s) selected")

    if uploaded_files and st.button("⚡ Process Documents", use_container_width=True):
        with st.spinner("Reading and indexing documents..."):
            all_texts = []
            filenames = []
            doc_texts = {}
            for uploaded_file in uploaded_files:
                with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.name)[1]) as tmp:
                    tmp.write(uploaded_file.read())
                    tmp_path = tmp.name
                text = load_document(tmp_path)
                all_texts.append(text)
                filenames.append(uploaded_file.name)
                doc_texts[uploaded_file.name] = text
                os.unlink(tmp_path)

            vectorstore, retriever = process_documents(all_texts, filenames)
            st.session_state.vectorstore = vectorstore
            st.session_state.retriever = retriever
            st.session_state.doc_names = filenames
            st.session_state.doc_texts = doc_texts
            st.session_state.chat_history = []
            st.success("Documents ready!")

    st.divider()

    if st.session_state.doc_names:
        st.subheader("📄 Loaded Documents")
        for name in st.session_state.doc_names:
            st.markdown(f"- {name}")

    st.divider()

    if st.session_state.chat_history:
        st.subheader("📊 Session Stats")
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Questions", len(st.session_state.chat_history))
        with col2:
            st.metric("Documents", len(st.session_state.doc_names))

    if st.button("🗑️ Clear Chat", use_container_width=True):
        st.session_state.chat_history = []
        st.rerun()

# Main area
if st.session_state.vectorstore is None:
    col1, col2, col3 = st.columns(3)
    with col1:
        st.info("**Step 1**\n\nEnter your Groq API key in the sidebar")
    with col2:
        st.info("**Step 2**\n\nUpload your documents (PDF, DOCX, TXT, PPTX)")
    with col3:
        st.info("**Step 3**\n\nAsk questions or summarize your documents!")
else:
    tab1, tab2 = st.tabs(["💬 Chat", "📋 Summarize & Suggest"])

    with tab1:
        st.markdown(
            '<div class="hybrid-badge">⚡ Hybrid Search Active — Vector + Keyword</div>',
            unsafe_allow_html=True
        )
        st.subheader("💬 Conversation")
        for msg in st.session_state.chat_history:
            with st.chat_message("user", avatar="👤"):
                st.write(msg["question"])
            with st.chat_message("assistant", avatar="🧠"):
                st.write(msg["answer"])
                if msg.get("sources"):
                    for source in msg["sources"]:
                        st.markdown(
                            f'<div class="source-badge">📄 Source: {source}</div>',
                            unsafe_allow_html=True
                        )

    with tab2:
        st.subheader("📋 Document Summarization")
        st.markdown("Select a document to get an instant AI summary.")
        selected_doc = st.selectbox(
            "Choose a document to summarize",
            st.session_state.doc_names
        )
        if st.button("📋 Generate Summary", use_container_width=True):
            if not api_key:
                st.warning("Please enter your Groq API key in the sidebar!")
            else:
                with st.spinner(f"Summarizing {selected_doc}..."):
                    text = st.session_state.doc_texts[selected_doc]
                    summary = summarize_document(text, selected_doc, api_key)
                    st.markdown(
                        f'<div class="summary-box">{summary}</div>',
                        unsafe_allow_html=True
                    )

        st.divider()

        st.subheader("💡 Smart Question Suggestions")
        st.markdown("Get AI-suggested questions based on your document.")
        selected_doc_q = st.selectbox(
            "Choose a document for question suggestions",
            st.session_state.doc_names,
            key="qsuggest"
        )
        if st.button("💡 Suggest Questions", use_container_width=True):
            if not api_key:
                st.warning("Please enter your Groq API key in the sidebar!")
            else:
                with st.spinner("Generating smart questions..."):
                    text = st.session_state.doc_texts[selected_doc_q]
                    questions = suggest_questions(text, selected_doc_q, api_key)
                    st.markdown("**Click any question to ask it directly:**")
                    for q in questions:
                        if st.button(f"❓ {q}", use_container_width=True, key=q):
                            if st.session_state.vectorstore is None:
                                st.warning("Please process documents first!")
                            else:
                                with st.spinner("🧠 Thinking..."):
                                    answer, sources = get_answer(
                                        st.session_state.vectorstore,
                                        q,
                                        api_key,
                                        st.session_state.chat_history,
                                        st.session_state.retriever
                                    )
                                    st.session_state.chat_history.append({
                                        "question": q,
                                        "answer": answer,
                                        "sources": sources
                                    })
                                    st.rerun()

# Chat input
question = st.chat_input("Ask a question about your documents...")

if question:
    if st.session_state.vectorstore is None:
        st.warning("Please upload and process documents first!")
    elif not api_key:
        st.warning("Please enter your Groq API key in the sidebar!")
    else:
        with st.spinner("🧠 Thinking..."):
            answer, sources = get_answer(
                st.session_state.vectorstore,
                question,
                api_key,
                st.session_state.chat_history,
                st.session_state.retriever
            )
            st.session_state.chat_history.append({
                "question": question,
                "answer": answer,
                "sources": sources
            })
            st.rerun()