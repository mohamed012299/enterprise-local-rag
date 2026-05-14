import os
import tempfile
import streamlit as st

from langchain_core.documents import Document

from langchain_community.document_loaders import (
    PyPDFLoader,
    TextLoader,
    Docx2txtLoader
)

from langchain_text_splitters import (
    RecursiveCharacterTextSplitter
)

from langchain_ollama import (
    OllamaEmbeddings,
    ChatOllama
)

from langchain_community.vectorstores import Chroma

# =========================================================
# IMPORT OCR MODULE
# =========================================================

from rag.ocr import extract_text_with_ocr

# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="Enterprise Local RAG",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =========================================================
# SESSION STATE
# =========================================================

if "messages" not in st.session_state:
    st.session_state.messages = []

if "retriever" not in st.session_state:
    st.session_state.retriever = None

if "llm" not in st.session_state:
    st.session_state.llm = None

if "processed_files" not in st.session_state:
    st.session_state.processed_files = []

if "theme" not in st.session_state:
    st.session_state.theme = "Dark"

# =========================================================
# THEMES
# =========================================================

themes = {

    "Dark": {
        "bg": "#0f1117",
        "sidebar": "#111827",
        "card": "#1f2937",
        "text": "#ffffff",
        "user": "#2563eb",
        "bot": "#1a1a1a",
        "border": "#374151",
    },

    "Light": {
        "bg": "#ffffff",
        "sidebar": "#f3f4f6",
        "card": "#f9fafb",
        "text": "#000000",
        "user": "#2563eb",
        "bot": "#f3f4f6",
        "border": "#d1d5db",
    },

    "Ocean": {
        "bg": "#07111f",
        "sidebar": "#0b1727",
        "card": "#122033",
        "text": "#ffffff",
        "user": "#2563eb",
        "bot": "#16263d",
        "border": "#274060",
    }
}

theme = themes[st.session_state.theme]

# =========================================================
# VECTOR DATABASE
# =========================================================

DB_FOLDER = "enterprise_vector_db"

if not os.path.exists(DB_FOLDER):
    os.makedirs(DB_FOLDER)

# =========================================================
# CUSTOM CSS
# =========================================================

st.markdown(f"""
<style>

.stApp {{
    background: {theme["bg"]};
    color: {theme["text"]};
}}

section[data-testid="stSidebar"] {{
    background: {theme["sidebar"]};
    border-right: 1px solid {theme["border"]};
}}

.main-title {{
    font-size: 54px;
    font-weight: 800;
    margin-bottom: 15px;
}}

.card {{
    background: {theme["card"]};
    border: 1px solid {theme["border"]};
    border-radius: 24px;
    padding: 22px;
    margin-bottom: 18px;
}}

.user-message {{
    background: linear-gradient(135deg,#2563eb,#1d4ed8);
    color: white;
    padding: 18px;
    border-radius: 20px;
    margin-left: 120px;
    margin-bottom: 16px;
}}

.bot-message {{
    background: {theme["bot"]};
    color: {theme["text"]};
    padding: 22px;
    border-radius: 20px;
    margin-right: 120px;
    margin-bottom: 18px;
    border: 1px solid {theme["border"]};
    line-height: 1.8;
}}

.stButton button {{
    background: linear-gradient(135deg,#111,#333);
    color: white;
    border-radius: 14px;
    border: none;
    padding: 12px 18px;
}}

[data-testid="stFileUploader"] {{
    background: {theme["card"]};
    border: 2px dashed {theme["border"]};
    border-radius: 20px;
    padding: 15px;
}}

</style>
""", unsafe_allow_html=True)

# =========================================================
# SIDEBAR
# =========================================================

with st.sidebar:

    st.markdown("""
    <h1 style='font-size:32px;'>
    🤖 Enterprise RAG
    </h1>
    """, unsafe_allow_html=True)

    st.markdown("---")

    # =====================================================
    # THEME
    # =====================================================

    selected_theme = st.selectbox(
        "🎨 Theme",
        ["Dark", "Light", "Ocean"]
    )

    st.session_state.theme = selected_theme

    st.markdown("---")

    # =====================================================
    # AI MODE
    # =====================================================

    st.markdown("## 🧠 AI Mode")

    ai_mode = st.radio(
        "Choose Assistant Mode",
        [
            "Chat",
            "Study",
            "Summarizer",
            "Business",
            "Legal"
        ]
    )

    st.markdown("---")

    # =====================================================
    # FILE UPLOAD
    # =====================================================

    st.markdown("## 📂 Upload Files")

    uploaded_files = st.file_uploader(
        "Drag & Drop Files",
        type=["pdf", "txt", "docx"],
        accept_multiple_files=True,
        key="multi_uploader"
    )

    # =====================================================
    # STATS
    # =====================================================

    st.markdown("---")

    st.markdown("## 📊 Workspace Stats")

    total_files = len(
        st.session_state.processed_files
    )

    total_messages = len(
        st.session_state.messages
    )

    col1, col2 = st.columns(2)

    with col1:

        st.metric(
            "Files",
            total_files
        )

    with col2:

        st.metric(
            "Chats",
            total_messages
        )

    # =====================================================
    # FILE MANAGER
    # =====================================================

    st.markdown("---")

    st.markdown("## 🗂 Loaded Files")

    if st.session_state.processed_files:

        for file in st.session_state.processed_files:

            extension = file.split(".")[-1]

            if extension == "pdf":
                icon = "📕"

            elif extension == "docx":
                icon = "📘"

            else:
                icon = "📄"

            st.markdown(f"""
            <div style="
                padding:12px;
                margin-bottom:10px;
                border-radius:14px;
                background:#1e293b;
                border:1px solid #334155;
                font-size:14px;
            ">
                {icon} {file}
            </div>
            """, unsafe_allow_html=True)

    else:

        st.info("No files uploaded yet.")

    # =====================================================
    # ACTIONS
    # =====================================================

    st.markdown("---")

    st.markdown("## ⚙ Actions")

    if st.button("🗑 Clear Chat"):

        st.session_state.messages = []

        st.rerun()

    if st.button("🧠 Reset Knowledge Base"):

        st.session_state.retriever = None

        st.session_state.llm = None

        st.session_state.processed_files = []

        st.session_state.messages = []

        st.success(
            "Knowledge Base Reset Complete"
        )

        st.rerun()

    # =====================================================
    # SYSTEM STATUS
    # =====================================================

    st.markdown("---")

    st.markdown("## 🟢 System Status")

    st.success("LLM Connected")

    st.success("Vector DB Ready")

    st.success("OCR Enabled")

# =========================================================
# AI MODES
# =========================================================

if ai_mode == "Chat":

    system_prompt = """
    You are a smart enterprise AI assistant.

    RULES:
    - Reply ONLY in Arabic or English.
    - Never use Chinese or any other language.
    - If user speaks Arabic reply in Arabic.
    - If user speaks English reply in English.
    - Use ONLY provided context.
    """

elif ai_mode == "Study":

    system_prompt = """
    You are an educational AI tutor.

    RULES:
    - Explain concepts simply.
    - Use bullet points.
    - Give examples.
    - Reply ONLY in Arabic or English.
    - Use ONLY provided context.
    """

elif ai_mode == "Summarizer":

    system_prompt = """
    You are a professional summarization assistant.

    RULES:
    - Summarize documents clearly.
    - Extract key points.
    - Keep answers concise.
    - Reply ONLY in Arabic or English.
    - Use ONLY provided context.
    """

elif ai_mode == "Business":

    system_prompt = """
    You are a business analyst AI assistant.

    RULES:
    - Analyze business information professionally.
    - Focus on insights and recommendations.
    - Reply ONLY in Arabic or English.
    - Use ONLY provided context.
    """

elif ai_mode == "Legal":

    system_prompt = """
    You are a legal AI assistant.

    RULES:
    - Analyze legal text carefully.
    - Highlight risks and important clauses.
    - Reply ONLY in Arabic or English.
    - Use ONLY provided context.
    """

# =========================================================
# HEADER
# =========================================================

st.markdown("""
<div class="main-title">
🚀 Enterprise Local RAG
</div>
""", unsafe_allow_html=True)

st.markdown("""
<div class="card">

<h3>📚 AI Company Knowledge Assistant</h3>

<p>
Upload company files and chat with them locally.
Supports scanned PDFs and OCR.
</p>

</div>
""", unsafe_allow_html=True)

# =========================================================
# PROCESS FILES
# =========================================================

if uploaded_files and st.session_state.retriever is None:

    with st.spinner("⚡ Building AI Knowledge Base..."):

        all_documents = []

        for uploaded_file in uploaded_files:

            file_extension = uploaded_file.name.split(".")[-1].lower()

            with tempfile.NamedTemporaryFile(
                delete=False,
                suffix=f".{file_extension}"
            ) as tmp_file:

                tmp_file.write(uploaded_file.read())
                temp_path = tmp_file.name

            try:

                # PDF
                if file_extension == "pdf":

                    loader = PyPDFLoader(temp_path)

                    documents = loader.load()

                    pdf_text = ""

                    for doc in documents:

                        if doc.page_content:
                            pdf_text += doc.page_content.strip()

                    # OCR fallback
                    if len(pdf_text) < 20:

                        st.warning(
                            f"⚡ Using OCR for scanned PDF: {uploaded_file.name}"
                        )

                        ocr_text = extract_text_with_ocr(temp_path)

                        if ocr_text.strip():

                            documents = [
                                Document(
                                    page_content=ocr_text
                                )
                            ]

                        else:

                            st.error(
                                f"❌ OCR failed for {uploaded_file.name}"
                            )

                            continue

                # TXT
                elif file_extension == "txt":

                    loader = TextLoader(
                        temp_path,
                        encoding="utf-8"
                    )

                    documents = loader.load()

                # DOCX
                elif file_extension == "docx":

                    loader = Docx2txtLoader(temp_path)

                    documents = loader.load()

                else:
                    continue

                all_documents.extend(documents)

                st.session_state.processed_files.append(
                    uploaded_file.name
                )

            except Exception:

                st.warning(
                    f"Could not process {uploaded_file.name}"
                )

        # =====================================================
        # SPLITTER
        # =====================================================

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1200,
            chunk_overlap=250
        )

        texts = text_splitter.split_documents(
            all_documents
        )

        texts = [
            t for t in texts
            if t.page_content
            and t.page_content.strip()
            and len(t.page_content.strip()) > 5
        ]

        # =====================================================
        # EMBEDDINGS
        # =====================================================

        embeddings = OllamaEmbeddings(
            model="nomic-embed-text"
        )

        # =====================================================
        # VECTOR STORE
        # =====================================================

        vectorstore = Chroma.from_documents(
            documents=texts,
            embedding=embeddings,
            persist_directory=DB_FOLDER
        )

        retriever = vectorstore.as_retriever(
            search_kwargs={"k": 5}
        )

        # =====================================================
        # LLM
        # =====================================================

        llm = ChatOllama(
            model="qwen2.5:3b",
            temperature=0.3
        )

        st.session_state.retriever = retriever
        st.session_state.llm = llm

    st.success("✅ Enterprise Knowledge Base Ready")

# =====================================================
# DISPLAY CHAT
# =====================================================

for message in st.session_state.messages:

    if message["role"] == "user":

        st.markdown(f"""
        <div class="user-message">
        {message["content"]}
        </div>
        """, unsafe_allow_html=True)

    else:

        st.markdown(f"""
        <div class="bot-message">
        {message["content"]}
        </div>
        """, unsafe_allow_html=True)

# =====================================================
# CHAT INPUT
# =====================================================

if st.session_state.retriever:

    question = st.chat_input(
        "💬 Ask anything about your files..."
    )

    if question:

        st.session_state.messages.append({
            "role": "user",
            "content": question
        })

        st.markdown(f"""
        <div class="user-message">
        {question}
        </div>
        """, unsafe_allow_html=True)

        docs = st.session_state.retriever.invoke(
            question
        )

        # =====================================================
        # CONTEXT + SOURCES
        # =====================================================

        context = ""

        sources = []

        for doc in docs:

            context += doc.page_content + "\n\n"

            source = doc.metadata.get(
                "source",
                "Unknown File"
            )

            page = doc.metadata.get(
                "page",
                "N/A"
            )

            source_text = (
                f"📄 {os.path.basename(source)} | Page {page}"
            )

            if source_text not in sources:

                sources.append(source_text)

        # =====================================================
        # HISTORY
        # =====================================================

        history = "\n".join([
            f"{m['role']}: {m['content']}"
            for m in st.session_state.messages[-8:]
        ])

        # =====================================================
        # PROMPT
        # =====================================================

        prompt = f"""
        {system_prompt}

        Conversation History:
        {history}

        Context:
        {context}

        User Question:
        {question}
        """

        # =====================================================
        # STREAMING RESPONSE
        # =====================================================

        response_placeholder = st.empty()

        full_response = ""

        for chunk in st.session_state.llm.stream(
            prompt
        ):

            if chunk.content:

                full_response += chunk.content

                response_placeholder.markdown(f"""
                <div class="bot-message">
                {full_response}▌
                </div>
                """, unsafe_allow_html=True)

        response_placeholder.markdown(f"""
        <div class="bot-message">
        {full_response}
        </div>
        """, unsafe_allow_html=True)

        answer = full_response

        # =====================================================
        # SAVE MESSAGE
        # =====================================================

        st.session_state.messages.append({
            "role": "assistant",
            "content": answer
        })

        # =====================================================
        # SOURCES
        # =====================================================

        if sources:

            sources_html = "<br>".join(sources)

            st.markdown(f"""
            <div class="card">

            <h4>📚 Sources</h4>

            {sources_html}

            </div>
            """, unsafe_allow_html=True)

else:

    st.markdown("""
    <div class="bot-message">

    📂 Upload files to start chatting.

    </div>
    """, unsafe_allow_html=True)