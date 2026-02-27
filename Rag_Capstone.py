import os
import shutil
import warnings
from pathlib import Path
import streamlit as st
from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.vectorstores import FAISS

# Gemini
from langchain_google_genai import (
    ChatGoogleGenerativeAI,
    GoogleGenerativeAIEmbeddings,
)

# OpenAI
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────
# Config
# ─────────────────────────────────────────
INDEX_DIR = "vector_store"

st.set_page_config(
    page_title="Document QA · RAG",
    page_icon="📄",
    layout="wide",
)

# ─────────────────────────────────────────
# Styling
# ─────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
*, html, body { font-family: 'Inter', sans-serif; }

/* Background */
.stApp { background: #0f1117; color: #e2e4ec; }

/* Headings */
h1 { color: #fff !important; font-size: 1.85rem !important; font-weight: 700 !important; margin-bottom: 2px !important; }
h3 { color: #fff !important; font-weight: 600 !important; }
p, label { color: #9aa0b4 !important; }

/* Tabs */
.stTabs [data-baseweb="tab-list"] { border-bottom: 1px solid #1e2130; gap: 0; }
.stTabs [data-baseweb="tab"] { background: transparent; color: #6b7280; border: none; padding: 10px 22px; font-weight: 500; font-size: 0.9rem; }
.stTabs [aria-selected="true"] { color: #e05c5c !important; border-bottom: 2px solid #e05c5c !important; background: transparent !important; }

/* Sidebar */
[data-testid="stSidebar"] { background: #0b0d14 !important; border-right: 1px solid #1a1d2e; }
[data-testid="stSidebar"] * { color: #9aa0b4; }
[data-testid="stSidebar"] h3 { color: #fff !important; }

/* Inputs */
.stTextInput > div > div > input {
    background: #161926 !important;
    border: 1px solid #252840 !important;
    color: #e2e4ec !important;
    border-radius: 8px !important;
    padding: 10px 14px !important;
    font-size: 0.95rem !important;
}
.stTextInput > div > div > input:focus { border-color: #3b82f6 !important; box-shadow: 0 0 0 2px rgba(59,130,246,0.15) !important; }

/* Buttons */
.stButton > button {
    background: #161926;
    border: 1px solid #252840;
    color: #c8cce0;
    border-radius: 8px;
    font-weight: 500;
    font-size: 0.88rem;
    transition: all 0.18s ease;
    padding: 8px 18px;
}
.stButton > button:hover { border-color: #3b82f6; color: #fff; background: #1a1f35; }

/* Selectbox */
.stSelectbox > div > div { background: #161926 !important; border-color: #252840 !important; color: #e2e4ec !important; border-radius: 8px !important; }

/* Answer card */
.answer-card {
    background: linear-gradient(135deg, #131726 0%, #161c2e 100%);
    border: 1px solid #1e2540;
    border-left: 4px solid #3b82f6;
    border-radius: 10px;
    padding: 20px 24px;
    margin: 14px 0 24px 0;
    color: #d4d8f0;
    line-height: 1.8;
    font-size: 0.97rem;
    white-space: pre-wrap;
    word-break: break-word;
}

/* Source card */
.source-card {
    background: #0f1220;
    border: 1px solid #1e2130;
    border-radius: 8px;
    padding: 14px 18px;
    margin: 10px 0;
    color: #a0a8c0;
    font-size: 0.875rem;
    line-height: 1.7;
}
.source-title { font-weight: 700; color: #e2e4ec; font-size: 0.9rem; margin-bottom: 8px; }
.source-meta { color: #414766; font-size: 0.76rem; margin-top: 10px; font-family: monospace; }

/* Status pills */
.pill-green { display:inline-block; background:#052e16; color:#4ade80; border:1px solid #166534; border-radius:20px; padding:3px 14px; font-size:0.8rem; font-weight:600; }
.pill-yellow { display:inline-block; background:#2d1a00; color:#fbbf24; border:1px solid #78350f; border-radius:20px; padding:3px 14px; font-size:0.8rem; font-weight:600; }
.pill-red   { display:inline-block; background:#2d0a0a; color:#f87171; border:1px solid #7f1d1d; border-radius:20px; padding:3px 14px; font-size:0.8rem; font-weight:600; }

/* Section micro-label */
.micro-label { color:#555e7a; font-size:0.75rem; font-weight:600; letter-spacing:0.07em; text-transform:uppercase; margin-bottom:5px; }

/* Info box */
.info-box { background:#0d1322; border:1px solid #1e2540; border-radius:8px; padding:14px 18px; color:#7a82a0; font-size:0.88rem; line-height:1.65; }

hr { border-color:#1a1d2e !important; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────
# Helpers Methods — environment & models
# ─────────────────────────────────────────
# -----------------------------------------
# os.getenv("KEY", "default_value") - if the variable is not found, it uses the default value.
# -----------------------------------------
def load_env():
    load_dotenv(".env")
    return {
        "GOOGLE_API_KEY":         os.getenv("GOOGLE_API_KEY"),
        "GOOGLE_CHAT_MODEL_NAME": os.getenv("GOOGLE_CHAT_MODEL_NAME", "gemini-1.5-flash"),
        "GOOGLE_EMBED_MODEL_NAME":os.getenv("GOOGLE_EMBED_MODEL_NAME", "gemini-embedding-001"),
        "OPENAI_API_KEY":         os.getenv("OPENAI_API_KEY"),
        "OPENAI_CHAT_MODEL_NAME": os.getenv("OPENAI_CHAT_MODEL_NAME",  "gpt-4o-mini"),
        "OPENAI_EMBED_MODEL_NAME":os.getenv("OPENAI_EMBED_MODEL_NAME", "text-embedding-3-small"),
    }


def make_embeddings(env, provider):
    if provider == "Gemini":
        return GoogleGenerativeAIEmbeddings(
            model=f"models/{env['GOOGLE_EMBED_MODEL_NAME']}",
            google_api_key=env["GOOGLE_API_KEY"],
        )
    return OpenAIEmbeddings(
        model=env["OPENAI_EMBED_MODEL_NAME"],
        api_key=env["OPENAI_API_KEY"],
    )


def make_llm(env, provider, temperature=0.2):
    if provider == "Gemini":
        return ChatGoogleGenerativeAI(
            model=env["GOOGLE_CHAT_MODEL_NAME"],
            google_api_key=env["GOOGLE_API_KEY"],
            temperature=temperature,
        )
    return ChatOpenAI(
        model=env["OPENAI_CHAT_MODEL_NAME"],
        api_key=env["OPENAI_API_KEY"],
        temperature=temperature,
    )


# ─────────────────────────────────────────
# Helpers — document processing- Chucking add into VectorDB
# ─────────────────────────────────────────
# -----------------------------------------
# Creates a text splitter (RecursiveCharacterTextSplitter) with two parameters:
# chunk_size=500: target size of each chunk (in characters).
# chunk_overlap=50: number of characters to overlap between consecutive chunks.
# Splits the input documents (docs) into smaller chunks using the splitter’s .split_documents(docs) method.
# Used RAG + LangChain To break larger Docs into smaller and manageable pieces for embeddings and retrival.
# -----------------------------------------
def chunk_documents(docs, chunk_size=500, chunk_overlap=50):
    return RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap
    ).split_documents(docs)


def build_index_from_pdfs(uploaded_files, embeddings):
    # Prepares a list to hold all the Document objects (one per page).
    # Ensures a temporary directory exists for writing uploaded bytes to files.
    all_docs = []
    tmp_dir = Path("tmp_uploads")
    tmp_dir.mkdir(parents=True, exist_ok=True)

    # uploaded_files likely contains file-like objects (e.g., from a web form).
    # The code writes the raw bytes into a temporary file on disk so file-based loaders (like PyPDFLoader) can parse the
    # PyPDFLoader(...).load() extracts text pages and returns a list of Document objects (one per page), each with page_content and metadata (e.g., page index, source).
    # Extends all_docs with these pages across all PDF.
    # Deletes the temporary file after reading (good hygiene). missing_ok=True avoids exceptions if the file was already remove
    for up in uploaded_files:
        tmp_path = tmp_dir / up.name
        with open(tmp_path, "wb") as f:
            f.write(up.read())
        all_docs.extend(PyPDFLoader(str(tmp_path)).load())
        tmp_path.unlink(missing_ok=True)

    if not all_docs:
        raise ValueError("No pages could be loaded from the uploaded PDFs.")

    # Builds a FAISS index (an efficient vector similarity search library)
    # so later you can retrieve the most similar chunks given a query vector
    # Ensures output directory exists
    # Persists FAISS index to disk so you can reload it later without recomping embeddings.
    chunks = chunk_documents(all_docs)
    db = FAISS.from_documents(chunks, embeddings)
    Path(INDEX_DIR).mkdir(parents=True, exist_ok=True)
    db.save_local(INDEX_DIR)
    return db, len(chunks)


def load_persistent_index(embeddings):
    if not Path(INDEX_DIR).exists(): # INDEX_DIR - Vector_Store folder/Directory
        return None
    try:
         # Loads your previously persisted FAISS vector store from disk.
        # embeddings must match the one used during save (same provider/model/dimensions).
        # FAISS stores only vectors and metadata, Not embedding function itself.
        # allow_dangerous_deserialization used in new langchain version to untrusted pickle files can
        # be a security risk. Only load indexes you trust.
        return FAISS.load_local(
            INDEX_DIR, embeddings, allow_dangerous_deserialization=True
        )
    except Exception as e:
        st.warning(f"⚠️ Saved index could not be loaded (provider mismatch?): {e}")
        return None

# Feature is not Working
def reset_embeddings_from_disk():
    """Delete saved FAISS index folder. Returns True if something was deleted."""
    if Path(INDEX_DIR).exists():
        shutil.rmtree(INDEX_DIR)
        return True
    return False


# ─────────────────────────────────────────
# Helpers — RAG chain
# ─────────────────────────────────────────
def format_docs(docs):pages = []
    # Simpler version
    # return "\n\n".join(doc.page_content for doc in docs)

    for doc in docs:
        pages.append(doc.page_content)
    return "\n\n".join(pages)


# Input mapping (dict of lambdas)
# It expects an input dict like:
# {"docs": [Document, ...], "question": "..."}
# It transforms that input into the exact variables needed by the prompt:
# "context" → produced by format_docs(x["docs"]) (joins retrieved doc chunks into a single text block separated by blank lines).
# "question" → passed through as‑is.
# Template has {context} and {questions} - LCEL fills these using the outputs of the mapping step.
def build_chain(llm):
    prompt = ChatPromptTemplate.from_template("""You are a helpful document assistant.

Rules:
- Respond in plain, readable text. Never respond with JSON or code.
- Use bullet points when listing multiple items.
- Be concise but complete.
- If the answer is not found in the context below, say exactly:
  "I could not find an answer in the provided documents."
- Never make up information.

Context:
{context}

Question: {question}

Answer:""")

    return (
        {
            "context":  lambda x: format_docs(x["docs"]), # The | operator is the LCEL “pipe” operator that composes runnable: output of left → input of right.
            "question": lambda x: x["question"],
        }
        | prompt
        | llm
    )


# Converst JSON Formate into user friendly Text
def extract_answer_text(raw) -> str:
    """
    Robustly pull a plain string from whatever LangChain returns.
    Handles: AIMessage, str, list of content blocks, dict.
    """
    # AIMessage / BaseChatMessage
    if hasattr(raw, "content"):
        content = raw.content
        # content can itself be a list of blocks (Gemini sometimes does this)
        if isinstance(content, list):
            parts = []
            for block in content:
                if isinstance(block, dict):
                    parts.append(block.get("text", ""))
                elif isinstance(block, str):
                    parts.append(block)
                elif hasattr(block, "text"):
                    parts.append(block.text)
            return "\n".join(p for p in parts if p).strip()
        return str(content).strip()

    # Plain string
    if isinstance(raw, str):
        return raw.strip()

    # List of blocks (rare fallback)
    if isinstance(raw, list):
        parts = []
        for block in raw:
            if isinstance(block, dict):
                parts.append(block.get("text", ""))
            elif isinstance(block, str):
                parts.append(block)
            elif hasattr(block, "text"):
                parts.append(block.text)
        return "\n".join(p for p in parts if p).strip()

    # Last resort — stringify but warn
    return str(raw).strip()


# ─────────────────────────────────────────
# Sidebar - UI 
# ─────────────────────────────────────────
env = load_env()

with st.sidebar:
    st.markdown("### ⚙️ Settings")
    st.markdown("---")

    provider    = st.selectbox("Model Provider", ["Gemini", "OpenAI"])
    temperature = st.slider("Temperature", 0.0, 1.0, 0.2, 0.05)
    top_k       = st.slider("Top-K Results", 2, 10, 5)
    use_persist = st.checkbox("Load saved index on startup", value=True)

    st.markdown("---")
    st.markdown("**📂 Upload PDFs**")
    uploaded_files = st.file_uploader(
        "upload", type=["pdf"],
        accept_multiple_files=True,
        label_visibility="collapsed",
    )
    rebuild = st.button("🔨 Build / Rebuild Index", use_container_width=True)

    st.markdown("---")
    st.markdown("**🗑️ Reset**")
    reset_clicked = st.button("Reset Embeddings", use_container_width=True)

    if reset_clicked:
        deleted = reset_embeddings_from_disk()
        for k in ("db", "answer", "sources"):
            st.session_state.pop(k, None)
        if deleted:
            st.success("✅ Embeddings cleared successfully.")
        else:
            st.info("No saved index found on disk.")
        st.rerun()


# ─────────────────────────────────────────
# API key      
# ─────────────────────────────────────────
if provider == "Gemini" and not env["GOOGLE_API_KEY"]:
    st.error("❌ GOOGLE_API_KEY not found in your .env file.")
    st.stop()

if provider == "OpenAI" and not env["OPENAI_API_KEY"]:
    st.error("❌ OPENAI_API_KEY not found in your .env file.")
    st.stop()


# ─────────────────────────────────────────
# Initialise models -- 
# ─────────────────────────────────────────
embeddings = make_embeddings(env, provider)
llm        = make_llm(env, provider, temperature=temperature)
chain      = build_chain(llm)


# ─────────────────────────────────────────
# Session state defaults
# ─────────────────────────────────────────
for key, default in [("db", None), ("answer", None), ("sources", [])]:
    if key not in st.session_state:
        st.session_state[key] = default

# ─────────────────────────────────────────
# Conversation memory (LangChain Buffer Memory) -- Need to added this feature 
# ─────────────────────────────────────────

# ─────────────────────────────────────────
# Load / build index
# ─────────────────────────────────────────
if use_persist and st.session_state.db is None:
    st.session_state.db = load_persistent_index(embeddings)

if rebuild:
    if not uploaded_files:
        st.sidebar.error("Upload at least one PDF before building the index.")
    else:
        with st.spinner("⏳ Building FAISS index from your PDFs…"):
            try:
                db, n = build_index_from_pdfs(uploaded_files, embeddings)
                st.session_state.db = db
                st.session_state.answer  = None
                st.session_state.sources = []
                st.sidebar.success(f"✅ Index ready — {n} chunks indexed.")
            except Exception as e:
                st.sidebar.error(f"❌ Build failed: {e}")


# ─────────────────────────────────────────
# Page header
# ─────────────────────────────────────────
st.markdown("# 📄 Document QA with RAG")
st.markdown("##### Retrieval-Augmented Generation · Ask questions about your PDFs")
st.markdown("---")

tab_setup, tab_ask = st.tabs(["📁  Indexing Setup", "❓  Ask Questions"])


# ─────────────────────────── Tab 1 ───────────────────────────
with tab_setup:
    st.markdown("### Index Status")

    if st.session_state.db is not None:
        st.markdown('<span class="pill-green">✓ Index loaded and ready</span>', unsafe_allow_html=True)
        st.markdown("")
        st.markdown('<div class="info-box">Your document index is live. Switch to the <strong style="color:#e2e4ec">Ask Questions</strong> tab to start querying.</div>', unsafe_allow_html=True)
    else:
        st.markdown('<span class="pill-yellow">⚠ No index loaded</span>', unsafe_allow_html=True)
        st.markdown("")
        st.markdown("""
<div class="info-box">
<strong style="color:#e2e4ec">Getting started:</strong><br>
1. Upload one or more PDF files using the sidebar uploader.<br>
2. Click <strong style="color:#e2e4ec">Build / Rebuild Index</strong>.<br>
3. Once ready, go to the <strong style="color:#e2e4ec">Ask Questions</strong> tab.
</div>
""", unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### How it works")
    st.markdown("""
<div class="info-box">
<strong style="color:#e2e4ec">1. Upload</strong> — Add your PDFs in the sidebar.<br>
<strong style="color:#e2e4ec">2. Index</strong> — Documents are split into chunks and embedded into a FAISS vector store.<br>
<strong style="color:#e2e4ec">3. Retrieve</strong> — Your question fetches the most relevant passages (MMR search).<br>
<strong style="color:#e2e4ec">4. Generate</strong> — The LLM reads those passages and writes a plain-text answer.
</div>
""", unsafe_allow_html=True)


# ─────────────────────────── Tab 2 ───────────────────────────
with tab_ask:

    if st.session_state.db is None:
        st.markdown('<span class="pill-yellow">⚠ No index available — upload PDFs and build an index first.</span>', unsafe_allow_html=True)

    else:
        st.markdown("### Ask your question")
        st.markdown('<div class="micro-label">Enter your question:</div>', unsafe_allow_html=True)

        col_q, col_btn = st.columns([5, 1])
        with col_q:
            query = st.text_input(
                "q", label_visibility="collapsed",
                placeholder="e.g. What are the duration options for hardware support packages?",
            )
        with col_btn:
            search = st.button("Search 🔍", use_container_width=True)

        # ── Run search ──
        if search:
            if not query.strip():
                st.warning("Please enter a question before searching.")
            else:
                retriever = st.session_state.db.as_retriever(
                    search_type="mmr",
                    search_kwargs={"k": top_k},
                )

                with st.spinner("🔎 Searching documents…"):
                    results = retriever.invoke(query)

                if not results:
                    st.session_state.answer  = None
                    st.session_state.sources = []
                    st.warning("No relevant content found for your query. Try rephrasing.")
                else:
                    with st.spinner("✍️ Generating answer…"):
                        try:
                            raw = chain.invoke({"docs": results, "question": query})
                            # ── KEY FIX: extract plain text, never display raw object ──
                            st.session_state.answer  = extract_answer_text(raw)
                            st.session_state.sources = results
                        except Exception as e:
                            st.session_state.answer  = None
                            st.session_state.sources = []
                            st.error(f"❌ Error generating answer: {e}")

        # ── Display answer ──
        if st.session_state.answer:
            st.markdown('<div class="micro-label">Answer</div>', unsafe_allow_html=True)
            st.markdown(
                f'<div class="answer-card">{st.session_state.answer}</div>',
                unsafe_allow_html=True,
            )

            # ── Collapsible sources ──
            with st.expander(f"📄 Show retrieved snippets (sources) — {len(st.session_state.sources)} chunks"):
                for i, doc in enumerate(st.session_state.sources, 1):
                    src  = doc.metadata.get("source", "Unknown")
                    page = doc.metadata.get("page", "?")
                    name = Path(src).name if src != "Unknown" else "Unknown"
                    st.markdown(f"""
<div class="source-card">
  <div class="source-title">Source {i}:</div>
  {doc.page_content}
  <div class="source-meta">File: {name} &nbsp;|&nbsp; Page: {page}</div>
</div>

""", unsafe_allow_html=True)
