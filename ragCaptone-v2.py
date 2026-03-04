"""
Document QA with RAG (Retrieval-Augmented Generation)
======================================================
This app lets you upload PDFs, index them into a vector store,
and ask natural-language questions about their content.

Flow:
  1. User uploads PDFs in the sidebar
  2. PDFs are split into chunks and stored in a FAISS vector database
  3. User asks a question
  4. The most relevant chunks are retrieved (MMR search)
  5. An LLM (Gemini or OpenAI) generates an answer from those chunks
"""

import os
import shutil
import warnings
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

# --- Document loading & splitting ---
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.vectorstores import FAISS

# --- Gemini (Google) models ---
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings

# --- OpenAI models ---
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

warnings.filterwarnings("ignore")


# ══════════════════════════════════════════════════════════
# CONSTANTS
# ══════════════════════════════════════════════════════════

# Folder where the FAISS vector index is saved between sessions
INDEX_FOLDER = "vector_store"

# How many past Q&A turns to include in the prompt for context
MAX_HISTORY_TURNS = 5


# ══════════════════════════════════════════════════════════
# PAGE CONFIG & CSS STYLING
# ══════════════════════════════════════════════════════════

st.set_page_config(
    page_title="Document QA · RAG",
    page_icon="📄",
    layout="wide",
)

# Dark-theme custom CSS (visual only — no logic here)
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
*, html, body { font-family: 'Inter', sans-serif; }

.stApp { background: #0f1117; color: #e2e4ec; }

h1 { color: #fff !important; font-size: 1.85rem !important; font-weight: 700 !important; margin-bottom: 2px !important; }
h3 { color: #fff !important; font-weight: 600 !important; }
p, label { color: #9aa0b4 !important; }

.stTabs [data-baseweb="tab-list"] { border-bottom: 1px solid #1e2130; gap: 0; }
.stTabs [data-baseweb="tab"] { background: transparent; color: #6b7280; border: none; padding: 10px 22px; font-weight: 500; font-size: 0.9rem; }
.stTabs [aria-selected="true"] { color: #e05c5c !important; border-bottom: 2px solid #e05c5c !important; background: transparent !important; }

[data-testid="stSidebar"] { background: #0b0d14 !important; border-right: 1px solid #1a1d2e; }
[data-testid="stSidebar"] * { color: #9aa0b4; }
[data-testid="stSidebar"] h3 { color: #fff !important; }

.stTextInput > div > div > input {
    background: #161926 !important; border: 1px solid #252840 !important;
    color: #e2e4ec !important; border-radius: 8px !important;
    padding: 10px 14px !important; font-size: 0.95rem !important;
}
.stTextInput > div > div > input:focus { border-color: #3b82f6 !important; box-shadow: 0 0 0 2px rgba(59,130,246,0.15) !important; }

.stButton > button {
    background: #161926; border: 1px solid #252840; color: #c8cce0;
    border-radius: 8px; font-weight: 500; font-size: 0.88rem;
    transition: all 0.18s ease; padding: 8px 18px;
}
.stButton > button:hover { border-color: #3b82f6; color: #fff; background: #1a1f35; }

.stSelectbox > div > div { background: #161926 !important; border-color: #252840 !important; color: #e2e4ec !important; border-radius: 8px !important; }

.answer-card {
    background: linear-gradient(135deg, #131726 0%, #161c2e 100%);
    border: 1px solid #1e2540; border-left: 4px solid #3b82f6;
    border-radius: 10px; padding: 20px 24px; margin: 14px 0 24px 0;
    color: #d4d8f0; line-height: 1.8; font-size: 0.97rem;
    white-space: pre-wrap; word-break: break-word;
}

.source-card {
    background: #0f1220; border: 1px solid #1e2130; border-radius: 8px;
    padding: 14px 18px; margin: 10px 0; color: #a0a8c0;
    font-size: 0.875rem; line-height: 1.7;
}
.source-title { font-weight: 700; color: #e2e4ec; font-size: 0.9rem; margin-bottom: 8px; }
.source-meta { color: #414766; font-size: 0.76rem; margin-top: 10px; font-family: monospace; }

.pill-green { display:inline-block; background:#052e16; color:#4ade80; border:1px solid #166534; border-radius:20px; padding:3px 14px; font-size:0.8rem; font-weight:600; }
.pill-yellow { display:inline-block; background:#2d1a00; color:#fbbf24; border:1px solid #78350f; border-radius:20px; padding:3px 14px; font-size:0.8rem; font-weight:600; }

.micro-label { color:#555e7a; font-size:0.75rem; font-weight:600; letter-spacing:0.07em; text-transform:uppercase; margin-bottom:5px; }
.info-box { background:#0d1322; border:1px solid #1e2540; border-radius:8px; padding:14px 18px; color:#7a82a0; font-size:0.88rem; line-height:1.65; }

hr { border-color:#1a1d2e !important; }
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════
# ENVIRONMENT / API KEYS
# ══════════════════════════════════════════════════════════

def load_api_keys() -> dict:
    """Read API keys and model names from the .env file."""
    load_dotenv(".env")
    return {
        # Gemini
        "GOOGLE_API_KEY":          os.getenv("GOOGLE_API_KEY"),
        "GOOGLE_CHAT_MODEL_NAME":  os.getenv("GOOGLE_CHAT_MODEL_NAME",  "gemini-1.5-flash"),
        "GOOGLE_EMBED_MODEL_NAME": os.getenv("GOOGLE_EMBED_MODEL_NAME", "gemini-embedding-001"),
        # OpenAI
        "OPENAI_API_KEY":          os.getenv("OPENAI_API_KEY"),
        "OPENAI_CHAT_MODEL_NAME":  os.getenv("OPENAI_CHAT_MODEL_NAME",  "gpt-4o-mini"),
        "OPENAI_EMBED_MODEL_NAME": os.getenv("OPENAI_EMBED_MODEL_NAME", "text-embedding-3-small"),
    }


# ══════════════════════════════════════════════════════════
# MODEL FACTORIES
# ══════════════════════════════════════════════════════════

def create_embeddings(keys: dict, provider: str):
    """
    Return an embedding model for the chosen provider.
    Embeddings convert text into numeric vectors for similarity search.
    """
    if provider == "Gemini":
        return GoogleGenerativeAIEmbeddings(
            model=f"models/{keys['GOOGLE_EMBED_MODEL_NAME']}",
            google_api_key=keys["GOOGLE_API_KEY"],
        )
    # Default: OpenAI
    return OpenAIEmbeddings(
        model=keys["OPENAI_EMBED_MODEL_NAME"],
        api_key=keys["OPENAI_API_KEY"],
    )


def create_llm(keys: dict, provider: str, temperature: float = 0.2):
    """
    Return a chat LLM for the chosen provider.
    Temperature controls randomness: 0 = deterministic, 1 = creative.
    """
    if provider == "Gemini":
        return ChatGoogleGenerativeAI(
            model=keys["GOOGLE_CHAT_MODEL_NAME"],
            google_api_key=keys["GOOGLE_API_KEY"],
            temperature=temperature,
        )
    # Default: OpenAI
    return ChatOpenAI(
        model=keys["OPENAI_CHAT_MODEL_NAME"],
        api_key=keys["OPENAI_API_KEY"],
        temperature=temperature,
    )


# ══════════════════════════════════════════════════════════
# DOCUMENT PROCESSING
# ══════════════════════════════════════════════════════════

def split_into_chunks(documents, chunk_size: int = 500, overlap: int = 50):
    """
    Break documents into smaller overlapping chunks.
    Overlap ensures important context near chunk boundaries is not lost.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=overlap,
    )
    return splitter.split_documents(documents)


def build_vector_index(uploaded_files, embeddings):
    """
    Load uploaded PDFs → split into chunks → embed → save FAISS index.
    Returns the FAISS database and the number of chunks created.
    """
    all_pages = []
    tmp_dir = Path("tmp_uploads")
    tmp_dir.mkdir(parents=True, exist_ok=True)

    # Save each uploaded file temporarily so PyPDFLoader can read it
    for file in uploaded_files:
        tmp_path = tmp_dir / file.name
        tmp_path.write_bytes(file.read())          # write bytes to disk
        all_pages.extend(PyPDFLoader(str(tmp_path)).load())  # load pages
        tmp_path.unlink(missing_ok=True)           # clean up temp file

    if not all_pages:
        raise ValueError("No pages could be loaded from the uploaded PDFs.")

    chunks = split_into_chunks(all_pages)

    # Build the FAISS vector store from the chunks
    vector_db = FAISS.from_documents(chunks, embeddings)

    # Persist to disk so it survives page refreshes
    Path(INDEX_FOLDER).mkdir(parents=True, exist_ok=True)
    vector_db.save_local(INDEX_FOLDER)

    return vector_db, len(chunks)


def load_saved_index(embeddings):
    """
    Load a previously built FAISS index from disk.
    Returns None if no saved index exists or if loading fails.
    """
    if not Path(INDEX_FOLDER).exists():
        return None  # No index saved yet

    try:
        return FAISS.load_local(
            INDEX_FOLDER,
            embeddings,
            allow_dangerous_deserialization=True,  # required by FAISS
        )
    except Exception as err:
        st.warning(f"⚠️ Could not load saved index (provider mismatch?): {err}")
        return None


def delete_saved_index():
    """Delete the FAISS index folder from disk. Returns True if it existed."""
    if Path(INDEX_FOLDER).exists():
        shutil.rmtree(INDEX_FOLDER)
        return True
    return False


# ══════════════════════════════════════════════════════════
# RAG CHAIN (Retrieval-Augmented Generation)
# ══════════════════════════════════════════════════════════

def build_rag_chain(llm):
    """
    Build the LangChain RAG pipeline:
      input dict → prompt → LLM → answer string

    The prompt includes:
      - {history}  : recent conversation turns
      - {context}  : retrieved document chunks
      - {question} : the user's question
    """
    prompt = ChatPromptTemplate.from_template("""You are a helpful document assistant.

Conversation History (last interactions):
{history}

Rules:
- Respond in plain, readable text.
- Use bullet points when listing multiple items.
- Be concise but complete.
- If the answer is not found in the context below, say exactly:
  "I could not find an answer in the provided documents."
- Never make up information.

Context:
{context}

Question: {question}

Answer:""")

    # Chain: map inputs → fill prompt → call LLM
    chain = (
        {
            "context":  lambda x: "\n\n".join(d.page_content for d in x["docs"]),
            "question": lambda x: x["question"],
            "history":  lambda x: x["history"],
        }
        | prompt
        | llm
    )
    return chain


# ══════════════════════════════════════════════════════════
# UTILITY HELPERS
# ══════════════════════════════════════════════════════════

def format_chat_history(history: list) -> str:
    """
    Convert the last N Q&A turns into a plain string for the prompt.
    Example output:
        User: What is X?
        Assistant: X is ...
    """
    recent_turns = history[-MAX_HISTORY_TURNS:]
    lines = []
    for turn in recent_turns:
        lines.append(f"User: {turn['question']}")
        lines.append(f"Assistant: {turn['answer']}")
    return "\n".join(lines)


def extract_text(raw_response) -> str:
    """
    Safely extract a plain string from whatever LangChain returns.
    LangChain can return: AIMessage, str, list of content blocks, or dict.
    This handles all cases so we never accidentally display a Python object.
    """
    # Case 1: AIMessage or similar object with a .content attribute
    if hasattr(raw_response, "content"):
        content = raw_response.content
        # Gemini sometimes returns content as a list of blocks
        if isinstance(content, list):
            return "\n".join(
                block.get("text", "") if isinstance(block, dict)
                else (block.text if hasattr(block, "text") else str(block))
                for block in content
            ).strip()
        return str(content).strip()

    # Case 2: Already a plain string
    if isinstance(raw_response, str):
        return raw_response.strip()

    # Case 3: List of content blocks (rare fallback)
    if isinstance(raw_response, list):
        return "\n".join(
            block.get("text", "") if isinstance(block, dict)
            else (block.text if hasattr(block, "text") else str(block))
            for block in raw_response
        ).strip()

    # Case 4: Unknown type — convert to string as last resort
    return str(raw_response).strip()


# ══════════════════════════════════════════════════════════
# SIDEBAR — Settings, Upload, Reset
# ══════════════════════════════════════════════════════════

api_keys = load_api_keys()

with st.sidebar:
    st.markdown("### ⚙️ Settings")
    st.markdown("---")

    # Choose between Gemini (Google) and OpenAI
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
    build_btn = st.button("🔨 Build / Rebuild Index", use_container_width=True)

    st.markdown("---")
    st.markdown("**🗑️ Reset**")
    reset_btn = st.button("Reset Embeddings", use_container_width=True)

    # Handle reset button click
    if reset_btn:
        was_deleted = delete_saved_index()
        # Clear relevant session state keys
        for key in ("db", "answer", "sources"):
            st.session_state.pop(key, None)
        if was_deleted:
            st.success("✅ Embeddings cleared successfully.")
        else:
            st.info("No saved index found on disk.")
        st.rerun()


# ══════════════════════════════════════════════════════════
# API KEY GUARD — stop early if keys are missing
# ══════════════════════════════════════════════════════════

if provider == "Gemini" and not api_keys["GOOGLE_API_KEY"]:
    st.error("❌ GOOGLE_API_KEY not found in your .env file.")
    st.stop()

if provider == "OpenAI" and not api_keys["OPENAI_API_KEY"]:
    st.error("❌ OPENAI_API_KEY not found in your .env file.")
    st.stop()


# ══════════════════════════════════════════════════════════
# INITIALISE MODELS
# ══════════════════════════════════════════════════════════

embeddings = create_embeddings(api_keys, provider)
llm        = create_llm(api_keys, provider, temperature=temperature)
chain      = build_rag_chain(llm)


# ══════════════════════════════════════════════════════════
# SESSION STATE DEFAULTS
# Streamlit reruns the whole script on every interaction,
# so we use st.session_state to persist data across reruns.
# ══════════════════════════════════════════════════════════

defaults = {
    "db":           None,   # the FAISS vector database
    "answer":       None,   # last generated answer string
    "sources":      [],     # last retrieved document chunks
    "chat_history": [],     # list of {"question": ..., "answer": ...} dicts
}
for key, default_value in defaults.items():
    if key not in st.session_state:
        st.session_state[key] = default_value


# ══════════════════════════════════════════════════════════
# LOAD OR BUILD INDEX
# ══════════════════════════════════════════════════════════

# Auto-load saved index from disk on first run (if option is enabled)
if use_persist and st.session_state.db is None:
    st.session_state.db = load_saved_index(embeddings)

# Handle "Build / Rebuild Index" button click
if build_btn:
    if not uploaded_files:
        st.sidebar.error("Upload at least one PDF before building the index.")
    else:
        with st.spinner("⏳ Building FAISS index from your PDFs…"):
            try:
                new_db, num_chunks = build_vector_index(uploaded_files, embeddings)
                st.session_state.db      = new_db
                st.session_state.answer  = None
                st.session_state.sources = []
                st.sidebar.success(f"✅ Index ready — {num_chunks} chunks indexed.")
            except Exception as err:
                st.sidebar.error(f"❌ Build failed: {err}")


# ══════════════════════════════════════════════════════════
# PAGE HEADER
# ══════════════════════════════════════════════════════════

st.markdown("# 📄 Document QA with RAG")
st.markdown("##### Retrieval-Augmented Generation · Ask questions about your PDFs")
st.markdown("---")

tab_setup, tab_ask = st.tabs(["📁  Indexing Setup", "❓  Ask Questions"])


# ══════════════════════════════════════════════════════════
# TAB 1 — Indexing Setup (status & instructions)
# ══════════════════════════════════════════════════════════

with tab_setup:
    st.markdown("### Index Status")

    if st.session_state.db is not None:
        st.markdown('<span class="pill-green">✓ Index loaded and ready</span>', unsafe_allow_html=True)
        st.markdown("")
        st.markdown(
            '<div class="info-box">Your document index is live. Switch to the '
            '<strong style="color:#e2e4ec">Ask Questions</strong> tab to start querying.</div>',
            unsafe_allow_html=True,
        )
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


# ══════════════════════════════════════════════════════════
# TAB 2 — Ask Questions (main RAG interface)
# ══════════════════════════════════════════════════════════

with tab_ask:

    # Show a warning if no index is loaded yet
    if st.session_state.db is None:
        st.markdown(
            '<span class="pill-yellow">⚠ No index available — upload PDFs and build an index first.</span>',
            unsafe_allow_html=True,
        )

    else:
        st.markdown("### Ask your question")
        st.markdown('<div class="micro-label">Enter your question:</div>', unsafe_allow_html=True)

        # Question input + Search button side by side
        col_question, col_button = st.columns([5, 1])
        with col_question:
            user_query = st.text_input(
                "q", label_visibility="collapsed",
                placeholder="e.g. What are the duration options for hardware support packages?",
            )
        with col_button:
            search_btn = st.button("Search 🔍", use_container_width=True)

        # ── When the user clicks "Search" ──
        if search_btn:
            if not user_query.strip():
                st.warning("Please enter a question before searching.")
            else:
                # Create a retriever using MMR (Maximal Marginal Relevance)
                # MMR balances relevance AND diversity in the retrieved chunks
                retriever = st.session_state.db.as_retriever(
                    search_type="mmr",
                    search_kwargs={"k": top_k},
                )

                # Step 1: Retrieve relevant chunks
                with st.spinner("🔎 Searching documents…"):
                    retrieved_docs = retriever.invoke(user_query)

                if not retrieved_docs:
                    st.session_state.answer  = None
                    st.session_state.sources = []
                    st.warning("No relevant content found. Try rephrasing your question.")
                else:
                    # Step 2: Generate answer using the LLM
                    with st.spinner("✍️ Generating answer…"):
                        try:
                            # Format past Q&A turns as a string for the prompt
                            history_text = format_chat_history(st.session_state.chat_history)

                            # Call the RAG chain
                            raw_response = chain.invoke({
                                "docs":     retrieved_docs,
                                "question": user_query,
                                "history":  history_text,
                            })

                            # Extract a clean string from the response
                            answer_text = extract_text(raw_response)

                            # Save results to session state
                            st.session_state.answer  = answer_text
                            st.session_state.sources = retrieved_docs

                            # Append this turn to chat history and keep last N turns
                            st.session_state.chat_history.append({
                                "question": user_query,
                                "answer":   answer_text,
                            })
                            st.session_state.chat_history = st.session_state.chat_history[-MAX_HISTORY_TURNS:]

                        except Exception as err:
                            st.session_state.answer  = None
                            st.session_state.sources = []
                            st.error(f"❌ Error generating answer: {err}")

        # ── Display the answer (if one exists) ──
        if st.session_state.answer:
            st.markdown('<div class="micro-label">Answer</div>', unsafe_allow_html=True)
            st.markdown(
                f'<div class="answer-card">{st.session_state.answer}</div>',
                unsafe_allow_html=True,
            )

            # Collapsible section showing the raw source chunks used
            num_sources = len(st.session_state.sources)
            with st.expander(f"📄 Show retrieved snippets (sources) — {num_sources} chunks"):
                for i, doc in enumerate(st.session_state.sources, start=1):
                    file_path = doc.metadata.get("source", "Unknown")
                    page_num  = doc.metadata.get("page", "?")
                    file_name = Path(file_path).name if file_path != "Unknown" else "Unknown"

                    st.markdown(f"""
<div class="source-card">
  <div class="source-title">Source {i}:</div>
  {doc.page_content}
  <div class="source-meta">File: {file_name} &nbsp;|&nbsp; Page: {page_num}</div>
</div>
""", unsafe_allow_html=True)
