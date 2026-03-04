"""
Simple Document QA with RAG
Upload PDFs → Build FAISS index → Ask questions
"""

import os
import shutil
import warnings
from pathlib import Path
import gc
import time

import streamlit as st
from dotenv import load_dotenv

# --- LangChain ---

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import (
    PyPDFLoader,
    CSVLoader,
    TextLoader,
)
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

warnings.filterwarnings("ignore")

# ==========================================================
# Configuration Section
# ==========================================================
INDEX_FOLDER = "vector_store"   # Folder to save FAISS index
MAX_HISTORY_TURNS = 5           # Limit conversation history to last 5 turns for context


# ==========================================================
# API KEYS
# ==========================================================

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


# ==========================================================
# Model Factory (LLMs and Embeddings)
# ==========================================================

def create_embeddings(keys, provider):
    if provider == "Gemini":
        return GoogleGenerativeAIEmbeddings(
            model=f"models/{keys['GOOGLE_EMBED_MODEL_NAME']}",
            google_api_key=keys["GOOGLE_API_KEY"],
        )
    return OpenAIEmbeddings(
        model=keys["OPENAI_EMBED_MODEL_NAME"],
        api_key=keys["OPENAI_API_KEY"],
    )


def create_llm(keys, provider, temperature):
    if provider == "Gemini":
        return ChatGoogleGenerativeAI(
            model=keys["GOOGLE_CHAT_MODEL_NAME"],
            google_api_key=keys["GOOGLE_API_KEY"],
            temperature=temperature,
        )
    return ChatOpenAI(
        model=keys["OPENAI_CHAT_MODEL_NAME"],
        api_key=keys["OPENAI_API_KEY"],
        temperature=temperature,
    )


# ==========================================================
# DOCUMENT PROCESSING
# ==========================================================

def split_into_chunks(documents):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100,
    )
    return splitter.split_documents(documents)

# This function processes uploaded files, extracts text, splits into chunks,and builds a FAISS vector index.
def build_vector_index(uploaded_files, embeddings):
    all_pages = []
    tmp_dir = Path("tmp_uploads")
    tmp_dir.mkdir(exist_ok=True)

    for file in uploaded_files:
        tmp_path = tmp_dir / file.name
        tmp_path.write_bytes(file.read())
        
        file_name = file.name.lower()

        if file_name.endswith(".pdf"):
            loader = PyPDFLoader(str(tmp_path))
        elif file_name.endswith(".csv"):
            loader = CSVLoader(file_path=str(tmp_path), encoding="utf-8")
        elif file_name.endswith(".txt"):
            loader = TextLoader(str(tmp_path), encoding="utf-8")
        else:
            st.warning(f"Unsupported file type: {file.name}")
            tmp_path.unlink(missing_ok=True)
            continue

        docs = loader.load()
        all_pages.extend(docs)
        tmp_path.unlink(missing_ok=True)

   
    if not all_pages:
        raise ValueError("No valid documents found in uploaded files.")

    chunks = split_into_chunks(all_pages)

    try:
        vector_db = FAISS.from_documents(chunks, embeddings)
    except Exception as e:
        st.error(f"Failed to build FAISS index: {e}")
        return None, 0  

    Path(INDEX_FOLDER).mkdir(exist_ok=True)
    vector_db.save_local(INDEX_FOLDER)

    return vector_db, len(chunks)

def delete_saved_index():
    if Path(INDEX_FOLDER).exists():
        shutil.rmtree(INDEX_FOLDER)


# ==========================================================
# Building RAG CHAIN
# ==========================================================

def build_rag_chain(llm):
    prompt = ChatPromptTemplate.from_template("""
You are a helpful AI assistant that answers questions using ONLY the provided document context.

Your goal is to provide accurate answers strictly based on the retrieved document content.

--------------------------------------------------
Conversation History
--------------------------------------------------
{history}

--------------------------------------------------
Document Context
--------------------------------------------------
{context}

--------------------------------------------------
User Question
--------------------------------------------------
{question}

Instructions:

1. Read the document context carefully before answering.
2. Use ONLY the information provided in the document context.
3. Do NOT use external knowledge or make assumptions.
4. If the question refers to previous conversation, use the conversation history to understand the context.
5. If the answer is not clearly present in the document context, respond exactly with:
   "I could not find an answer in the provided documents."
6. Provide clear, concise answers.
7. When listing multiple items, use bullet points.
8. Combine information from multiple context sections if needed.

Answer:
""")

    chain = (
        {
            "context": lambda x: "\n\n".join(d.page_content for d in x["docs"]),
            "question": lambda x: x["question"],
            "history": lambda x: x["history"],
        }
        | prompt
        | llm
    )

    return chain

# ==========================================================
# Conversation Memory
# ==========================================================

# Convert chat history into a readable format for the LLM prompt, limiting to the last N turns.
# Stores last 5 conversation turns.
def format_chat_history(history):
    history = history[-MAX_HISTORY_TURNS:]
    lines = []
    for turn in history:
        lines.append(f"User: {turn['question']}")
        lines.append(f"Assistant: {turn['answer']}")
    return "\n".join(lines)


# =========================================================
# Response Cleaning
# ========================================================
def extract_text(response):
    """
    Cleanly extract readable text from LangChain / Gemini response.
    Handles:
    - AIMessage
    - list of blocks
    - dict blocks
    - plain string
    """

    # If response has .content (AIMessage)
    if hasattr(response, "content"):
        content = response.content
    else:
        content = response

    # If content is already string
    if isinstance(content, str):
        return content.strip()

    # If content is a list (Gemini block format)
    if isinstance(content, list):
        texts = []
        for block in content:
            if isinstance(block, dict) and "text" in block:
                texts.append(block["text"])
            else:
                texts.append(str(block))
        return "\n".join(texts).strip()

    # If content is dict
    if isinstance(content, dict):
        if "text" in content:
            return content["text"].strip()

    # Fallback
    return str(content).strip()


# ==========================================================
# STREAMLIT UI and App Logic
# ==========================================================

# Main page styling
st.markdown("""
<style>
.stApp {
   background: linear-gradient(to right, #0f2027, #203a43, #2c5364);
}
</style>
""", unsafe_allow_html=True)

# Header styling
st.markdown("""
<style>
[data-testid="stHeader"] {
    background: linear-gradient(to right, #0f2027, #203a43, #2c5364);
}
/* Remove header border line */
[data-testid="stHeader"]::after {
    background: none;
}
</style>
""", unsafe_allow_html=True)

st.set_page_config(page_title="Document QA", layout="centered")
st.title("📄 Document QA with RAG")
st.divider()
st.markdown("### How it works")

# Instruction box styling
st.markdown("""
<div class="info-box">
<strong style="color:#e2e4ec">1. Choose your API model provider from sidebar.</strong><br>
<strong style="color:#e2e4ec">2. Upload</strong> — Add your Documents in the sidebar and click on build Index.<br>
<strong style="color:#e2e4ec">3. Index</strong> — Documents are split into chunks and embedded into a FAISS vector store.<br>
<strong style="color:#e2e4ec">4. Retrieve</strong> — Your question fetches the most relevant passages (MMR search).<br>
<strong style="color:#e2e4ec">5. Generate</strong> — The LLM reads those passages and writes a plain-text answer.
</div>
""", unsafe_allow_html=True)
st.divider()


# ---------------------------------------------------------
# Sidebar - File Upload and Indexing
# ----------------------------------------------------------
with st.sidebar:
    
    st.header("⚙️Settings")
    provider = st.selectbox("Model Provider", ["Gemini", "OpenAI"])
    st.divider()

    uploaded_files = st.file_uploader(
        " 📚 Upload Documents",
        type=["pdf", "csv", "txt"],
        accept_multiple_files=True,
    )

#   streamlit columns for Build and Reset buttons
    col1, col2 = st.columns(2)
    with col1:
        build_btn = st.button("Build Index")
    with col2:
        reset_btn = st.button("Reset Index")
    st.divider ()

api_keys = load_api_keys()

# *************************************************************
# GuardRail: Ensure API keys are present
# *************************************************************
if provider == "Gemini" and not api_keys["GOOGLE_API_KEY"]:
    st.error("Missing GOOGLE_API_KEY in .env")
    st.stop()

if provider == "OpenAI" and not api_keys["OPENAI_API_KEY"]:
    st.error("Missing OPENAI_API_KEY in .env")
    st.stop()
# Initialize models
FIXED_TEMPERATURE = 0.2
embeddings = create_embeddings(api_keys, provider)
llm = create_llm(api_keys, provider, FIXED_TEMPERATURE)
chain = build_rag_chain(llm)

# ---------------------------------
# Session state - Streamlit reruns the script every interaction.
# ---------------------------------
if "db" not in st.session_state:
    st.session_state.db = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Reset
def delete_saved_index():
    """Safely delete FAISS index on Windows"""

    # 1️⃣ Remove DB reference from session state
    if "db" in st.session_state:
        st.session_state.db = None

    # 2️⃣ Force garbage collection
    gc.collect()
    time.sleep(0.5)  # small delay for Windows file release

    # 3️⃣ Delete folder
    if Path(INDEX_FOLDER).exists():
        shutil.rmtree(INDEX_FOLDER, ignore_errors=True)


# Reset the index and chat history
if reset_btn:
    delete_saved_index()
    st.session_state.chat_history = []
    st.success("Index deleted successfully.")
    st.rerun()

# ---------------------------------
# Build index
# ---------------------------------
if build_btn:
    if not uploaded_files:
        st.error("Upload at least one PDF, CSV, or TXT file.")
    else:
        with st.spinner(" Building index..."):
            db, count = build_vector_index(uploaded_files, embeddings)
            if db is None:
                st.stop()
            st.session_state.db = db
        st.success(f"Index built ({count} chunks).")

# --------------------------------
# Question section
# 
if st.session_state.db is None:
    st.info("Upload a document in the sidebar and click 'Build Index'.")
else:
    question = st.text_input("Ask a question")

    if st.button("Search"):

        if not question.strip():
            st.warning("Please enter a question.")
            st.stop()

            # ----------------------------------------------------
            # Retrieval
            # ---------------------------------------------------
            # similarity - Returns the most similar chunks by fAISS
            # mmr - Returns relevant + diverse chunks by MMR (Maximal Marginal Relevance)
        retriever = st.session_state.db.as_retriever(
            search_type="mmr",
            search_kwargs={"k": 6},
        )

        docs = retriever.invoke(question)
        history_text = format_chat_history(st.session_state.chat_history)

        try:
            # LLM reads the retrieved document context and generates the answer.
            response = chain.invoke({
                "docs": docs,
                "question": question,
                "history": history_text,
            })
        except Exception as e:
            st.error(f"Error during generation: {e}")
            st.stop()

        answer = extract_text(response)

        st.session_state.chat_history.append({
            "question": question,
            "answer": answer,
        })

        st.subheader("🤖 Answer")
        st.write(answer)

        with st.expander("Sources"):
            for i, doc in enumerate(docs, 1):

                source = doc.metadata.get("source", "unknown")
                page = doc.metadata.get("page", "?")

                st.markdown(f"**Source {i}**")
                st.markdown(f"**File:** {Path(source).name}")
                st.markdown(f"**Page:** {page}")

                st.markdown("**Content:**")
                st.write(doc.page_content)

                st.divider()

