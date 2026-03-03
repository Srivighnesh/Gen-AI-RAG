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

INDEX_FOLDER = "vector_store"
MAX_HISTORY_TURNS = 5


# ==========================================================
# API KEYS
# ==========================================================

def load_api_keys():
    load_dotenv(".env")
    return {
        "GOOGLE_API_KEY": os.getenv("GOOGLE_API_KEY"),
        "GOOGLE_CHAT_MODEL_NAME": os.getenv("GOOGLE_CHAT_MODEL_NAME", "gemini-1.5-flash"),
        "GOOGLE_EMBED_MODEL_NAME": os.getenv("GOOGLE_EMBED_MODEL_NAME", "gemini-embedding-001"),
        "OPENAI_API_KEY": os.getenv("OPENAI_API_KEY"),
        "OPENAI_CHAT_MODEL_NAME": os.getenv("OPENAI_CHAT_MODEL_NAME", "gpt-4o-mini"),
        "OPENAI_EMBED_MODEL_NAME": os.getenv("OPENAI_EMBED_MODEL_NAME", "text-embedding-3-small"),
    }


# ==========================================================
# MODEL FACTORIES
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

def split_into_chunks(documents, chunk_size=500, overlap=50):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=overlap,
    )
    return splitter.split_documents(documents)


def build_vector_index(uploaded_files, embeddings):
    all_pages = []
    tmp_dir = Path("tmp_uploads")
    tmp_dir.mkdir(exist_ok=True)

    for file in uploaded_files:
        tmp_path = tmp_dir / file.name
        tmp_path.write_bytes(file.read())
        
        File_name = file.name.lower()

        # Choose loader based on file type
        if File_name.endswith(".pdf"):
            loader = PyPDFLoader(str(tmp_path))

        elif File_name.endswith(".csv"):
            loader = CSVLoader(
                file_path=str(tmp_path),
                encoding="utf-8",
            )
        elif File_name.endswith(".txt"):
            loader = TextLoader(str(tmp_path), encoding="utf-8")
        else:
            st.warning(f"Unsupported file type: {file.name}")
            tmp_path.unlink(missing_ok=True)  # Clean up unsupported file
            continue
        
        docs = loader.load()
        all_pages.extend(docs)
        tmp_path.unlink(missing_ok=True)  # Clean up after loading
        if not all_pages:
            raise ValueError("No valid documents found in uploaded files.")

        chunks = split_into_chunks(all_pages)
        vector_db = FAISS.from_documents(chunks, embeddings)

        Path(INDEX_FOLDER).mkdir(exist_ok=True)
        vector_db.save_local(INDEX_FOLDER)
    return vector_db, len(chunks)


def delete_saved_index():
    if Path(INDEX_FOLDER).exists():
        shutil.rmtree(INDEX_FOLDER)


# ==========================================================
# RAG CHAIN
# ==========================================================

def build_rag_chain(llm):
    prompt = ChatPromptTemplate.from_template("""
You are a helpful document assistant.

Conversation History:
{history}

Context:
{context}

Question:
{question}

Rules:
- Answer only from context.
- If answer not found, say:
  "I could not find an answer in the provided documents."

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


def format_chat_history(history):
    history = history[-MAX_HISTORY_TURNS:]
    lines = []
    for turn in history:
        lines.append(f"User: {turn['question']}")
        lines.append(f"Assistant: {turn['answer']}")
    return "\n".join(lines)


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
# STREAMLIT UI (SIMPLE)
# ==========================================================

st.set_page_config(page_title="Document QA", layout="centered")
st.title("📄 Document QA with RAG")
st.divider()
st.markdown("### How it works")
st.markdown("""
<div class="info-box">
<strong style="color:#e2e4ec">1. Upload</strong> — Add your Documents in the sidebar.<br>
<strong style="color:#e2e4ec">2. Index</strong> — Documents are split into chunks and embedded into a FAISS vector store.<br>
<strong style="color:#e2e4ec">3. Retrieve</strong> — Your question fetches the most relevant passages (MMR search).<br>
<strong style="color:#e2e4ec">4. Generate</strong> — The LLM reads those passages and writes a plain-text answer.
</div>
""", unsafe_allow_html=True)
st.divider()
api_keys = load_api_keys()

with st.sidebar:
    st.header("Settings")
    provider = st.selectbox("Model Provider", ["Gemini", "OpenAI"])
    temperature = st.slider("Temperature", 0.0, 1.0, 0.2)
    top_k = st.slider("Top-K", 2, 10, 5)

    st.divider()

    uploaded_files = st.file_uploader(
        "Upload Documents",
        type=["pdf", "csv", "txt"],
        accept_multiple_files=True,
    )

    build_btn = st.button("Build Index")
    reset_btn = st.button("Reset Index")

# Initialize models
embeddings = create_embeddings(api_keys, provider)
llm = create_llm(api_keys, provider, temperature)
chain = build_rag_chain(llm)

# Session state
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


# Build index
if build_btn:
    if not uploaded_files:
        st.error("Upload at least one PDF.")
    else:
        with st.spinner("Building index..."):
            db, count = build_vector_index(uploaded_files, embeddings)
            st.session_state.db = db
        st.success(f"Index built ({count} chunks).")

# Question section
if st.session_state.db is None:
    st.info("Upload PDFs and build the index to begin.")
else:
    question = st.text_input("Ask a question")

    if st.button("Search"):
        retriever = st.session_state.db.as_retriever(
            search_type="mmr",
            search_kwargs={"k": top_k},
        )

        docs = retriever.invoke(question)

        history_text = format_chat_history(st.session_state.chat_history)

        response = chain.invoke({
            "docs": docs,
            "question": question,
            "history": history_text,
        })

        answer = extract_text(response)

        st.session_state.chat_history.append({
            "question": question,
            "answer": answer,
        })

        st.subheader("Answer")
        st.write(answer)

        with st.expander("Sources"):
            for i, doc in enumerate(docs, 1):
                st.markdown(f"**Source {i}**")
                st.write(doc.page_content)
                st.write(f"Page no: {doc.metadata.get('page', '?')}")
                st.divider()
