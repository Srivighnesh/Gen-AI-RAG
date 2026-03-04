# 📄 AI Research Assistant — RAG (Retrieval-Augmented Generation)

A Streamlit app that lets you upload documents, index them with FAISS, and ask questions using Gemini or OpenAI — with source citations and conversational memory.

---

## 📁 Project Structure

```
your-project/
│
├── app.py                 # Main Streamlit app
├── requirements.txt       # Python dependencies
├── .env                   # API keys (DO NOT commit this)
├── .gitignore             # Ignores .env and vector_store/
├── tmp_uploads/           # Auto-created temporarily during indexing (auto-cleaned)
└── vector_store/          # Auto-created when you build index (ignored by git)
```

---

## ⚙️ Setup Instructions

### 1. Clone the repo

```bash
git clone https://github.com/YOUR_USERNAME/YOUR_REPO_NAME.git
cd YOUR_REPO_NAME
```

### 2. Create a virtual environment

```bash
python -m venv .venv
```

### 3. Activate the virtual environment

**Windows:**
```bash
.venv\Scripts\activate
```

**Mac / Linux:**
```bash
source .venv/bin/activate
```

### 4. Install dependencies

```bash
pip install -r requirements.txt
```

### 5. Create your `.env` file

Create a file called `.env` in the project root:

```
GOOGLE_API_KEY=your_gemini_api_key_here
OPENAI_API_KEY=your_openai_api_key_here
```

> Get Gemini key: https://aistudio.google.com/app/apikey  
> Get OpenAI key: https://platform.openai.com/api-keys

### 6. Run the app

```bash
streamlit run app.py
```

---

## 🚀 How to Use

1. **Choose a model provider** — select **Gemini** or **OpenAI** from the sidebar dropdown
2. **Upload documents** — upload one or more **PDF**, **CSV**, or **TXT** files via the sidebar
3. **Build the index** — click **Build Index** to chunk, embed, and save the FAISS vector store
4. **Ask a question** — type your question in the main area and click **Search**
5. **View the answer** — the LLM's response is shown immediately below
6. **Inspect sources** — expand the **Sources** section to see which chunks and pages were retrieved
7. **Reset when done** — click **Reset Index** in the sidebar to clear the vector store and chat history

> ⚠️ Unsupported file types are skipped with a warning. Only PDF, CSV, and TXT are accepted.

---

## 🔑 Environment Variables

| Variable | Description | Default |
|---|---|---|
| `GOOGLE_API_KEY` | Gemini API key from Google AI Studio | *(required for Gemini)* |
| `OPENAI_API_KEY` | OpenAI API key | *(required for OpenAI)* |
| `GOOGLE_CHAT_MODEL_NAME` | Gemini chat model | `gemini-1.5-flash` |
| `GOOGLE_EMBED_MODEL_NAME` | Gemini embedding model | `gemini-embedding-001` |
| `OPENAI_CHAT_MODEL_NAME` | OpenAI chat model | `gpt-4o-mini` |
| `OPENAI_EMBED_MODEL_NAME` | OpenAI embedding model | `text-embedding-3-small` |

Only the API key for the provider you select is required. The other can be left blank.

---

## 🧠 How It Works

```
Upload Files → Parse (Loader) → Split into Chunks → Embed → FAISS Index
                                                                   ↓
              Answer ← extract_text() ← LLM ← Prompt ← MMR Retrieve (k=6)
                                                ↑
                                       Manual Chat History (last 5 turns)
```

### Chunking
Documents are split using `RecursiveCharacterTextSplitter` with a chunk size of **800 characters** and an overlap of **100 characters** to avoid losing context at boundaries.

### Retrieval
Retrieval uses **MMR (Maximal Marginal Relevance)** search (`k=6`) which balances relevance with diversity — avoiding near-duplicate chunks when multiple subtopics are relevant.

### Conversation Memory
The app maintains a rolling window of the **last 5 Q&A turns** in `st.session_state.chat_history`. History is serialised to a plain string and injected into the prompt on each turn.

> **Note:** This app does **not** use LangChain's built-in Memory classes (`ConversationBufferMemory`, `ConversationSummaryMemory`, etc.). History is managed manually via `st.session_state`.

### LLM Temperature
All models run at a fixed temperature of **0.2** to keep answers factual and consistent.

---

## 📦 Tech Stack

| Layer | Technology |
|---|---|
| **UI** | Streamlit |
| **LLM (Gemini)** | `gemini-1.5-flash` via `langchain-google-genai` |
| **LLM (OpenAI)** | `gpt-4o-mini` via `langchain-openai` |
| **Embeddings (Gemini)** | `gemini-embedding-001` |
| **Embeddings (OpenAI)** | `text-embedding-3-small` |
| **Vector Store** | FAISS (persistent local storage via `langchain-community`) |
| **Document Loaders** | `PyPDFLoader`, `CSVLoader`, `TextLoader` |
| **Text Splitting** | `RecursiveCharacterTextSplitter` |
| **Chain** | LangChain LCEL (`ChatPromptTemplate \| LLM`) |
| **Config** | `python-dotenv` |

---

## ⚠️ Known Limitations

- The FAISS index is **rebuilt from scratch** each time — incremental indexing is not supported
- Chat history is **lost on page refresh** (stored only in `st.session_state`)
- The LLM is instructed to answer **only from the provided documents** — it will not use external knowledge
- On **Windows**, index reset uses a small delay (`gc.collect()` + `time.sleep(0.5)`) to release file locks before deletion

---

## 👥 Contributors

- **Chilaka Raghava** — Thanks for the laptop 🙏
