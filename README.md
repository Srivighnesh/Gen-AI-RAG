# 🧠 AI Research Assistant — RAG (Retrieval-Augmented Generation)

A Streamlit app that lets you upload PDF documents, index them with FAISS, and ask questions using Gemini or OpenAI — with source citations.

---

## 📁 Project Structure

```
your-project/
│
├── app.py                 # Main Streamlit app
├── requirements.txt       # Python dependencies
├── .env                   # API keys (DO NOT commit this)
├── .gitignore             # Ignores .env and vector_store/
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

1. Upload one or more **PDF** files in the sidebar
2. Click **Build / Rebuild Index** to create the FAISS vector store
3. Go to the **Ask Questions** tab
4. Type your question and click **Search**
5. View the answer + collapsible source citations

---

## 🔑 Environment Variables

| Variable | Description |
|---|---|
| `GOOGLE_API_KEY` | Gemini API key from Google AI Studio |
| `OPENAI_API_KEY` | OpenAI API key (optional if using Gemini) |
| `GOOGLE_CHAT_MODEL_NAME` | Default: `gemini-1.5-flash` |
| `GOOGLE_EMBED_MODEL_NAME` | Default: `gemini-embedding-001` |
| `OPENAI_CHAT_MODEL_NAME` | Default: `gpt-4o-mini` |
| `OPENAI_EMBED_MODEL_NAME` | Default: `text-embedding-3-small` |

---

## 📦 Tech Stack

- **UI:** Streamlit
- **LLM:** Gemini 1.5 Flash / GPT-4o-mini (via LangChain)
- **Embeddings:** Gemini Embedding 001 / text-embedding-3-small
- **Vector Store:** FAISS (persistent local storage)
- **Document Loader:** PyPDFLoader
- **Framework:** LangChain

## Contributors 

- **Chilaka Raghava THanks for laptop.
