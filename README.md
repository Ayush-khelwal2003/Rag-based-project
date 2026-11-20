# 📄 RAG-Based Document Question Answering System

A **Retrieval-Augmented Generation (RAG)** chatbot that lets you upload any PDF and ask natural-language questions about its content — powered by **Cohere embeddings**, **Pinecone vector search**, and a **Streamlit** UI.

---

## 🏗️ Architecture

```
PDF Upload
   │
   ▼
pdf_processor.py  ──►  Text Extraction (PyPDF2)
                        Word-based Chunking (500 words / 50 overlap)
   │
   ▼
vector_store.py   ──►  Cohere embed-english-v3.0  (1024-dim embeddings)
                        Pinecone Serverless Index  (cosine similarity)
   │
   ▼
rag_engine.py     ──►  User Question → embed → Pinecone query (top-K)
                        Build context prompt → Cohere command-r-plus
                        Return answer + source excerpts
   │
   ▼
app.py            ──►  Streamlit interactive UI
```

---

## 📁 Project Structure

```
RAG_Project/
├── app.py               # Streamlit UI (main entry point)
├── pdf_processor.py     # PDF extraction + chunking
├── vector_store.py      # Cohere embeddings + Pinecone operations
├── rag_engine.py        # RAG orchestration + answer generation
├── requirements.txt     # Python dependencies
├── .env.example         # Environment variable template
└── README.md
```

---

## ⚙️ Setup

### 1. Clone / navigate to project
```bash
cd RAG_Project
```

### 2. Create a virtual environment
```bash
python3 -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Configure API keys

Copy the example env file and fill in your keys:
```bash
cp .env.example .env
```

Edit `.env`:
```env
COHERE_API_KEY=<your Cohere API key>
PINECONE_API_KEY=<your Pinecone API key>
PINECONE_INDEX_NAME=rag-document-qa
```

| Key | Where to get it |
|-----|----------------|
| `COHERE_API_KEY` | [dashboard.cohere.com](https://dashboard.cohere.com) → API Keys |
| `PINECONE_API_KEY` | [app.pinecone.io](https://app.pinecone.io) → API Keys |

> **Pinecone free tier** supports 1 serverless index — the app auto-creates it on first run.

### 5. Run the app
```bash
streamlit run app.py
```

The app opens at **http://localhost:8501** in your browser.

---

## 🚀 Usage

1. **Upload** a PDF using the file uploader.
2. Wait for the ✅ success message — the document is chunked, embedded, and stored in Pinecone.
3. **Type a question** in the chat box and press **Send 🚀**.
4. The assistant retrieves the most relevant excerpts and generates a grounded answer.
5. Click **"View source excerpts"** under any answer to see which parts of the document were used.
6. Use **🗑️ Remove Document** (sidebar) to clear the index and upload a new PDF.

---

## 🔧 Configuration

| Setting | File | Default |
|---------|------|---------|
| Chunk size (words) | `pdf_processor.py` → `CHUNK_SIZE` | 500 |
| Chunk overlap | `pdf_processor.py` → `CHUNK_OVERLAP` | 50 |
| Embedding model | `vector_store.py` → `EMBED_MODEL` | `embed-english-v3.0` |
| Generation model | `rag_engine.py` → `GENERATE_MODEL` | `command-r-plus` |
| Max answer tokens | `rag_engine.py` → `MAX_TOKENS` | 800 |
| Top-K chunks | Sidebar slider | 5 (1–10) |

---

## 📦 Dependencies

| Package | Purpose |
|---------|---------|
| `streamlit` | Interactive web UI |
| `cohere` | Embeddings + text generation |
| `pinecone-client` | Vector database |
| `PyPDF2` | PDF text extraction |
| `python-dotenv` | Environment variable loading |

# Rag-based-project
