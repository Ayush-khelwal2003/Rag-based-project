# 🧠 DocMind AI — RAG-Based Document Question Answering System

[![Live Demo](https://img.shields.io/badge/Streamlit-Live_Demo-FF4B4B?style=for-the-badge&logo=Streamlit&logoColor=white)](https://rag-based-project-jsaymum9zccpzfvxucbmmh.streamlit.app)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=Python&logoColor=white)](https://www.python.org/)
[![Cohere](https://img.shields.io/badge/Cohere-Embeddings_%26_LLM-39594C?style=for-the-badge)](https://cohere.com/)
[![Pinecone](https://img.shields.io/badge/Pinecone-Serverless_Vector_DB-000000?style=for-the-badge)](https://www.pinecone.io/)

> **🌐 Live Application:** [https://rag-based-project-jsaymum9zccpzfvxucbmmh.streamlit.app](https://rag-based-project-jsaymum9zccpzfvxucbmmh.streamlit.app)

A full-stack **Retrieval-Augmented Generation (RAG)** chatbot that allows users to upload PDF documents and ask context-aware questions to receive document-specific answers with real-time source citations and similarity scores.

---

## ✨ Features

- **📄 Document Processing**: Text extraction and sliding-window chunking via PyPDF2.
- **⚡ Semantic Search**: Powered by **Cohere `embed-english-v3.0`** (1024 dimensions) and **Pinecone Serverless Vector Database**.
- **🎯 Context-Aware Generation**: Leverages **Cohere `command-a-03-2025`** with multi-turn prompt augmentation to guarantee grounded, hallucination-free answers.
- **🔍 Source Verification**: Interactive citation accordions displaying retrieved chunk IDs, excerpt text, and cosine similarity match percentages.
- **🎨 Modern Dark UI**: Polished glassmorphic Streamlit interface with suggested queries, analytics metrics, and session export tools.

---

## 🏗️ System Architecture

```
PDF Document
    │
    ▼
pdf_processor.py   ──►  Extract text (PyPDF2)
                         Sliding-window chunking (500 words / 50 overlap)
    │
    ▼
vector_store.py    ──►  Embed via Cohere (embed-english-v3.0)
                         Upsert vectors into Pinecone Serverless Index
    │
    ▼
rag_engine.py      ──►  Embed query → Cosine similarity retrieval (top-K)
                         Augment prompt with retrieved excerpts
                         Generate grounded answer via Cohere (command-a-03-2025)
    │
    ▼
app.py             ──►  Streamlit interactive frontend (chat, sources, analytics)
```

---

## 📁 Project Structure

```
RAG_Project/
├── app.py               # Streamlit application UI & chat interface
├── pdf_processor.py     # PDF text extraction and chunking pipeline
├── vector_store.py      # Cohere embeddings & Pinecone vector DB integration
├── rag_engine.py        # RAG pipeline orchestration & generation
├── requirements.txt     # Python dependencies
├── .env.example         # Environment variable template
├── .gitignore           # Prevents secrets (.env) from leaking
└── README.md            # Project documentation & live links
```

---

## 🚀 Quickstart (Local Setup)

### 1. Clone repository
```bash
git clone https://github.com/Ayush-khelwal2003/Rag-based-project.git
cd Rag-based-project
```

### 2. Set up virtual environment
```bash
python3 -m venv venv
source venv/bin/activate      # On Windows: venv\Scripts\activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Configure environment variables
Create a `.env` file based on `.env.example`:
```bash
cp .env.example .env
```

Add your API keys inside `.env`:
```env
COHERE_API_KEY=your_cohere_api_key_here
PINECONE_API_KEY=your_pinecone_api_key_here
PINECONE_INDEX_NAME=rag-document-qa
```

| Key | Description | Source |
|---|---|---|
| `COHERE_API_KEY` | Required for embeddings & answer generation | [Cohere Dashboard](https://dashboard.cohere.com/api-keys) |
| `PINECONE_API_KEY` | Vector database API key | [Pinecone Console](https://app.pinecone.io) |

### 5. Launch the application
```bash
streamlit run app.py
```
Open **http://localhost:8501** in your browser.

---

## ☁️ Deployment on Streamlit Cloud

1. Fork or push this repository to GitHub.
2. Sign in to [Streamlit Community Cloud](https://share.streamlit.io/).
3. Deploy with:
   - **Repository:** `Ayush-khelwal2003/Rag-based-project`
   - **Branch:** `main`
   - **Main file:** `app.py`
4. Add your API keys under **App Settings ➔ Secrets**:
   ```toml
   COHERE_API_KEY = "your_cohere_key"
   PINECONE_API_KEY = "your_pinecone_key"
   PINECONE_INDEX_NAME = "rag-document-qa"
   ```

---

## ⚙️ Configuration & Models

| Component | Technology / Model | Default |
|---|---|---|
| Embedding Model | Cohere `embed-english-v3.0` | 1024 Dimensions |
| Vector Database | Pinecone Serverless | Cosine Metric |
| Generation LLM | Cohere `command-a-03-2025` | Temperature = 0.3 |
| Chunk Strategy | Sliding Window | 500 words / 50 overlap |
| Top-K Retrieval | Dynamic slider | 5 chunks |
