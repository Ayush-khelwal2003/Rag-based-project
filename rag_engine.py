"""
rag_engine.py
─────────────
Orchestrates retrieval + Cohere generation to answer questions
about an uploaded PDF document.
"""

import os
from typing import List, Dict, Optional

import cohere
from dotenv import load_dotenv

from vector_store import VectorStore

load_dotenv()

# ─── Generation config ────────────────────────────────────────────────────────
GENERATE_MODEL = "command-a-03-2025"   # Cohere chat / generation model
MAX_TOKENS     = 800
TEMPERATURE    = 0.3
TOP_K_CHUNKS   = 5


def _get_secret(key: str, default: str = "") -> str:
    val = os.getenv(key, "")
    if not val:
        try:
            import streamlit as st
            if key in st.secrets:
                val = str(st.secrets[key])
        except Exception:
            pass
    return val or default


class RAGEngine:
    """
    End-to-end RAG pipeline:
      query → retrieve relevant chunks → build prompt → generate answer
    """

    def __init__(self, vector_store: VectorStore):
        self.vs = vector_store
        self.co = cohere.Client(_get_secret("COHERE_API_KEY"))

    # ── Context builder ───────────────────────────────────────────────────────

    @staticmethod
    def _build_context(chunks: List[Dict]) -> str:
        parts = []
        for i, chunk in enumerate(chunks, 1):
            parts.append(f"[Excerpt {i}]\n{chunk['text']}")
        return "\n\n".join(parts)

    # ── Prompt builder ────────────────────────────────────────────────────────

    @staticmethod
    def _build_prompt(question: str, context: str) -> str:
        return (
            "You are a helpful assistant that answers questions strictly based "
            "on the provided document excerpts.\n\n"
            "If the answer is not contained in the excerpts, say: "
            "\"I could not find relevant information in the document.\"\n\n"
            f"Document Excerpts:\n{context}\n\n"
            f"Question: {question}\n\n"
            "Answer:"
        )

    # ── Main QA method ────────────────────────────────────────────────────────

    def answer(
        self,
        question: str,
        doc_id: str,
        top_k: int = TOP_K_CHUNKS,
        temperature: float = TEMPERATURE,
        chat_history: Optional[List[Dict]] = None,
    ) -> Dict:
        """
        Answer *question* about *doc_id*.

        Returns:
            {
                "answer": str,
                "source_chunks": List[Dict],
                "context": str,
            }
        """
        # 1. Retrieve relevant chunks
        source_chunks = self.vs.query(question, doc_id, top_k=top_k)

        if not source_chunks:
            return {
                "answer": "No relevant content found in the document for your question.",
                "source_chunks": [],
                "context": "",
            }

        # 2. Build context & prompt
        context = self._build_context(source_chunks)
        prompt  = self._build_prompt(question, context)

        # 3. Generate answer with Cohere
        response = self.co.chat(
            model=GENERATE_MODEL,
            message=prompt,
            max_tokens=MAX_TOKENS,
            temperature=temperature,
        )

        return {
            "answer": response.text.strip(),
            "source_chunks": source_chunks,
            "context": context,
        }

