"""
vector_store.py
───────────────
Manages Cohere embeddings + Pinecone vector database operations.
"""

import os
import time
from typing import List, Dict, Optional

import cohere
from pinecone import Pinecone, ServerlessSpec
from dotenv import load_dotenv

load_dotenv()

# ─── Constants ───────────────────────────────────────────────────────────────
EMBED_MODEL      = "embed-english-v3.0"   # Cohere embedding model
EMBED_DIMENSION  = 1024                   # dimension for embed-english-v3.0
UPSERT_BATCH     = 100                    # vectors per Pinecone upsert batch


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


class VectorStore:
    """Wraps Cohere embeddings and a Pinecone index for RAG retrieval."""

    def __init__(self):
        cohere_key   = _get_secret("COHERE_API_KEY")
        pinecone_key = _get_secret("PINECONE_API_KEY")
        self.index_name = _get_secret("PINECONE_INDEX_NAME", "rag-document-qa")

        if not cohere_key or not pinecone_key:
            raise ValueError(
                "Missing API keys. Set COHERE_API_KEY and PINECONE_API_KEY "
                "in your .env file."
            )

        self.co = cohere.Client(cohere_key)
        self.pc = Pinecone(api_key=pinecone_key)
        self.index = self._get_or_create_index()

    # ── Index management ──────────────────────────────────────────────────────

    def _get_or_create_index(self):
        existing = [i.name for i in self.pc.list_indexes()]
        if self.index_name not in existing:
            self.pc.create_index(
                name=self.index_name,
                dimension=EMBED_DIMENSION,
                metric="cosine",
                spec=ServerlessSpec(cloud="aws", region="us-east-1"),
            )
            # Wait until the index is ready
            while not self.pc.describe_index(self.index_name).status["ready"]:
                time.sleep(1)
        return self.pc.Index(self.index_name)

    # ── Embedding ─────────────────────────────────────────────────────────────

    def embed_texts(self, texts: List[str], input_type: str = "search_document") -> List[List[float]]:
        """Generate embeddings for a list of texts via Cohere."""
        response = self.co.embed(
            texts=texts,
            model=EMBED_MODEL,
            input_type=input_type,
        )
        return response.embeddings

    # ── Upsert ────────────────────────────────────────────────────────────────

    def upsert_chunks(self, chunks: List[Dict], doc_id: str) -> int:
        """
        Embed and upsert chunks into Pinecone.

        Parameters
        ----------
        chunks  : list of {"chunk_id": int, "text": str, ...}
        doc_id  : unique identifier for this document (used as namespace)

        Returns the number of vectors upserted.
        """
        texts = [c["text"] for c in chunks]
        embeddings = self.embed_texts(texts, input_type="search_document")

        vectors = []
        for chunk, embedding in zip(chunks, embeddings):
            vectors.append({
                "id": f"{doc_id}_chunk_{chunk['chunk_id']}",
                "values": embedding,
                "metadata": {
                    "text": chunk["text"],
                    "chunk_id": chunk["chunk_id"],
                    "doc_id": doc_id,
                },
            })

        # Upsert in batches
        total = 0
        for i in range(0, len(vectors), UPSERT_BATCH):
            batch = vectors[i : i + UPSERT_BATCH]
            self.index.upsert(vectors=batch, namespace=doc_id)
            total += len(batch)

        return total

    # ── Query ─────────────────────────────────────────────────────────────────

    def query(self, question: str, doc_id: str, top_k: int = 5) -> List[Dict]:
        """
        Retrieve the *top_k* most relevant chunks for *question*.

        Returns a list of metadata dicts with added 'score' field.
        """
        q_embedding = self.embed_texts([question], input_type="search_query")[0]
        results = self.index.query(
            vector=q_embedding,
            top_k=top_k,
            include_metadata=True,
            namespace=doc_id,
        )
        matches = []
        for match in results.matches:
            item = dict(match.metadata) if match.metadata else {}
            if hasattr(match, "score") and match.score is not None:
                item["score"] = round(float(match.score), 4)
            matches.append(item)
        return matches

    # ── Namespace helpers ─────────────────────────────────────────────────────

    def delete_namespace(self, doc_id: str) -> None:
        """Remove all vectors for a document namespace."""
        self.index.delete(delete_all=True, namespace=doc_id)

    def namespace_exists(self, doc_id: str) -> bool:
        """Return True if the namespace already has vectors."""
        stats = self.index.describe_index_stats()
        return doc_id in stats.get("namespaces", {})

