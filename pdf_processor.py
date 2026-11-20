"""
pdf_processor.py
────────────────
Handles PDF text extraction and intelligent chunking.
"""

import re
import PyPDF2
import io
from typing import List, Dict


# ─── Chunking config ─────────────────────────────────────────────────────────
CHUNK_SIZE = 500       # target words per chunk
CHUNK_OVERLAP = 50     # overlap words between consecutive chunks


def extract_text_from_pdf(file_bytes: bytes) -> str:
    """Extract raw text from a PDF byte stream."""
    reader = PyPDF2.PdfReader(io.BytesIO(file_bytes))
    pages_text = []
    for page in reader.pages:
        text = page.extract_text() or ""
        text = re.sub(r"\s+", " ", text).strip()
        if text:
            pages_text.append(text)
    return "\n\n".join(pages_text)


def chunk_text(text: str,
               chunk_size: int = CHUNK_SIZE,
               overlap: int = CHUNK_OVERLAP) -> List[Dict]:
    """
    Split *text* into overlapping word-based chunks.

    Returns a list of dicts:
        { "chunk_id": int, "text": str, "word_count": int }
    """
    words = text.split()
    chunks: List[Dict] = []
    start = 0
    chunk_id = 0

    while start < len(words):
        end = min(start + chunk_size, len(words))
        chunk_words = words[start:end]

        chunks.append({
            "chunk_id": chunk_id,
            "text": " ".join(chunk_words),
            "word_count": len(chunk_words),
        })

        chunk_id += 1
        step = max(chunk_size - overlap, 1)
        start += step

    return chunks


def process_pdf(file_bytes: bytes):
    """
    Full pipeline: extract → chunk.
    Returns (raw_text, chunks).
    """
    raw_text = extract_text_from_pdf(file_bytes)
    chunks = chunk_text(raw_text)
    return raw_text, chunks

