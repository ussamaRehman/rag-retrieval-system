# src/rag/chunking.py

from typing import List, Dict


def chunk_text(
    text: str,
    doc_id: str,
    chunk_size: int = 500,
    overlap: int = 0,
) -> List[Dict]:
    """
    Deterministically split text into fixed-size character chunks.
    """
    if chunk_size <= 0:
        raise ValueError("chunk_size must be > 0")
    if overlap < 0:
        raise ValueError("overlap must be >= 0")
    if overlap >= chunk_size:
        raise ValueError("overlap must be < chunk_size")

    chunks = []
    start = 0
    chunk_id = 0

    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunk_text = text[start:end]

        chunks.append(
            {
                "doc_id": doc_id,
                "chunk_id": f"{doc_id}_{chunk_id}",
                "text": chunk_text,
                "start_offset": start,
                "end_offset": end,
            }
        )

        chunk_id += 1
        start = end - overlap

    return chunks
