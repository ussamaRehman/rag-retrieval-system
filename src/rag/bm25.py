# src/rag/bm25.py

from typing import List, Dict
from rank_bm25 import BM25Okapi


class BM25Retriever:
    def __init__(self, chunks: List[Dict], *, strict: bool = True):
        self.strict = strict
        self.chunks: List[Dict] = []
        self.tokenized_corpus: List[List[str]] = []
        self.bm25 = None

        if strict:
            self._init_strict(chunks)
        else:
            self._init_permissive(chunks)

    def _init_strict(self, chunks: List[Dict]) -> None:
        if not chunks:
            raise ValueError("chunks must be non-empty")

        for idx, chunk in enumerate(chunks):
            if not isinstance(chunk, dict):
                raise ValueError(f"chunk at index {idx} must be a dict")
            if "text" not in chunk:
                raise ValueError(f"chunk at index {idx} missing 'text'")
            text = chunk["text"]
            if not isinstance(text, str):
                raise ValueError(f"chunk at index {idx} text must be a string")
            if not text.strip():
                raise ValueError(f"chunk at index {idx} text must be non-empty")

        self.chunks = chunks
        self.tokenized_corpus = [c["text"].lower().split() for c in chunks]
        self.bm25 = BM25Okapi(self.tokenized_corpus)

    def _init_permissive(self, chunks: List[Dict]) -> None:
        valid_chunks: List[Dict] = []
        tokenized: List[List[str]] = []

        for chunk in chunks or []:
            if not isinstance(chunk, dict):
                continue
            text = chunk.get("text")
            if not isinstance(text, str):
                continue
            if not text.strip():
                continue
            valid_chunks.append(chunk)
            tokenized.append(text.lower().split())

        self.chunks = valid_chunks
        self.tokenized_corpus = tokenized
        if tokenized:
            self.bm25 = BM25Okapi(tokenized)

    def search(self, query: str, top_k: int = 5) -> List[Dict]:
        if top_k <= 0:
            if self.strict:
                raise ValueError("top_k must be > 0")
            return []
        if not isinstance(query, str):
            if self.strict:
                raise ValueError("query must be a string")
            return []
        if not query.strip():
            if self.strict:
                raise ValueError("query must be non-empty")
            return []
        if self.bm25 is None:
            return []

        tokenized_query = query.lower().split()
        if not tokenized_query:
            return []
        scores = self.bm25.get_scores(tokenized_query)

        scored = list(zip(self.chunks, scores))
        scored.sort(key=lambda x: x[1], reverse=True)

        results = []
        for chunk, score in scored[:top_k]:
            result = dict(chunk)
            result["score"] = float(score)
            results.append(result)

        return results
