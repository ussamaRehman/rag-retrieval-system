# src/rag/dense.py

from typing import List, Dict

import numpy as np
from sentence_transformers import SentenceTransformer


class DenseRetriever:
    def __init__(
        self,
        chunks: List[Dict],
        model_name: str = "all-MiniLM-L6-v2",
        *,
        strict: bool = True,
    ):
        self.strict = strict
        self.chunks: List[Dict] = []
        self.texts: List[str] = []
        self.embeddings = np.empty((0, 0), dtype=np.float32)

        if strict:
            self._init_strict(chunks)
        else:
            self._init_permissive(chunks)

        self.model = SentenceTransformer(model_name)
        if self.texts:
            self.embeddings = self.model.encode(
                self.texts,
                normalize_embeddings=True,
                show_progress_bar=False,
            )

    def _init_strict(self, chunks: List[Dict]) -> None:
        if chunks is None:
            raise ValueError("chunks must be provided")
        if not isinstance(chunks, list):
            raise ValueError("chunks must be a list")

        for idx, chunk in enumerate(chunks):
            if not isinstance(chunk, dict):
                raise ValueError(f"chunk at index {idx} must be a dict")
            text = chunk.get("text")
            if not isinstance(text, str):
                raise ValueError(f"chunk at index {idx} text must be a string")
            if not text.strip():
                raise ValueError(f"chunk at index {idx} text must be non-empty")
            self.chunks.append(chunk)
            self.texts.append(text)

    def _init_permissive(self, chunks: List[Dict]) -> None:
        for chunk in chunks or []:
            if not isinstance(chunk, dict):
                continue
            text = chunk.get("text")
            if not isinstance(text, str):
                continue
            if not text.strip():
                continue
            self.chunks.append(chunk)
            self.texts.append(text)

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
        if self.embeddings.size == 0:
            return []

        query_emb = self.model.encode(
            [query],
            normalize_embeddings=True,
            show_progress_bar=False,
        )

        scores = np.dot(query_emb, self.embeddings.T)[0]
        scored = list(zip(self.chunks, scores))
        scored.sort(key=lambda x: x[1], reverse=True)

        results = []
        for chunk, score in scored[:top_k]:
            result = dict(chunk)
            result["score"] = float(score)
            results.append(result)

        return results
