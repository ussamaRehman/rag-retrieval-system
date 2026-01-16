from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
from rank_bm25 import BM25Okapi

DEFAULT_MODEL_NAME = "all-MiniLM-L6-v2"


class RetrievalService:
    def __init__(
        self,
        index_dir: Path,
        *,
        api_version: str,
        max_top_k: int,
        snippet_chars: int,
        default_mode: str,
    ) -> None:
        self.index_dir = Path(index_dir)
        self.api_version = api_version
        self.max_top_k = max_top_k
        self.snippet_chars = snippet_chars
        self.default_mode = default_mode

        self.chunks = self._load_metadata()
        self.texts = [chunk["text"] for chunk in self.chunks]
        self.doc_ids = [chunk["doc_id"] for chunk in self.chunks]

        self.bm25 = BM25Okapi([text.lower().split() for text in self.texts]) if self.texts else None
        self.embeddings = self._load_embeddings()
        self.embed_model_name = self._load_embed_model_name()
        self._dense_model = None

    def _load_metadata(self) -> List[Dict]:
        metadata_path = self.index_dir / "metadata.jsonl"
        chunks: List[Dict] = []
        if not metadata_path.exists():
            return chunks
        with metadata_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                chunks.append(json.loads(line))
        return chunks

    def _load_embeddings(self) -> np.ndarray:
        embeddings_path = self.index_dir / "embeddings.npy"
        if not embeddings_path.exists():
            return np.empty((0, 0), dtype=np.float32)
        embeddings = np.load(embeddings_path)
        if embeddings.ndim == 1:
            embeddings = embeddings.reshape(1, -1)
        if embeddings.size and embeddings.shape[0] != len(self.texts):
            return np.empty((0, 0), dtype=np.float32)
        return embeddings.astype(np.float32, copy=False)

    def _load_embed_model_name(self) -> str:
        params_path = self.index_dir / "params.json"
        if not params_path.exists():
            return DEFAULT_MODEL_NAME
        with params_path.open("r", encoding="utf-8") as handle:
            params = json.load(handle)
        return params.get("embed_model_name", DEFAULT_MODEL_NAME)

    def versions(self) -> Dict[str, str]:
        return {
            "api": self.api_version,
            "embed_model": self.embed_model_name,
            "index_dir": str(self.index_dir),
        }

    def retrieve(self, query: str, mode: Optional[str], top_k: int) -> List[Dict]:
        if not query or not query.strip():
            return []
        k = max(1, min(top_k, self.max_top_k))
        chosen_mode = mode or self.default_mode

        if chosen_mode == "bm25":
            return self._retrieve_bm25(query, k)
        if chosen_mode == "dense":
            return self._retrieve_dense(query, k)
        raise ValueError(f"Unknown retrieval mode: {chosen_mode}")

    def _retrieve_bm25(self, query: str, k: int) -> List[Dict]:
        if not self.bm25:
            return []
        scores = self.bm25.get_scores(query.lower().split())
        ranked = sorted(enumerate(scores), key=lambda pair: (-pair[1], pair[0]))
        return self._build_citations(ranked[:k])

    def _retrieve_dense(self, query: str, k: int) -> List[Dict]:
        if self.embeddings.size == 0:
            return []
        model = self._get_dense_model()
        if model is None:
            return []
        query_emb = model.encode([query], normalize_embeddings=True, show_progress_bar=False)
        scores = np.dot(query_emb, self.embeddings.T)[0]
        ranked = sorted(enumerate(scores), key=lambda pair: (-pair[1], pair[0]))
        return self._build_citations(ranked[:k])

    def _get_dense_model(self):
        if self._dense_model is None:
            from sentence_transformers import SentenceTransformer

            self._dense_model = SentenceTransformer(self.embed_model_name)
        return self._dense_model

    def _build_citations(self, ranked: List[tuple[int, float]]) -> List[Dict]:
        citations: List[Dict] = []
        for idx, score in ranked:
            chunk = self.chunks[idx]
            snippet = " ".join(chunk["text"].split())[: self.snippet_chars]
            citations.append(
                {
                    "doc_id": chunk["doc_id"],
                    "chunk_id": chunk["chunk_id"],
                    "score": float(score),
                    "snippet": snippet,
                    "start_offset": chunk["start_offset"],
                    "end_offset": chunk["end_offset"],
                }
            )
        return citations
