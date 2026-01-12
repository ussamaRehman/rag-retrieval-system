import numpy as np
import pytest

import rag.dense as dense


class FakeModel:
    def __init__(self, name: str):
        self.name = name

    def encode(
        self,
        texts,
        normalize_embeddings: bool = True,
        show_progress_bar: bool = False,
    ):
        vectors = []
        for text in texts:
            if text == "alpha":
                vec = np.array([1.0, 0.0], dtype=np.float32)
            elif text == "beta":
                vec = np.array([0.0, 1.0], dtype=np.float32)
            elif text == "alpha beta":
                vec = np.array([0.6, 0.8], dtype=np.float32)
            else:
                vec = np.array([0.0, 0.0], dtype=np.float32)

            if normalize_embeddings:
                norm = np.linalg.norm(vec)
                if norm > 0:
                    vec = vec / norm
            vectors.append(vec)

        if not vectors:
            return np.empty((0, 2), dtype=np.float32)
        return np.vstack(vectors)


@pytest.fixture(autouse=True)
def _patch_sentence_transformer(monkeypatch):
    monkeypatch.setattr(dense, "SentenceTransformer", FakeModel)


def test_strict_rejects_missing_text() -> None:
    with pytest.raises(ValueError):
        dense.DenseRetriever([{"doc_id": "doc-1"}], strict=True)


def test_strict_rejects_non_string_query() -> None:
    retriever = dense.DenseRetriever([{"doc_id": "doc-1", "text": "alpha"}])
    with pytest.raises(ValueError):
        retriever.search(None)  # type: ignore[arg-type]


def test_strict_rejects_blank_query() -> None:
    retriever = dense.DenseRetriever([{"doc_id": "doc-1", "text": "alpha"}])
    with pytest.raises(ValueError):
        retriever.search("   ")


def test_strict_rejects_non_positive_top_k() -> None:
    retriever = dense.DenseRetriever([{"doc_id": "doc-1", "text": "alpha"}])
    with pytest.raises(ValueError):
        retriever.search("alpha", top_k=0)


def test_empty_corpus_returns_empty() -> None:
    retriever = dense.DenseRetriever([], strict=True)
    assert retriever.search("alpha") == []


def test_permissive_skips_invalid_chunks() -> None:
    chunks = [
        {"doc_id": "doc-1"},
        {"doc_id": "doc-2", "text": "alpha"},
    ]
    retriever = dense.DenseRetriever(chunks, strict=False)
    results = retriever.search("alpha", top_k=5)
    assert len(results) == 1
    assert results[0]["doc_id"] == "doc-2"
    assert "score" in results[0]


def test_ranking_prefers_closest_match() -> None:
    chunks = [
        {"doc_id": "doc-1", "text": "alpha"},
        {"doc_id": "doc-2", "text": "beta"},
    ]
    retriever = dense.DenseRetriever(chunks, strict=True)
    results = retriever.search("alpha", top_k=1)
    assert results[0]["doc_id"] == "doc-1"
