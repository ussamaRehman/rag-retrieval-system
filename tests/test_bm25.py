import pytest

from rag.bm25 import BM25Retriever


def test_strict_requires_non_empty_chunks() -> None:
    with pytest.raises(ValueError):
        BM25Retriever([], strict=True)


def test_strict_requires_text_key() -> None:
    with pytest.raises(ValueError):
        BM25Retriever([{"doc_id": "doc-1"}], strict=True)


def test_strict_top_k_must_be_positive() -> None:
    retriever = BM25Retriever([{"doc_id": "doc-1", "text": "hello world"}])
    with pytest.raises(ValueError):
        retriever.search("hello", top_k=0)


def test_strict_requires_non_empty_query() -> None:
    retriever = BM25Retriever([{"doc_id": "doc-1", "text": "hello world"}])
    with pytest.raises(ValueError):
        retriever.search("   ")


def test_permissive_empty_chunks_returns_empty() -> None:
    retriever = BM25Retriever([], strict=False)
    assert retriever.search("hello") == []


def test_permissive_skips_invalid_chunks() -> None:
    chunks = [
        {"doc_id": "doc-1"},
        {"doc_id": "doc-2", "text": "hello world"},
    ]
    retriever = BM25Retriever(chunks, strict=False)
    results = retriever.search("hello", top_k=5)
    assert len(results) == 1
    assert results[0]["doc_id"] == "doc-2"
    assert "score" in results[0]
