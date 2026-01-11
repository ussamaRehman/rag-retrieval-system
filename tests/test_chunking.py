import pytest

from rag.chunking import chunk_text


def test_chunk_size_must_be_positive() -> None:
    with pytest.raises(ValueError):
        chunk_text("abc", "doc", chunk_size=0)


def test_overlap_must_be_non_negative() -> None:
    with pytest.raises(ValueError):
        chunk_text("abc", "doc", chunk_size=3, overlap=-1)


def test_overlap_must_be_less_than_chunk_size() -> None:
    with pytest.raises(ValueError):
        chunk_text("abcdef", "doc", chunk_size=3, overlap=3)
