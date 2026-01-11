import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rag.chunking import chunk_text


class TestChunkTextValidation(unittest.TestCase):
    def test_chunk_size_must_be_positive(self) -> None:
        with self.assertRaises(ValueError):
            chunk_text("abc", "doc", chunk_size=0)

    def test_overlap_must_be_non_negative(self) -> None:
        with self.assertRaises(ValueError):
            chunk_text("abc", "doc", chunk_size=3, overlap=-1)

    def test_overlap_must_be_less_than_chunk_size(self) -> None:
        with self.assertRaises(ValueError):
            chunk_text("abcdef", "doc", chunk_size=3, overlap=3)

