"""Build an offline retrieval index from raw text files."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

import numpy as np
from sentence_transformers import SentenceTransformer

from src.rag.chunking import chunk_text

DEFAULT_MODEL_NAME = "all-MiniLM-L6-v2"


def _iter_input_files(input_dir: Path) -> Iterable[Path]:
    patterns = ["*.txt", "*.md"]
    files: list[Path] = []
    for pattern in patterns:
        files.extend(input_dir.rglob(pattern))
    return sorted(files, key=lambda path: str(path))


def _write_metadata(metadata_path: Path, chunks: list[dict]) -> None:
    with metadata_path.open("w", encoding="utf-8") as handle:
        for chunk in chunks:
            record = {
                "doc_id": chunk["doc_id"],
                "chunk_id": chunk["chunk_id"],
                "text": chunk["text"],
                "start_offset": chunk["start_offset"],
                "end_offset": chunk["end_offset"],
            }
            json.dump(record, handle, ensure_ascii=True)
            handle.write("\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build offline retrieval index.")
    parser.add_argument("--input", required=True, help="Input directory of raw files.")
    parser.add_argument(
        "--output", required=True, help="Output directory for index artifacts."
    )
    parser.add_argument(
        "--chunk-size", type=int, default=500, help="Chunk size in characters."
    )
    parser.add_argument("--overlap", type=int, default=0, help="Chunk overlap.")
    parser.add_argument(
        "--model", default=DEFAULT_MODEL_NAME, help="SentenceTransformer model name."
    )
    args = parser.parse_args()

    input_dir = Path(args.input)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    files = _iter_input_files(input_dir) if input_dir.exists() else []
    chunks: list[dict] = []
    chunk_texts: list[str] = []
    doc_ids: set[str] = set()

    for path in files:
        doc_id = path.stem
        doc_ids.add(doc_id)
        text = path.read_text(encoding="utf-8")
        doc_chunks = chunk_text(
            text, doc_id, chunk_size=args.chunk_size, overlap=args.overlap
        )
        chunks.extend(doc_chunks)
        chunk_texts.extend(chunk["text"] for chunk in doc_chunks)

    if chunk_texts:
        model = SentenceTransformer(args.model)
        embeddings = model.encode(
            chunk_texts, normalize_embeddings=True, show_progress_bar=False
        )
        embeddings = np.asarray(embeddings, dtype=np.float32)
    else:
        embeddings = np.empty((0, 0), dtype=np.float32)

    _write_metadata(output_dir / "metadata.jsonl", chunks)
    np.save(output_dir / "embeddings.npy", embeddings)

    params = {
        "embed_model_name": args.model,
        "chunk_size": args.chunk_size,
        "overlap": args.overlap,
        "num_docs": len(doc_ids),
        "num_chunks": len(chunks),
    }
    with (output_dir / "params.json").open("w", encoding="utf-8") as handle:
        json.dump(params, handle, indent=2, sort_keys=True)

    print(f"Indexed {len(doc_ids)} docs and {len(chunks)} chunks.")
    print(f"Wrote artifacts to {output_dir}")


if __name__ == "__main__":
    main()
