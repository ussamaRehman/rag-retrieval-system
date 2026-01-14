"""Evaluate BM25 and dense retrieval on a small synthetic set."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

DEFAULT_MODEL_NAME = "all-MiniLM-L6-v2"
EVAL_SET = [
    {"query": "refund policy", "expected_doc_id": "refunds"},
    {"query": "shipping times", "expected_doc_id": "shipping"},
    {"query": "reset my password", "expected_doc_id": "account"},
    {"query": "contact support", "expected_doc_id": "support"},
    {"query": "pricing tiers", "expected_doc_id": "pricing"},
    {"query": "data retention", "expected_doc_id": "security"},
]


def _load_metadata(index_dir: Path) -> list[dict]:
    metadata_path = index_dir / "metadata.jsonl"
    chunks: list[dict] = []
    if not metadata_path.exists():
        return chunks
    with metadata_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            chunks.append(json.loads(line))
    return chunks


def _compute_metrics(rank: int | None) -> tuple[float, float, float]:
    if rank is None:
        return 0.0, 0.0, 0.0
    recall = 1.0
    mrr = 1.0 / rank
    ndcg = 1.0 / math.log2(rank + 1)
    return recall, mrr, ndcg


def _evaluate(
    queries: list[dict],
    retrieve_fn,
    doc_ids: list[str],
    k: int,
) -> tuple[float, float, float]:
    recall_total = 0.0
    mrr_total = 0.0
    ndcg_total = 0.0

    for example in queries:
        expected = example["expected_doc_id"]
        indices = retrieve_fn(example["query"], k)
        rank = None
        for position, idx in enumerate(indices, start=1):
            if doc_ids[idx] == expected:
                rank = position
                break
        recall, mrr, ndcg = _compute_metrics(rank)
        recall_total += recall
        mrr_total += mrr
        ndcg_total += ndcg

    count = len(queries)
    if count == 0:
        return 0.0, 0.0, 0.0
    return recall_total / count, mrr_total / count, ndcg_total / count


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate retrieval on a small set.")
    parser.add_argument("--index", required=True, help="Index directory path.")
    args = parser.parse_args()

    index_dir = Path(args.index)
    chunks = _load_metadata(index_dir)
    texts = [chunk["text"] for chunk in chunks]
    doc_ids = [chunk["doc_id"] for chunk in chunks]

    embeddings_path = index_dir / "embeddings.npy"
    embeddings = (
        np.load(embeddings_path)
        if embeddings_path.exists()
        else np.empty((0, 0), dtype=np.float32)
    )
    if embeddings.ndim == 1:
        embeddings = embeddings.reshape(1, -1)
    if embeddings.size and embeddings.shape[0] != len(texts):
        raise ValueError("Embeddings row count does not match metadata.")

    params_path = index_dir / "params.json"
    if params_path.exists():
        with params_path.open("r", encoding="utf-8") as handle:
            params = json.load(handle)
        model_name = params.get("embed_model_name", DEFAULT_MODEL_NAME)
    else:
        model_name = DEFAULT_MODEL_NAME

    print(f"Loaded {len(texts)} chunks from {index_dir}")

    bm25 = BM25Okapi([text.lower().split() for text in texts]) if texts else None

    def bm25_retrieve(query: str, k: int) -> list[int]:
        if not bm25 or not query.strip():
            return []
        scores = bm25.get_scores(query.lower().split())
        order = np.argsort(scores)[::-1]
        return order[:k].tolist()

    if embeddings.size:
        model = SentenceTransformer(model_name)
    else:
        model = None

    def dense_retrieve(query: str, k: int) -> list[int]:
        if model is None or not query.strip():
            return []
        query_emb = model.encode([query], normalize_embeddings=True, show_progress_bar=False)
        scores = np.dot(query_emb, embeddings.T)[0]
        order = np.argsort(scores)[::-1]
        return order[:k].tolist()

    bm25_metrics = _evaluate(EVAL_SET, bm25_retrieve, doc_ids, k=10)
    dense_metrics = _evaluate(EVAL_SET, dense_retrieve, doc_ids, k=10)

    print(
        "BM25 Recall@10={:.3f} MRR@10={:.3f} nDCG@10={:.3f}".format(
            *bm25_metrics
        )
    )
    print(
        "Dense Recall@10={:.3f} MRR@10={:.3f} nDCG@10={:.3f}".format(
            *dense_metrics
        )
    )


if __name__ == "__main__":
    main()
