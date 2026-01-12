# rag-retrieval-system

A hiring-first, production-oriented **retrieval system** for a RAG pipeline.
Focus: deterministic chunking, BM25 + dense retrieval baselines, reproducible offline artifacts, and measurable evaluation.

## Status
- Gate 1 (Retrieval): chunking ✅, BM25 ✅, dense ✅, index build ⏳, eval ⏳
- API + citations: not started (out of Gate 1 scope)

## Modules
- `src/rag/chunking.py`: deterministic fixed-size chunking (with overlap)
- `src/rag/bm25.py`: BM25 retriever (strict by default, permissive optional)
- `src/rag/dense.py`: dense retriever (MiniLM embeddings)
- `src/rag/build_index.py`: offline index builder *(planned)*
- `src/rag/eval_retrieval.py`: offline retrieval evaluation *(planned)*

## Quickstart

### Install
```sh
uv sync
```

### Tests
```sh
pytest
```
