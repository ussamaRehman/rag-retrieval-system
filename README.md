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
uv run pytest -q
```

### Install troubleshooting
`sentence-transformers` pulls heavy deps (PyTorch), and `uv sync` may time out on slow connections.

```sh
UV_HTTP_TIMEOUT=120 uv sync
```

Notes:
- Unit tests do NOT download models (they use a fake embedder).
- Model downloads (if needed) happen during the offline indexing step (`build_index.py`).

### Manual sanity check (BM25 + Dense)
```sh
uv run python -c "from src.rag.chunking import chunk_text; from src.rag.bm25 import BM25Retriever; from src.rag.dense import DenseRetriever; text='Refunds are issued within five days. Contact support to get your money back.'; chunks=chunk_text(text,'doc-1',chunk_size=40,overlap=5); bm=BM25Retriever(chunks); dn=DenseRetriever(chunks, strict=False); print('bm25', [c['chunk_id'] for c in bm.search('refund', top_k=2)]); print('dense', [c['chunk_id'] for c in dn.search('get my money back', top_k=2)])"
```
