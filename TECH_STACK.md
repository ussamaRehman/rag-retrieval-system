# TECH_STACK.md

This document defines the **approved tech stack** for this project.
Changes require explicit discussion.

---

## Language & runtime
- Python 3.11+
- Environment & dependency management: **uv**

Why:
- fast installs
- reproducible lockfiles
- modern Python workflow

---

## API
- FastAPI
- Uvicorn

Why:
- async-friendly
- strong typing with Pydantic
- industry standard for ML inference services

---

## Retrieval
- BM25: `rank-bm25`
- Dense embeddings: `sentence-transformers` (MiniLM)
- Vector index: `faiss-cpu`

Why:
- transparent, controllable retrieval
- CPU-friendly
- no managed services

---

## Data & evaluation
- `datasets` (HF) for public datasets
- `numpy`, `pandas`, `pyarrow`

Why:
- reproducibility
- efficient artifact handling

---

## Observability
- Metrics: `prometheus-client`
- Logging: `structlog`

Why:
- explicit metrics > logs
- common in production MLOps stacks

---

## Reliability & hardening
- In-memory token-bucket rate limiter
- Timeouts via `anyio` / asyncio
- Strict Pydantic validation

Why:
- no external infra dependencies
- clear failure modes

---

## Quality & CI
- Tests: `pytest`, `httpx`
- Lint/format: `ruff`
- CI: GitHub Actions

Why:
- fast feedback loops
- hiring-signal tooling

---

## Explicitly disallowed
- LangChain
- LlamaIndex
- Managed vector DBs
- Implicit magic abstractions