# PRD — Production RAG System

## Goal
Build a **production-grade Retrieval-Augmented Generation (RAG) system** that demonstrates:
- end-to-end ML engineering thinking,
- reproducible offline pipelines,
- measurable retrieval quality,
- production-ready inference APIs,
- observability, reliability, and safety controls.

---

## Build
- Offline pipeline:
  - ingest → chunk → index → evaluate → versioned artifacts
- Online system:
  - FastAPI inference API
  - retrieval + grounded answers with citations
  - optional tool calling via strict schemas
- Engineering rigor:
  - Docker, CI, tests, metrics, logs
  - deterministic builds and evaluations

---

## Non-goals
- Frontend or UI
- Cloud deployment (AWS/GCP/K8s)
- LangChain / LlamaIndex abstractions
- Conversational memory across turns
- Fine-tuning large LLMs
- Internet browsing tools

---


---

## Success criteria (Definition of Done)
A reviewer can:

1. Clone the repo
2. Run a command to build an index from a demo corpus
3. Reproduce retrieval metrics via an eval script
4. Run the API locally (or via Docker)
5. Query `/predict` and receive:
   - an answer
   - structured citations
   - or a correct abstention (`no_answer=true`)
6. Inspect `/metrics` and see counters/latency histograms
7. See CI green with tests passing
8. Understand design tradeoffs via docs

---

## MVP scope (locked)

### Retrieval
- BM25 baseline
- Dense retrieval (MiniLM + FAISS)
- Optional hybrid (post-MVP)

### Generation
- Deterministic, extractive, grounded synthesis (CI-safe)
- Pluggable interface for external LLMs (optional runtime)

### Citations
- Mandatory
- Include chunk offsets and snippets

### Tool calling
- Strict JSON schema
- Allowlisted tools only
- Per-request tool budget
- Timeouts + safe execution

### Hardening
- Rate limiting
- Timeouts
- Input size caps
- Consistent JSON errors

### Observability
- Structured logs
- Prometheus metrics

### Evaluation
- Retrieval metrics:
  - nDCG@10
  - MRR@10
  - Recall@10
- Threshold tuning for abstention
- Error analysis report

---

## Explicitly out of scope (do not implement)
- Authentication / multi-tenancy
- Vector DB services
- Cross-encoder rerankers (initially)
- Human labeling pipelines