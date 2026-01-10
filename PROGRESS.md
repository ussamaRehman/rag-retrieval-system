# PROGRESS.md

This file tracks **what is done**, **what is in progress**, and **what is next**.
It should be updated at the end of each meaningful work session.

---

## Current phase
Phase 0 — Project setup & contracts

---

## Completed
- Repo initialized with uv and pushed to GitHub 
- PRD.md written
- AGENTS.md written
- TECH_STACK.md written
- PROGRESS.md written

---

## In progress
- (none)

---

## Next steps (locked order)
1. Scaffold retrieval modules (`src/rag/`)
2. Implement deterministic chunking
3. Implement BM25 retrieval
4. Implement dense retrieval (MiniLM + FAISS)
5. Build index CLI
6. Print top-k retrieval results
7. Compute offline retrieval metrics

---

## Review gates
- [ ] Gate 1 — Retrieval
- [ ] Gate 2 — API + citations
- [ ] Gate 3 — Hardening + metrics
- [ ] Gate 4 — Eval report

---

## Notes / decisions log
- Tool calling and Text-to-SQL will be separate projects
- This project focuses on RAG + retrieval rigor