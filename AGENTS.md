# AGENTS.md — Agentic Workflow Contract

This repo is designed to be **agent-friendly** and optimized for
human + AI pair-programming (Codex).

---

## Roles

### Human (Owner / Executor)
- Writes and reviews code (When needed)
- Runs commands and verifies outputs (When needed)
- Commits changes (When needed)
- Decides when to proceed to the next gate 

### Assistant (PM / Senior MLE Reviewer)
- Locks scope and MVP boundaries
- Makes architectural decisions
- Prevents scope creep
- Reviews milestones and artifacts
- Does NOT micromanage code

### Codex (Implementation Agent)
- Writes and reviews code
- Runs commands and verifies outputs
- Commits changes
- Implements code within agreed scope
- Scaffolds files and modules
- Writes tests
- Must follow repo rules and constraints

---

## Non-negotiable workflow rules

1. **CLI first**
   - No API until CLI pipeline works end-to-end.
   - No UI until API behavior is correct.

2. **Tight loops**
   - plan → implement → run → fix → document → commit

3. **Agent-friendly repo**
   - clear structure: `src/`, `tests/`, `docs/`
   - short docs added when subsystems change

4. **Boring, popular dependencies**
   - Prefer widely used, well-maintained libraries
   - Every non-obvious dependency must be justified

5. **Every change includes**
   - tests OR a clear manual verification command
   - updated docs if behavior changed

6. **No hand-waving**
   - If something fails, run commands, show errors, then fix

7. **Small commits**
   - One concern per commit
   - Clear, descriptive messages

---

## Where code is allowed

✅ Real code lives in:

src/
app/
rag/
tools/

---

## Review gates (must stop and review)

### Gate 1 — Retrieval
- BM25 + Dense retrieval works
- Index build CLI exists
- Offline metrics printed

### Gate 2 — API + citations
- `/predict` returns answer + citations
- Abstention works

### Gate 3 — Hardening + metrics
- Rate limits enforced
- Timeouts enforced
- `/metrics` works

### Gate 4 — Evaluation report
- `reports/eval.md`
- `reports/error_analysis.md`

Do not move past a gate without explicit approval.

---

## How to ask for help
Use one of these formats:
- **Scope check:** “Should we add X?”
- **Tradeoff:** “Option A vs B?”
- **Gate review:** “Gate N ready”
- **Constraint violation:** “Codex suggests Y — allowed?”