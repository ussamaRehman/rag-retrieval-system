# Shipping at Inference Speed 

**Here is the full article on this workflow:** https://steipete.me/posts/2025/shipping-at-inference-speed

This repo follows an **Inference-Speed workflow**:
optimize for **tight feedback loops**, **clear artifacts**, and **shippable systems**.

This document is a standing instruction for:
- human contributors,
- AI coding agents (Codex),
- future maintainers.

---

## Core principle

> Ship working, observable systems fast —  
> then iterate using metrics, not opinions.

We prioritize:
- correctness over cleverness,
- explicit code over abstractions,
- reproducibility over novelty.

---

## Project intent

- offline pipelines must be reproducible,
- online services must be observable and safe,
- tradeoffs must be documented.

---

## Non-negotiable workflow rules

1. **CLI first**
   - Every subsystem must work via CLI before it appears in an API.
   - No API endpoints until CLI output is correct end-to-end.
   - No UI until API behavior is locked.

2. **Tight loops**