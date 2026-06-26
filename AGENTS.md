# AGENTS.md

This file provides guidance to coding agents when working with code in this repository.

## Project Overview

Tentanator is an AI-powered exam grading system that combines manual grading with in-context learning (few-shot prompting). The workflow: grade sample responses manually → use graded examples as few-shot context → AI suggests grades for remaining responses via Cerebras inference. Works natively with Excel files (.xlsx) for both input and output.

## Current architecture (read ARCHITECTURE.md first)

The live system is a shared model with two interchangeable clients — see `ARCHITECTURE.md`:

- `backend/` (Rust + Axum) — the single source of truth. Owns all domain logic, persistence (Turso), sampling, ICL, LLM calls, exports. The HTTP API *is* the model.
- `tui/` (Python + Textual) — thin client over the HTTP API.
- `web/` (TanStack Start + React) — thin client over the same HTTP API.

The original single-file Python app (`tentanator.py` and friends) has been
removed; its logic now lives in `backend/`. The grade-parsing and sampling
ports note their provenance in code comments, and the on-disk data it produced
is still importable on demand (see ARCHITECTURE.md "Legacy import").

## Client parity (TUI ⇄ Web) — non-negotiable

**Core principle: the TUI must be able to do everything the Web GUI can — and vice versa.** The web is not the primary client with the TUI as a reduced companion; they are equals. Any user-facing capability shipped in `web/` is incomplete until the same task is doable in `tui/`, exactly as a backend route is incomplete until both clients cover it.

The TUI and Web never talk to each other; they are both thin clients over the backend. Parity has two layers, both required:

- **Contract parity** — both clients fully and correctly cover the backend contract. The backend is authoritative; no business logic lives in a client.
- **Capability parity** — any task a user can complete in the Web GUI can also be completed in the TUI. Covering a route in `api.py` is necessary but not sufficient: the capability must be reachable through the TUI's own UI, not just its API layer.

Rules when touching the contract:

- Adding or changing a backend route (`backend/src/routes.rs`) is incomplete until **both** `web/src/lib/api.ts` and `tui/api.py` are updated and the `ARCHITECTURE.md` contract tables list it, or the omission is added to the explicit parity allowlist with a reason.
- DTO shapes are generated from the Rust structs via ts-rs — do not hand-edit `web/src/lib/generated/`; change the struct and run `cd backend && cargo test export_bindings`. The TUI consumes the same JSON; keep its types in sync.
- A change is "done" only when the backend, both client API layers, and the `ARCHITECTURE.md` contract tables all agree.
- A consumer may intentionally omit an endpoint (e.g. browser-only blob downloads). Record such exclusions in `ALLOW_MISSING` rather than leaving them as silent gaps.

Enforcement: `scripts/check_api_parity.py` extracts the `(method, path)` surface from `routes.rs`, both clients, and the doc tables, and fails if any drifts. It runs in CI (`py-check`); run it locally after any contract change. It is the cheapest layer — a backend-emitted OpenAPI spec with generated clients, then conformance tests driven from it, are the stronger follow-ons. Do not let the TUI or the docs fall behind the Web client. Note the script checks **contract** parity only; **capability** parity (every GUI feature having a TUI equivalent) is not machine-checked, so it is on you when shipping any user-facing feature.

Capability parity status: the TUI now covers the full Web feature set — scheme + question-config editor, grade import with conflict resolution, results table, results PDF with cover pages, per-question settings, Daisy/CSV exports, course editing, named sessions, ungrade, archive/unarchive, workspace import, and column auto-detect. To keep both clients thin, the readable scheme DSL parse/emit (`POST /api/scheme/parse`, `/api/scheme/emit`) and the `Response N`/`Points N` column heuristic (`GET /api/exam-files/{file}/detect`) live in the backend, not in either client. Remaining backend routes deliberately surfaced in neither UI: exam and session deletion (destructive; archive covers the common case) — add to both clients together if/when needed.

## Development & commands

ARCHITECTURE.md is authoritative for the running system: the HTTP API contract,
the examination engine, exams/sessions, and how to run the stack (Docker quick
start and the three-process local setup). Read it first.

```bash
# Backend (the model) — tests and DTO bindings
cd backend && cargo test                     # unit + roundtrip tests
cd backend && cargo test export_bindings     # regen web/src/lib/generated/ from Rust structs
cd backend && cargo clippy --all-targets     # lint

# Contract parity (runs in CI as py-check); run after any contract change
python scripts/check_api_parity.py

# TUI client (Python + Textual)
cd tui && pip install -r requirements.txt && python app.py

# Web client (TanStack Start + React)
cd web && npm install && npm run dev

# Results-PDF renderer (Python, LaTeX/poppler-heavy)
cd results-renderer && pip install -r requirements.txt
```

## Code Style

- **Rust (`backend/`)**: keep `cargo clippy` clean; domain logic stays here, not
  in clients. DTOs that cross the wire derive ts-rs bindings — change the struct
  and run `cargo test export_bindings`, never hand-edit `web/src/lib/generated/`.
- **Python (`tui/`, `results-renderer/`, `scripts/`)**: type hints on all
  functions, dataclasses for data structures, async/await for I/O. Keep pylint
  clean (100 char line length); run pylint in the background so it does not flood
  the terminal, as the last step before finishing.
- **TypeScript (`web/`)**: consume the generated DTOs; gate on `tsc`.
- All files end with a final newline.
