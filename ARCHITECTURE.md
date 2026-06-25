# Tentanator architecture (model/controller split)

Tentanator is being re-shaped from a single Python CLI into a shared model with
two interchangeable controllers (clients):

```
                 ┌─────────────────────────────┐
                 │      backend/  (Rust)        │
                 │  Axum HTTP API = the model   │
                 │  domain · sampling · ICL ·   │
                 │  persistence · LLM clients   │
                 └──────────────┬──────────────┘
                                │ HTTP/JSON
                 ┌──────────────┴──────────────┐
                 │                              │
        ┌────────┴────────┐          ┌──────────┴──────────┐
        │  tui/  (Python) │          │  web/  (TanStack)   │
        │  Textual client │          │  React client       │
        └─────────────────┘          └─────────────────────┘
```

- **backend/** — Rust + Axum. Owns all domain logic and is the single source of
  truth: reads exam files, runs sampling, builds in-context-learning prompts,
  calls the LLM providers, persists sessions, exports graded Excel. No business
  logic lives in the clients.
- **tui/** — Python [Textual](https://textual.textualize.io/) app. Thin client
  over the HTTP API.
- **web/** — TanStack Start React app. Thin client over the same HTTP API.

The legacy Python files (`tentanator.py`, `sampling.py`, etc.) remain at the
repo root as the reference implementation while the port proceeds.

## Scope changes from the legacy app

- **Fine-tuning is dropped.** ICL (few-shot prompting) is the only grading path.
  `openai_trainer.py` and the JSONL training export are not ported.
- **Sampling is reduced to two embedding-based strategies**: `random` and
  `maximin` (max-spread / farthest-first). KMeans, IsolationForest+GMM, PCA and
  GPTSort are not ported.

## LLM providers

- **Embeddings**: OpenAI `text-embedding-3-large` (for `maximin` sampling).
- **Grading**: Cerebras `gpt-oss-120b` (OpenAI-compatible chat API,
  `reasoning_effort=high`), plus a low-effort summary pass to condense the
  reasoning chain. Keys come from `OPENAI_API_KEY` / `CEREBRAS_API_KEY`.

## On-disk state (unchanged formats, owned by backend)

Relative to the backend's data directory (default = repo root, override with
`TENTANATOR_DATA_DIR`):

- `exams/` — input `.xlsx`/`.csv`.
- `graded_exams/` — exported `.xlsx`.
- `.tentanator_sessions/` — `<name>.json` session + `<name>.cache.json` embeddings.
- `.tentanator_sessions/archive/` — archived sessions.
- `global_bank/graded_pool/<gq_id>.jsonl` — cross-session graded-example pool.

Sessions are the unit of work and carry an optional `course` tag for grouping.
There is no workspace/project directory concept — the old `workspaces/<name>/`
folders only exist to be imported (see below).

## Backwards compatibility

The backend reads the legacy on-disk formats unchanged — existing data imports
with no conversion step:

- **Sessions**: old `.tentanator_sessions/<name>.json` load as-is. Caches in a
  side-car `<name>.cache.json` are preferred, but caches embedded in the main
  file (the older layout) are still honoured. Unknown legacy fields (e.g. a
  per-item `embedding`) are ignored, and every field has a default so partial
  files never fail to load.
- **Oldest single-file session**: a `.tentanator_session.json` at the data root
  is migrated into `.tentanator_sessions/<name>.json` on startup (and the
  original renamed `.backup`), mirroring the legacy Python migration.
- **Workspaces → import**: the legacy workspace concept (swapping
  `workspaces/<name>/` folders in and out of the root) was a band-aid for the
  lack of a session model and is gone. Old workspace folders are instead
  *imported* one-time: `GET /api/legacy-workspaces` lists them and
  `POST /api/legacy-workspaces/{name}/import` copies their sessions, caches and
  exams into the flat store, tagging the imported sessions with the workspace
  name as their `course` (and merging the graded pool). Existing exams/sessions
  are never overwritten.
- **Graded pool**: `global_bank/graded_pool/<gq_id>.jsonl` is read and appended
  in the same format, so cross-session ICL examples carry over.

This is covered by `backend` tests (`cargo test`): a legacy session with
embedded caches + extra fields, the single-file migration, and a workspace
import that tags the course.

## HTTP API contract

JSON in/out. Base path `/api`. Errors return `{ "error": "..." }` with a 4xx/5xx
status.

### Health, legacy import & exams
| Method | Path | Body | Returns |
|---|---|---|---|
| GET | `/api/health` | — | `{ "status": "ok" }` |
| GET | `/api/legacy-workspaces` | — | `[{ name, sessions }]` (importable) |
| POST | `/api/legacy-workspaces/{name}/import` | — | `{ imported_sessions[], imported_exams, skipped_exams }` |
| GET | `/api/exams` | — | `string[]` filenames in `exams/` |
| GET | `/api/exams/{file}/columns` | — | `string[]` header names |
| GET | `/api/exams/{file}/rows` | — | `{ rows: object[] }` (cells as strings) |

### Sessions
| Method | Path | Body | Returns |
|---|---|---|---|
| GET | `/api/sessions?archived=false&course=` | — | `SessionSummary[]` (optional `course` filter) |
| POST | `/api/sessions` | `CreateSession` | `Session` |
| GET | `/api/sessions/{name}` | — | `Session` |
| PUT | `/api/sessions/{name}` | `{ course? }` | `Session` |
| DELETE | `/api/sessions/{name}` | — | `204` |
| POST | `/api/sessions/{name}/archive` | — | `204` |
| POST | `/api/sessions/{name}/unarchive` | — | `204` |

### Grading workflow (per output column / question)
| Method | Path | Body | Returns |
|---|---|---|---|
| PUT | `/api/sessions/{name}/questions/{col}` | `QuestionMeta` | `Question` |
| POST | `/api/sessions/{name}/questions/{col}/sampling` | `{ algorithm, n_samples }` | `SamplingResult` |
| POST | `/api/sessions/{name}/questions/{col}/grade` | `{ row_id, grade }` | `Question` |
| DELETE | `/api/sessions/{name}/questions/{col}/grade/{row_id}` | — | `Question` |
| POST | `/api/sessions/{name}/questions/{col}/suggest` | `{ row_id }` | `AIGradeSuggestion` |
| POST | `/api/sessions/{name}/export` | — | `{ path }` |

`CreateSession = { csv_file, id_columns[], input_columns[], output_columns[], name?, course? }`
`QuestionMeta = { exam_question?, sample_answer?, global_question_id? }`
`algorithm ∈ { "random", "maximin" }`.

Grades accept the legacy signed-sum syntax (`"2+1.5+2.5"`, `"7.5"`, `"2+2.5-0.5"`);
the backend validates and stores both the raw expression (for ICL context) and
the evaluated numeric total (for export).

## Running locally

Three processes. Keys (`OPENAI_API_KEY`, `CEREBRAS_API_KEY`) come from the
repo-root `.env`; the backend walks up to find it.

```bash
# 1. Backend (serves the model on :8787). Point it at the repo root so it sees
#    exams/, .tentanator_sessions/, etc.
cd backend
TENTANATOR_DATA_DIR=.. cargo run            # or: cargo run --release
#    embeddings/AI need the keys; random sampling + manual grading work without.

# 2. TUI (separate terminal)
cd tui && python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
TENTANATOR_API=http://127.0.0.1:8787 python app.py

# 3. Web (separate terminal)
cd web && npm install
VITE_API_BASE=http://127.0.0.1:8787 npm run dev   # http://localhost:3000
```

Backend tests: `cd backend && cargo test`.
