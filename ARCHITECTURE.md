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
  calls the LLM providers, persists sessions to an embedded [Turso](https://github.com/tursodatabase/turso)
  database (the Rust SQLite rewrite), exports graded Excel. No business logic
  lives in the clients.
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

## State & concurrency

All session state — sessions, questions, graded items, embedding caches and the
cross-session graded pool — lives in a single Turso database at
`<data_dir>/.tentanator.db` (`db.rs` defines the schema; vectors are stored as
f32 BLOBs). Only exam inputs (`exams/`) and exported Excel (`graded_exams/`)
stay as files. Data directory defaults to the repo root; override with
`TENTANATOR_DATA_DIR`.

Sessions are the unit of work and carry an optional `course` tag for grouping.
There is no workspace/project directory concept — the old `workspaces/<name>/`
folders only exist to be imported (see below).

**Concurrency**: the web app and TUI hit the store at the same time, so DB work
goes through one connection behind an async mutex, held only across DB calls and
released across LLM/file I/O. (Turso 0.4 still errors on concurrent
multi-connection writes; serializing the brief DB sections avoids that while
keeping LLM calls parallel. The lock can be relaxed as Turso's MVCC matures.)

## Backwards compatibility

On startup the backend imports any legacy on-disk data into the DB, idempotently
(existing sessions are skipped), so upgrading from the file-based or Python app
just works:

- **Sessions**: old `.tentanator_sessions/<name>.json` are read as-is — side-car
  `<name>.cache.json` preferred, caches embedded in the main file honoured,
  unknown legacy fields (e.g. a per-item `embedding`) ignored, and every field
  defaulted so partial files never fail. Active and `archive/` are both imported.
- **Oldest single-file session**: a `.tentanator_session.json` at the data root
  is imported and the original renamed `.backup`.
- **Workspaces → import**: the legacy workspace concept (swapping
  `workspaces/<name>/` folders in and out of the root) was a band-aid for the
  lack of a session model and is gone. Old workspace folders are *imported*
  one-time instead: `GET /api/legacy-workspaces` lists them and
  `POST /api/legacy-workspaces/{name}/import` loads their sessions (into the DB)
  and exams (into `exams/`), tagging the sessions with the workspace name as
  their `course` and merging the graded pool. Existing exams/sessions are never
  overwritten.
- **Graded pool**: `global_bank/graded_pool/<gq_id>.jsonl` is merged into the DB
  pool, so cross-session ICL examples carry over.

Covered by `backend` tests (`cargo test`): create/grade/load roundtrip, pool
hydration across sessions, BLOB vector roundtrip, and a workspace import that
tags the course.

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

## Quick start (Docker)

The whole stack runs from `docker-compose.yml`. You only need Docker.

```bash
# 1. (optional) API keys — manual grading + random sampling work without them;
#    embeddings (maximin) and AI suggestions need them.
cp .env.example .env        # then edit in your OPENAI_API_KEY / CEREBRAS_API_KEY

# 2. Drop exam files (.xlsx / .csv) into the data dir
mkdir -p data/exams
cp /path/to/your-exam.xlsx data/exams/

# 3. Start the backend + web UI
docker compose up -d --build

# 4. Use it
open http://localhost:3000          # web UI (grade, sample, export)
docker compose run --rm tui         # interactive terminal UI

# graded exports appear in ./data/graded_exams/ ; the DB is ./data/.tentanator.db

docker compose logs -f backend      # tail logs
docker compose down                 # stop everything
```

The interactive **TUI** is its own command (`docker compose run --rm tui`) rather
than part of `up`, because it needs a terminal. It starts the backend if it
isn't already running. Quit the TUI with `Ctrl+Q`.

Everything in `./data/` (exams, graded exports, the Turso DB) persists on the
host. Files written by the backend container are root-owned; `sudo chown -R
"$USER" data` if you need to edit them directly.

## Running locally (without Docker)

Three processes; keys come from the repo-root `.env` (the backend walks up to
find it).

```bash
# 1. Backend (serves the model on :8787). Point it at the repo root.
cd backend && TENTANATOR_DATA_DIR=.. cargo run        # or cargo run --release

# 2. TUI (separate terminal)
cd tui && python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
TENTANATOR_API=http://127.0.0.1:8787 python app.py

# 3. Web (separate terminal)
cd web && npm install
VITE_API_BASE=http://127.0.0.1:8787 npm run dev       # http://localhost:3000
```

Backend tests: `cd backend && cargo test`.
