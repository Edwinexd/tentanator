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
  calls the LLM providers, persists exams and grading sessions to an embedded
  [Turso](https://github.com/tursodatabase/turso) database (the Rust SQLite
  rewrite), exports graded Excel. No business logic lives in the clients.
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

## Exams & sessions

An **exam** is the central object and owns everything durable: the response file
(`exam_file`), its column mapping (id / input / output), per-question config, the
grade scheme, and the single canonical grade set. Exams carry an optional
`course` tag for grouping and an `archived` flag.

A **session** is a lightweight named grading pass *under* an exam (table keyed by
`(exam, name)`). Every exam starts with a `default` session. Grades recorded in a
session land in the exam's one grade set, tagged with the session name (the
`session` column on `graded_items`), so sessions group/track work — they are not
separate grade sets. `GET /api/exams/{exam}/sessions` reports each session's
`graded_count`.

## State & concurrency

All exam state — exams, sessions, questions, graded items, embedding caches and
the cross-exam graded pool — lives in a single Turso database at
`<data_dir>/.tentanator.db` (`db.rs` defines the schema; vectors are stored as
f32 BLOBs). Only exam inputs (`exams/`) and exported Excel (`graded_exams/`)
stay as files. Data directory defaults to the repo root; override with
`TENTANATOR_DATA_DIR`.

**Concurrency**: the web app and TUI hit the store at the same time, so DB work
goes through one connection behind an async mutex, held only across DB calls and
released across LLM/file I/O. (Turso 0.4 still errors on concurrent
multi-connection writes; serializing the brief DB sections avoids that while
keeping LLM calls parallel. The lock can be relaxed as Turso's MVCC matures.)

## Legacy import (on demand)

The backend does **not** import anything on startup and does not migrate old DB
rows — start fresh. But the old Python-app on-disk data can be pulled into the
new format on request, so in-progress exams carry over. Each old session becomes
one exam (`session_name`->`name`, `csv_file`->`exam_file`) with a `default`
grading session; its graded items land in that session, caches and the graded
pool are merged. Existing exams/files are never overwritten (name collisions are
suffixed).

- **Loose sessions**: `GET /api/legacy-sessions` reports how many old
  `.tentanator_sessions/*.json` (active + `archive/`) sit at the data root;
  `POST /api/legacy-sessions/import` imports them (archived ones stay archived),
  preferring the side-car `<stem>.cache.json`, honouring embedded caches, and
  ignoring unknown legacy fields so partial files never fail.
- **Workspaces**: `GET /api/legacy-workspaces` lists leftover `workspaces/<name>/`
  folders; `POST /api/legacy-workspaces/{name}/import` imports their sessions as
  exams tagged `course = <name>`, copies their exam files into `exams/`, and
  merges their graded pool.

Covered by `backend` tests (`cargo test`): create/grade/load roundtrip, session
graded-count tracking, pool hydration across exams, BLOB vector roundtrip, and a
workspace import that creates an exam + default session and tags the course.

## HTTP API contract

JSON in/out. Base path `/api`. Errors return `{ "error": "..." }` with a 4xx/5xx
status.

### Health, legacy import & exam files
| Method | Path | Body | Returns |
|---|---|---|---|
| GET | `/api/health` | — | `{ "status": "ok" }` |
| GET | `/api/legacy-workspaces` | — | `[{ name, exams }]` (importable) |
| POST | `/api/legacy-workspaces/{name}/import` | — | `{ imported_exams[], imported_files, skipped_files }` |
| GET | `/api/legacy-sessions` | — | `{ count }` (loose `.tentanator_sessions/`) |
| POST | `/api/legacy-sessions/import` | — | `{ imported_exams[] }` |
| GET | `/api/exam-files` | — | `string[]` filenames in `exams/` |
| GET | `/api/scans` | — | `string[]` scanned PDF filenames in `scans/` |
| PUT | `/api/files/{kind}/{filename}` | raw bytes | `{ filename }` (`kind` ∈ `exams`/`scans`) |
| GET | `/api/exam-files/{file}/columns` | — | `string[]` header names |
| GET | `/api/exam-files/{file}/rows` | — | `{ rows: object[] }` (cells as strings) |

### Exams
| Method | Path | Body | Returns |
|---|---|---|---|
| GET | `/api/exams?archived=false&course=` | — | `ExamSummary[]` (optional `course` filter) |
| POST | `/api/exams` | `CreateExam` | `Exam` |
| GET | `/api/exams/{name}` | — | `Exam` |
| PUT | `/api/exams/{name}` | `{ course? }` | `Exam` |
| DELETE | `/api/exams/{name}` | — | `204` |
| POST | `/api/exams/{name}/archive` | — | `204` |
| POST | `/api/exams/{name}/unarchive` | — | `204` |
| PUT | `/api/exams/{name}/columns` | `{ id_columns[]?, input_columns[], output_columns[] }` | `Exam` |

### Sessions (grading passes under an exam)
| Method | Path | Body | Returns |
|---|---|---|---|
| GET | `/api/exams/{name}/sessions` | — | `SessionSummary[]` (with `graded_count`) |
| POST | `/api/exams/{name}/sessions` | `{ name? }` | `Session` |
| DELETE | `/api/exams/{name}/sessions/{session}` | — | `204` (not `default`) |

### Grading workflow (per output column / question)
| Method | Path | Body | Returns |
|---|---|---|---|
| PUT | `/api/exams/{name}/questions/{col}` | `QuestionMeta` | `Question` |
| POST | `/api/exams/{name}/questions/{col}/sampling` | `{ algorithm, n_samples }` | `SamplingResult` |
| POST | `/api/exams/{name}/questions/{col}/grade` | `{ row_id, grade, session? }` | `Question` |
| DELETE | `/api/exams/{name}/questions/{col}/grade/{row_id}` | — | `Question` |
| POST | `/api/exams/{name}/questions/{col}/suggest` | `{ row_id }` | `AIGradeSuggestion` |
| GET | `/api/exams/{name}/questions/{col}/status` | — | `QuestionStatus` (graded counts, ICL readiness) |
| POST | `/api/exams/{name}/export` | — | graded `.xlsx` file (attachment, not JSON) |

`CreateExam = { exam_file, id_columns[], input_columns[], output_columns[], name?, course? }`
`QuestionMeta = { exam_question?, sample_answer?, global_question_id? }`
`grade.session` defaults to `"default"`; the session is created if absent.
`algorithm ∈ { "random", "maximin" }`.

Grades accept the legacy signed-sum syntax (`"2+1.5+2.5"`, `"7.5"`, `"2+2.5-0.5"`);
the backend validates and stores both the raw expression (for ICL context) and
the evaluated numeric total (for export).

### Scheme, results, import & engine exports

| Method | Path | Body | Returns |
|---|---|---|---|
| PUT | `/api/exams/{name}/scheme` | `GradeScheme` | `204` |
| PUT | `/api/exams/{name}/questions-config` | `QuestionConfigUpdate[]` | `Exam` |
| GET | `/api/exams/{name}/results` | — | `ResultsResponse` |
| POST | `/api/exams/{name}/results` | `GradeScheme` | `ResultsResponse` (live preview of an unsaved scheme) |
| POST | `/api/exams/{name}/import/preview` | `ImportReq` | `ImportSummary` |
| POST | `/api/exams/{name}/import/apply` | `ImportReq` | `ImportSummary` |
| GET | `/api/exams/{name}/conflicts` | — | `GradeConflict[]` |
| POST | `/api/exams/{name}/conflicts/resolve` | `ResolveReq` | `204` |
| GET | `/api/exams/{name}/render-data` | — | `RenderData` (per-student renderer contract) |
| GET | `/api/exams/{name}/scans` | — | `ScanMatch[]` (scans eligible for cover pages) |
| POST | `/api/exams/{name}/export/daisy` | — | Daisy `id,grade` `.xlsx` (attachment) |
| POST | `/api/exams/{name}/export/csv` | — | per-question `.csv` (attachment) |
| POST | `/api/exams/{name}/export/results-pdf` | `{ scanned_pdf? }` | renderer JSON (proxied; PDF lands in `graded_exams/`) |
| GET | `/api/graded/{filename}` | — | streams a file from `graded_exams/` |

`ImportReq = { file, id_column, mappings: [{ column, output_col }], label? }`
`ResolveReq = { output_col, row_id, choose }` (`choose ∈ existing | incoming`).

## Examination engine (v2)

An examination is the central object: a roster + responses (the exam file), per-
question config, grades from any source, a grade scheme, and computed results.
Nothing about a specific exam's shape is hardcoded — sections, question counts,
types and the grade formula are all per-examination config (`scheme.rs`).

- **Question config** (`PUT /api/exams/{name}/questions-config`): per question
  `var` (expression identifier), `group` (section tag), `qtype`, `max_points`,
  `position`, optional `estimate` expression.
- **Grade scheme** (`PUT /api/exams/{name}/scheme`): tunable `constants`,
  named `vars` (expressions over question vars / prior vars / `groupsum("tag")`),
  and ordered guarded `rules` (`when <bool> -> <grade>`, first match wins).
  Expressions run via `evalexpr`. Reproduces PVT's gated ECTS rules; collapses to
  trivial for simple exams (tested against `reference/pvt/data/grades.json`).
- **Results** (`GET …/results`, `POST …/results` for live preview with an
  unsaved scheme): `StudentResult[]` (id, grade, total, estimated, complete) +
  distribution + unresolved-conflict count.
- **Import & merge** (`POST …/import/preview`, `…/import/apply`): map a graded
  sheet's columns to questions; non-conflicting cells merge with `source`
  provenance, disagreements become `grade_conflicts`
  (`GET …/conflicts`, `POST …/conflicts/resolve`).
- **Exports**: `…/export` (full graded xlsx), `…/export/daisy` (Daisy `id,grade`
  xlsx), `…/export/csv` (per-question), `…/export/results-pdf` (proxies the
  renderer). `GET …/render-data` is the per-student contract the renderer reads.

### Results-PDF renderer (`results-renderer/`)

A separate Python service (LaTeX/poppler/zxing/pikepdf — too heavy to reimplement
in Rust). `POST /render { exam, scanned_pdf? }` fetches `render-data` from the
backend, renders a LaTeX answer sheet per student (responses, marks, grade, a
Code128 id barcode), and — when a scanned exam PDF is provided in `data/scans/` —
prepends each student's original cover page (matched by PDF417 barcode), then
concatenates to `graded_exams/<exam>_results.pdf`. The backend reaches it via
`RENDERER_URL`. Generalized from `reference/pvt/scripts/01-03`.

## Quick start (Docker)

The whole stack runs from `docker-compose.yml`. You only need Docker.

```bash
# 1. (optional) API keys — manual grading + random sampling work without them;
#    embeddings (maximin) and AI suggestions need them.
cp .env.example .env        # then edit in your OPENAI_API_KEY / CEREBRAS_API_KEY

# 2. Drop exam files (.xlsx / .csv) into the data dir (the repo root)
mkdir -p exams
cp /path/to/your-exam.xlsx exams/
#    For results PDFs with cover pages, also: cp scanned-exam.pdf scans/

# 3. Start the backend + web UI
docker compose up -d --build

# 4. Use it
open http://localhost:3000          # web UI (grade, sample, export)
docker compose run --rm tui         # interactive terminal UI

# graded exports appear in ./graded_exams/ ; the DB is ./.tentanator.db
# (the repo root is mounted into the backend at /workspace and used as the data
#  dir; legacy .tentanator_sessions/, workspaces/ and global_bank/ at the root are
#  importable on demand from the home screen - not on startup)

docker compose logs -f backend      # tail logs
docker compose down                 # stop everything
```

The interactive **TUI** is its own command (`docker compose run --rm tui`) rather
than part of `up`, because it needs a terminal. It starts the backend if it
isn't already running. Quit the TUI with `Ctrl+Q`.

Everything at the repo root (exams, graded exports, the Turso DB) persists on the
host. Files written by the backend container are root-owned; `sudo chown -R
"$USER" .` if you need to edit them directly.

**CI/CD**: `.github/workflows/ci.yml` runs `cargo test` and the web typecheck on
every push/PR, then on `master` and `v*` tags builds and publishes the three
images to GHCR — `ghcr.io/<owner>/tentanator-{backend,web,tui}` (tagged
`latest`, the commit SHA, and the semver on tags). To run the published images
instead of building locally, point the `image:` keys in `docker-compose.yml` at
the GHCR tags and drop the `build:` lines.

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
