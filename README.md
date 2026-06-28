# Tentanator

AI-assisted exam grading. Grade a representative sample of responses by hand,
then let the graded examples drive in-context-learning (few-shot) suggestions for
the rest. Reads and writes Excel (`.xlsx`) and CSV natively, and can render a
results PDF per student.

The grading loop:

1. Import an exam file (student responses) and map its ID / input / output
   columns.
2. Sample a representative subset per question (`random` or embedding-based
   `maximin`).
3. Grade the sample by hand.
4. Once enough examples exist, the backend builds a few-shot prompt and asks the
   LLM to suggest grades for the remaining responses; you review and accept or
   override.
5. Define a grade scheme, compute results, and export Excel / Daisy / CSV / a
   results PDF.

## Architecture

Tentanator is a shared model with two interchangeable thin clients. Read
[ARCHITECTURE.md](ARCHITECTURE.md) for the full contract and design.

```
                 ┌─────────────────────────────┐
                 │      backend/  (Rust)        │
                 │  Axum HTTP API = the model   │
                 │  domain · sampling · ICL ·   │
                 │  persistence · LLM clients   │
                 └──────────────┬──────────────┘
                                │ HTTP/JSON
                 ┌──────────────┴──────────────┐
        ┌────────┴────────┐          ┌──────────┴──────────┐
        │  tui/  (Python) │          │  web/  (TanStack)   │
        │  Textual client │          │  React client       │
        └─────────────────┘          └─────────────────────┘
```

- **`backend/`** (Rust + Axum) is the single source of truth: column mapping,
  sampling, ICL prompts, LLM calls, the examination/scheme engine, persistence
  (an embedded Turso database), and all exports. No business logic lives in a
  client.
- **`tui/`** (Python + Textual) and **`web/`** (TanStack Start + React) are thin
  clients over the same HTTP API. They have parity: anything one can do, the
  other can too.
- **`results-renderer/`** (Python) renders the per-student results PDF (LaTeX /
  poppler / barcodes), too heavy to reimplement in Rust.

DTO types shared with the web client are generated from the Rust structs via
ts-rs; do not hand-edit `web/src/lib/generated/`. The TUI, web client and docs
are kept in lockstep with the backend contract, checked by
`scripts/check_api_parity.py` in CI.

## Quick start (Docker)

The whole stack runs from `docker-compose.yml`; you only need Docker.

```bash
# 1. (optional) API keys. Manual grading + random sampling work without them;
#    embeddings (maximin) and AI suggestions need them.
cp .env.example .env        # then set OPENAI_API_KEY / CEREBRAS_API_KEY

# 2. Drop exam files (.xlsx / .csv) into the data dir (the repo root)
mkdir -p exams && cp /path/to/your-exam.xlsx exams/
#    For results PDFs with cover pages, also: cp scanned-exam.pdf scans/

# 3. Start the backend + web UI
docker compose up -d --build

# 4. Use it
open http://localhost:3000          # web UI (grade, sample, export)
docker compose run --rm tui         # interactive terminal UI (Ctrl+Q to quit)

docker compose logs -f backend      # tail logs
docker compose down                 # stop everything
```

Graded exports land in `./graded_exams/`; the database is `./.tentanator.db`.
Files written by the backend container are root-owned, so `sudo chown -R "$USER" .`
if you need to edit them directly.

## Running locally (without Docker)

Three processes. Keys come from the repo-root `.env`.

```bash
# Backend (serves the model on :8787)
cd backend && TENTANATOR_DATA_DIR=.. cargo run        # or cargo run --release

# TUI (separate terminal)
cd tui && python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt
TENTANATOR_API=http://127.0.0.1:8787 python app.py

# Web (separate terminal)
cd web && npm install
VITE_API_BASE=http://127.0.0.1:8787 npm run dev       # http://localhost:3000
```

Backend tests: `cd backend && cargo test`.

## LLM providers

- **Embeddings**: OpenAI `text-embedding-3-large` (for `maximin` sampling).
- **Grading**: Cerebras `gpt-oss-120b` (OpenAI-compatible chat API), with a
  low-effort summary pass to condense the reasoning chain.

Keys come from `OPENAI_API_KEY` / `CEREBRAS_API_KEY`.

## Data & legacy import

Exam state (exams, sessions, questions, graded items, embedding caches, the
cross-exam graded pool) lives in the Turso database. Only exam inputs (`exams/`),
scans (`scans/`) and exported Excel (`graded_exams/`) stay as files. The data
directory defaults to the repo root; override with `TENTANATOR_DATA_DIR`.

Tentanator started as a single-file Python CLI; that code has been removed and
replaced by `backend/` (the pre-rearchitecture source is in git history). The
on-disk data the old app produced (`.tentanator_sessions/`, `workspaces/`) is
not migrated on startup but can be imported on demand from the home screen. See
[ARCHITECTURE.md](ARCHITECTURE.md) "Legacy import".

## CI/CD

`.github/workflows/ci.yml` runs `cargo test`, the contract-parity check and the
web typecheck on every push/PR, then on `master` and `v*` tags builds and
publishes `ghcr.io/<owner>/tentanator-{backend,web,tui}`.

## License

GNU Affero General Public License. See [LICENSE](LICENSE).

## Contributing

Contributions are welcome. The backend owns all domain logic; keep the TUI and
web client at parity and the contract tables in ARCHITECTURE.md in sync (run
`python scripts/check_api_parity.py`). DTOs are generated from the Rust structs,
so change the struct and run `cd backend && cargo test export_bindings` rather
than editing generated files.
