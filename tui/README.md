# Tentanator TUI

A [Textual](https://textual.textualize.io/) terminal client for Tentanator. It
is a thin client: all grading, sampling, persistence and LLM work happens in the
Rust backend (`../backend`). See `../ARCHITECTURE.md` for the API contract.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Run

Start the backend first (defaults to `http://127.0.0.1:8787`), then:

```bash
# point at the backend if it isn't on the default port
export TENTANATOR_API=http://127.0.0.1:8787
python app.py
```

## Workflow

- **Sessions screen** lists existing sessions. Press `n` to start a new one.
- **New session** lets you choose an exam file, then the ID / input / output columns.
- **Grading** lets you select a question, optionally run `random` or `maximin`
  sampling to prioritise representative responses, then grade. Press `a` for an
  AI (in-context-learning) suggestion, edit if needed, and save. `e` exports the
  graded Excel file.
