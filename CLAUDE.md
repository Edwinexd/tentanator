# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Tentanator is a Python-based AI-powered exam grading system that combines manual grading with in-context learning (few-shot prompting). The workflow: grade sample responses manually → use graded examples as few-shot context → AI suggests grades for remaining responses via Cerebras inference. Works natively with Excel files (.xlsx) for both input and output.

## Development Setup

```bash
# Create and activate virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Setup API keys
echo "OPENAI_API_KEY=your-openai-key-here" > .env   # For embeddings only
echo "CEREBRAS_API_KEY=your-cerebras-key-here" >> .env  # For chat/grading inference
```

## Common Commands

```bash
# Run main grading application
python tentanator.py

# Lint code
pylint tentanator.py openai_trainer.py sampling.py embeddings.py combine_moodle_dumps.py workspace.py

# Check syntax
python -m py_compile tentanator.py
```

## Architecture

### Core Modules

**tentanator.py** (main application)
- Main grading interface and orchestration
- Data classes: `GradedItem`, `QuestionGrades`, `GradingSession`
- Key functions:
  - `grade_questions()`: Interactive grading CLI
  - `export_to_excel()`: Export graded data directly to .xlsx
  - `get_ai_grade_suggestion()`: Get AI suggestions via in-context learning (few-shot)
  - `_build_icl_messages()`: Build few-shot prompt from graded examples
  - `read_exam_data()`: Read .xlsx or .csv files as list of dicts
  - `list_exam_files()`: List .xlsx and .csv files in exams/ directory
- Uses Cerebras (qwen-3-235b) for chat completions via OpenAI-compatible API
- Uses OpenAI only for text embeddings (sampling algorithms)
- Session persistence in `.tentanator_sessions/` directory
- Embeddings caching for sampling algorithms
- Async/await architecture for API calls

**openai_trainer.py** (legacy, no longer used in main workflow)
- OpenAI fine-tuning integration (kept for reference/manual use)
- Not called from tentanator.py — replaced by in-context learning

**sampling.py**
- Sample selection algorithms for representative grading
- Implements: KMeans (auto/fixed k), random sampling, maximin diversity sampling, GPT-based quality sorting
- `SamplingAlgorithm` type: `"kmeans_auto" | "kmeans_fixed" | "random" | "maximin" | "gptsort" | "iforest_gmm"`
- Uses scikit-learn with silhouette scoring for optimal k selection
- GPTSort uses Cerebras to sort responses by quality without embeddings

**embeddings.py**
- OpenAI text embeddings wrapper
- Uses `text-embedding-3-large` model
- Async API calls for performance

**combine_moodle_dumps.py**
- Merges Moodle grades + responses xlsx dumps into a single exam xlsx
- Input: files in `exams_in_raw/`; Output: `exams/{base_name}.xlsx`

**workspace.py**
- Workspace management (load/unload/create/delete) — moves per-workspace directories between root and `workspaces/<name>/`

### Data Flow

1. **Input**: Place Excel files (.xlsx) in `exams/` directory
2. **Manual Grading**: Excel → tentanator.py → session saved to `.tentanator_sessions/`
3. **AI Grading**: Graded examples used as few-shot context → `get_ai_grade_suggestion()` → suggested grades via Cerebras
4. **Export**: Completed session → `export_to_excel()` → `graded_exams/*.xlsx`

### Key Configuration

In `tentanator.py`:
- `GRADING_THRESHOLD = 5`: Minimum manual grades required before AI suggestions
- `MIN_ICL_EXAMPLES = 5`: Minimum graded items before in-context learning kicks in
- `MAX_ICL_EXAMPLES = 25`: Maximum few-shot examples included in prompt
- `CEREBRAS_MODEL`: Reasoning model used for grading inference (default: `gpt-oss-120b`, `reasoning_effort="high"`)
- `CEREBRAS_SUMMARY_MODEL`: Lightweight model used to condense the reasoning chain (default: `llama3.1-8b`)
- `NUM_REPRESENTATIVE_SAMPLES = 5`: Number of samples for selection algorithms
- `SAMPLING_ALGORITHM`: Choose from `"kmeans_auto"`, `"kmeans_fixed"`, `"random"`, `"maximin"`, `"gptsort"`, `"iforest_gmm"`
- `BASE_SYSTEM_PROMPT`: Template for AI grading prompts

### Directory Structure

- `exams/`: Input Excel (.xlsx) or CSV files with student responses
- `exams_in_raw/`: Raw Moodle grade/response dumps (input to `combine_moodle_dumps.py`)
- `graded_exams/`: Output Excel (.xlsx) files with completed grades
- `.tentanator_sessions/`: Saved grading sessions (JSON)
- `global_bank/`: Downloaded question bank CSV files (internal)
- `backups/`: Archived sessions and models
- `workspaces/<name>/`: Per-workspace snapshots of the above dirs (managed by `workspace.py`)

### Session Management

Sessions are automatically saved after each grade to `.tentanator_sessions/{name}_{timestamp}.json`. Sessions track:
- ID/input/output column mappings
- All graded items with timestamps
- Embeddings cache for sampling algorithms
- Exam question text for each output column

## Code Style

- Use typed Python with type hints on all functions
- Use dataclasses (with `@dataclass` decorator) for data structures
- Use attr.s if available (currently not in use)
- All files must end with a final newline
- Follow pylint conventions (100 char line length)
- Async/await for API calls (Cerebras for chat, OpenAI for embeddings)
- Always run and handle pylint checks before finishing edits, this should always be your last task in the todo. Always run pylint in the background to not blow up the users terminal.
