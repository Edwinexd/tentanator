# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Tentanator is a Python-based AI-powered exam grading system that combines manual grading with OpenAI fine-tuning. The workflow: grade sample responses manually → train custom AI models → use AI to suggest grades for remaining responses.

## Development Setup

```bash
# Create and activate virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Setup API key
echo "OPENAI_API_KEY=your-api-key-here" > .env
```

## Common Commands

```bash
# Run main grading application
python tentanator.py

# Train AI models from graded data
python openai_trainer.py

# Export graded CSVs to Excel format
python make_excel.py

# Lint code
pylint tentanator.py openai_trainer.py sampling.py embeddings.py make_excel.py

# Check syntax
python -m py_compile tentanator.py
```

## Architecture

### Core Modules

**tentanator.py** (main application, ~1250 lines)
- Main grading interface and orchestration
- Data classes: `GradedItem`, `QuestionGrades`, `GradingSession`
- Key functions:
  - `grade_questions()`: Interactive grading CLI (lines ~522-900)
  - `export_to_csv()`: Export graded data
  - `export_to_jsonl()`: Export training data for fine-tuning
  - `get_ai_grade_suggestion()`: Get AI suggestions from trained models
- Session persistence in `.tentanator_sessions/` directory
- Embeddings caching for sampling algorithms
- Async/await architecture for OpenAI API calls

**openai_trainer.py**
- OpenAI fine-tuning integration
- Data classes: `FineTuningConfig`, `TrainingFile`, `FineTuningJob`, `ModelRegistry`
- Handles JSONL validation, file upload, job creation, and monitoring
- Maintains `models.json` registry of trained models

**sampling.py**
- Sample selection algorithms for representative grading
- Implements: KMeans (auto/fixed k), random sampling, maximin diversity sampling, GPT-based quality sorting
- `SamplingAlgorithm` type: `"kmeans_auto" | "kmeans_fixed" | "random" | "maximin" | "gptsort"`
- Uses scikit-learn with silhouette scoring for optimal k selection
- GPTSort uses ChatGPT to sort responses by quality without embeddings

**embeddings.py**
- OpenAI text embeddings wrapper
- Uses `text-embedding-3-large` model
- Async API calls for performance

**make_excel.py**
- Batch converts CSV files from `graded_exams/` to Excel format
- Auto-adjusts column widths for readability

### Data Flow

1. **Manual Grading**: CSV → tentanator.py → session saved to `.tentanator_sessions/`
2. **Training Data**: Session → export_to_jsonl() → `training_data/*.jsonl`
3. **Model Training**: JSONL → openai_trainer.py → fine-tuned model → `models.json`
4. **AI Grading**: Ungraded responses → get_ai_grade_suggestion() → suggested grades
5. **Export**: Completed session → export_to_csv() → `graded_exams/*.csv` → make_excel.py → Excel

### Key Configuration

In `tentanator.py`:
- `GRADING_THRESHOLD = 25`: Minimum manual grades required before AI training
- `NUM_REPRESENTATIVE_SAMPLES = 25`: Number of samples for selection algorithms
- `SAMPLING_ALGORITHM`: Choose from `"kmeans_auto"`, `"kmeans_fixed"`, `"random"`, `"maximin"`, `"gptsort"`
- `BASE_SYSTEM_PROMPT`: Template for AI grading prompts

### Directory Structure

- `exams/`: Input CSV files with student responses
- `graded_exams/`: Output CSV files with completed grades
- `graded_exams_out/`: Excel exports
- `training_data/`: JSONL files for OpenAI fine-tuning
- `.tentanator_sessions/`: Saved grading sessions (JSON)
- `models.json`: Registry of trained AI models
- `backups/`: Archived sessions and models

### Session Management

Sessions are automatically saved after each grade to `.tentanator_sessions/{csv_name}_{timestamp}.json`. Sessions track:
- ID/input/output column mappings
- All graded items with timestamps
- Embeddings cache for sampling algorithms
- Exam question text for each output column

Model registry (`models.json`) maps normalized question names to OpenAI fine-tuned model IDs.

## Code Style

- Use typed Python with type hints on all functions
- Use dataclasses (with `@dataclass` decorator) for data structures
- Use attr.s if available (currently not in use)
- All files must end with a final newline
- Follow pylint conventions (100 char line length)
- Async/await for OpenAI API calls
- Always run and handle pylint checks before finishing edits, this should always be your last task in the todo. Always run pylint in the background to not blow up the users terminal.
