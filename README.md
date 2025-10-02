# Tentanator - AI-Powered CSV Exam Grading Assistant

Tentanator is a Python-based intelligent grading system that combines manual grading with OpenAI fine-tuning to streamline exam grading workflows. It enables educators to grade a sample of responses manually, train custom AI models on that data, and automatically grade the remaining responses with AI assistance.

## Features

- **Interactive CSV Grading Interface**: Grade exam responses directly from CSV files with a user-friendly CLI
- **AI-Assisted Grading**: After grading 50 sample responses, use fine-tuned GPT models to suggest grades for remaining responses
- **OpenAI Fine-Tuning Integration**: Automatically export graded data to JSONL format and train custom grading models
- **Session Persistence**: Resume grading sessions at any time with automatic session saving
- **Smart Auto-Grading**: Automatically assigns grade "0" to blank or dash responses
- **Batch Export**: Export fully graded CSV files with all grades filled in
- **Model Registry**: Track and manage all fine-tuned models for different questions

## Installation

### Prerequisites

- Python 3.8 or higher
- OpenAI API key (for AI-assisted grading features)

### Setup

1. Clone the repository:
```bash
git clone <repository-url>
cd tentanator
```

2. Create a virtual environment:
```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Set up your OpenAI API key:
Create a `.env` file in the project root:
```env
OPENAI_API_KEY=your-api-key-here
```

## Usage

### Main Grading Workflow

1. **Start the grading application**:
```bash
python tentanator.py
```

2. **Select your CSV file**:
   - Place exam CSV files in the `exams/` directory
   - The program will list available files for selection

3. **Configure columns**:
   - Select ID columns (e.g., student ID, name)
   - Select input columns (student responses to grade)
   - Select output columns (where grades will be stored)

4. **Grade responses**:
   - Grade at least 50 valid responses per question
   - Use commands:
     - `q` - quit and save session
     - `s` - skip current response
     - `b` - go back to previous response
     - `[Enter]` - accept AI suggestion (if available)
     - Type grade value directly

5. **Export training data** (optional):
   - After grading 50 samples, export to JSONL for model training
   - Training data is saved in `training_data/` directory

6. **AI-assisted grading** (if models available):
   - After manual grading, use trained models to grade remaining responses
   - Review and modify AI suggestions as needed

### Training Custom Models

1. **Run the training module**:
```bash
python openai_trainer.py
```

2. **Select JSONL file to train**:
   - Choose from available training data files
   - Each question gets its own fine-tuned model

3. **Monitor training**:
   - Training typically takes 10-30 minutes
   - Models are automatically registered when complete

### Project Structure

```
tentanator/
├── tentanator.py              # Main grading application
├── openai_trainer.py          # OpenAI fine-tuning module
├── requirements.txt           # Python dependencies
├── .env                       # API keys (not in version control)
├── exams/                     # Input CSV files
│   └── *.csv
├── graded_exams/              # Output CSV files with grades
│   └── *.csv
├── training_data/             # JSONL files for fine-tuning
│   └── *.jsonl
├── models.json                # Registry of fine-tuned models
├── .tentanator_session.json   # Saved grading session
└── .tentanator_training_session.json  # Saved training session
```

## Key Components

### tentanator.py

The main application module containing:

- **Data Classes**:
  - `GradedItem`: Individual graded response
  - `QuestionGrades`: Grades for a single question
  - `GradingSession`: Complete grading session data

- **Core Functions**:
  - `grade_questions()`: Main interactive grading interface
  - `export_to_csv()`: Export graded data to CSV
  - `export_to_jsonl()`: Export training data for fine-tuning
  - `get_ai_grade_suggestion()`: Get grade suggestions from trained models

### openai_trainer.py

The OpenAI fine-tuning module containing:

- **Data Classes**:
  - `FineTuningConfig`: Configuration for training jobs
  - `TrainingFile`: Uploaded training file metadata
  - `FineTuningJob`: Fine-tuning job tracking
  - `ModelRegistry`: Registry of trained models

- **OpenAITrainer Class**:
  - `validate_jsonl_file()`: Validate training data format
  - `upload_training_file()`: Upload data to OpenAI
  - `create_fine_tuning_job()`: Start fine-tuning
  - `monitor_job()`: Track job progress
  - `batch_grade_with_model()`: Grade multiple responses

## Workflow Example

1. **Initial Manual Grading**:
   - Load exam CSV with student responses
   - Select columns for identification and grading
   - Grade 50 sample responses per question
   - Export to JSONL format

2. **Model Training**:
   - Run `openai_trainer.py`
   - Select JSONL file to train
   - Wait for fine-tuning to complete (10-30 mins)
   - Model is automatically registered

3. **AI-Assisted Completion**:
   - Resume grading session
   - System detects available trained model
   - AI suggests grades for remaining responses
   - Review and accept/modify suggestions
   - Export final graded CSV

## Features in Detail

### Session Persistence
- All grading progress is automatically saved after each grade
- Sessions can be resumed at any time
- Tracks graded items, timestamps, and configuration

### Smart Grading Logic
- Blank or dash responses auto-graded as "0"
- Valid response counter excludes auto-graded items
- Threshold of 50 valid responses required for training

### AI Integration
- Pre-computes suggestions for smoother grading experience
- Rolling window of 5 suggestions cached for performance
- Models matched to questions by normalized naming

### Export Options
- **CSV Export**: Complete graded exam file
- **JSONL Export**: OpenAI fine-tuning format
- Both exports maintain data integrity and relationships

## Tips and Best Practices

1. **Consistent Grading**: Be consistent in your manual grading as the AI will learn from your patterns

2. **Quality Training Data**: Ensure your 50 sample grades are representative of the full range of responses

3. **Review AI Suggestions**: Always review AI-suggested grades, especially for edge cases

4. **Backup Sessions**: Session files are automatically created but consider backing up important grading data

5. **Model Management**: Use the `models.json` registry to track which models are trained for which questions

## Troubleshooting

- **Missing API Key**: Ensure `.env` file exists with valid `OPENAI_API_KEY`
- **Session Recovery**: If a session is corrupted, backup and delete `.tentanator_session.json`
- **Training Failures**: Check OpenAI dashboard for quota or billing issues
- **CSV Format Issues**: Ensure CSV files have proper headers and UTF-8 encoding

## Requirements

- Python 3.8+
- openai
- python-dotenv>=1.0.0

## License

GNU Affero General Public License - See LICENSE file for details

## Contributing

Contributions are welcome! Please ensure all code follows the existing patterns:
- Type hints for all functions
- Dataclasses for data structures
- Comprehensive error handling
- Session persistence for long-running operations
