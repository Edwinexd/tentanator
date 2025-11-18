# Tentanator - AI-Powered CSV Exam Grading Assistant

Tentanator is a Python-based intelligent grading system that combines manual grading with OpenAI fine-tuning to streamline exam grading workflows. It enables educators to grade a sample of responses manually, train custom AI models on that data, and automatically grade the remaining responses with AI assistance.

## Features

- **Interactive CSV Grading Interface**: Grade exam responses directly from CSV files with a user-friendly CLI
- **AI-Assisted Grading**: After grading sample responses, use fine-tuned GPT models to suggest grades for remaining responses
- **Content Moderation**: Automatic filtering of harmful content before training with OpenAI's moderation API
- **OpenAI Fine-Tuning Integration**: Automatically export graded data to JSONL format and train custom grading models
- **Session Persistence**: Resume grading sessions at any time with automatic session saving
- **Smart Sampling**: Choose from multiple sampling algorithms (KMeans, maximin, random, GPTSort) to select representative responses
- **Smart Auto-Grading**: Automatically assigns grade "0" to blank or dash responses
- **Batch Export**: Export fully graded CSV files with all grades filled in
- **Excel Export**: Convert graded CSV files to Excel format with auto-adjusted column widths
- **Model Registry**: Track and manage all fine-tuned models for different questions
- **Global Question Bank**: Link questions across multiple exams to build comprehensive training datasets

## Installation

### Prerequisites

- Python 3.8 or higher
- OpenAI API key (for AI-assisted grading features)

### Setup

1. Clone the repository:
```bash
git clone https://github.com/Edwinexd/tentanator.git
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

### Complete Grading Workflow

#### Step 1: Initial Setup and Sampling

1. **Start the grading application**:
```bash
python tentanator.py
```

2. **Select your CSV file**:
   - Place exam CSV files in the `exams/` directory
   - The program will list available files for selection

3. **Configure column mappings**:
   - Select ID columns (e.g., student ID, name)
   - Select input columns (student responses to grade)
   - Select output columns (where grades will be stored)
   - Link to global question bank (optional, for cross-exam model reuse)

4. **Choose sampling method** (configurable in `tentanator.py`):
   - `kmeans_auto`: Automatically determines optimal clusters (recommended)
   - `kmeans_fixed`: Fixed number of clusters
   - `maximin`: Diversity-based sampling
   - `random`: Random selection
   - `gptsort`: GPT-based quality sorting
   - Default: 25 representative samples per question

#### Step 2: Grade Sample Responses

5. **Grade the selected samples**:
   - Grade the representative samples shown by the system
   - Minimum 25 valid responses required per question (configurable)
   - Use commands:
     - `q` - quit and save session
     - `s` - skip current response
     - `b` - go back to previous response
     - Type grade value directly
   - Session is auto-saved after each grade

6. **Export training data**:
   - After reaching the minimum sample threshold (default: 25)
   - Choose to export to JSONL format for OpenAI fine-tuning
   - Training data saved in `training_data/` directory
   - Enter exam question text when prompted (used in training)

#### Step 3: Train the AI Model

7. **Run the training module**:
```bash
python openai_trainer.py
```

8. **Configure and start training**:
   - Select JSONL file(s) to train (or choose "all" for batch training)
   - **Content moderation runs automatically**:
     - Each training example is checked for harmful content
     - Flagged examples are excluded from training
     - Statistics displayed: total examples, flagged count, categories
     - Training aborted if >50% of content is flagged
   - Upload proceeds with clean examples only
   - Fine-tuning job is created and monitored

9. **Monitor training progress**:
   - Training typically takes 10-30 minutes per model
   - OpenAI allows up to 6 concurrent fine-tuning jobs
   - Models are automatically registered in `models.json` when complete
   - Global question IDs link models across exams

#### Step 4: AI-Assisted Grading

10. **Resume grading with trained model**:
```bash
python tentanator.py
```

11. **Select the same session**:
   - System detects available trained models
   - AI automatically suggests grades for remaining responses
   - Pre-computes suggestions for smooth grading experience

12. **Review and finalize**:
   - Review AI suggestions (shown before each response)
   - Press `[Enter]` to accept suggestion
   - Type grade value to override
   - All grades are recorded with timestamps
   - Export final CSV when complete

### Export to Excel

After grading is complete, convert CSV files to Excel format:

```bash
python make_excel.py
```

**What it does**:
- Converts all CSV files from `graded_exams/` directory
- Creates Excel files in `graded_exams_out/` directory
- Auto-adjusts column widths for better readability
- Preserves all data and formatting from the CSV files

### Configuration Options

Key settings in `tentanator.py`:

```python
GRADING_THRESHOLD = 25              # Minimum manual grades before training
NUM_REPRESENTATIVE_SAMPLES = 25     # Number of samples to grade
SAMPLING_ALGORITHM = "kmeans_auto"  # Sampling method
```

Available sampling algorithms:
- `kmeans_auto`: Automatically determines optimal number of clusters
- `kmeans_fixed`: Uses fixed number of clusters
- `maximin`: Maximizes diversity in selected samples
- `random`: Random selection
- `gptsort`: Uses GPT to sort responses by quality

### Content Moderation

All training data is automatically moderated before upload to OpenAI:

**Moderation Categories Checked**:
- Harassment and threatening content
- Hate speech and threatening hate
- Illicit content and violent instructions
- Self-harm content, intent, and instructions
- Sexual content and minors
- Violence and graphic violence

**Moderation Behavior**:
- Individual messages (system, user, assistant) are checked
- Flagged examples are automatically excluded from training
- Detailed statistics shown: total, flagged count, categories
- Training prevented if >50% of content is flagged
- Fails open if moderation API encounters errors

**To disable moderation** (not recommended):
```python
trainer.upload_training_file(filepath, question_name, moderate=False)
```

### Project Structure

```
tentanator/
├── tentanator.py              # Main grading application
├── openai_trainer.py          # OpenAI fine-tuning module with moderation
├── sampling.py                # Sampling algorithms (KMeans, maximin, etc.)
├── embeddings.py              # OpenAI embeddings wrapper
├── make_excel.py              # CSV to Excel converter utility
├── global_bank.py             # Global question bank management
├── test_moderation.py         # Moderation testing suite
├── requirements.txt           # Python dependencies
├── .env                       # API keys (not in version control)
├── exams/                     # Input CSV files
│   └── *.csv
├── graded_exams/              # Output CSV files with grades
│   └── *.csv
├── graded_exams_out/          # Excel exports of graded exams
│   └── *.xlsx
├── training_data/             # JSONL files for fine-tuning
│   ├── *.jsonl                # Combined training files
│   └── partials/              # Per-exam training data
│       └── *.jsonl
├── .tentanator_sessions/      # Saved grading sessions
│   └── *.json
├── global_bank.json           # Global question bank registry
├── models.json                # Registry of fine-tuned models
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

### make_excel.py

The CSV to Excel converter utility containing:
- Automatic conversion of all CSV files from `graded_exams/` directory
- Excel file generation with proper formatting
- Auto-adjusted column widths for optimal readability
- Batch processing of multiple CSV files

### openai_trainer.py

The OpenAI fine-tuning module with content moderation:

- **Data Classes**:
  - `FineTuningConfig`: Configuration for training jobs
  - `TrainingFile`: Uploaded training file metadata
  - `FineTuningJob`: Fine-tuning job tracking
  - `ModelRegistry`: Registry of trained models

- **OpenAITrainer Class**:
  - `moderate_content()`: Check content using OpenAI moderation API
  - `validate_and_moderate_jsonl()`: Validate format and filter harmful content
  - `validate_jsonl_file()`: Validate training data format (no moderation)
  - `upload_training_file()`: Upload data to OpenAI (with moderation by default)
  - `create_fine_tuning_job()`: Start fine-tuning
  - `monitor_job()`: Track job progress
  - `batch_grade_with_model()`: Grade multiple responses

### sampling.py

Implements various sampling algorithms for selecting representative responses:

- **SamplingAlgorithm**: Type definition for available algorithms
- **Functions**:
  - `kmeans_sample()`: K-means clustering with auto/fixed cluster selection
  - `maximin_sample()`: Maximize diversity using maximin distance
  - `random_sample()`: Simple random sampling
  - `gptsort_sample()`: GPT-based quality sorting
  - `select_representative_samples()`: Main interface for all algorithms

### embeddings.py

Wrapper for OpenAI text embeddings:

- Uses `text-embedding-3-large` model
- Async API calls for performance
- Caching for repeated requests

## Quick Start Workflow

### First Time Setup

```bash
# 1. Setup and configure
python tentanator.py
# → Choose CSV file from exams/
# → Map ID, input, and output columns
# → Link to global question bank (optional)
# → Grade 25 representative samples (default)
# → Export to JSONL when prompted

# 2. Train AI models
python openai_trainer.py
# → Select "all" to train all untrained files
# → Content moderation runs automatically
# → Wait 10-30 minutes per model
# → Models registered automatically

# 3. Complete grading with AI assistance
python tentanator.py
# → Select the same session
# → AI suggests grades for remaining responses
# → Review and accept/modify suggestions
# → Export final CSV when complete

# 4. Convert to Excel (optional)
python make_excel.py
```

### Detailed Workflow Example

**Day 1: Setup and Initial Grading**
1. Place `exam1.csv` in `exams/` directory
2. Run `python tentanator.py`
3. Configure column mappings (ID: "Student ID", Input: "Response Q1", Output: "Grade Q1")
4. Link to global question "Calculus Derivatives" in global bank
5. System selects 25 representative samples using KMeans clustering
6. Grade the 25 samples (takes ~10-15 minutes)
7. Export to JSONL → creates `gq1_exam1_Grade_Q1.jsonl`
8. Quit and save session

**Day 1: Train Model**
1. Run `python openai_trainer.py`
2. Content moderation checks all 25 examples
   - Example output: "Valid: 24 training examples (excluded 1 flagged)"
   - Flagged categories: harassment (1 example excluded)
3. Upload proceeds with 24 clean examples
4. Fine-tuning job created (Job ID: ftjob-xxx)
5. Wait 15-20 minutes for completion
6. Model registered as `ft:gpt-4-mini:...:tentanator_grade_q1:xxx`

**Day 2: Complete Grading**
1. Run `python tentanator.py`
2. Select existing session "exam1_..."
3. System loads trained model for "Grade Q1"
4. AI pre-computes suggestions for next 5 responses
5. Review each suggestion, press Enter to accept or type override
6. Complete all 200 remaining responses (~15-20 minutes)
7. Export final CSV to `graded_exams/exam1.csv`
8. Run `python make_excel.py` → creates `graded_exams_out/exam1.xlsx`

**Future Exams: Reuse Model**
1. Load `exam2.csv` with same question
2. Link to same global question "Calculus Derivatives"
3. Grade 25 new samples → adds to existing training data
4. Retrain model with combined data from both exams
5. Use improved model for grading

## Features in Detail

### Session Persistence
- All grading progress is automatically saved after each grade
- Sessions stored in `.tentanator_sessions/` directory
- Sessions can be resumed at any time
- Tracks graded items, timestamps, embeddings cache, and configuration
- Multiple sessions can be active for different exams

### Smart Grading Logic
- Blank or dash responses auto-graded as "0"
- Valid response counter excludes auto-graded items
- Configurable threshold (default: 25 valid responses) required for training
- Representative sampling reduces manual grading workload

### Sampling Algorithms
- **KMeans Auto**: Automatically determines optimal clusters using silhouette scoring
- **KMeans Fixed**: Uses specified number of clusters
- **Maximin**: Selects diverse samples by maximizing minimum distance
- **Random**: Simple random selection (baseline)
- **GPTSort**: Uses GPT to sort responses by quality without embeddings

### Global Question Bank
- Links identical questions across multiple exams
- Combines training data from all linked exams
- Single model trained on data from multiple exam iterations
- Improves model accuracy with larger, more diverse datasets
- Tracked in `global_bank.json`

### Content Moderation
- **Automatic**: Runs by default on all training data before upload
- **Categories**: 13 moderation categories checked (harassment, hate, violence, etc.)
- **Statistics**: Detailed reporting of flagged content and categories
- **Safety**: Training prevented if >50% of content is flagged
- **Transparent**: Shows which examples were excluded and why
- **Fail-Safe**: Fails open if moderation API encounters errors

### AI Integration
- Pre-computes suggestions for smoother grading experience
- Rolling window of 5 suggestions cached for performance
- Models matched to questions by global question ID or normalized naming
- Base system prompt includes exam question for context

### Export Options
- **CSV Export**: Complete graded exam file with all grades
- **JSONL Export**: OpenAI fine-tuning format with moderation
- **Partial Files**: Per-exam training data in `partials/` subdirectory
- **Combined Files**: Merged training data for global questions
- **Excel Export**: Formatted .xlsx files with auto-adjusted columns

## Tips and Best Practices

1. **Use Representative Sampling**: The default `kmeans_auto` algorithm selects diverse samples, reducing the amount of manual grading needed while maintaining quality

2. **Consistent Grading**: Be consistent in your manual grading as the AI will learn from your patterns

3. **Link Global Questions**: Use the global question bank to combine training data across multiple exam iterations for more accurate models

4. **Monitor Content Moderation**: Check moderation statistics to ensure your training data is appropriate and unbiased

5. **Review AI Suggestions**: Always review AI-suggested grades, especially for edge cases or unusual responses

6. **Gradual Improvement**: Models improve with more training data - consider retraining after grading multiple exams

7. **Backup Sessions**: Session files are automatically created in `.tentanator_sessions/` but consider backing up important grading data

8. **Model Management**: Use `models.json` to track which models are trained for which questions and when they were created

9. **Batch Training**: Use "all" option in `openai_trainer.py` to train multiple models simultaneously (up to 6 concurrent jobs)

10. **Quality Over Quantity**: 25 well-chosen representative samples often perform better than 50+ random samples

## Troubleshooting

### Common Issues

**Missing API Key**
- Ensure `.env` file exists in project root with valid `OPENAI_API_KEY`
- Test with: `python -c "import dotenv; dotenv.load_dotenv(); import os; print('OK' if os.getenv('OPENAI_API_KEY') else 'MISSING')"`

**Session Recovery**
- Sessions stored in `.tentanator_sessions/` directory
- If corrupted, backup and delete the specific session JSON file
- Start fresh by selecting "New session" in tentanator.py

**Training Failures**
- Check OpenAI dashboard for quota or billing issues
- Verify moderation didn't exclude too many examples (>50%)
- Ensure minimum 10 examples remain after moderation
- OpenAI limits: 6 concurrent fine-tuning jobs

**Content Moderation Blocking Training**
- Review which categories are being flagged
- Check if student responses contain inappropriate content
- Consider if content is genuinely problematic or false positive
- If false positive, contact OpenAI support or manually review

**CSV Format Issues**
- Ensure CSV files have proper headers in first row
- Use UTF-8 encoding (not ASCII or Latin-1)
- Avoid special characters in column names
- Check for consistent delimiter (comma vs semicolon)

**Model Not Found**
- Verify model is registered in `models.json`
- Check global question ID matches between session and model
- Ensure fine-tuning job completed successfully
- Run `python openai_trainer.py` to check job status

**Slow Performance**
- Embeddings are cached after first use
- First-time sampling may take 1-2 minutes for large datasets
- AI suggestions are pre-computed in batches of 5
- Consider using `random` sampling for faster initial selection

## Requirements

### Python Dependencies

- Python 3.8+
- openai>=1.0.0 (for fine-tuning and moderation APIs)
- python-dotenv>=1.0.0 (for environment variable management)
- pandas>=2.0.0 (for CSV processing)
- openpyxl>=3.1.0 (for Excel export)
- scikit-learn (for KMeans clustering and embeddings)
- numpy (for numerical operations)

### OpenAI API Access

- Valid OpenAI API key with access to:
  - Fine-tuning API (GPT-4 Mini recommended)
  - Moderation API (free)
  - Embeddings API (for sampling algorithms)
  - Chat completions (for GPTSort sampling)

### Costs Estimate

- **Embeddings**: ~$0.13 per 1M tokens (text-embedding-3-large)
- **Fine-tuning**: ~$3.00 per 1M tokens training (gpt-4.1-mini)
- **Inference**: ~$0.30 per 1M tokens (fine-tuned model)
- **Moderation**: Free
- **Example**: 200 responses × 100 words each = ~26,700 tokens
  - Embeddings: <$0.01
  - Fine-tuning (25 samples): <$0.10
  - Inference (175 graded): ~$0.01
  - **Total per exam: ~$0.12**

## License

GNU Affero General Public License - See LICENSE file for details

## Contributing

Contributions are welcome! Please ensure all code follows the existing patterns:
- Type hints for all functions
- Dataclasses for data structures
- Comprehensive error handling
- Session persistence for long-running operations
