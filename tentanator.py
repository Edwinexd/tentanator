#!/usr/bin/env python3
"""
Tentanator - CSV grading assistant with AI fine-tuning support
"""

import asyncio
import csv
import json
import os
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import dotenv
from openai import AsyncOpenAI

from sampling import SamplingAlgorithm, get_samples
from embeddings import get_embedding

dotenv.load_dotenv()

# Initialize OpenAI client globally
client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Constants
NUM_REPRESENTATIVE_SAMPLES = 25  # Number of representative samples (used for kmeans_fixed)
GRADING_THRESHOLD = 25  # Minimum number of manual grades required
assert NUM_REPRESENTATIVE_SAMPLES <= GRADING_THRESHOLD, \
    "Representative samples cannot exceed grading threshold"

# Sampling algorithm configuration
# Available algorithms:
#   - SamplingAlgorithm.KMEANS_AUTO: Auto-finds optimal k (2-20) by silhouette score
#   - SamplingAlgorithm.KMEANS_FIXED: Fixed k=NUM_REPRESENTATIVE_SAMPLES
#   - SamplingAlgorithm.RANDOM: Random sampling with n=NUM_REPRESENTATIVE_SAMPLES
#   - SamplingAlgorithm.MAXIMIN: Maximin diversity (greedy farthest-first)
#   - SamplingAlgorithm.GPTSORT: GPT-based quality sorting (top n=NUM_REPRESENTATIVE_SAMPLES)
#   - SamplingAlgorithm.IFOREST_GMM: Isolation Forest + Gaussian Mixture Model (strongly recommended - see examples in README.md of samplings)
SAMPLING_ALGORITHM: SamplingAlgorithm = SamplingAlgorithm.IFOREST_GMM

# Base system prompt for grading
BASE_SYSTEM_PROMPT = """You are an experienced teacher assistant helping grade \
student exam responses.
Your task is to evaluate the student's answer to the following question and \
provide a grade.
Be consistent, fair, and objective in your grading. All respones will be \
reviewed by a human teacher.

Exam Question: {exam_question}

Grading Criteria:
- Correctness of the answer
- Completeness of the response

Provide only the grade value as your response."""


@dataclass
class GradedItem:
    """Represents a single graded item"""
    row_id: str  # Unique identifier from ID columns
    input_text: str
    grade: str
    timestamp: str


@dataclass
class SamplingResult:
    """Stores results of a sampling operation"""
    algorithm: str  # Algorithm used (e.g., "iforest_gmm", "kmeans_auto", etc.)
    selected_ids: List[str]  # Row IDs selected
    quality_score: float  # Quality score from the algorithm (if applicable)
    num_samples: int  # Number of samples selected
    timestamp: str  # When the sampling was performed


@dataclass
class QuestionGrades:
    """Grades for a single question/output column"""
    question_name: str
    input_column: str
    exam_question: str = ""  # The actual exam question text
    graded_items: List[GradedItem] = field(default_factory=list)
    sampling_result: Optional[SamplingResult] = None  # Sampling result for this question


@dataclass
class GradingSession:
    """Represents a grading session with all necessary data"""
    csv_file: str
    id_columns: List[str]  # Columns used to identify rows
    input_columns: List[str]
    output_columns: List[str]
    questions: Dict[str, QuestionGrades]  # output_column -> QuestionGrades
    last_updated: str = ""
    # input_column -> {row_id -> embedding}
    embeddings_cache: Dict[str, Dict[str, List[float]]] = field(default_factory=dict)


def get_sessions_dir() -> Path:
    """Get the sessions directory path, creating it if necessary"""
    sessions_dir = Path(".tentanator_sessions")
    sessions_dir.mkdir(exist_ok=True)

    # Migrate old session if exists
    old_session_file = Path(".tentanator_session.json")
    if old_session_file.exists():
        print("📦 Migrating existing session to new format...")
        try:
            with open(old_session_file, 'r', encoding='utf-8') as f:
                data = json.load(f)

            # Generate session name from CSV file and timestamp
            csv_base = Path(data.get("csv_file", "unknown")).stem
            timestamp = data.get("last_updated", datetime.now().isoformat())[:19]
            timestamp = timestamp.replace(":", "").replace("-", "").replace("T", "_")
            session_name = f"{csv_base}_migrated_{timestamp}"
            session_name = "".join(c if c.isalnum() or c in "_-" else "_" for c in session_name)

            # Save to new location
            new_path = sessions_dir / f"{session_name}.json"
            data["session_name"] = session_name
            with open(new_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2)

            # Rename old file to backup
            old_session_file.rename(".tentanator_session.json.backup")
            print(f"✓ Migrated to: {session_name}")

        except (json.JSONDecodeError, OSError, KeyError) as e:
            print(f"⚠️  Could not migrate old session: {e}")

    return sessions_dir


def list_sessions() -> List[Tuple[str, Dict[str, Any]]]:
    """List all available sessions with their metadata"""
    sessions_dir = get_sessions_dir()
    sessions = []

    for session_file in sessions_dir.glob("*.json"):
        try:
            with open(session_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                session_name = session_file.stem
                metadata = {
                    "csv_file": data.get("csv_file", "Unknown"),
                    "last_updated": data.get("last_updated", "Unknown"),
                    "num_questions": len(data.get("questions", {}))
                }
                sessions.append((session_name, metadata))
        except (json.JSONDecodeError, KeyError):
            continue

    return sorted(sessions, key=lambda x: x[1].get("last_updated", ""), reverse=True)


def save_session(session: GradingSession, session_name: Optional[str] = None) -> str:
    """Save the current session to a JSON file"""
    session.last_updated = datetime.now().isoformat()

    # If no session name provided, generate one from CSV filename
    if not session_name:
        csv_base = Path(session.csv_file).stem
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        session_name = f"{csv_base}_{timestamp}"

    # Ensure session name is filesystem-safe
    session_name = "".join(c if c.isalnum() or c in "_-" else "_" for c in session_name)

    sessions_dir = get_sessions_dir()
    filename = sessions_dir / f"{session_name}.json"

    # Convert to dict for JSON serialization
    session_dict = {
        "session_name": session_name,
        "csv_file": session.csv_file,
        "id_columns": session.id_columns,
        "input_columns": session.input_columns,
        "output_columns": session.output_columns,
        "last_updated": session.last_updated,
        "embeddings_cache": session.embeddings_cache,
        "questions": {}
    }

    for col, question in session.questions.items():
        question_dict = {
            "question_name": question.question_name,
            "input_column": question.input_column,
            "exam_question": question.exam_question,
            "graded_items": [asdict(item) for item in question.graded_items],
            "sampling_result": asdict(question.sampling_result) if question.sampling_result else None
        }
        session_dict["questions"][col] = question_dict

    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(session_dict, f, indent=2)
    print(f"✓ Session saved as '{session_name}'")
    return session_name


async def pregenerate_embeddings_for_csv(session: GradingSession, csv_data: List[Dict[str, str]],
                                          session_name: str) -> None:
    """
    Pre-generate embeddings for all responses in the CSV data

    Args:
        session: The GradingSession to populate with embeddings
        csv_data: List of CSV rows as dictionaries
        session_name: Name of the session for saving
    """
    print("\n🔢 Pre-generating embeddings for all responses...")

    total_generated = 0
    total_skipped = 0

    for input_column in session.input_columns:
        if input_column not in session.embeddings_cache:
            session.embeddings_cache[input_column] = {}

        print(f"\n  Processing column: {input_column}")

        # Collect all rows that need embeddings
        tasks = []
        row_ids_to_process = []

        for row in csv_data:
            row_id = get_row_id(row, session.id_columns)

            # Skip if already cached
            if row_id in session.embeddings_cache[input_column]:
                total_skipped += 1
                continue

            response_text = row.get(input_column, "")

            # Skip blank/empty responses
            if not response_text.strip() or response_text.strip() in ['-', 'N/A']:
                total_skipped += 1
                continue

            # Create task for generating embedding
            tasks.append(get_embedding(response_text))
            row_ids_to_process.append(row_id)

        # Generate all embeddings concurrently
        if tasks:
            print(f"    Generating {len(tasks)} embeddings concurrently...")
            results = await asyncio.gather(*tasks, return_exceptions=True)

            for row_id, result in zip(row_ids_to_process, results):
                if isinstance(result, Exception):
                    print(f"    ⚠️  Failed for row {row_id}: {result}")
                    continue

                # Type check: result should be List[float] if not an Exception
                if isinstance(result, list):
                    session.embeddings_cache[input_column][row_id] = result
                total_generated += 1

                if total_generated % 10 == 0:
                    print(f"    Generated {total_generated} embeddings...")

    if total_generated > 0:
        print(f"\n✓ Generated {total_generated} embeddings, skipped {total_skipped}")
        save_session(session, session_name)
    else:
        print(f"✓ All embeddings already cached ({total_skipped} total)")


async def create_graded_item_with_embedding(row_id: str, input_text: str, grade: str,
                                             session: Optional[GradingSession] = None,
                                             input_column: Optional[str] = None) -> GradedItem:
    """
    Create a GradedItem and ensure embedding is cached

    Args:
        row_id: Unique identifier for the row
        input_text: The student's response text
        grade: The grade assigned
        session: Optional session to cache embeddings in
        input_column: Optional column name for embedding cache

    Returns:
        GradedItem (embeddings stored separately in session.embeddings_cache)
    """
    # Ensure embedding is cached if session provided
    if session and input_column and input_text.strip() and input_text.strip() not in ['-', 'N/A']:
        if input_column not in session.embeddings_cache:
            session.embeddings_cache[input_column] = {}

        # Generate embedding if not already cached
        if row_id not in session.embeddings_cache[input_column]:
            try:
                embedding = await get_embedding(input_text)
                session.embeddings_cache[input_column][row_id] = embedding
            except (OSError, ValueError, RuntimeError) as e:
                print(f"⚠️  Failed to generate embedding: {e}")

    return GradedItem(
        row_id=row_id,
        input_text=input_text,
        grade=grade,
        timestamp=datetime.now().isoformat()
    )


def load_session(session_name: str) -> Optional[GradingSession]:
    """Load a session from a JSON file"""
    sessions_dir = get_sessions_dir()
    filename = sessions_dir / f"{session_name}.json"

    if not filename.exists():
        return None

    try:
        with open(filename, 'r', encoding='utf-8') as f:
            data = json.load(f)

        # Reconstruct the session
        questions = {}
        for col, q_data in data.get("questions", {}).items():
            graded_items = []
            for item in q_data["graded_items"]:
                # Create GradedItem (ignore embedding field from old sessions)
                graded_items.append(GradedItem(
                    row_id=item["row_id"],
                    input_text=item["input_text"],
                    grade=item["grade"],
                    timestamp=item["timestamp"]
                ))
            # Load sampling result if present
            sampling_result = None
            if "sampling_result" in q_data and q_data["sampling_result"]:
                sr_data = q_data["sampling_result"]
                sampling_result = SamplingResult(
                    algorithm=sr_data["algorithm"],
                    selected_ids=sr_data["selected_ids"],
                    quality_score=sr_data["quality_score"],
                    num_samples=sr_data["num_samples"],
                    timestamp=sr_data["timestamp"]
                )

            questions[col] = QuestionGrades(
                question_name=q_data["question_name"],
                input_column=q_data["input_column"],
                exam_question=q_data.get("exam_question", ""),
                graded_items=graded_items,
                sampling_result=sampling_result
            )

        session = GradingSession(
            csv_file=data["csv_file"],
            id_columns=data["id_columns"],
            input_columns=data["input_columns"],
            output_columns=data["output_columns"],
            questions=questions,
            last_updated=data.get("last_updated", ""),
            embeddings_cache=data.get("embeddings_cache", {})
        )

        return session
    except (json.JSONDecodeError, OSError, KeyError) as e:
        print(f"Error loading session: {e}")
        return None


def list_csv_files(directory: str = "exams") -> List[str]:
    """List all CSV files in the specified directory."""
    csv_path = Path(directory)
    if not csv_path.exists():
        return []
    return sorted([f.name for f in csv_path.glob("*.csv")])


def select_csv_file(csv_files: List[str]) -> Optional[str]:
    """Prompt user to select a CSV file."""
    if not csv_files:
        print("No CSV files found in the exams/ directory.")
        return None

    print("\nAvailable CSV files:")
    for idx, filename in enumerate(csv_files, 1):
        print(f"{idx}. {filename}")

    while True:
        try:
            choice = input("\nSelect a CSV file (enter number): ").strip()
            choice_idx = int(choice) - 1
            if 0 <= choice_idx < len(csv_files):
                return csv_files[choice_idx]
            print(f"Please enter a number between 1 and {len(csv_files)}")
        except ValueError:
            print("Please enter a valid number")
        except KeyboardInterrupt:
            print("\n\nExiting...")
            return None


def get_csv_columns(filepath: Path) -> List[str]:
    """Read and return column headers from CSV file."""
    with open(filepath, 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        headers = next(reader)
    return headers


def select_columns(columns: List[str], prompt: str, allow_multiple: bool = True) -> List[str]:
    """Prompt user to select columns from the list."""
    print(f"\n{prompt}")
    print("Available columns:")
    for idx, col in enumerate(columns, 1):
        print(f"{idx}. {col}")

    while True:
        try:
            if allow_multiple:
                choice = input("\nEnter column numbers separated by commas (e.g., 1,3,5): ").strip()
                indices = [int(x.strip()) - 1 for x in choice.split(',')]
            else:
                choice = input("\nEnter column number: ").strip()
                indices = [int(choice) - 1]

            if all(0 <= idx < len(columns) for idx in indices):
                return [columns[idx] for idx in indices]
            print(f"Please enter numbers between 1 and {len(columns)}")
        except ValueError:
            print("Please enter valid numbers")
        except KeyboardInterrupt:
            print("\n\nExiting...")
            return []


def read_csv_data(filepath: Path) -> List[Dict[str, str]]:
    """Read all rows from CSV file as list of dictionaries"""
    with open(filepath, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        return list(reader)


def get_row_id(row: Dict[str, str], id_columns: List[str]) -> str:
    """Create a unique ID for a row based on ID columns"""
    return "_".join([row.get(col, "") for col in id_columns])


def load_model_registry() -> Dict[str, Dict[str, Any]]:
    """Load the models.json registry"""
    models_file = Path("models.json")
    if models_file.exists():
        try:
            with open(models_file, 'r', encoding='utf-8') as f:
                registry = json.load(f)
                print(f"📚 Loaded {len(registry)} models from registry")
                return registry
        except (json.JSONDecodeError, OSError) as e:
            print(f"⚠️  Failed to load models.json: {e}")
    else:
        print("📚 No models.json found - no trained models available")
    return {}


async def get_ai_grade_suggestion(model_id: str, question: QuestionGrades,
                                   response_text: str) -> Optional[str]:
    """Get AI-suggested grade from fine-tuned model"""
    try:
        # Build system prompt with exam question
        if question.exam_question:
            system_content = BASE_SYSTEM_PROMPT.format(exam_question=question.exam_question)
        else:
            system_content = f"You are grading responses for: {question.question_name}"

        # Get grade from model
        response = await client.chat.completions.create(
            model=model_id,
            messages=[
                {"role": "system", "content": system_content},
                {"role": "user", "content": response_text}
            ],
            max_tokens=10,
            temperature=0.0
        )

        return (response.choices[0].message.content or "").strip()

    except (OSError, ValueError, RuntimeError, AttributeError) as e:
        print(f"⚠️  AI suggestion failed: {e}")
        return None


def ask_sampling_algorithm(question: QuestionGrades) -> Optional[SamplingAlgorithm]:
    """
    Ask the user which sampling algorithm to use.
    Returns None if user wants to skip sampling.
    """
    # Check if we already have sampling results for this question
    if question.sampling_result:
        print(f"\n📊 Previous sampling: {question.sampling_result.algorithm} "
              f"(selected {question.sampling_result.num_samples} samples)")
        response = input(
            "Use previous sampling? [y]es/[n]o to select new algorithm: "
        ).strip().lower()
        if response in ['y', 'yes', '']:
            return None  # Will use existing sampling

    print("\n📊 Select sampling algorithm for representative samples:")
    print("1. iforest_gmm - Isolation Forest + GMM (recommended but slow)")
    print("2. kmeans_auto - KMeans with automatic k optimization")
    print("3. kmeans_fixed - KMeans with fixed k=25")
    print("4. random - Random sampling")
    print("5. maximin - Maximin diversity sampling")
    print("6. gptsort - GPT-based quality sorting")
    print("7. [s]kip - Skip sampling (grade all manually)")

    while True:
        choice = input("Enter choice (1-7 or 's'): ").strip().lower()

        if choice in ['s', 'skip']:
            return None

        algorithm_map = {
            '1': SamplingAlgorithm.IFOREST_GMM,
            '2': SamplingAlgorithm.KMEANS_AUTO,
            '3': SamplingAlgorithm.KMEANS_FIXED,
            '4': SamplingAlgorithm.RANDOM,
            '5': SamplingAlgorithm.MAXIMIN,
            '6': SamplingAlgorithm.GPTSORT,
        }

        if choice in algorithm_map:
            return algorithm_map[choice]

        print("Invalid choice. Please enter 1-7 or 's' to skip.")


def select_representative_samples(
    session: GradingSession,
    input_column: str,
    output_column: Optional[str] = None,
    algorithm: Optional[SamplingAlgorithm] = None,
    n_samples: Optional[int] = None
) -> Tuple[List[str], float, int]:
    """
    Select representative samples using specified sampling algorithm.

    Args:
        session: The GradingSession with cached embeddings
        input_column: The input column to select samples from
        output_column: The output column (question) being graded (required for gptsort)
        algorithm: Sampling algorithm to use (defaults to SAMPLING_ALGORITHM constant)
        n_samples: Number of samples to select (required for some algorithms, optional for others)

    Returns:
        Tuple of (list of row_ids for representative samples, quality score, number of samples selected)
    """
    # Use default algorithm if not specified
    if algorithm is None:
        algorithm = SAMPLING_ALGORITHM

    # Get embeddings for this column
    if input_column not in session.embeddings_cache:
        print(f"⚠️  No embeddings found for column {input_column}")
        return [], 0.0, 0

    embeddings_dict = session.embeddings_cache[input_column]

    # Filter out empty embeddings
    valid_embeddings = {
        row_id: emb for row_id, emb in embeddings_dict.items()
        if emb is not None and len(emb) > 0
    }

    if len(valid_embeddings) < 2:
        print(f"⚠️  Not enough valid embeddings ({len(valid_embeddings)}) for sampling")
        return list(valid_embeddings.keys()), 0.0, len(valid_embeddings)

    # Use the get_samples function with the specified algorithm
    try:
        print(f"\n📊 Sampling with algorithm: {algorithm}")

        # If n_samples not specified and algorithm requires it, use NUM_REPRESENTATIVE_SAMPLES
        effective_n_samples = n_samples
        algorithms_needing_n = (
            SamplingAlgorithm.KMEANS_FIXED,
            SamplingAlgorithm.RANDOM,
            SamplingAlgorithm.MAXIMIN,
            SamplingAlgorithm.GPTSORT,
            SamplingAlgorithm.IFOREST_GMM
        )
        if effective_n_samples is None and algorithm in algorithms_needing_n:
            effective_n_samples = NUM_REPRESENTATIVE_SAMPLES

        # For gptsort, we need text data and question text
        text_data = None
        question_text = None
        if algorithm == SamplingAlgorithm.GPTSORT:
            if output_column is None:
                raise ValueError("output_column is required for gptsort algorithm")

            # Load CSV data to get text responses
            csv_data = read_csv_data(Path(session.csv_file))

            # Build text_data dict mapping row_id to response text
            text_data = {}
            for row in csv_data:
                row_id = get_row_id(row, session.id_columns)
                if row_id in valid_embeddings:  # Only include rows with valid embeddings
                    text_data[row_id] = row.get(input_column, "")

            # Get question text from session
            if output_column in session.questions:
                question = session.questions[output_column]
                question_text = question.exam_question or question.question_name
            else:
                question_text = output_column

        selected_ids, quality_score, num_selected = get_samples(
            valid_embeddings,
            algorithm=algorithm,
            n_samples=effective_n_samples,
            text_data=text_data,
            question_text=question_text
        )
        return selected_ids, quality_score, num_selected

    except (ValueError, RuntimeError) as e:
        print(f"⚠️  Sampling failed: {e}")
        # Fallback to returning first n samples or all if n not specified
        fallback_n = n_samples if n_samples else min(NUM_REPRESENTATIVE_SAMPLES, len(valid_embeddings))
        return list(valid_embeddings.keys())[:fallback_n], 0.0, fallback_n


async def perform_sampling_for_all_questions(
    session: GradingSession,
    threshold: int = GRADING_THRESHOLD,
    session_name: Optional[str] = None
) -> None:
    """
    Perform sampling for all questions upfront before grading starts.
    """
    print("\n=== Sampling Phase ===")
    print("Setting up representative samples for each question...")
    print("-" * 50)

    # Initialize questions and perform sampling for each
    for col_idx, output_col in enumerate(session.output_columns):
        # Get or create QuestionGrades
        if output_col not in session.questions:
            if col_idx < len(session.input_columns):
                input_col = session.input_columns[col_idx]
            else:
                input_col = session.input_columns[0]
            session.questions[output_col] = QuestionGrades(
                question_name=output_col,
                input_column=input_col,
                graded_items=[]
            )

        question = session.questions[output_col]

        # Count only valid (non-blank) graded items
        valid_graded_count = sum(1 for item in question.graded_items
                                if item.input_text.strip() not in ["", "-", "N/A"])

        print(f"\n📋 Question {col_idx + 1}/{len(session.output_columns)}: {output_col}")
        print(f"   Already graded: {valid_graded_count}/{threshold} responses")

        if valid_graded_count >= threshold:
            print("   ✓ Already has enough graded samples")
            continue

        # Check if we already have sampling results
        if question.sampling_result:
            print(f"   ✓ Already has sampling: {question.sampling_result.algorithm} "
                  f"({question.sampling_result.num_samples} samples)")
            continue

        # Ask user for sampling algorithm for this question
        print(f"   Needs {threshold - valid_graded_count} more graded responses")
        chosen_algorithm = ask_sampling_algorithm(question)

        if chosen_algorithm:
            # Run sampling with chosen algorithm
            representative_row_ids, quality_score, num_selected = select_representative_samples(
                session, question.input_column, output_column=output_col,
                algorithm=chosen_algorithm
            )

            if representative_row_ids:
                # Save sampling results
                question.sampling_result = SamplingResult(
                    algorithm=chosen_algorithm.value,
                    selected_ids=representative_row_ids,
                    quality_score=quality_score,
                    num_samples=num_selected,
                    timestamp=datetime.now().isoformat()
                )

                # Save session immediately after sampling
                if session_name:
                    save_session(session, session_name)

                print(f"   ✓ Selected {num_selected} representative samples")
                if quality_score > 0:
                    print(f"     Quality score: {quality_score:.3f}")
        else:
            print("   ⚠️  No sampling - will grade all responses manually")

    # Show sampling summary
    print("\n" + "=" * 50)
    print("📊 SAMPLING SUMMARY")
    print("=" * 50)

    for output_col in session.output_columns:
        question = session.questions[output_col]
        print(f"\n{output_col}:")
        if question.sampling_result:
            print(f"  Algorithm: {question.sampling_result.algorithm}")
            print(f"  Samples: {question.sampling_result.num_samples}")
            if question.sampling_result.quality_score > 0:
                print(f"  Quality: {question.sampling_result.quality_score:.3f}")
        else:
            print("  No sampling (manual grading)")

    print("\n" + "=" * 50)
    input("\nPress Enter to start grading...")


async def grade_questions(session: GradingSession, csv_data: List[Dict[str, str]],
                          threshold: int = GRADING_THRESHOLD,
                          session_name: Optional[str] = None) -> GradingSession:
    """Interactive grading interface - grades one question at a time across all rows"""
    # Pre-generate embeddings for all responses before grading starts
    if session_name:
        await pregenerate_embeddings_for_csv(session, csv_data, session_name)

    # Perform sampling for all questions upfront
    await perform_sampling_for_all_questions(session, threshold, session_name)

    print("\n=== Manual Grading Mode ===")
    print(f"Grade at least {threshold} valid responses per question (excluding blank/dash)")
    print("Commands: [q]uit, [s]kip, [b]ack, or enter grade value")
    print("AI models (if available) will grade remaining responses after threshold is reached")
    print("-" * 50)

    # Load model registry to check for trained models
    model_registry = load_model_registry()

    # Cache for pre-computed AI suggestions (rolling window)
    ai_suggestion_cache: Dict[str, Dict[str, str]] = {}  # question_name -> {row_id: grade}
    window_size = 5  # Pre-compute next 5 responses

    # Process one output column at a time
    for col_idx, output_col in enumerate(session.output_columns):

        # Get question (already initialized in perform_sampling_for_all_questions)
        question = session.questions[output_col]

        # Use sampling results if available
        representative_row_ids = []

        # Count only valid (non-blank) graded items
        valid_graded_count = sum(1 for item in question.graded_items
                                if item.input_text.strip() not in ["", "-", "N/A"])

        if valid_graded_count < threshold and question.sampling_result:
            # Use the pre-computed sampling results
            representative_row_ids = question.sampling_result.selected_ids
            print(f"\n📊 Using {question.sampling_result.algorithm} sampling")
            print(f"   {question.sampling_result.num_samples} representative samples to grade")

        # Check if there's a trained model for this question
        model_id = None
        if model_registry:
            # Extract exam identifier from session
            exam_id = Path(session.csv_file).stem if session.csv_file else ""

            # Look for a model trained on this question and exam
            print(f"\n🔍 Looking for model for: {output_col} (Exam: {exam_id})")

            for mid, info in model_registry.items():
                model_question = info.get('question_name', '')
                model_exam = info.get('exam_id', '')

                # Display available models with their exam IDs
                print(f"   Available model: {model_question} (Exam: {model_exam}) -> {mid[:30]}...")

                # More flexible matching - handle both "Points 27" and "Points_27" formats
                normalized_output = output_col.replace(' ', '_').lower()
                normalized_model = model_question.replace(' ', '_').lower()

                # Match if the output_col appears in the model's question_name
                # (handles cases like "Points_27" matching "Exam123_Points_27")
                if normalized_output in normalized_model:
                    # Check if exam matches (if exam_id is present in model registry)
                    if not model_exam or exam_id.lower() in model_exam.lower() or model_exam.lower() in exam_id.lower():
                        model_id = mid
                        print(f"\n✅ MATCHED! Using model for {output_col} from exam {model_exam or 'unknown'}")
                        print(f"   Model ID: {model_id}")
                        break
                    print(f"   ⚠️  Question matches but exam doesn't ({model_exam} != {exam_id})")

            if not model_id:
                print(f"❌ No matching model found - manual grading only for {output_col}")

        # Count only valid (non-blank) graded items
        valid_graded_count = sum(1 for item in question.graded_items
                                if item.input_text.strip() not in ["", "-", "N/A"])

        # Skip if we already have enough valid grades for this question
        if valid_graded_count >= threshold:
            print(f"\n✓ Already have {threshold} manual grades for {output_col}")

            # Check if there are ungraded rows that could benefit from AI grading
            graded_ids = {item.row_id for item in question.graded_items}
            ungraded_rows = []
            for row_idx, row in enumerate(csv_data):
                row_id = get_row_id(row, session.id_columns)
                if row_id not in graded_ids:
                    response_text = row.get(question.input_column, "N/A")
                    # Skip blank responses
                    if response_text.strip() not in ["", "-", "N/A"]:
                        ungraded_rows.append((row_idx, row))

            if ungraded_rows and model_id:
                print(f"🤖 Model available for remaining {len(ungraded_rows)} responses")
                use_ai = input("Use AI to grade remaining responses? [y/n]: ").strip().lower()

                if use_ai == 'y':
                    print("\nGrading with AI - you can review and modify each suggestion...")

                    for row_idx, row in ungraded_rows:
                        row_id = get_row_id(row, session.id_columns)
                        response_text = row.get(question.input_column, "N/A")

                        # Get AI suggestion
                        ai_suggestion = await get_ai_grade_suggestion(
                            model_id, question, response_text
                        )

                        if ai_suggestion:
                            print(f"\n{'-'*60}")
                            print(f"Student {row_idx + 1}/{len(csv_data)}")
                            print("\nResponse:")
                            print("-" * 40)
                            print(response_text)  # Full response, no truncation
                            print("-" * 40)
                            print(f"\n🤖 AI Grade: {ai_suggestion}")

                            # Let user confirm or modify
                            prompt = "Accept grade? [ENTER=yes, b=back, q=quit, or type new grade]: "
                            user_input = input(prompt).strip()

                            # Handle back command
                            if user_input.lower() == 'b' and len(question.graded_items) > 0:
                                # Remove the last graded item
                                removed = question.graded_items.pop()
                                print(f"Removed grade for ID: {removed.row_id}")
                                save_session(session, session_name)
                                # Restart the grading process to go back
                                return await grade_questions(session, csv_data, threshold, session_name)
                            if user_input.lower() == 'q':
                                save_session(session, session_name)
                                print("\nSaved and exiting...")
                                return session
                            if user_input:
                                final_grade = user_input
                                print(f"✓ Modified to: {final_grade}")
                            else:
                                final_grade = ai_suggestion
                                print(f"✓ Accepted: {final_grade}")

                            graded_item = await create_graded_item_with_embedding(
                                row_id, response_text, final_grade, session, question.input_column
                            )
                            question.graded_items.append(graded_item)

                            # Save after EACH grade for safety
                            save_session(session, session_name)
                            print(f"💾 Saved (Total graded: {len(question.graded_items)})")

                    print(f"\n✅ Completed AI grading for {output_col}")

                    # Auto-zero any remaining ungraded responses
                    print("\n🔄 Auto-zeroing remaining ungraded responses...")
                    auto_zeroed = 0
                    graded_ids_set = {item.row_id for item in question.graded_items}
                    tasks = []
                    rows_to_grade = []
                    for row in csv_data:
                        row_id = get_row_id(row, session.id_columns)
                        if row_id not in graded_ids_set:
                            response_text = row.get(question.input_column, "-")
                            # Auto-grade any ungraded response as 0
                            tasks.append(create_graded_item_with_embedding(
                                row_id, response_text, "0", session, question.input_column
                            ))
                            rows_to_grade.append(row_id)

                    if tasks:
                        graded_items = await asyncio.gather(*tasks)
                        question.graded_items.extend(graded_items)
                        auto_zeroed = len(graded_items)

                    if auto_zeroed > 0:
                        print(f"✓ Auto-zeroed {auto_zeroed} remaining responses")
                        save_session(session, session_name)

                    # Export to CSV after AI review and auto-zeroing
                    print("\n📝 Exporting graded CSV...")
                    export_to_csv(session, csv_data)

            continue

        print(f"\n{'='*60}")
        print(f"GRADING QUESTION {col_idx + 1}/{len(session.output_columns)}: {output_col}")
        print(f"Current progress: {valid_graded_count}/{threshold} valid responses graded")
        auto_graded = len(question.graded_items) - valid_graded_count
        if auto_graded > 0:
            print(f"(Plus {auto_graded} auto-graded blank/dash responses)")

        # Create set of already graded row IDs for this question
        graded_ids = {item.row_id for item in question.graded_items}

        # Initialize cache for this question if needed
        if output_col not in ai_suggestion_cache:
            ai_suggestion_cache[output_col] = {}

        # Pre-compute AI suggestions for rolling window (if model available)
        async def precompute_suggestions(
            start_idx: int,
            win_size: int,
            curr_model_id: Optional[str],
            curr_graded_ids: set,
            curr_output_col: str,
            curr_question: QuestionGrades
        ):
            """Pre-compute AI suggestions for upcoming responses concurrently"""
            if not curr_model_id:
                return

            # Collect all responses that need AI suggestions
            tasks = []
            row_ids_to_compute = []

            end_idx = min(start_idx + win_size, len(csv_data))
            for idx in range(start_idx, end_idx):
                if idx >= len(csv_data):
                    break

                future_row = csv_data[idx]
                future_row_id = get_row_id(future_row, session.id_columns)

                # Skip if already graded or cached
                if future_row_id in curr_graded_ids or future_row_id in ai_suggestion_cache[curr_output_col]:
                    continue

                future_response = future_row.get(curr_question.input_column, "N/A")

                # Skip blank responses
                if future_response.strip() in ["", "-", "N/A"]:
                    continue

                # Create task for getting AI suggestion
                tasks.append(get_ai_grade_suggestion(
                    curr_model_id, curr_question, future_response
                ))
                row_ids_to_compute.append(future_row_id)

            # Execute all AI grading calls concurrently
            if tasks:
                results = await asyncio.gather(*tasks, return_exceptions=True)
                for row_id, ai_grade in zip(row_ids_to_compute, results):
                    if isinstance(ai_grade, Exception):
                        continue
                    # Type check: ai_grade should be str if not an Exception
                    if ai_grade and isinstance(ai_grade, str):
                        ai_suggestion_cache[curr_output_col][row_id] = ai_grade

        if model_id:
            print("🤖 AI model available - pre-computing suggestions for smoother grading")
            # Pre-load first window
            await precompute_suggestions(0, window_size, model_id, graded_ids, output_col, question)
        print(f"{'='*60}")

        # Create prioritized list of rows to grade (representative samples first)
        representative_set = set(representative_row_ids)
        prioritized_rows = []

        # First add representative samplesƒ‹
        for row_idx, row in enumerate(csv_data):
            row_id = get_row_id(row, session.id_columns)
            if row_id in representative_set:
                prioritized_rows.append((row_idx, row))

        # Then add remaining rows
        for row_idx, row in enumerate(csv_data):
            row_id = get_row_id(row, session.id_columns)
            if row_id not in representative_set:
                prioritized_rows.append((row_idx, row))

        # Grade this column for each row (prioritizing representative samples)
        for row_idx, row in prioritized_rows:
            # Check if we've reached the threshold for valid grades
            if valid_graded_count >= threshold:
                print(f"\n✓ Reached {threshold} valid grades for {output_col}!")
                break

            # Pre-compute suggestions for next window of responses
            if model_id and row_idx % 3 == 0:  # Update window every 3 responses
                await precompute_suggestions(
                    row_idx + 1, window_size, model_id, graded_ids, output_col, question
                )

            # Get row ID
            row_id = get_row_id(row, session.id_columns)

            # Skip if already graded
            if row_id in graded_ids:
                continue

            print(f"\n{'-'*60}")
            is_representative = row_id in representative_set
            representative_marker = " 🎯 REPRESENTATIVE SAMPLE" if is_representative else ""
            print(f"Student {row_idx + 1}/{len(csv_data)} | {output_col}{representative_marker}")

            # Show ID columns for reference
            id_info = " | ".join([f"{col}: {row.get(col, 'N/A')}" for col in session.id_columns])
            print(f"ID: {id_info}")
            print(f"Progress: {valid_graded_count}/{threshold} valid responses")
            print(f"{'-'*60}")

            # Display the response
            response_text = row.get(question.input_column, "N/A")
            print(f"\n{question.input_column}:")
            print("-" * 40)
            print(response_text)
            print("-" * 40)

            # Auto-grade blank or "-" responses as 0
            if response_text.strip() in ["", "-", "N/A"]:
                graded_item = await create_graded_item_with_embedding(
                    row_id, response_text, "0", session, question.input_column
                )
                question.graded_items.append(graded_item)
                print(f"\n✓ Auto-graded as 0 (blank/dash response - not counted toward {threshold} goal)")
                save_session(session, session_name)
                # Don't increment valid_graded_count for auto-graded blank responses
                continue

            # Show existing value if present in CSV
            existing = row.get(output_col, "")
            if existing:
                print(f"\nCurrent grade in CSV: {existing}")

            # Check if we have a pre-computed AI suggestion
            ai_suggestion = ai_suggestion_cache.get(output_col, {}).get(row_id)

            if ai_suggestion:
                print(f"\n🤖 AI Suggestion: {ai_suggestion}")
                print("   Press [ENTER] to accept, or type a different grade")

            while True:
                if ai_suggestion:
                    grade = input(f"\nGrade for {output_col} [{ai_suggestion}]: ").strip()
                    # If empty, accept AI suggestion
                    if not grade:
                        grade = ai_suggestion
                        print(f"✓ Accepted AI suggestion: {grade}")
                else:
                    grade = input(f"\nEnter grade for {output_col}: ").strip()

                if grade.lower() == 'q':
                    save_session(session, session_name)
                    print("\nSaved and exiting...")
                    return session
                if grade.lower() == 's':
                    print("  Skipping this response...")
                    break
                if grade.lower() == 'b' and len(question.graded_items) > 0:
                    # Remove the last graded item
                    removed = question.graded_items.pop()
                    print(f"Removed grade for ID: {removed.row_id}")
                    save_session(session, session_name)
                    return await grade_questions(session, csv_data, threshold, session_name)
                if grade:
                    graded_item = await create_graded_item_with_embedding(
                        row_id, response_text, grade, session, question.input_column
                    )
                    question.graded_items.append(graded_item)
                    valid_graded_count += 1  # Increment valid count for manually graded items

                    # Clear from cache once used
                    if row_id in ai_suggestion_cache.get(output_col, {}):
                        del ai_suggestion_cache[output_col][row_id]

                    if grade == ai_suggestion:
                        print(f"✓ Accepted AI suggestion: {grade}")
                    else:
                        print(f"✓ Graded as: {grade}")

                    # Save after each grade
                    save_session(session, session_name)
                    break
                print("  Please enter a grade or command")

    # Check if all questions are complete (based on valid grades only)
    all_complete = all(
        sum(1 for item in q.graded_items if item.input_text.strip() not in ["", "-", "N/A"]) >= threshold
        for q in session.questions.values()
    )
    if all_complete:
        print(f"\n{'='*60}")
        print("✓ THRESHOLD REACHED!")
        num_questions = len(session.output_columns)
        print(f"Graded {threshold} valid responses for all {num_questions} questions")
        print(f"{'='*60}")

        # Export to JSONL for training
        print("\n📤 Exporting training data to JSONL...")
        export_to_jsonl(session, session_name=session_name)

        print("\n💡 Next steps:")
        print("   1. Train a model using the exported JSONL files")
        print("   2. Reopen this session - the model will be auto-detected")
        print("   3. Continue grading remaining responses with AI assistance")

        return session  # Return here to avoid duplicate export prompt

    return session


def export_to_csv(session: GradingSession, csv_data: List[Dict[str, str]],
                  output_dir: str = "graded_exams") -> str:
    """Export graded data to CSV file with grades filled in"""
    Path(output_dir).mkdir(exist_ok=True)

    # Use the same filename as the original CSV
    output_file = Path(output_dir) / session.csv_file

    # Create a mapping of row_id to grades for all questions
    grades_by_row: Dict[str, Dict[str, str]] = {}  # row_id -> {output_col -> grade}

    for output_col, question in session.questions.items():
        for item in question.graded_items:
            if item.row_id not in grades_by_row:
                grades_by_row[item.row_id] = {}
            grades_by_row[item.row_id][output_col] = item.grade

    # Update the CSV data with grades
    updated_rows = []
    for row in csv_data:
        row_id = get_row_id(row, session.id_columns)
        updated_row = row.copy()

        # Add grades for this row
        if row_id in grades_by_row:
            for output_col, grade in grades_by_row[row_id].items():
                updated_row[output_col] = grade

        updated_rows.append(updated_row)

    # Write the updated CSV
    if updated_rows:
        fieldnames = list(updated_rows[0].keys())
        with open(output_file, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(updated_rows)

        print(f"✓ Exported graded CSV to: {output_file}")
        return str(output_file)

    return ""


def export_to_jsonl(
    session: GradingSession,
    output_dir: str = "training_data",
    session_name: Optional[str] = None
) -> None:
    """Export graded data to JSONL format for fine-tuning"""
    Path(output_dir).mkdir(exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Extract exam identifier from session name or CSV filename
    if session_name:
        exam_id = session_name.split('_')[0] if '_' in session_name else session_name
    else:
        exam_id = Path(session.csv_file).stem

    # Sanitize exam_id for filename
    exam_id = "".join(c if c.isalnum() or c in "_-" else "_" for c in exam_id)

    # Create one JSONL file per question
    for output_col, question in session.questions.items():
        if not question.graded_items:
            continue

        # Ask for exam question if not already set
        if not question.exam_question:
            print(f"\n{'='*60}")
            print(f"Question: {output_col}")
            print(f"Input column: {question.input_column}")
            print(f"{'='*60}")
            print("Please enter the exam question that students were answering:")
            print("(This will be included in the training data for context)")
            while True:
                exam_question = input("> ").strip()
                if exam_question:
                    question.exam_question = exam_question
                    # Save the updated session with the exam question
                    save_session(session, session_name)
                    break
                print("⚠ Exam question cannot be empty. Please enter the question:")

        # Include exam ID in the filename for uniqueness
        sanitized_col = output_col.replace(' ', '_').replace('/', '-')
        jsonl_file = Path(output_dir) / f"{exam_id}_{sanitized_col}_{timestamp}.jsonl"
        exported_count = 0

        with open(jsonl_file, 'w', encoding='utf-8') as f:
            for item in question.graded_items:
                # Skip blank/dash responses as they are irrelevant for training
                if item.input_text.strip() in ["", "-", "N/A"]:
                    continue

                # Build system prompt with exam question
                if question.exam_question:
                    system_content = BASE_SYSTEM_PROMPT.format(exam_question=question.exam_question)
                else:
                    system_content = f"You are grading responses for: {output_col}"

                # Build the training example
                training_example = {
                    "messages": [
                        {
                            "role": "system",
                            "content": system_content
                        },
                        {
                            "role": "user",
                            "content": item.input_text
                        },
                        {
                            "role": "assistant",
                            "content": item.grade
                        }
                    ]
                }

                f.write(json.dumps(training_example) + '\n')
                exported_count += 1

        if exported_count > 0:
            print(f"✓ Exported {exported_count} examples for {output_col} to {jsonl_file}")
            if len(question.graded_items) - exported_count > 0:
                print(f"  (Excluded {len(question.graded_items) - exported_count} blank/dash responses)")
        else:
            print(f"⚠ No valid examples to export for {output_col} (all were blank/dash responses)")
            # Remove empty file
            jsonl_file.unlink()


async def main() -> None:
    """Main function to run the CLI tool."""
    print("=== Tentanator - CSV Grading Assistant ===\n")

    # Check for existing sessions
    existing_sessions = list_sessions()
    selected_session = None
    current_session_name = None

    if existing_sessions:
        print("Found existing sessions:\n")
        for i, (name, metadata) in enumerate(existing_sessions, 1):
            print(f"{i}. {name}")
            print(f"   CSV: {metadata['csv_file']}")
            print(f"   Last updated: {metadata['last_updated']}")
            print(f"   Questions: {metadata['num_questions']}\n")

        print(f"{len(existing_sessions) + 1}. Create new session")

        while True:
            choice = input(f"\nSelect session [1-{len(existing_sessions) + 1}]: ").strip()
            try:
                choice_num = int(choice)
                if 1 <= choice_num <= len(existing_sessions):
                    current_session_name, _ = existing_sessions[choice_num - 1]
                    selected_session = load_session(current_session_name)
                    if selected_session:
                        print(f"\n✓ Loaded session '{current_session_name}'")
                        break
                    print(f"Error loading session '{current_session_name}'")
                elif choice_num == len(existing_sessions) + 1:
                    # Create new session
                    break
                else:
                    print("Invalid choice. Please try again.")
            except ValueError:
                print("Invalid input. Please enter a number.")

    if selected_session:
        print("\n=== Resuming Session ===")
        print(f"CSV: {selected_session.csv_file}")

        # Count valid grades only (excluding blank/dash)
        valid_graded = sum(
            sum(1 for item in q.graded_items if item.input_text.strip() not in ["", "-", "N/A"])
            for q in selected_session.questions.values()
        )
        total_graded = sum(len(q.graded_items) for q in selected_session.questions.values())
        auto_graded = total_graded - valid_graded

        print(f"Progress: {valid_graded} valid responses graded")
        if auto_graded > 0:
            print(f"         ({auto_graded} blank/dash responses auto-graded as 0)")
        print(f"Questions: {len(selected_session.questions)} questions in progress\n")

        # Load CSV data
        filepath = Path("exams") / selected_session.csv_file
        csv_data = read_csv_data(filepath)

        session = await grade_questions(selected_session, csv_data,
                                         session_name=current_session_name)

        # Check if we only did manual grading (50) without AI review
        # If so, ask about JSONL export for training
        all_fully_graded = all(
            len(q.graded_items) >= len(csv_data)  # All rows graded
            for q in session.questions.values()
        )

        if not all_fully_graded:
            # We only graded the manual threshold, ask about JSONL export for training
            export_now = input("\nExport to JSONL for fine-tuning? [y/n]: ").strip().lower()
            if export_now == 'y':
                export_to_jsonl(session, session_name=current_session_name)
        # If all_fully_graded is True, we already exported CSV and don't need JSONL

        return

    # Start new session
    csv_files = list_csv_files()
    selected_file = select_csv_file(csv_files)

    if not selected_file:
        return

    filepath = Path("exams") / selected_file
    print(f"\nSelected: {selected_file}")

    # Get columns from CSV
    try:
        columns = get_csv_columns(filepath)
        csv_data = read_csv_data(filepath)
    except (OSError, csv.Error, UnicodeDecodeError) as e:
        print(f"Error reading CSV file: {e}")
        return

    # Select ID columns
    prompt = "Select columns to use as unique IDENTIFIERS (e.g., student ID, name):"
    id_columns = select_columns(columns, prompt, allow_multiple=True)
    if not id_columns:
        print("Warning: No ID columns selected. Using row number as identifier.")
        id_columns = ["_row_number"]

    print(f"\nID columns: {', '.join(id_columns)}")

    # Select input columns
    prompt = "Select columns to use as INPUT (student responses):"
    input_columns = select_columns(columns, prompt, allow_multiple=True)
    if not input_columns:
        return

    print(f"\nInput columns: {', '.join(input_columns)}")

    # Select output columns (one per question)
    prompt = "Select columns to use as OUTPUT (grading targets, one per question):"
    output_columns = select_columns(columns, prompt, allow_multiple=True)
    if not output_columns:
        return

    print(f"\nOutput columns: {', '.join(output_columns)}")

    # Verify matching input/output columns
    if len(input_columns) != len(output_columns):
        print(f"\nWarning: {len(input_columns)} input columns but {len(output_columns)} output columns.")
        print("Each output column will be paired with the corresponding input column by index.")

    print("\n=== Configuration Complete ===")
    print(f"CSV File: {selected_file}")
    print(f"ID Columns: {', '.join(id_columns)}")
    print(f"Input Columns: {', '.join(input_columns)}")
    print(f"Output Columns: {', '.join(output_columns)}")

    # Ask for session name
    session_name_input = input("\nEnter a name for this session (or press Enter for auto-generated): ").strip()
    if not session_name_input:
        csv_base = Path(selected_file).stem
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        session_name = f"{csv_base}_{timestamp}"
    else:
        # Ensure session name is filesystem-safe
        session_name = "".join(c if c.isalnum() or c in "_-" else "_" for c in session_name_input)

    print(f"✓ Creating session: {session_name}")

    # Create new session
    session = GradingSession(
        csv_file=selected_file,
        id_columns=id_columns,
        input_columns=input_columns,
        output_columns=output_columns,
        questions={},
        last_updated=datetime.now().isoformat()
    )

    # Start grading
    session = await grade_questions(session, csv_data, session_name=session_name)

    # Check if we only did manual grading (50) without AI review
    # If so, ask about JSONL export for training
    if session.questions:
        all_fully_graded = all(
            len(q.graded_items) >= len(csv_data)  # All rows graded
            for q in session.questions.values()
        )

        if not all_fully_graded:
            # We only graded the manual threshold, ask about JSONL export for training
            export_now = input("\nExport to JSONL for fine-tuning? [y/n]: ").strip().lower()
            if export_now == 'y':
                export_to_jsonl(session, session_name=session_name)
        # If all_fully_graded is True, we already exported CSV and don't need JSONL


if __name__ == "__main__":
    asyncio.run(main())
