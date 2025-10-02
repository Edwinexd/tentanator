#!/usr/bin/env python3
"""
Tentanator - CSV grading assistant with AI fine-tuning support
"""

import csv
import json
import os
from dataclasses import dataclass, asdict, field
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Dict, Any, Tuple

from openai import OpenAI

import dotenv

dotenv.load_dotenv()

# Base system prompt for grading
BASE_SYSTEM_PROMPT = """You are an experienced teacher assistant helping grade student exam responses.
Your task is to evaluate the student's answer to the following question and provide a grade.
Be consistent, fair, and objective in your grading. All respones will be reviewed by a human teacher.

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
class QuestionGrades:
    """Grades for a single question/output column"""
    question_name: str
    input_column: str
    exam_question: str = ""  # The actual exam question text
    graded_items: List[GradedItem] = field(default_factory=list)


@dataclass
class GradingSession:
    """Represents a grading session with all necessary data"""
    csv_file: str
    id_columns: List[str]  # Columns used to identify rows
    input_columns: List[str]
    output_columns: List[str]
    questions: Dict[str, QuestionGrades]  # output_column -> QuestionGrades
    last_updated: str = ""


def get_sessions_dir() -> Path:
    """Get the sessions directory path, creating it if necessary"""
    sessions_dir = Path(".tentanator_sessions")
    sessions_dir.mkdir(exist_ok=True)

    # Migrate old session if exists
    old_session_file = Path(".tentanator_session.json")
    if old_session_file.exists():
        print("📦 Migrating existing session to new format...")
        try:
            with open(old_session_file, 'r') as f:
                data = json.load(f)

            # Generate session name from CSV file and timestamp
            csv_base = Path(data.get("csv_file", "unknown")).stem
            timestamp = data.get("last_updated", datetime.now().isoformat())[:19].replace(":", "").replace("-", "").replace("T", "_")
            session_name = f"{csv_base}_migrated_{timestamp}"
            session_name = "".join(c if c.isalnum() or c in "_-" else "_" for c in session_name)

            # Save to new location
            new_path = sessions_dir / f"{session_name}.json"
            data["session_name"] = session_name
            with open(new_path, 'w') as f:
                json.dump(data, f, indent=2)

            # Rename old file to backup
            old_session_file.rename(".tentanator_session.json.backup")
            print(f"✓ Migrated to: {session_name}")

        except Exception as e:
            print(f"⚠️  Could not migrate old session: {e}")

    return sessions_dir


def list_sessions() -> List[Tuple[str, Dict[str, Any]]]:
    """List all available sessions with their metadata"""
    sessions_dir = get_sessions_dir()
    sessions = []

    for session_file in sessions_dir.glob("*.json"):
        try:
            with open(session_file, 'r') as f:
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
        "questions": {}
    }

    for col, question in session.questions.items():
        session_dict["questions"][col] = {
            "question_name": question.question_name,
            "input_column": question.input_column,
            "exam_question": question.exam_question,
            "graded_items": [asdict(item) for item in question.graded_items]
        }

    with open(filename, 'w') as f:
        json.dump(session_dict, f, indent=2)
    print(f"✓ Session saved as '{session_name}'")
    return session_name


def load_session(session_name: str) -> Optional[GradingSession]:
    """Load a session from a JSON file"""
    sessions_dir = get_sessions_dir()
    filename = sessions_dir / f"{session_name}.json"

    if not filename.exists():
        return None

    try:
        with open(filename, 'r') as f:
            data = json.load(f)

        # Reconstruct the session
        questions = {}
        for col, q_data in data.get("questions", {}).items():
            graded_items = [GradedItem(**item) for item in q_data["graded_items"]]
            questions[col] = QuestionGrades(
                question_name=q_data["question_name"],
                input_column=q_data["input_column"],
                exam_question=q_data.get("exam_question", ""),
                graded_items=graded_items
            )

        return GradingSession(
            csv_file=data["csv_file"],
            id_columns=data["id_columns"],
            input_columns=data["input_columns"],
            output_columns=data["output_columns"],
            questions=questions,
            last_updated=data.get("last_updated", "")
        )
    except Exception as e:
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
        except Exception as e:
            print(f"⚠️  Failed to load models.json: {e}")
    else:
        print("📚 No models.json found - no trained models available")
    return {}


def get_ai_grade_suggestion(client: Any, model_id: str, question: QuestionGrades,
                            response_text: str) -> Optional[str]:
    """Get AI-suggested grade from fine-tuned model"""
    try:
        # Build system prompt with exam question
        if question.exam_question:
            system_content = BASE_SYSTEM_PROMPT.format(exam_question=question.exam_question)
        else:
            system_content = f"You are grading responses for: {question.question_name}"

        # Get grade from model
        response = client.chat.completions.create(
            model=model_id,
            messages=[
                {"role": "system", "content": system_content},
                {"role": "user", "content": response_text}
            ],
            max_tokens=10,
            temperature=0.0
        )

        return response.choices[0].message.content.strip()

    except Exception as e:
        print(f"⚠️  AI suggestion failed: {e}")
        return None


def grade_questions(session: GradingSession, csv_data: List[Dict[str, str]],
                   threshold: int = 50, openai_client: Optional[Any] = None,
                   session_name: Optional[str] = None) -> GradingSession:
    """Interactive grading interface - grades one question at a time across all rows"""
    print("\n=== Manual Grading Mode ===")
    print(f"Grade at least {threshold} valid responses per question (excluding blank/dash)")
    print("Commands: [q]uit, [s]kip, [b]ack, or enter grade value")
    print("AI models (if available) will grade remaining responses after threshold is reached")
    print("-" * 50)

    # Load model registry to check for trained models
    model_registry = load_model_registry()

    # Cache for pre-computed AI suggestions (rolling window)
    ai_suggestion_cache: Dict[str, Dict[str, str]] = {}  # question_name -> {row_id: grade}
    WINDOW_SIZE = 5  # Pre-compute next 5 responses

    # Process one output column at a time
    for col_idx, output_col in enumerate(session.output_columns):

        # Get or create QuestionGrades
        if output_col not in session.questions:
            input_col = session.input_columns[col_idx] if col_idx < len(session.input_columns) else session.input_columns[0]
            session.questions[output_col] = QuestionGrades(
                question_name=output_col,
                input_column=input_col,
                graded_items=[]
            )

        question = session.questions[output_col]

        # Check if there's a trained model for this question
        model_id = None
        if openai_client and model_registry:
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

                # Match both question name AND exam ID
                if normalized_output == normalized_model:
                    # Check if exam matches (if exam_id is present in model registry)
                    if not model_exam or exam_id.lower() in model_exam.lower() or model_exam.lower() in exam_id.lower():
                        model_id = mid
                        print(f"\n✅ MATCHED! Using model for {output_col} from exam {model_exam or 'unknown'}")
                        print(f"   Model ID: {model_id}")
                        break
                    else:
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

            if ungraded_rows and model_id and openai_client:
                print(f"🤖 Model available for remaining {len(ungraded_rows)} responses")
                use_ai = input("Use AI to grade remaining responses? [y/n]: ").strip().lower()

                if use_ai == 'y':
                    print(f"\nGrading with AI - you can review and modify each suggestion...")

                    for row_idx, row in ungraded_rows:
                        row_id = get_row_id(row, session.id_columns)
                        response_text = row.get(question.input_column, "N/A")

                        # Get AI suggestion
                        ai_suggestion = get_ai_grade_suggestion(
                            openai_client, model_id, question, response_text
                        )

                        if ai_suggestion:
                            print(f"\n{'-'*60}")
                            print(f"Student {row_idx + 1}/{len(csv_data)}")
                            print(f"\nResponse:")
                            print("-" * 40)
                            print(response_text)  # Full response, no truncation
                            print("-" * 40)
                            print(f"\n🤖 AI Grade: {ai_suggestion}")

                            # Let user confirm or modify
                            user_input = input(f"Accept grade? [ENTER=yes, b=back, q=quit, or type new grade]: ").strip()

                            # Handle back command
                            if user_input.lower() == 'b' and len(question.graded_items) > 0:
                                # Remove the last graded item
                                removed = question.graded_items.pop()
                                print(f"Removed grade for ID: {removed.row_id}")
                                save_session(session, session_name)
                                # Restart the grading process to go back
                                return grade_questions(session, csv_data, threshold, openai_client)
                            elif user_input.lower() == 'q':
                                save_session(session, session_name)
                                print("\nSaved and exiting...")
                                return session
                            elif user_input:
                                final_grade = user_input
                                print(f"✓ Modified to: {final_grade}")
                            else:
                                final_grade = ai_suggestion
                                print(f"✓ Accepted: {final_grade}")

                            graded_item = GradedItem(
                                row_id=row_id,
                                input_text=response_text,
                                grade=final_grade,
                                timestamp=datetime.now().isoformat()
                            )
                            question.graded_items.append(graded_item)

                            # Save after EACH grade for safety
                            save_session(session, session_name)
                            print(f"💾 Saved (Total graded: {len(question.graded_items)})")

                    print(f"\n✅ Completed AI grading for {output_col}")

                    # Auto-zero any remaining ungraded responses
                    print("\n🔄 Auto-zeroing remaining ungraded responses...")
                    auto_zeroed = 0
                    for row in csv_data:
                        row_id = get_row_id(row, session.id_columns)
                        if row_id not in {item.row_id for item in question.graded_items}:
                            response_text = row.get(question.input_column, "-")
                            # Auto-grade any ungraded response as 0
                            graded_item = GradedItem(
                                row_id=row_id,
                                input_text=response_text,
                                grade="0",
                                timestamp=datetime.now().isoformat()
                            )
                            question.graded_items.append(graded_item)
                            auto_zeroed += 1

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
        def precompute_suggestions(start_idx: int, window_size: int):
            """Pre-compute AI suggestions for upcoming responses"""
            if not model_id or not openai_client:
                return

            end_idx = min(start_idx + window_size, len(csv_data))
            for idx in range(start_idx, end_idx):
                if idx >= len(csv_data):
                    break

                future_row = csv_data[idx]
                future_row_id = get_row_id(future_row, session.id_columns)

                # Skip if already graded or cached
                if future_row_id in graded_ids or future_row_id in ai_suggestion_cache[output_col]:
                    continue

                future_response = future_row.get(question.input_column, "N/A")

                # Skip blank responses
                if future_response.strip() in ["", "-", "N/A"]:
                    continue

                # Get AI suggestion in background
                ai_grade = get_ai_grade_suggestion(
                    openai_client, model_id, question, future_response
                )
                if ai_grade:
                    ai_suggestion_cache[output_col][future_row_id] = ai_grade

        if model_id:
            print(f"🤖 AI model available - pre-computing suggestions for smoother grading")
            # Pre-load first window
            precompute_suggestions(0, WINDOW_SIZE)
        print(f"{'='*60}")


        # Grade this column for each row
        for row_idx, row in enumerate(csv_data):
            # Check if we've reached the threshold for valid grades
            if valid_graded_count >= threshold:
                print(f"\n✓ Reached {threshold} valid grades for {output_col}!")
                break

            # Pre-compute suggestions for next window of responses
            if model_id and openai_client and row_idx % 3 == 0:  # Update window every 3 responses
                precompute_suggestions(row_idx + 1, WINDOW_SIZE)

            # Get row ID
            row_id = get_row_id(row, session.id_columns)

            # Skip if already graded
            if row_id in graded_ids:
                continue

            print(f"\n{'-'*60}")
            print(f"Student {row_idx + 1}/{len(csv_data)} | {output_col}")

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
                graded_item = GradedItem(
                    row_id=row_id,
                    input_text=response_text,
                    grade="0",
                    timestamp=datetime.now().isoformat()
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
                elif grade.lower() == 's':
                    print("  Skipping this response...")
                    break
                elif grade.lower() == 'b' and len(question.graded_items) > 0:
                    # Remove the last graded item
                    removed = question.graded_items.pop()
                    print(f"Removed grade for ID: {removed.row_id}")
                    save_session(session, session_name)
                    return grade_questions(session, csv_data, threshold, openai_client)
                elif grade:
                    graded_item = GradedItem(
                        row_id=row_id,
                        input_text=response_text,
                        grade=grade,
                        timestamp=datetime.now().isoformat()
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
                else:
                    print("  Please enter a grade or command")

    # Check if all questions are complete (based on valid grades only)
    all_complete = all(
        sum(1 for item in q.graded_items if item.input_text.strip() not in ["", "-", "N/A"]) >= threshold
        for q in session.questions.values()
    )
    if all_complete:
        print(f"\n{'='*60}")
        print(f"✓ THRESHOLD REACHED!")
        print(
            f"Graded {threshold} valid responses for all "
            f"{len(session.output_columns)} questions"
        )
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


def export_to_jsonl(session: GradingSession, output_dir: str = "training_data", session_name: Optional[str] = None) -> None:
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
            exam_question = input("> ").strip()
            if exam_question:
                question.exam_question = exam_question
                # Save the updated session with the exam question
                save_session(session, session_name)
            else:
                print("⚠ Using generic prompt without specific exam question")

        # Include exam ID in the filename for uniqueness
        sanitized_col = output_col.replace(' ', '_').replace('/', '-')
        jsonl_file = Path(output_dir) / f"{exam_id}_{sanitized_col}_{timestamp}.jsonl"
        exported_count = 0

        with open(jsonl_file, 'w') as f:
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


def main() -> None:
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
                    else:
                        print(f"Error loading session '{current_session_name}'")
                elif choice_num == len(existing_sessions) + 1:
                    # Create new session
                    break
                else:
                    print("Invalid choice. Please try again.")
            except ValueError:
                print("Invalid input. Please enter a number.")

    if selected_session:
        print(f"\n=== Resuming Session ===")
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

        # Initialize OpenAI client
        openai_client = None
        api_key = os.getenv("OPENAI_API_KEY")
        if api_key:
            openai_client = OpenAI(api_key=api_key)
            print("✓ OpenAI client initialized for AI-assisted grading")
        else:
            print("⚠️  OPENAI_API_KEY not found, AI suggestions disabled")

        session = grade_questions(selected_session, csv_data, openai_client=openai_client,
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
    except Exception as e:
        print(f"Error reading CSV file: {e}")
        return

    # Select ID columns
    id_columns = select_columns(columns, "Select columns to use as unique IDENTIFIERS (e.g., student ID, name):", allow_multiple=True)
    if not id_columns:
        print("Warning: No ID columns selected. Using row number as identifier.")
        id_columns = ["_row_number"]

    print(f"\nID columns: {', '.join(id_columns)}")

    # Select input columns
    input_columns = select_columns(columns, "Select columns to use as INPUT (student responses):", allow_multiple=True)
    if not input_columns:
        return

    print(f"\nInput columns: {', '.join(input_columns)}")

    # Select output columns (one per question)
    output_columns = select_columns(columns, "Select columns to use as OUTPUT (grading targets, one per question):", allow_multiple=True)
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

    # Initialize OpenAI client if available
    openai_client = None
    api_key = os.getenv("OPENAI_API_KEY")
    if api_key:
        openai_client = OpenAI(api_key=api_key)
        print("✓ OpenAI client initialized for AI-assisted grading")
    else:
        print("⚠️  OPENAI_API_KEY not found, AI suggestions disabled")

    # Start grading
    session = grade_questions(session, csv_data, openai_client=openai_client, session_name=session_name)

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
    main()
