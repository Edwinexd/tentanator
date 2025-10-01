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

# Base system prompt for grading
BASE_SYSTEM_PROMPT = """You are an experienced teacher grading student exam responses.
Your task is to evaluate the student's answer to the following question and provide a grade.
Be consistent, fair, and objective in your grading.

Exam Question: {exam_question}

Grading Criteria:
- Correctness of the answer
- Completeness of the response
- Clarity of explanation (if applicable)

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
    graded_items: List[GradedItem] = field(default_factory=list)


@dataclass
class GradingSession:
    """Represents a grading session with all necessary data"""
    csv_file: str
    id_columns: List[str]  # Columns used to identify rows
    input_columns: List[str]
    output_columns: List[str]
    questions: Dict[str, QuestionGrades]  # output_column -> QuestionGrades
    current_question_index: int = 0
    last_updated: str = ""


def save_session(session: GradingSession, filename: str = ".tentanator_session.json") -> None:
    """Save the current session to a JSON file"""
    session.last_updated = datetime.now().isoformat()

    # Convert to dict for JSON serialization
    session_dict = {
        "csv_file": session.csv_file,
        "id_columns": session.id_columns,
        "input_columns": session.input_columns,
        "output_columns": session.output_columns,
        "current_question_index": session.current_question_index,
        "last_updated": session.last_updated,
        "questions": {}
    }

    for col, question in session.questions.items():
        session_dict["questions"][col] = {
            "question_name": question.question_name,
            "input_column": question.input_column,
            "graded_items": [asdict(item) for item in question.graded_items]
        }

    with open(filename, 'w') as f:
        json.dump(session_dict, f, indent=2)
    print(f"✓ Session saved to {filename}")


def load_session(filename: str = ".tentanator_session.json") -> Optional[GradingSession]:
    """Load a session from a JSON file"""
    if not Path(filename).exists():
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
                graded_items=graded_items
            )

        return GradingSession(
            csv_file=data["csv_file"],
            id_columns=data["id_columns"],
            input_columns=data["input_columns"],
            output_columns=data["output_columns"],
            questions=questions,
            current_question_index=data.get("current_question_index", 0),
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


def grade_questions(session: GradingSession, csv_data: List[Dict[str, str]], threshold: int = 50) -> GradingSession:
    """Interactive grading interface - grades one question at a time across all rows"""
    print("\n=== Manual Grading Mode ===")
    print(f"Grade at least {threshold} valid responses per question (excluding blank/dash)")
    print("Commands: [q]uit, [s]kip, [b]ack, or enter grade value")
    print("-" * 50)

    # Process one output column at a time
    for col_idx in range(session.current_question_index, len(session.output_columns)):
        output_col = session.output_columns[col_idx]
        session.current_question_index = col_idx

        # Get or create QuestionGrades
        if output_col not in session.questions:
            input_col = session.input_columns[col_idx] if col_idx < len(session.input_columns) else session.input_columns[0]
            session.questions[output_col] = QuestionGrades(
                question_name=output_col,
                input_column=input_col,
                graded_items=[]
            )

        question = session.questions[output_col]

        # Count only valid (non-blank) graded items
        valid_graded_count = sum(1 for item in question.graded_items
                                if item.input_text.strip() not in ["", "-", "N/A"])

        # Skip if we already have enough valid grades for this question
        if valid_graded_count >= threshold:
            print(f"\n✓ Already have {threshold} valid grades for {output_col}, skipping to next question")
            continue

        print(f"\n{'='*60}")
        print(f"GRADING QUESTION {col_idx + 1}/{len(session.output_columns)}: {output_col}")
        print(f"Current progress: {valid_graded_count}/{threshold} valid responses graded")
        auto_graded = len(question.graded_items) - valid_graded_count
        if auto_graded > 0:
            print(f"(Plus {auto_graded} auto-graded blank/dash responses)")
        print(f"{'='*60}")

        # Create set of already graded row IDs for this question
        graded_ids = {item.row_id for item in question.graded_items}

        # Grade this column for each row
        for row_idx, row in enumerate(csv_data):
            # Check if we've reached the threshold for valid grades
            if valid_graded_count >= threshold:
                print(f"\n✓ Reached {threshold} valid grades for {output_col}!")
                break

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
                save_session(session)
                # Don't increment valid_graded_count for auto-graded blank responses
                continue

            # Show existing value if present in CSV
            existing = row.get(output_col, "")
            if existing:
                print(f"\nCurrent grade in CSV: {existing}")

            while True:
                grade = input(f"\nEnter grade for {output_col}: ").strip()

                if grade.lower() == 'q':
                    save_session(session)
                    print("\nSaved and exiting...")
                    return session
                elif grade.lower() == 's':
                    print("  Skipping this response...")
                    break
                elif grade.lower() == 'b' and len(question.graded_items) > 0:
                    # Remove the last graded item
                    removed = question.graded_items.pop()
                    print(f"Removed grade for ID: {removed.row_id}")
                    save_session(session)
                    return grade_questions(session, csv_data, threshold)
                elif grade:
                    graded_item = GradedItem(
                        row_id=row_id,
                        input_text=response_text,
                        grade=grade,
                        timestamp=datetime.now().isoformat()
                    )
                    question.graded_items.append(graded_item)
                    valid_graded_count += 1  # Increment valid count for manually graded items
                    print(f"✓ Graded as: {grade}")

                    # Save after each grade
                    save_session(session)
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
        print(f"✓ GRADING COMPLETE!")
        print(f"Graded {threshold} valid responses for all {len(session.output_columns)} questions")
        print(f"{'='*60}")

    return session


def export_to_jsonl(session: GradingSession, output_dir: str = "training_data") -> None:
    """Export graded data to JSONL format for fine-tuning"""
    Path(output_dir).mkdir(exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Create one JSONL file per question
    for output_col, question in session.questions.items():
        if not question.graded_items:
            continue

        jsonl_file = Path(output_dir) / f"{output_col.replace(' ', '_')}_{timestamp}.jsonl"
        exported_count = 0

        with open(jsonl_file, 'w') as f:
            for item in question.graded_items:
                # Skip blank/dash responses as they are irrelevant for training
                if item.input_text.strip() in ["", "-", "N/A"]:
                    continue

                # Build the training example
                training_example = {
                    "messages": [
                        {
                            "role": "system",
                            "content": f"You are grading responses for: {output_col}"
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
            print(f"✓ Exported {exported_count} examples for {output_col} to {jsonl_file} (excluded {len(question.graded_items) - exported_count} blank/dash responses)")
        else:
            print(f"⚠ No valid examples to export for {output_col} (all were blank/dash responses)")
            # Remove empty file
            jsonl_file.unlink()


def main() -> None:
    """Main function to run the CLI tool."""
    print("=== Tentanator - CSV Grading Assistant ===\n")

    # Check for existing session
    existing_session = load_session()
    if existing_session:
        print(f"Found existing session from {existing_session.last_updated}")
        print(f"CSV: {existing_session.csv_file}")

        # Count valid grades only (excluding blank/dash)
        valid_graded = sum(
            sum(1 for item in q.graded_items if item.input_text.strip() not in ["", "-", "N/A"])
            for q in existing_session.questions.values()
        )
        total_graded = sum(len(q.graded_items) for q in existing_session.questions.values())
        auto_graded = total_graded - valid_graded

        print(f"Progress: {valid_graded} valid responses graded")
        if auto_graded > 0:
            print(f"         ({auto_graded} blank/dash responses auto-graded as 0)")
        print(f"Questions: {len(existing_session.questions)} questions in progress")

        resume = input("\nResume this session? [y/n]: ").strip().lower()
        if resume == 'y':
            filepath = Path("exams") / existing_session.csv_file
            csv_data = read_csv_data(filepath)
            session = grade_questions(existing_session, csv_data)

            # Ask if they want to export
            export_now = input("\nExport to JSONL for fine-tuning? [y/n]: ").strip().lower()
            if export_now == 'y':
                export_to_jsonl(session)
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

    # Create new session
    session = GradingSession(
        csv_file=selected_file,
        id_columns=id_columns,
        input_columns=input_columns,
        output_columns=output_columns,
        questions={},
        current_question_index=0,
        last_updated=datetime.now().isoformat()
    )

    # Start grading
    session = grade_questions(session, csv_data)

    # Ask if they want to export
    if session.questions:
        export_now = input("\nExport to JSONL for fine-tuning? [y/n]: ").strip().lower()
        if export_now == 'y':
            export_to_jsonl(session)


if __name__ == "__main__":
    main()