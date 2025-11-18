"""
Combine Moodle grade and response dump files into a single Excel file.

This script merges two Excel exports from Moodle:
1. Grades file: Contains student info and point scores
2. Responses file: Contains student responses and question text

Output format matches existing exam CSVs, saved as Excel in exams_in/ directory.
"""
import sys
from pathlib import Path
import pandas as pd


def extract_question_number(col_name: str) -> int:
    """Extract question number from column name like 'Q. 1 /1.00' or 'Response 1'."""
    if col_name.startswith('Q. '):
        # Format: 'Q. 1 /1.00' or 'Q. Kommentarer /0.00'
        parts = col_name.split('/')
        if len(parts) == 2:
            q_part = parts[0].replace('Q.', '').strip()
            try:
                return int(q_part)
            except ValueError:
                return -1  # For non-numeric like "Kommentarer"
    elif col_name.startswith('Response '):
        # Format: 'Response 1'
        try:
            return int(col_name.replace('Response ', ''))
        except ValueError:
            return -1
    return -1


def is_zero_max_column(df: pd.DataFrame, col_name: str) -> bool:
    """Check if a column has max points of 0 (should be removed)."""
    if not col_name.startswith('Q. '):
        return False

    # Extract max points from format 'Q. X /0.00'
    if '/0.00' in col_name:
        try:
            max_val = df[col_name].max()
            return max_val == 0 or pd.isna(max_val)
        except (KeyError, TypeError):
            return True
    return False


def combine_moodle_dumps(grades_path: str, responses_path: str, output_path: str) -> None:
    """
    Combine Moodle grades and responses files into single Excel file.

    Args:
        grades_path: Path to grades Excel file
        responses_path: Path to responses Excel file
        output_path: Path for output Excel file
    """
    # pylint: disable=too-many-locals,too-many-branches,too-many-statements
    print(f"Reading grades file: {grades_path}")
    grades_df = pd.read_excel(grades_path)

    print(f"Reading responses file: {responses_path}")
    responses_df = pd.read_excel(responses_path)

    # Remove "Overall average" row if it exists (last row in grades file)
    if len(grades_df) > 0 and str(grades_df.iloc[-1]['Last name']).strip() == 'Overall average':
        print("\nRemoving 'Overall average' row from grades")
        grades_df = grades_df.iloc[:-1]

    print(f"\nGrades shape: {grades_df.shape}")
    print(f"Responses shape: {responses_df.shape}")

    # Identify columns to remove (max points = 0)
    zero_max_cols = []
    for col in grades_df.columns:
        if is_zero_max_column(grades_df, col):
            zero_max_cols.append(col)

    print(f"\nRemoving {len(zero_max_cols)} zero-max columns:")
    for col in zero_max_cols:
        print(f"  - {col}")

    # CRITICAL: Merge on Daisy ID to handle different row orders
    # Start with student info from grades file - include ALL identifying columns
    student_info_cols = ['Daisy ID', 'Last name', 'First name']

    # Add optional identifying columns if they exist
    optional_cols = ['Username', 'Email address', 'ID number']
    for col in optional_cols:
        if col in grades_df.columns:
            student_info_cols.append(col)

    output_df = grades_df[student_info_cols].copy()

    print("\nMerging on Daisy ID to handle row order differences...")
    print(f"Including student info columns: {list(output_df.columns)}")

    # Get list of question numbers (excluding Kommentarer sections)
    question_cols_grades = [col for col in grades_df.columns
                           if col.startswith('Q. ') and col not in zero_max_cols]

    # Build ordered list of (response_col, grade_col, output_grade_col) tuples
    question_pairs = []

    for grade_col in question_cols_grades:
        # Parse question number/name from grade column
        # Format: 'Q. 1 /1.00' or 'Q. Kommentarer /0.00.1'
        q_part = grade_col.split('/')[0].replace('Q.', '').strip()

        # Check if this is a Kommentarer column
        if 'Kommentarer' in q_part or 'Kommentarer' in grade_col:
            # Find corresponding response in responses_df
            # Try to match based on suffix
            suffix = ''
            if '/0.00.' in grade_col:
                suffix_num = grade_col.split('/0.00.')[-1]
                suffix = '.' + suffix_num
            # Note: komm_name not needed in this refactored version

            resp_col_name = f'Response Kommentarer{suffix}'

            if resp_col_name in responses_df.columns:
                question_pairs.append((resp_col_name, None, None))
        else:
            # Regular question number
            try:
                q_num = int(q_part)
            except ValueError:
                continue

            # Find corresponding response column
            resp_col_name = f'Response {q_num}'

            if resp_col_name in responses_df.columns:
                question_pairs.append((resp_col_name, grade_col, f'Points {q_num}'))

    # Merge response and grade pairs one by one to maintain column order
    for resp_col, grade_col, output_grade_col in question_pairs:
        # Merge response column
        resp_subset = responses_df[['Daisy ID', resp_col]].copy()
        output_df = output_df.merge(resp_subset, on='Daisy ID', how='left')

        # Merge grade column if it exists
        if grade_col and output_grade_col:
            grade_subset = grades_df[['Daisy ID', grade_col]].copy()
            grade_subset.rename(columns={grade_col: output_grade_col}, inplace=True)
            output_df = output_df.merge(grade_subset, on='Daisy ID', how='left')

    print(f"\nOutput shape: {output_df.shape}")
    print(f"Output columns: {len(output_df.columns)}")

    # Save to Excel
    print(f"\nSaving to: {output_path}")
    output_df.to_excel(output_path, index=False, engine='openpyxl')
    print("Done!")


def find_matching_files(directory: str = 'exams_in_raw') -> tuple:
    """
    Find matching grades and responses files in directory.

    Returns:
        Tuple of (grades_file, responses_file, base_name) or (None, None, None)
    """
    input_dir = Path(directory)
    if not input_dir.exists():
        return None, None, None

    # Find all Excel files in directory
    excel_files = list(input_dir.glob('*.xlsx'))

    # Look for files ending with -grades.xlsx and -responses.xlsx
    grades_files = [f for f in excel_files if 'grades' in f.name.lower()]
    responses_files = [f for f in excel_files if 'responses' in f.name.lower()]

    # Try to match files with same base name
    for grades_file in grades_files:
        # Extract base name by removing -grades.xlsx suffix
        base_name = grades_file.stem.replace('-grades', '').replace('grades', '')
        base_name = base_name.strip('-').strip()

        # Find corresponding responses file
        for responses_file in responses_files:
            resp_base = responses_file.stem.replace('-responses', '').replace('responses', '')
            resp_base = resp_base.strip('-').strip()

            if base_name == resp_base:
                return str(grades_file), str(responses_file), base_name

    # If no exact match, just pair first grades with first responses
    if grades_files and responses_files:
        base_name = grades_files[0].stem.replace('-grades', '').replace('grades', '')
        base_name = base_name.strip('-').strip()
        return str(grades_files[0]), str(responses_files[0]), base_name

    return None, None, None


def main() -> None:
    """Main entry point."""
    # Check if files were provided as arguments
    if len(sys.argv) >= 3:
        # Manual mode: user provided file paths
        grades_file = sys.argv[1]
        responses_file = sys.argv[2]
        output_file = sys.argv[3] if len(sys.argv) > 3 else 'combined_exam.xlsx'
    else:
        # Auto mode: find files in exams_in_raw/
        print("No arguments provided. Auto-detecting files in exams_in_raw/...\n")
        grades_file, responses_file, base_name = find_matching_files()

        if not grades_file or not responses_file:
            print("Error: Could not find matching grades and responses files "
                  "in exams_in_raw/")
            print("\nUsage: python combine_moodle_dumps.py <grades_file.xlsx> "
                  "<responses_file.xlsx> [output.xlsx]")
            print("\nOr place files in exams_in_raw/ directory with "
                  "'grades' and 'responses' in filenames")
            sys.exit(1)

        # Use base name from input files for output
        output_file = f'exams_in/{base_name}.xlsx'
        print("Found files:")
        print(f"  Grades: {grades_file}")
        print(f"  Responses: {responses_file}")
        print(f"  Output: {output_file}\n")

    if not Path(grades_file).exists():
        print(f"Error: Grades file not found: {grades_file}")
        sys.exit(1)

    if not Path(responses_file).exists():
        print(f"Error: Responses file not found: {responses_file}")
        sys.exit(1)

    # Create exams directory if it doesn't exist
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    combine_moodle_dumps(grades_file, responses_file, output_file)


if __name__ == '__main__':
    main()
