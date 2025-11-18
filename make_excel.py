#!/usr/bin/env python3
"""Convert between CSV and Excel formats for exam data.

This module handles bidirectional conversion:
- CSV -> Excel: Graded exams from graded_exams/ to graded_exams_out/
- Excel -> CSV: Ungraded exams from exams_in/ to exams/
"""

from pathlib import Path
from typing import List, cast

import pandas as pd


def convert_csv_to_excel() -> None:
    """Convert all CSV files from graded_exams to Excel files in graded_exams_out."""
    # Define source and destination directories
    source_dir: Path = Path("graded_exams")
    output_dir: Path = Path("graded_exams_out")

    # Check if source directory exists
    if not source_dir.exists():
        print(f"Source directory '{source_dir}' does not exist. Skipping CSV->Excel conversion.")
        return

    # Create output directory if it doesn't exist
    output_dir.mkdir(exist_ok=True)

    # Find all CSV files in the source directory
    csv_files: List[Path] = list(source_dir.glob("*.csv"))

    if not csv_files:
        print(f"No CSV files found in '{source_dir}'.")
        return

    print("\n=== CSV to Excel Conversion ===")
    print(f"Found {len(csv_files)} CSV file(s) to convert.")

    # Process each CSV file
    for csv_file in csv_files:
        try:
            # Read CSV file
            df: pd.DataFrame = pd.read_csv(csv_file)

            # Create output filename (replace .csv with .xlsx)
            excel_filename: str = csv_file.stem + ".xlsx"
            excel_path: Path = output_dir / excel_filename

            # Write to Excel file
            with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
                df.to_excel(writer, index=False, sheet_name='Sheet1')

                # Auto-adjust column widths
                worksheet = writer.sheets['Sheet1']
                for column in df.columns:
                    column_width = max(
                        df[column].astype(str).map(len).max(),
                        len(str(column))
                    )
                    # Limit max width to avoid excessively wide columns
                    column_width = min(column_width, 50)
                    col_idx = cast(int, df.columns.get_loc(column))
                    worksheet.column_dimensions[
                        worksheet.cell(1, col_idx + 1).column_letter
                    ].width = column_width + 2

            print(f"  Converted: {csv_file.name} -> {excel_filename}")

        except (OSError, ValueError, pd.errors.ParserError) as e:
            print(f"  Error converting {csv_file.name}: {e}")
            continue

    print(f"CSV->Excel conversion complete. Excel files saved to '{output_dir}'.")


def convert_excel_to_csv() -> None:
    """Convert all Excel files from exams_in to CSV files in exams."""
    # Define source and destination directories
    source_dir: Path = Path("exams_in")
    output_dir: Path = Path("exams")

    # Check if source directory exists
    if not source_dir.exists():
        print(f"Source directory '{source_dir}' does not exist. Skipping Excel->CSV conversion.")
        return

    # Create output directory if it doesn't exist
    output_dir.mkdir(exist_ok=True)

    # Find all Excel files in the source directory
    excel_files: List[Path] = list(source_dir.glob("*.xlsx")) + list(source_dir.glob("*.xls"))

    if not excel_files:
        print(f"No Excel files found in '{source_dir}'.")
        return

    print("\n=== Excel to CSV Conversion ===")
    print(f"Found {len(excel_files)} Excel file(s) to convert.")

    # Process each Excel file
    for excel_file in excel_files:
        try:
            # Read Excel file (first sheet by default)
            df: pd.DataFrame = pd.read_excel(excel_file, sheet_name=0)

            # Create output filename (replace .xlsx/.xls with .csv)
            csv_filename: str = excel_file.stem + ".csv"
            csv_path: Path = output_dir / csv_filename

            # Write to CSV file
            df.to_csv(csv_path, index=False)

            print(f"  Converted: {excel_file.name} -> {csv_filename}")

        except (OSError, ValueError) as e:
            print(f"  Error converting {excel_file.name}: {e}")
            continue

    print(f"Excel->CSV conversion complete. CSV files saved to '{output_dir}'.")


def make_excel() -> None:
    """Run both conversion processes: Excel->CSV (input) and CSV->Excel (output)."""
    print("Starting file conversions...")

    # First convert ungraded Excel files to CSV for processing
    convert_excel_to_csv()

    # Then convert graded CSV files to Excel for distribution
    convert_csv_to_excel()

    print("\nAll conversions complete.")


if __name__ == "__main__":
    make_excel()
