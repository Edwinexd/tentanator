#!/usr/bin/env python3
"""Convert CSV files from graded_exams folder to Excel files in graded_exams_out folder."""

import sys
from pathlib import Path
from typing import List, cast

import pandas as pd


def make_excel() -> None:
    """Convert all CSV files from graded_exams to Excel files in graded_exams_out."""
    # Define source and destination directories
    source_dir: Path = Path("graded_exams")
    output_dir: Path = Path("graded_exams_out")

    # Check if source directory exists
    if not source_dir.exists():
        print(f"Error: Source directory '{source_dir}' does not exist.")
        sys.exit(1)

    # Create output directory if it doesn't exist
    output_dir.mkdir(exist_ok=True)

    # Find all CSV files in the source directory
    csv_files: List[Path] = list(source_dir.glob("*.csv"))

    if not csv_files:
        print(f"No CSV files found in '{source_dir}'.")
        return

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

            print(f" Converted: {csv_file.name} -> {excel_filename}")

        except Exception as e:
            print(f" Error converting {csv_file.name}: {e}")
            continue

    print(f"\nConversion complete. Excel files saved to '{output_dir}'.")


if __name__ == "__main__":
    make_excel()