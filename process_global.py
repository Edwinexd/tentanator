#!/usr/bin/env python3
"""
Process global bank CSV files and generate embeddings for questions in Swedish and English.
"""

import asyncio
import csv
import json
import os
from pathlib import Path
from typing import Dict, List
import logging

import dotenv
from embeddings import get_embedding

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Load environment variables
dotenv.load_dotenv()

GLOBAL_BANK_DIR = Path("global_bank")
OUTPUT_DIR = Path("global_banks_embeddings")


async def process_csv_file(filepath: Path) -> Dict[str, Dict[str, List[float]]]:
    """
    Process a single CSV file and generate embeddings for q_se and q_en columns.

    Args:
        filepath: Path to the CSV file to process

    Returns:
        Dictionary mapping id to embeddings: {id: {"en": [...], "se": [...]}}
    """
    embeddings_map = {}

    logger.info("Processing file: %s", filepath)

    with open(filepath, 'r', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile)
        rows = list(reader)

        # Collect all tasks for parallel processing
        tasks = []
        row_data = []

        for row in rows:
            row_id = row.get('id')
            q_se = row.get('q_se', '').strip()
            q_en = row.get('q_en', '').strip()

            if not row_id:
                logger.warning("Skipping row without id in %s", filepath)
                continue

            # Only process rows that have at least one question
            if q_se or q_en:
                row_data.append((row_id, q_se, q_en))

                # Create embedding tasks
                if q_se:
                    tasks.append(get_embedding(q_se))
                else:
                    tasks.append(None)

                if q_en:
                    tasks.append(get_embedding(q_en))
                else:
                    tasks.append(None)

        # Execute all embedding tasks in parallel
        if tasks:
            logger.info("Generating embeddings for %d questions...", len(row_data))
            results = await asyncio.gather(*[
                task if task else asyncio.create_task(asyncio.sleep(0))
                for task in tasks
            ])

            # Map results back to IDs
            for i, (row_id, q_se, q_en) in enumerate(row_data):
                se_embedding = results[i * 2] if results[i * 2] and q_se else []
                en_embedding = results[i * 2 + 1] if results[i * 2 + 1] and q_en else []

                embeddings_map[row_id] = {
                    "se": se_embedding,
                    "en": en_embedding
                }

                logger.debug("Generated embeddings for ID %s", row_id)

    logger.info("Processed %d questions from %s", len(embeddings_map), filepath)
    return embeddings_map


async def process_all_global_banks() -> Dict[str, Dict[str, Dict[str, List[float]]]]:
    """
    Process all CSV files in the global_bank directory and generate embeddings.
    Save each CSV's embeddings to a separate file.

    Returns:
        Dictionary mapping CSV filenames to their embeddings
    """
    all_results = {}

    if not GLOBAL_BANK_DIR.exists():
        logger.error("Directory %s does not exist", GLOBAL_BANK_DIR)
        return all_results

    # Create output directory if it doesn't exist
    OUTPUT_DIR.mkdir(exist_ok=True)
    logger.info("Output directory: %s", OUTPUT_DIR)

    csv_files = list(GLOBAL_BANK_DIR.glob("*.csv"))

    if not csv_files:
        logger.warning("No CSV files found in %s", GLOBAL_BANK_DIR)
        return all_results

    logger.info("Found %d CSV file(s) to process", len(csv_files))

    for csv_file in csv_files:
        file_embeddings = await process_csv_file(csv_file)

        # Save embeddings for this CSV file
        output_file = OUTPUT_DIR / f"{csv_file.stem}_embeddings.json"
        save_embeddings_to_json(file_embeddings, str(output_file))

        all_results[csv_file.name] = file_embeddings

    return all_results


def save_embeddings_to_json(embeddings: Dict[str, Dict[str, List[float]]],
                           output_file: str) -> None:
    """
    Save embeddings dictionary to a JSON file.

    Args:
        embeddings: Dictionary of embeddings to save
        output_file: Path to the output JSON file
    """
    logger.info("Saving embeddings to %s", output_file)

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(embeddings, f, indent=2)

    logger.info("Successfully saved %d embeddings to %s", len(embeddings), output_file)


def load_embeddings_from_json(input_file: str) -> Dict[str, Dict[str, List[float]]]:
    """
    Load embeddings from a JSON file.

    Args:
        input_file: Path to the JSON file to load

    Returns:
        Dictionary of embeddings
    """
    if not os.path.exists(input_file):
        logger.warning("File %s does not exist", input_file)
        return {}

    with open(input_file, 'r', encoding='utf-8') as f:
        embeddings = json.load(f)

    logger.info("Loaded %d embeddings from %s", len(embeddings), input_file)
    return embeddings


def load_all_embeddings() -> Dict[str, Dict[str, Dict[str, List[float]]]]:
    """
    Load all embeddings from the output directory.

    Returns:
        Dictionary mapping CSV filenames to their embeddings
    """
    all_embeddings = {}

    if not OUTPUT_DIR.exists():
        logger.warning("Directory %s does not exist", OUTPUT_DIR)
        return all_embeddings

    json_files = list(OUTPUT_DIR.glob("*_embeddings.json"))

    for json_file in json_files:
        csv_name = json_file.stem.replace("_embeddings", "") + ".csv"
        embeddings = load_embeddings_from_json(str(json_file))
        all_embeddings[csv_name] = embeddings

    logger.info("Loaded embeddings for %d CSV files", len(all_embeddings))
    return all_embeddings


async def main() -> None:
    """
    Main function to process all global bank CSV files and generate embeddings.
    """
    try:
        # Process all CSV files
        all_results = await process_all_global_banks()

        if all_results:
            # Print summary statistics
            print("\n✅ Processing completed successfully!")
            print(f"   Output directory: {OUTPUT_DIR}")
            print(f"   Processed {len(all_results)} CSV file(s):\n")

            total_questions = 0
            total_with_se = 0
            total_with_en = 0

            for csv_name, embeddings in all_results.items():
                questions = len(embeddings)
                with_se = sum(1 for emb in embeddings.values() if emb.get("se"))
                with_en = sum(1 for emb in embeddings.values() if emb.get("en"))

                total_questions += questions
                total_with_se += with_se
                total_with_en += with_en

                print(f"   • {csv_name}:")
                print(f"     - Questions processed: {questions}")
                print(f"     - With Swedish text: {with_se}")
                print(f"     - With English text: {with_en}")

            print("\n   Total across all files:")
            print(f"     - Total questions: {total_questions}")
            print(f"     - Total with Swedish: {total_with_se}")
            print(f"     - Total with English: {total_with_en}")
        else:
            print("❌ No data was processed. Check if CSV files exist and contain valid data.")

    except Exception as e:
        logger.error("Error during processing: %s", str(e), exc_info=True)
        raise


if __name__ == "__main__":
    asyncio.run(main())
