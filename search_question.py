#!/usr/bin/env python3
"""
Interactive search tool for global bank questions using embeddings.
Allows users to search for similar questions by entering natural language queries.
"""

import asyncio
import csv
import json
import shutil
import textwrap
from pathlib import Path
from typing import Any, Dict, List, Tuple
import logging

import numpy as np
import dotenv
from aioconsole import ainput
from openai import AsyncOpenAI
from embeddings import get_embedding

# Configure logging
logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)

# Load environment variables
dotenv.load_dotenv()

# Initialize OpenAI client
client = AsyncOpenAI()

GLOBAL_BANK_DIR = Path("global_bank")
EMBEDDINGS_DIR = Path("global_banks_embeddings")


def cosine_similarity(vec1: List[float], vec2: List[float]) -> float:
    """
    Calculate cosine similarity between two vectors.

    Args:
        vec1: First embedding vector
        vec2: Second embedding vector

    Returns:
        Cosine similarity score (0-1, higher is more similar)
    """
    if not vec1 or not vec2:
        return 0.0

    a = np.array(vec1)
    b = np.array(vec2)

    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


def load_csv_data(csv_file: Path) -> Dict[str, Dict[str, Any]]:
    """
    Load question data from a CSV file.

    Args:
        csv_file: Path to the CSV file

    Returns:
        Dictionary mapping id to full row data
    """
    data: Dict[str, Dict[str, Any]] = {}

    if not csv_file.exists():
        logger.warning("CSV file not found: %s", csv_file)
        return data

    with open(csv_file, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            row_id = row.get('id')
            if row_id:
                data[row_id] = row

    return data


def load_all_data() -> Tuple[
    Dict[Tuple[str, str], Dict[str, List[float]]],
    Dict[Tuple[str, str], Dict[str, Any]]
]:
    """
    Load all embeddings and CSV data from global banks.

    Returns:
        Tuple of (embeddings_dict, csv_data_dict) where:
        - embeddings_dict maps (csv_name, id) to {"se": [...], "en": [...]}
        - csv_data_dict maps (csv_name, id) to full row data
    """
    all_embeddings: Dict[Tuple[str, str], Dict[str, List[float]]] = {}
    all_csv_data: Dict[Tuple[str, str], Dict[str, Any]] = {}

    if not EMBEDDINGS_DIR.exists():
        print(f"Error: Embeddings directory '{EMBEDDINGS_DIR}' does not exist.")
        print("Please run 'python process_global.py' first to generate embeddings.")
        return all_embeddings, all_csv_data

    # Load each embeddings file
    for embeddings_file in EMBEDDINGS_DIR.glob("*_embeddings.json"):
        csv_name = embeddings_file.stem.replace("_embeddings", "")
        csv_file = GLOBAL_BANK_DIR / f"{csv_name}.csv"

        # Load embeddings
        with open(embeddings_file, 'r', encoding='utf-8') as f:
            embeddings = json.load(f)

        # Load CSV data
        csv_data = load_csv_data(csv_file)

        # Store with csv_name prefix for unique keys
        for question_id, emb in embeddings.items():
            all_embeddings[(csv_name, question_id)] = emb

        for question_id, data in csv_data.items():
            all_csv_data[(csv_name, question_id)] = data

    return all_embeddings, all_csv_data


async def detect_language(text: str) -> str:
    """
    Use GPT to detect if the text is in Swedish or English.

    Args:
        text: Text to analyze

    Returns:
        "se" for Swedish, "en" for English
    """
    try:
        response = await client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a language detector. "
                 "Respond with only 'se' for Swedish or 'en' for English."},
                {"role": "user", "content": f"What language is this text in? Text: {text}"}
            ],
            max_tokens=10,
            temperature=0
        )

        content = response.choices[0].message.content
        if content:
            result = content.strip().lower()
            return "se" if result == "se" else "en"
        return "en"
    except Exception as e:  # pylint: disable=broad-exception-caught
        logger.warning("Language detection failed: %s. Defaulting to English.", str(e))
        return "en"


async def search_questions(
    query: str,
    embeddings: Dict[Tuple[str, str], Dict[str, List[float]]],
    csv_data: Dict[Tuple[str, str], Dict[str, Any]],
    language: str = "en",
    top_k: int = 5
) -> List[Tuple[float, Tuple[str, str], Dict[str, Any]]]:
    """
    Search for similar questions using embedding similarity.

    Args:
        query: User's search query
        embeddings: Dictionary of all question embeddings
        csv_data: Dictionary of all question data
        language: Language to search in ("en" or "se")
        top_k: Number of top results to return

    Returns:
        List of (similarity_score, question_id, question_data) tuples
    """
    # Generate embedding for the query
    query_embedding = await get_embedding(query)

    if not query_embedding:
        return []

    # Calculate similarities with all questions
    similarities = []
    lang_key = "en" if language == "en" else "se"

    for key, emb_data in embeddings.items():
        question_embedding = emb_data.get(lang_key)

        if not question_embedding:
            continue

        similarity = cosine_similarity(query_embedding, question_embedding)
        question_data = csv_data.get(key)

        if question_data:
            similarities.append((similarity, key, question_data))

    # Sort by similarity (descending) and return top K
    similarities.sort(reverse=True, key=lambda x: x[0])
    return similarities[:top_k]


def truncate_text(text: str, max_length: int) -> str:
    """
    Truncate text to max_length, adding ellipsis if needed.

    Args:
        text: Text to truncate
        max_length: Maximum length

    Returns:
        Truncated text with ellipsis if needed
    """
    if len(text) <= max_length:
        return text
    return text[:max_length-1] + "…"


def display_results(  # pylint: disable=too-many-locals
        results: List[Tuple[float, Tuple[str, str], Dict[str, Any]]],
        language: str
    ) -> None:
    """
    Display search results in a compact multi-column format.

    Args:
        results: List of (similarity_score, key, question_data) tuples
        language: Language preference for display
    """
    if not results:
        print("\nNo results found.\n")
        return

    # Get terminal width
    term_width = shutil.get_terminal_size().columns
    term_width = max(80, min(term_width, 150))  # Clamp between 80 and 150

    q_col = "q_en" if language == "en" else "q_se"
    ans_col = "ans_en" if language == "en" else "ans_se"

    # Fixed column widths
    num_col = 2
    score_col = 5
    id_col = 4
    ch_col = 3
    subj_col = 4
    spacing = 6  # Spaces between columns

    # Calculate remaining space for question and answer columns
    fixed_width = num_col + score_col + id_col + ch_col + subj_col + spacing
    remaining = term_width - fixed_width

    # Split remaining space between question and answer (roughly equal)
    q_width = remaining // 2
    a_width = remaining - q_width

    # Ensure minimum widths
    q_width = max(20, q_width)
    a_width = max(20, a_width)

    print(f"\n{'═'*term_width}")
    print(f"Top {len(results)} Results (Language: {language.upper()})")
    print(f"{'═'*term_width}")
    print(f"{'#':<{num_col}} {'Scr':<{score_col}} {'ID':<{id_col}} "
          f"{'Ch':<{ch_col}} {'Subj':<{subj_col}} {'Question':<{q_width}} {'Answer':<{a_width}}")
    print(f"{'─'*term_width}")

    for i, (score, key, data) in enumerate(results, 1):
        _, question_id = key
        question = str(data.get(q_col, "N/A")).strip().replace("\n", " ")
        answer = str(data.get(ans_col, "N/A")).strip().replace("\n", " ")
        chapter = str(data.get("chapter", "N/A"))[:3]
        subject = str(data.get("subject", "N/A"))[:4]

        # Truncate for column display
        q_short = truncate_text(question, q_width)
        a_short = truncate_text(answer, a_width)

        print(f"{i:<{num_col}} {score:.3f} {question_id:<{id_col}} "
              f"{chapter:<{ch_col}} {subject:<{subj_col}} "
              f"{q_short:<{q_width}} {a_short:<{a_width}}")

    # Show detailed view option
    print(f"{'─'*term_width}")
    print("\nEnter result # for details (1-5) or press Enter to search again: ", end="")


def show_detail(  # pylint: disable=too-many-locals
        result: Tuple[float, Tuple[str, str], Dict[str, Any]],
        language: str,
        index: int
    ) -> None:
    """
    Display detailed view of a single result.

    Args:
        result: Single (similarity_score, key, question_data) tuple
        language: Language preference for display
        index: Result number
    """
    # Get terminal width
    term_width = shutil.get_terminal_size().columns
    term_width = max(80, min(term_width, 120))  # Clamp between 80 and 120
    content_width = term_width - 4  # Leave margin for indentation

    score, key, data = result
    _, question_id = key
    q_col = "q_en" if language == "en" else "q_se"
    ans_col = "ans_en" if language == "en" else "ans_se"

    question = str(data.get(q_col, "N/A")).strip()
    answer = str(data.get(ans_col, "N/A")).strip()
    chapter = str(data.get("chapter", "N/A"))
    subject = str(data.get("subject", "N/A"))
    q_type = str(data.get("type", "N/A"))

    print(f"\n{'═'*term_width}")
    print(f"Result #{index} - Full Details")
    print(f"{'═'*term_width}")
    print(f"ID: {question_id} | Chapter: {chapter} | Subject: {subject} | "
          f"Type: {q_type} | Score: {score:.4f}")
    print(f"{'─'*term_width}")
    print("\nQuestion:")
    wrapped_q = textwrap.fill(question, width=content_width)
    print(f"  {wrapped_q}")
    print("\nAnswer:")
    wrapped_a = textwrap.fill(answer, width=content_width)
    print(f"  {wrapped_a}")
    print(f"{'─'*term_width}\n")


async def main() -> None:
    """
    Main interactive loop for searching questions.
    """
    print("Loading embeddings and question data...")
    embeddings, csv_data = load_all_data()

    if not embeddings:
        print("No embeddings found. Exiting.")
        return

    num_csvs = len(set(k[0] for k in embeddings))
    print(f"Loaded {len(embeddings)} questions from {num_csvs} CSV file(s).")
    print("\nWelcome to the Question Search Tool!")
    print("Enter your search query to find similar questions.")
    print("Language is automatically detected. Type 'quit', 'exit', or 'q' to exit.\n")

    while True:
        try:
            # Get user input
            query = (await ainput("Search: ")).strip()

            if not query:
                continue

            # Check for exit commands
            if query.lower() in ['quit', 'exit', 'q']:
                print("\nGoodbye!")
                break

            # Auto-detect language
            print("Detecting language and searching...")
            language = await detect_language(query)

            # Perform search
            results = await search_questions(query, embeddings, csv_data, language)

            # Display results
            display_results(results, language)

            # Handle detail view requests
            while True:
                try:
                    detail_input = (await ainput()).strip()

                    if not detail_input:
                        break

                    # Try to parse as number
                    try:
                        result_num = int(detail_input)
                        if 1 <= result_num <= len(results):
                            show_detail(results[result_num - 1], language, result_num)
                            print("Enter another number or press Enter to search again: ", end="")
                        else:
                            print(f"Please enter a number between 1 and {len(results)}: ", end="")
                    except ValueError:
                        print("Invalid input. Enter a number or press Enter: ", end="")

                except KeyboardInterrupt:
                    print()
                    break

        except KeyboardInterrupt:
            print("\n\nInterrupted. Goodbye!")
            break
        except Exception as e:  # pylint: disable=broad-exception-caught
            logger.error("Error during search: %s", str(e), exc_info=True)
            print(f"\nError occurred: {e}")
            print("Please try again.\n")


if __name__ == "__main__":
    asyncio.run(main())
