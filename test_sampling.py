"""
Test script to evaluate how different sampling strategies cover point classes.

Loads graded exam CSVs and tests each sampling algorithm to see how well
they capture the diversity of grade values (point classes) in their samples.
Finds the minimum number of samples needed for each strategy.
"""
import asyncio
import json
import os
from typing import Dict, List, Tuple, Set, Optional
from collections import Counter

import pandas as pd
import dotenv

from sampling import (
    SamplingAlgorithm,
    get_samples,
    get_features,
    optimize_features,
)

dotenv.load_dotenv()

# Configuration
GRADED_EXAMS_DIR = "graded_exams"
SAMPLE_SIZES = list(range(5, 50))  # Test sample sizes from 5 to 49
MIN_SAMPLES_PER_CLASS = 1  # Minimum samples needed per point class
MIN_SAMPLES_INCLUDE = 1  # Minimum total samples for a class to be included in criteria
TARGET_COVERAGE = 1.0  # Must cover all point classes (that meet MIN_SAMPLES_INCLUDE)
STRATEGIES = [
    SamplingAlgorithm.KMEANS_AUTO,
    SamplingAlgorithm.KMEANS_FIXED,
    SamplingAlgorithm.RANDOM,
    SamplingAlgorithm.MAXIMIN,
    SamplingAlgorithm.IFOREST_GMM,
]


def load_sampling_config() -> Tuple[Dict[str, Dict[str, str]], Dict[str, Dict[str, str]]]:
    """
    Load question texts and sample answers from config file.

    Returns:
        Tuple of (question_texts, sample_answers) dictionaries
    """
    config_path = "test_sampling_config.json"
    if os.path.exists(config_path):
        with open(config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)
            return config.get("question_texts", {}), config.get("sample_answers", {})
    return {}, {}


# Load question texts and sample answers from config file (if it exists)
# Map CSV filename -> question name -> text/answer
# If not specified, will use question name only
QUESTION_TEXTS, SAMPLE_ANSWERS = load_sampling_config()


async def get_features_for_responses(
    responses: Dict[str, str],
    question_text: Optional[str] = None,
    sample_answer: Optional[str] = None
) -> Dict[str, List[float]]:
    """
    Get feature vectors for all responses.

    Args:
        responses: Dictionary mapping IDs to response text
        question_text: Optional question text for context
        sample_answer: Optional sample answer for context

    Returns:
        Dictionary mapping IDs to feature vectors
    """
    # Create tasks for all feature vectors
    tasks = [
        get_features(text, question_text=question_text, sample_answer=sample_answer)
        for text in responses.values()
    ]
    embeddings_list = await asyncio.gather(*tasks)

    # Map back to IDs
    embeddings_dict = dict(zip(responses.keys(), embeddings_list))
    return embeddings_dict


def load_graded_data(
    csv_path: str
) -> List[Tuple[str, Dict[str, str], Dict[str, float]]]:
    """
    Load graded data from CSV file.

    Args:
        csv_path: Path to graded exam CSV

    Returns:
        List of (question_name, responses_dict, points_dict) tuples where:
        - question_name: Name of the question column
        - responses_dict: Maps student ID to response text
        - points_dict: Maps student ID to points awarded
    """
    df = pd.read_csv(csv_path)

    # Find response and points columns
    response_cols = [col for col in df.columns if col.startswith("Response ")]
    points_cols = [col for col in df.columns if col.startswith("Points ")]

    questions_data = []

    for resp_col, pts_col in zip(response_cols, points_cols):
        # Extract question number/name
        question_name = resp_col.replace("Response ", "Q")

        # Build dictionaries for this question
        responses_dict = {}
        points_dict = {}

        for _, row in df.iterrows():
            student_id = str(row['AnonCode'])
            response = str(row[resp_col])
            points = row[pts_col]

            # Skip empty/missing responses
            if pd.isna(response) or response.strip() in ['-', '', 'nan']:
                continue
            if pd.isna(points):
                continue

            responses_dict[student_id] = response
            points_dict[student_id] = float(points)

        # Only include questions with graded responses
        if responses_dict and points_dict:
            questions_data.append((question_name, responses_dict, points_dict))

    return questions_data


def analyze_point_coverage(  # pylint: disable=too-many-locals
    selected_ids: List[str],
    points_dict: Dict[str, float],
    min_samples_per_class: int,
    min_samples_include: int,
) -> Tuple[Set[float], float, bool, Dict[float, int], Set[float]]:
    """
    Analyze how well a sample covers different point classes.

    Args:
        selected_ids: List of selected student IDs
        points_dict: Dictionary mapping all student IDs to points
        min_samples_per_class: Minimum samples required per class
        min_samples_include: Minimum total samples for a class to be included in criteria

    Returns:
        Tuple of (
            set of unique points in sample,
            coverage ratio (of included classes),
            meets_criteria (all included classes have min samples or all available),
            dict mapping point value to count in sample,
            set of excluded point values (classes with < min_samples_include)
        )
    """
    # Count all available samples per point class
    all_points_counter = Counter(points_dict.values())
    all_unique_points = set(all_points_counter.keys())

    # Filter classes based on min_samples_include
    included_classes = {
        pt for pt in all_unique_points
        if all_points_counter[pt] >= min_samples_include
    }
    excluded_classes = all_unique_points - included_classes

    # Count selected samples per point class
    selected_points_list = [points_dict[id_] for id_ in selected_ids if id_ in points_dict]
    selected_points_counter = Counter(selected_points_list)
    selected_unique_points = set(selected_points_counter.keys())

    # Calculate coverage ratio (only for included classes)
    coverage_ratio = (
        len(selected_unique_points & included_classes) / len(included_classes)
        if included_classes else 0.0
    )

    # Check if criteria is met: each INCLUDED class has min samples OR all available samples
    meets_criteria = True
    for point_value in included_classes:
        available_count = all_points_counter[point_value]
        selected_count = selected_points_counter.get(point_value, 0)
        required_count = min(min_samples_per_class, available_count)

        if selected_count < required_count:
            meets_criteria = False
            break

    return (
        selected_unique_points,
        coverage_ratio,
        meets_criteria,
        dict(selected_points_counter),
        excluded_classes
    )


async def test_strategy_with_sample_size(  # pylint: disable=too-many-arguments,too-many-positional-arguments
    strategy: SamplingAlgorithm,
    n_samples: int,
    embeddings_dict: Dict[str, List[float]],
    responses_dict: Dict[str, str],
    points_dict: Dict[str, float],
    question_text: str,
    min_samples_per_class: int,
) -> Optional[Tuple[float, bool, Dict[float, int], int, Set[float]]]:
    """
    Test a single strategy with a specific sample size.

    Args:
        strategy: Sampling strategy to test
        n_samples: Number of samples to select
        embeddings_dict: Embeddings for all responses
        responses_dict: Response text for all responses
        points_dict: Points for all responses
        question_text: Text of the question (for GPTSort)
        min_samples_per_class: Minimum samples required per class

    Returns:
        Tuple of (coverage, meets_criteria, class_counts, num_selected, excluded) or None
    """
    try:
        if strategy == SamplingAlgorithm.KMEANS_AUTO:
            selected_ids, _, num_selected = get_samples(
                embeddings_dict,
                algorithm=strategy,
            )
        elif strategy == SamplingAlgorithm.GPTSORT:
            # GPTSort requires text data and question text
            selected_ids, _, num_selected = get_samples(
                embeddings_dict,
                algorithm=strategy,
                n_samples=n_samples,
                text_data=responses_dict,
                question_text=question_text,
            )
        else:
            # Other strategies require n_samples
            selected_ids, _, num_selected = get_samples(
                embeddings_dict,
                algorithm=strategy,
                n_samples=n_samples,
            )

        # Analyze coverage
        _, coverage_ratio, meets_criteria, class_counts, excluded = analyze_point_coverage(
            selected_ids, points_dict, min_samples_per_class, MIN_SAMPLES_INCLUDE
        )
        return coverage_ratio, meets_criteria, class_counts, num_selected, excluded

    except Exception:  # pylint: disable=broad-except
        return None


async def test_sampling_for_question(  # pylint: disable=too-many-locals,too-many-branches,too-many-statements,too-many-nested-blocks
    csv_file: str,
    question_name: str,
    responses_dict: Dict[str, str],
    points_dict: Dict[str, float],
    min_samples_per_class: int,
) -> Dict[str, Dict[int, Tuple[float, bool, Dict[float, int], int]]]:
    """
    Test all sampling strategies with different sample sizes for a single question.

    Args:
        csv_file: Name of the CSV file being processed
        question_name: Name of the question
        responses_dict: Maps student ID to response text
        points_dict: Maps student ID to points awarded
        min_samples_per_class: Minimum samples required per class

    Returns:
        Dictionary mapping strategy name to {sample_size: (coverage, meets_criteria, counts, num)}
    """
    # Get question text and sample answer from config (if available)
    csv_questions = QUESTION_TEXTS.get(csv_file, {})
    question_text = csv_questions.get(question_name, question_name)
    csv_answers = SAMPLE_ANSWERS.get(csv_file, {})
    sample_answer = csv_answers.get(question_name, None)

    # Get feature vectors for all responses
    print(f"\nGetting feature vectors for {len(responses_dict)} responses...")
    if question_text != question_name or sample_answer:
        print(f"   Using question context: {question_text[:80]}...")
        if sample_answer:
            print(f"   Using sample answer: {sample_answer[:80]}...")
    embeddings_dict = await get_features_for_responses(
        responses_dict, question_text=question_text, sample_answer=sample_answer
    )
    print(f"✓ Got {len(embeddings_dict)} feature vectors")

    # Apply PCA dimensionality reduction (90% variance retention)
    embeddings_dict = optimize_features(embeddings_dict, variance_ratio=0.9)

    # Calculate unique points for coverage display
    unique_points = set(points_dict.values())

    results = {}

    # Test each strategy
    for strategy in STRATEGIES:
        print(f"\n--- Testing strategy: {strategy.value} ---")
        results[strategy.value] = {}

        # For KMEANS_AUTO, we don't control sample size
        if strategy == SamplingAlgorithm.KMEANS_AUTO:
            result = await test_strategy_with_sample_size(
                strategy, 0, embeddings_dict, responses_dict,
                points_dict, question_text, min_samples_per_class
            )
            if result:
                coverage_ratio, meets_criteria, class_counts, num_selected, excluded = result
                results[strategy.value][num_selected] = (
                    coverage_ratio, meets_criteria, class_counts, num_selected
                )
                marker = "✓" if meets_criteria else "✗"

                # Build per-class status string
                all_points_counter = Counter(points_dict.values())
                class_status_parts = []
                for pt in sorted(unique_points):
                    selected = class_counts.get(pt, 0)
                    available = all_points_counter[pt]
                    needed = min(min_samples_per_class, available)
                    if pt in excluded:
                        class_status_parts.append(f"{pt}:{selected}/{needed}⊗")
                    else:
                        class_marker = "✓" if selected >= needed else "✗"
                        class_status_parts.append(f"{pt}:{selected}/{needed}{class_marker}")
                class_status = ", ".join(class_status_parts)

                print(f"  {marker} Selected {num_selected} samples, "
                      f"Coverage: {coverage_ratio:.1%}")
                print(f"     Classes: [{class_status}]")
                if excluded:
                    print(f"     Excluded (< {MIN_SAMPLES_INCLUDE} total): {sorted(excluded)}")
        else:
            # Test with different sample sizes
            for n_samples in SAMPLE_SIZES:
                # Skip if sample size is larger than available responses
                if n_samples >= len(responses_dict):
                    continue

                print(f"  Testing with n={n_samples} samples...")
                result = await test_strategy_with_sample_size(
                    strategy, n_samples, embeddings_dict, responses_dict,
                    points_dict, question_text, min_samples_per_class
                )
                if result:
                    coverage_ratio, meets_criteria, class_counts, num_selected, excluded = result
                    results[strategy.value][n_samples] = (
                        coverage_ratio, meets_criteria, class_counts, num_selected
                    )
                    # Calculate included classes for correct display
                    all_points_counter = Counter(points_dict.values())
                    included_classes = {
                        pt for pt in unique_points
                        if all_points_counter[pt] >= MIN_SAMPLES_INCLUDE
                    }
                    n_covered = int(coverage_ratio * len(included_classes))
                    marker = "✓" if meets_criteria else "✗"

                    # Build per-class status string
                    class_status_parts = []
                    for pt in sorted(unique_points):
                        selected = class_counts.get(pt, 0)
                        available = all_points_counter[pt]
                        needed = min(min_samples_per_class, available)
                        if pt in excluded:
                            class_status_parts.append(f"{pt}:{selected}/{needed}⊗")
                        else:
                            class_marker = "✓" if selected >= needed else "✗"
                            class_status_parts.append(f"{pt}:{selected}/{needed}{class_marker}")
                    class_status = ", ".join(class_status_parts)

                    print(f"  {marker} Coverage: {coverage_ratio:.1%} "
                          f"({n_covered}/{len(included_classes)} included classes)")
                    print(f"     Classes: [{class_status}]")

    return results


def find_min_samples_for_target(
    results_by_size: Dict[int, Tuple[float, bool, Dict[float, int], int]]
) -> Optional[int]:
    """
    Find minimum sample size that meets the criteria.

    Args:
        results_by_size: Dictionary mapping sample_size to result tuple

    Returns:
        Minimum sample size meeting criteria, or None if criteria not met
    """
    for sample_size in sorted(results_by_size.keys()):
        _, meets_criteria, _, _ = results_by_size[sample_size]
        if meets_criteria:
            return sample_size
    return None


async def main() -> None:  # pylint: disable=too-many-locals,too-many-branches,too-many-statements
    """Main function to test sampling strategies on all graded exams."""
    print("="*80)
    print("SAMPLING STRATEGY EVALUATION - MINIMUM SAMPLES FINDER")
    print("="*80)
    print(f"\nTesting {len(STRATEGIES)} strategies")
    print(f"Sample sizes: {min(SAMPLE_SIZES)} to {max(SAMPLE_SIZES)}")
    print(f"Min samples per class: {MIN_SAMPLES_PER_CLASS}")
    print(f"Target coverage: {TARGET_COVERAGE:.0%}")
    print("\nStrategies:")
    for strategy in STRATEGIES:
        print(f"  - {strategy.value}")

    # Find all graded CSV files
    csv_files = [
        f for f in os.listdir(GRADED_EXAMS_DIR)
        if f.endswith('.csv')
    ]

    if not csv_files:
        print(f"\n⚠️  No CSV files found in {GRADED_EXAMS_DIR}/")
        return

    print(f"\nFound {len(csv_files)} graded exam files")

    all_results = []

    # Process each CSV file
    for csv_file in csv_files:
        csv_path = os.path.join(GRADED_EXAMS_DIR, csv_file)
        print(f"\n{'#'*80}")
        print(f"Processing: {csv_file}")
        print(f"{'#'*80}")
        questions_data = load_graded_data(csv_path)
        print(f"Loaded {len(questions_data)} questions with graded responses")

        # Test each question
        for question_name, responses_dict, points_dict in questions_data:
            print(f"\n{'='*80}")
            print(f"Testing question: {question_name}")
            print(f"{'='*80}")

            # Get point class info
            all_points = list(points_dict.values())
            unique_points = set(all_points)
            point_counts = Counter(all_points)

            print(f"Total responses: {len(responses_dict)}")
            print(f"Point classes: {len(unique_points)} - {sorted(unique_points)}")
            print(f"Distribution: {dict(sorted(point_counts.items()))}")

            results = await test_sampling_for_question(
                csv_file, question_name, responses_dict, points_dict, MIN_SAMPLES_PER_CLASS
            )
            all_results.append((
                csv_file, question_name, len(responses_dict),
                unique_points, point_counts, results
            ))

    print("\n\n" + "="*80)
    print("DETAILED RESULTS")
    print("="*80)

    # Print detailed results for each question
    for csv_file, q_name, n_resp, unique_pts, pt_counts, results in all_results:
        print(f"\n{csv_file} - {q_name}")
        print(f"  Total responses: {n_resp}")
        print(f"  Point classes: {len(unique_pts)} - {sorted(unique_pts)}")
        print(f"  Distribution: {dict(sorted(pt_counts.items()))}")

        for strategy_name, results_by_size in results.items():
            if not results_by_size:
                continue

            print(f"\n  {strategy_name}:")
            for sample_size in sorted(results_by_size.keys()):
                coverage, meets_criteria, class_counts, _ = results_by_size[sample_size]
                marker = "✓" if meets_criteria else " "
                # Show per-class counts
                counts_str = ", ".join(
                    f"{pt}:{class_counts.get(pt, 0)}/{pt_counts[pt]}"
                    for pt in sorted(unique_pts)
                )
                print(f"    {marker} n={sample_size:2d}: coverage={coverage:5.1%}, "
                      f"criteria={meets_criteria}")
                print(f"       Class counts: {counts_str}")

            # Find minimum for target
            min_samples = find_min_samples_for_target(results_by_size)
            if min_samples:
                print(f"    → Minimum to meet criteria: {min_samples} samples")
            else:
                print("    → Criteria not met at any tested sample size")

    # Print summary statistics
    print("\n" + "="*80)
    print("SUMMARY - MINIMUM SAMPLES NEEDED")
    print("="*80)

    # Aggregate minimum samples needed across all questions
    # Build dynamically from results to handle variant strategy names
    strategy_min_samples = {}
    strategy_coverages = {}

    for _, _, _, _, _, results in all_results:
        for strategy_name, results_by_size in results.items():
            if not results_by_size:
                continue

            # Initialize if not seen before
            if strategy_name not in strategy_min_samples:
                strategy_min_samples[strategy_name] = []
            if strategy_name not in strategy_coverages:
                strategy_coverages[strategy_name] = {size: [] for size in SAMPLE_SIZES}

            # Track minimum samples needed
            min_samples = find_min_samples_for_target(results_by_size)
            if min_samples:
                strategy_min_samples[strategy_name].append(min_samples)

            # Track coverage at each sample size
            for sample_size, (coverage, _, _, _) in results_by_size.items():
                if sample_size in SAMPLE_SIZES:
                    strategy_coverages[strategy_name][sample_size].append(coverage)

    print(f"\nMinimum samples to meet criteria "
          f"({MIN_SAMPLES_PER_CLASS} samples/class, {TARGET_COVERAGE:.0%} coverage):")
    for strategy_name in sorted(strategy_min_samples.keys()):
        min_samples = strategy_min_samples[strategy_name]

        if min_samples:
            avg_min = sum(min_samples) / len(min_samples)
            min_of_min = min(min_samples)
            max_of_min = max(min_samples)
            success_rate = len(min_samples) / len(all_results)
            print(f"  {strategy_name:20s}: avg={avg_min:5.1f}, "
                  f"min={min_of_min:2d}, max={max_of_min:2d} "
                  f"(achieved in {success_rate:.0%} of questions)")
        else:
            print(f"  {strategy_name:20s}: Criteria not met in any question")

    print("\nAverage coverage by strategy and sample size (selected sizes):")
    # Show subset of sample sizes for cleaner output
    display_sizes = [5, 10, 15, 20, 25, 30]
    display_sizes = [s for s in display_sizes if s in SAMPLE_SIZES]

    print(f"{'Strategy':<20s} ", end="")
    for size in display_sizes:
        print(f"n={size:2d}  ", end="")
    print()
    print("-" * 80)

    for strategy_name in sorted(strategy_coverages.keys()):
        print(f"{strategy_name:<20s} ", end="")

        for size in display_sizes:
            coverages = strategy_coverages[strategy_name][size]
            if coverages:
                avg_cov = sum(coverages) / len(coverages)
                print(f"{avg_cov:5.1%} ", end="")
            else:
                print("  -   ", end="")
        print()

    print("\n" + "="*80)
    print("EVALUATION COMPLETE")
    print("="*80)


if __name__ == "__main__":
    asyncio.run(main())
