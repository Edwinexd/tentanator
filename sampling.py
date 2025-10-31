"""
Sampling algorithms for selecting representative samples from student responses.

Includes clustering-based (KMeans), diversity-based (maximin), random sampling,
GPT-based quality sorting, and outlier-aware clustering approaches.
"""
import os
from typing import Dict, List, Tuple, Optional
from enum import Enum
import asyncio

import numpy as np
import pandas as pd
from sklearn import pipeline
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
from sklearn.mixture import GaussianMixture
from openai import AsyncOpenAI, APIError, APIConnectionError
import dotenv

dotenv.load_dotenv()


class SamplingAlgorithm(str, Enum):
    """Enum for sampling algorithm selection."""
    KMEANS_AUTO = "kmeans_auto"
    KMEANS_FIXED = "kmeans_fixed"
    RANDOM = "random"
    MAXIMIN = "maximin"
    GPTSORT = "gptsort"
    IFOREST_GMM = "iforest_gmm"

# Initialize OpenAI client forf GPTSort algorithm
openai_client = AsyncOpenAI()


def find_optimal_k(
    data: Dict[str, List[float]], min_k: int = 2, max_k: Optional[int] = None
) -> Tuple[int, float]:
    """
    Find the optimal number of clusters (k) that maximizes the silhouette score.

    Args:
        data: Dictionary mapping IDs to feature vectors
        min_k: Minimum number of clusters to try (default: 2)
        max_k: Maximum number of clusters to try (default: sqrt(n) or n//2, whichever is smaller)

    Returns:
        Tuple of (optimal k, best silhouette score)
    """
    n_samples = len(data)

    if n_samples < 2:
        return 1, 0.0

    # Set max_k if not provided - try up to 20 clusters (or n_samples - 1 if fewer samples)
    if max_k is None:
        max_k = min(n_samples - 1, 20)
    else:
        # Ensure max_k doesn't exceed n_samples - 1
        max_k = min(max_k, n_samples - 1)

    # Ensure min_k is valid
    min_k = max(2, min(min_k, max_k))

    if min_k > max_k:
        # Not enough samples for clustering
        return min(n_samples, 1), 0.0

    best_k = min_k
    best_score = -1.0

    print(f"\n🔍 Finding optimal number of clusters (trying k={min_k} to k={max_k})...")

    for k in range(min_k, max_k + 1):
        try:
            _, _, _, score = cluster_with_kmean(data, k)
            print(f"   k={k:2d}: silhouette score = {score:.3f}")

            if score > best_score:
                best_score = score
                best_k = k
        except (ValueError, RuntimeError) as e:
            print(f"   k={k:2d}: failed ({e})")
            continue

    print(f"✓ Optimal k={best_k} with silhouette score={best_score:.3f}")
    return best_k, best_score


def cluster_with_kmean(
    data: Dict[str, List[float]], k: int
) -> Tuple[Dict[str, int], List[str], np.ndarray, float]:
    """
    Cluster data using KMeans algorithm.

    Returns dict of 'id': cluster, representative IDs (closest to centers),
    center of cluster, and the silhouette score of the clustering.

    Args:
        data: Dictionary mapping IDs to feature vectors
        k: Number of clusters

    Returns:
        Tuple of (cluster assignments dict, representative IDs list,
                  cluster centers, silhouette score)
    """
    df = pd.DataFrame.from_dict(data, orient='index')

    pipe = pipeline.Pipeline(
        [('Scaler', StandardScaler()),
        ('KMeans', KMeans(n_clusters=k, init='k-means++', random_state=42, n_init=10))]
    )
    pipe.fit(df)

    df['cluster'] = pipe.predict(df)
    scaled_data = pipe.named_steps['Scaler'].transform(
        df.drop(axis=1, columns='cluster')
    )
    score = silhouette_score(scaled_data, df['cluster'])

    # Find closest data points to cluster centers (medoid-like IDs)
    centers_scaled = pipe.named_steps["KMeans"].cluster_centers_
    medoid_indices = []
    for center in centers_scaled:
        distances = np.linalg.norm(scaled_data - center, axis=1)
        medoid_indices.append(np.argmin(distances))
    medoid_ids = [str(df.index[idx]) for idx in medoid_indices]

    centers_orig = pipe.named_steps["Scaler"].inverse_transform(centers_scaled)

    return df['cluster'].to_dict(), medoid_ids, centers_orig, float(score)


def maximin_sampling(  # pylint: disable=too-many-locals
    data: Dict[str, List[float]], n_samples: int
) -> List[str]:
    """
    Select samples using maximin sampling strategy.
    Iteratively selects the point that is farthest from already selected points.

    Args:
        data: Dictionary mapping IDs to feature vectors
        n_samples: Number of samples to select

    Returns:
        List of selected sample IDs
    """
    all_ids = list(data.keys())
    n_total = len(all_ids)

    if n_samples >= n_total:
        return all_ids

    # Convert to numpy array for efficient distance computation
    embeddings_array = np.array([data[id_] for id_ in all_ids])

    # Normalize embeddings for better distance computation
    scaler = StandardScaler()
    embeddings_scaled = scaler.fit_transform(embeddings_array)

    selected_indices = []

    # Start with the point closest to the centroid
    centroid = np.mean(embeddings_scaled, axis=0)
    distances_to_centroid = np.linalg.norm(
        embeddings_scaled - centroid, axis=1
    )
    first_idx = np.argmin(distances_to_centroid)
    selected_indices.append(first_idx)

    # Iteratively select points that maximize minimum distance to selected points
    for _ in range(n_samples - 1):
        # Compute minimum distance from each point to any selected point
        min_distances = np.full(n_total, np.inf)

        for idx in range(n_total):
            if idx in selected_indices:
                min_distances[idx] = -np.inf  # Already selected
                continue

            # Find minimum distance to any selected point
            distances = np.linalg.norm(
                embeddings_scaled[idx] - embeddings_scaled[selected_indices],
                axis=1
            )
            min_distances[idx] = np.min(distances)

        # Select the point with maximum minimum distance
        next_idx = np.argmax(min_distances)
        selected_indices.append(next_idx)

    # Convert indices back to IDs
    selected_ids = [all_ids[idx] for idx in selected_indices]

    return selected_ids


def iforest_gmm_sampling(  # pylint: disable=too-many-locals,too-many-branches,too-many-statements
    data: Dict[str, List[float]], n_samples: int, contamination: float = 0.15
) -> Tuple[List[str], Dict[str, int], np.ndarray]:
    """
    Select samples using Isolation Forest outlier detection + Gaussian Mixture Model clustering.

    First identifies outliers using Isolation Forest, then clusters normal points with GMM.
    Returns representatives from both outliers and clusters.

    Args:
        data: Dictionary mapping IDs to feature vectors
        n_samples: Number of samples to select
        contamination: Expected proportion of outliers (default: 0.15)

    Returns:
        Tuple of (list of selected sample IDs, cluster assignments dict, GMM centers)
    """
    all_ids = list(data.keys())
    n_total = len(all_ids)

    if n_samples >= n_total:
        # Return all with dummy cluster assignments
        return all_ids, {id_: 0 for id_ in all_ids}, np.array([])

    if n_total < 2:
        return all_ids, {all_ids[0]: 0} if all_ids else {}, np.array([])

    # Convert to numpy array
    embeddings_array = np.array([data[id_] for id_ in all_ids])

    # Normalize embeddings
    scaler = StandardScaler()
    embeddings_scaled = scaler.fit_transform(embeddings_array)

    print(f"\n🔍 Isolation Forest + GMM sampling for {n_samples} samples...")

    # Step 1: Identify outliers using Isolation Forest
    iso_forest = IsolationForest(contamination=contamination, random_state=42)
    outlier_labels = iso_forest.fit_predict(embeddings_scaled)

    outlier_indices = np.where(outlier_labels == -1)[0]
    normal_indices = np.where(outlier_labels == 1)[0]

    n_outliers = len(outlier_indices)
    n_normal = len(normal_indices)

    print(f"   Found {n_outliers} outliers ({n_outliers/n_total:.1%}) and {n_normal} normal points")

    # Determine how many samples to take from outliers vs clusters
    # Allocate proportionally to ensure outliers are represented
    outlier_samples = min(n_outliers, max(1, int(n_samples * contamination)))
    cluster_samples = n_samples - outlier_samples

    print(f"   Selecting {outlier_samples} outliers and {cluster_samples} from clusters")

    selected_indices = []
    cluster_assignments = {}

    # Select outliers (choose most extreme ones)
    if n_outliers > 0 and outlier_samples > 0:
        # Get outlier scores and select the most anomalous ones
        outlier_scores = iso_forest.score_samples(embeddings_scaled[outlier_indices])
        # Lower (more negative) scores = more anomalous
        most_anomalous_indices = np.argsort(outlier_scores)[:outlier_samples]
        selected_outlier_indices = outlier_indices[most_anomalous_indices]
        selected_indices.extend(selected_outlier_indices)

        # Assign outliers to cluster -1
        for idx in outlier_indices:
            cluster_assignments[all_ids[idx]] = -1

        print(f"   ✓ Selected {len(selected_outlier_indices)} most anomalous outliers")

    # Step 2: Cluster normal points with GMM
    if n_normal > 1 and cluster_samples > 0:
        normal_embeddings = embeddings_scaled[normal_indices]

        # Determine optimal number of GMM components
        n_components = min(cluster_samples, n_normal, 10)

        # Use BIC to find optimal number of components if we have enough data
        if n_normal >= 10 and n_components > 2:
            bic_scores = []
            test_components = range(1, min(n_components + 1, n_normal))

            for n_comp in test_components:
                gmm = GaussianMixture(
                    n_components=n_comp,
                    covariance_type='full',
                    random_state=42,
                    max_iter=100
                )
                gmm.fit(normal_embeddings)
                bic_scores.append(gmm.bic(normal_embeddings))

            # Lower BIC is better
            optimal_n_components = list(test_components)[np.argmin(bic_scores)]
            print(f"   BIC optimization selected {optimal_n_components} components")
        else:
            optimal_n_components = n_components

        # Fit GMM with optimal components
        gmm = GaussianMixture(
            n_components=optimal_n_components,
            covariance_type='full',
            random_state=42,
            max_iter=100
        )
        gmm.fit(normal_embeddings)
        gmm_labels = gmm.predict(normal_embeddings)

        # Assign cluster labels to normal points
        for i, idx in enumerate(normal_indices):
            cluster_assignments[all_ids[idx]] = int(gmm_labels[i])

        # Select representatives from each GMM cluster
        # Distribute cluster_samples across clusters
        samples_per_cluster = cluster_samples // optimal_n_components
        remainder = cluster_samples % optimal_n_components

        cluster_representatives = []
        for cluster_id in range(optimal_n_components):
            cluster_mask = gmm_labels == cluster_id
            cluster_member_indices = normal_indices[cluster_mask]

            if len(cluster_member_indices) > 0:
                # Calculate how many samples to take from this cluster
                n_from_cluster = samples_per_cluster + (1 if cluster_id < remainder else 0)
                n_from_cluster = min(n_from_cluster, len(cluster_member_indices))

                # Find points distributed from center to ring (periphery)
                cluster_embeddings = embeddings_scaled[cluster_member_indices]
                cluster_mean = np.mean(cluster_embeddings, axis=0)
                distances = np.linalg.norm(cluster_embeddings - cluster_mean, axis=1)

                # Sort by distance from center
                sorted_by_distance = np.argsort(distances)

                if n_from_cluster == 1:
                    # Just take the center point
                    selected_from_cluster = [cluster_member_indices[sorted_by_distance[0]]]
                else:
                    # Take center + evenly distributed points across the distance spectrum
                    # This captures center + ring (periphery) of the cluster
                    step = len(sorted_by_distance) / n_from_cluster
                    selected_indices = [int(i * step) for i in range(n_from_cluster)]
                    selected_from_cluster = [cluster_member_indices[sorted_by_distance[i]]
                                           for i in selected_indices]

                cluster_representatives.extend(selected_from_cluster)

        # Take exactly cluster_samples representatives (in case of rounding)
        selected_cluster_indices = cluster_representatives[:cluster_samples]
        selected_indices.extend(selected_cluster_indices)

        print(f"   ✓ Selected {len(selected_cluster_indices)} representatives from "
              f"{optimal_n_components} GMM clusters")

        # Get GMM centers (in original space)
        centers_scaled = gmm.means_
        centers_orig = scaler.inverse_transform(centers_scaled)

    elif n_normal == 1 and cluster_samples > 0:
        # Only one normal point, just select it
        selected_indices.append(normal_indices[0])
        cluster_assignments[all_ids[normal_indices[0]]] = 0
        centers_orig = np.array([])
    else:
        centers_orig = np.array([])

    # Convert indices to IDs
    selected_ids = [all_ids[idx] for idx in selected_indices]

    print(f"✓ IForest+GMM complete: selected {len(selected_ids)} samples "
          f"({outlier_samples} outliers + {len(selected_ids) - outlier_samples} from clusters)")

    return selected_ids, cluster_assignments, centers_orig


async def _gpt_sort_chunk(
    responses: List[Tuple[str, str]],
    question_text: str,
) -> List[str]:
    """
    Sort a chunk of responses by quality using ChatGPT.

    Args:
        responses: List of (id, response_text) tuples
        question_text: The question being graded

    Returns:
        List of IDs sorted from best to worst quality
    """
    if len(responses) <= 1:
        return [r[0] for r in responses]

    if os.environ.get("DISABLE_GPTSORT", "0") == "1":
        print("⚠️  GPTSort is disabled via DISABLE_GPTSORT environment variable.")
        return [r[0] for r in responses]

    # Create prompt for sorting
    response_list = "\n\n".join([
        f"[{i+1}] ID: {id_}\n{text}"
        for i, (id_, text) in enumerate(responses)
    ])

    prompt = f"""You are grading exam responses. Sort the following responses by \
quality from BEST to WORST.

Question: {question_text}

Responses to sort:
{response_list}

Return ONLY a comma-separated list of the response numbers in order from best to worst.
For example: 3,1,5,2,4

Your ranking (numbers only, comma-separated):"""

    try:
        response = await openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=100,
        )

        # Parse the response
        ranking_text = (response.choices[0].message.content or "").strip()

        # Extract numbers from the response, filtering out empty strings
        ranking_numbers = []
        for part in ranking_text.split(","):
            part = part.strip()
            if part:  # Skip empty strings
                try:
                    num = int(part)
                    # Validate the number is in valid range
                    if 1 <= num <= len(responses):
                        ranking_numbers.append(num)
                except ValueError:
                    # Skip non-numeric parts
                    continue

        # If we didn't get all numbers, fill in missing ones
        if len(ranking_numbers) != len(responses):
            all_nums = set(range(1, len(responses) + 1))
            missing = all_nums - set(ranking_numbers)
            ranking_numbers.extend(sorted(missing))

        # Convert to IDs (1-indexed to 0-indexed)
        sorted_ids = [responses[num - 1][0] for num in ranking_numbers]

        return sorted_ids

    except (APIError, APIConnectionError) as e:
        print(f"⚠️  OpenAI API error sorting chunk: {e}")
        # Fallback to original order
        return [r[0] for r in responses]
    except (AttributeError, IndexError, KeyError) as e:
        print(f"⚠️  Error parsing API response: {e}")
        # Fallback to original order
        return [r[0] for r in responses]


def _select_representative_from_sorted(
    sorted_ids: List[str],
    n_samples: int
) -> List[str]:
    """
    Select representative samples from a sorted list (best to worst).

    Selects evenly from top third, middle third, and bottom third to get
    diverse quality representation.

    Args:
        sorted_ids: List of IDs sorted from best to worst quality
        n_samples: Number of samples to select

    Returns:
        List of selected IDs maintaining order from best to worst
    """
    n_total = len(sorted_ids)

    if n_total <= n_samples:
        return sorted_ids

    # Divide into thirds
    samples_per_third = n_samples // 3
    remainder = n_samples % 3

    # Allocate remainder to top third first, then middle, then bottom
    top_n = samples_per_third + (1 if remainder > 0 else 0)
    mid_n = samples_per_third + (1 if remainder > 1 else 0)
    bot_n = samples_per_third

    # Define thirds
    third_size = n_total // 3
    top_third = sorted_ids[:third_size]
    mid_third = sorted_ids[third_size:2*third_size]
    bot_third = sorted_ids[2*third_size:]

    # Select evenly from each third
    selected = []

    # Top third - take first top_n samples (best quality)
    selected.extend(top_third[:top_n])

    # Middle third - take evenly spaced samples
    if mid_n > 0 and len(mid_third) > 0:
        step = len(mid_third) / mid_n
        mid_indices = [int(i * step) for i in range(mid_n)]
        selected.extend([mid_third[i] for i in mid_indices])

    # Bottom third - take last bot_n samples (worst quality)
    if bot_n > 0 and len(bot_third) > 0:
        selected.extend(bot_third[-bot_n:])

    return selected


async def _gpt_merge_sorted(
    list1: List[Tuple[str, str]],
    list2: List[Tuple[str, str]],
    question_text: str,
) -> List[str]:
    """
    Merge two sorted lists of responses using ChatGPT to compare.

    Args:
        list1: First sorted list of (id, response_text) tuples
        list2: Second sorted list of (id, response_text) tuples
        question_text: The question being graded

    Returns:
        List of IDs from merged and sorted lists
    """
    # Combine and sort using GPT
    combined = list1 + list2
    return await _gpt_sort_chunk(combined, question_text)


async def gptsort_sampling(  # pylint: disable=too-many-locals
    data: Dict[str, str],
    question_text: str,
    chunk_size: int = 25,
    n_samples: Optional[int] = None,
) -> List[str]:
    """
    Sort responses by quality using GPT-based sorting with chunk merging.

    This approach:
    1. Splits responses into chunks of chunk_size
    2. Asks ChatGPT to sort each chunk by quality
    3. Successively merges sorted chunks to get final ranking
    4. Selects representative samples from top/middle/bottom thirds

    Args:
        data: Dictionary mapping IDs to response text (NOT embeddings)
        question_text: The question being graded
        chunk_size: Size of chunks to sort independently (default: 25)
        n_samples: Number of representative samples to return (default: all)
                   Samples are selected evenly from top, middle, and bottom thirds

    Returns:
        List of IDs representing quality distribution (best to worst within each third)
    """
    all_ids = list(data.keys())
    n_total = len(all_ids)

    if n_total == 0:
        return []

    if n_total == 1:
        return all_ids

    print(f"\n🤖 GPTSort: Sorting {n_total} responses...")

    # Step 1: Split into chunks
    chunks: List[List[Tuple[str, str]]] = []
    for i in range(0, n_total, chunk_size):
        chunk_ids = all_ids[i:i + chunk_size]
        chunk = [(id_, data[id_]) for id_ in chunk_ids]
        chunks.append(chunk)

    print(f"   Split into {len(chunks)} chunks of size ~{chunk_size}")

    # Step 2: Sort each chunk
    print(f"   Sorting {len(chunks)} chunks...")
    sorted_chunk_ids = await asyncio.gather(*[
        _gpt_sort_chunk(chunk, question_text)
        for chunk in chunks
    ])

    # Convert back to list of (id, text) tuples
    sorted_chunks = [
        [(id_, data[id_]) for id_ in chunk_ids]
        for chunk_ids in sorted_chunk_ids
    ]

    print("   ✓ All chunks sorted")

    # Step 3: Successively merge chunks
    print(f"   Merging {len(sorted_chunks)} sorted chunks...")
    while len(sorted_chunks) > 1:
        merged_chunks = []

        # Merge pairs of chunks
        for i in range(0, len(sorted_chunks), 2):
            if i + 1 < len(sorted_chunks):
                # Merge two chunks
                merged_ids = await _gpt_merge_sorted(
                    sorted_chunks[i],
                    sorted_chunks[i + 1],
                    question_text
                )
                merged_chunk = [(id_, data[id_]) for id_ in merged_ids]
                merged_chunks.append(merged_chunk)
            else:
                # Odd chunk, carry forward
                merged_chunks.append(sorted_chunks[i])

        sorted_chunks = merged_chunks
        print(f"   Merged to {len(sorted_chunks)} chunk(s)")

    # Extract final sorted IDs
    final_sorted_ids = [id_ for id_, _ in sorted_chunks[0]]

    # Select representative samples from top/middle/bottom if n_samples specified
    if n_samples is not None and n_samples < len(final_sorted_ids):
        result_ids = _select_representative_from_sorted(final_sorted_ids, n_samples)
        print(f"✓ GPTSort complete: selected {len(result_ids)} representative samples")
        print(f"   (from top/middle/bottom thirds of {len(final_sorted_ids)} sorted responses)")
        return result_ids

    print(f"✓ GPTSort complete: returning all {len(final_sorted_ids)} samples")
    return final_sorted_ids


def get_samples(  # pylint: disable=too-many-return-statements,too-many-branches
    data: Dict[str, List[float]],
    algorithm: SamplingAlgorithm = SamplingAlgorithm.KMEANS_AUTO,
    n_samples: Optional[int] = None,
    text_data: Optional[Dict[str, str]] = None,
    question_text: Optional[str] = None,
) -> Tuple[List[str], float, int]:
    """
    Select representative samples from data using specified algorithm.

    Args:
        data: Dictionary mapping IDs to feature vectors (embeddings)
        algorithm: Sampling algorithm to use:
            - "kmeans_auto": KMeans with automatic k optimization (default)
            - "kmeans_fixed": KMeans with fixed k (requires n_samples)
            - "random": Random sampling (requires n_samples)
            - "maximin": Maximin diversity sampling (requires n_samples)
            - "gptsort": GPT-based quality sorting (requires text_data and question_text)
            - "iforest_gmm": Isolation Forest + GMM (requires n_samples)
        n_samples: Number of samples to select (required for some algorithms)
        text_data: Dictionary mapping IDs to response text (required for gptsort)
        question_text: Question text for grading context (required for gptsort)

    Returns:
        Tuple of (list of selected sample IDs, quality score, number of samples selected)
    """
    n_total = len(data)

    if n_total == 0:
        return [], 0.0, 0

    if n_total == 1:
        return list(data.keys()), 0.0, 1

    if algorithm == SamplingAlgorithm.KMEANS_AUTO:
        # Automatically find optimal k
        optimal_k, _ = find_optimal_k(data)
        _, medoid_ids, _, silhouette = cluster_with_kmean(data, optimal_k)
        print(f"✓ KMeans auto-optimization selected k={optimal_k}")
        return medoid_ids, silhouette, optimal_k

    if algorithm == SamplingAlgorithm.KMEANS_FIXED:
        # Use fixed k
        if n_samples is None:
            raise ValueError("n_samples is required for kmeans_fixed algorithm")

        if n_samples > n_total:
            print(f"⚠️  Requested {n_samples} samples but only {n_total} available")
            return list(data.keys()), 0.0, n_total

        if n_samples < 2:
            # Just return first sample
            return [list(data.keys())[0]], 0.0, 1

        print(f"\n🔬 KMeans clustering with fixed k={n_samples}...")
        _, medoid_ids, _, silhouette = cluster_with_kmean(data, n_samples)
        print(f"✓ Clustering complete with silhouette score: {silhouette:.3f}")
        return medoid_ids, silhouette, n_samples

    if algorithm == SamplingAlgorithm.RANDOM:
        # Random sampling
        if n_samples is None:
            raise ValueError("n_samples is required for random algorithm")

        n_samples = min(n_samples, n_total)
        all_ids = list(data.keys())

        # Use numpy for reproducible random sampling
        rng = np.random.default_rng(42)
        selected_indices = rng.choice(len(all_ids), size=n_samples, replace=False)
        selected_ids = [all_ids[i] for i in selected_indices]

        print(f"✓ Random sampling selected {n_samples} samples")
        return selected_ids, 0.0, n_samples  # No quality score for random

    if algorithm == SamplingAlgorithm.MAXIMIN:
        # Maximin diversity sampling
        if n_samples is None:
            raise ValueError("n_samples is required for maximin algorithm")

        n_samples = min(n_samples, n_total)

        print(f"\n🎯 Maximin diversity sampling for {n_samples} samples...")
        selected_ids = maximin_sampling(data, n_samples)
        print(f"✓ Maximin sampling selected {n_samples} diverse samples")
        return selected_ids, 0.0, n_samples  # No quality score for maximin

    if algorithm == SamplingAlgorithm.GPTSORT:
        # GPT-based quality sorting
        if text_data is None:
            raise ValueError("text_data is required for gptsort algorithm")
        if question_text is None:
            raise ValueError("question_text is required for gptsort algorithm")

        # Run async function synchronously
        selected_ids = asyncio.run(
            gptsort_sampling(text_data, question_text, n_samples=n_samples)
        )
        actual_count = len(selected_ids)
        print(f"✓ GPTSort selected {actual_count} samples")
        return selected_ids, 0.0, actual_count  # No quality score for GPTSort

    if algorithm == SamplingAlgorithm.IFOREST_GMM:
        # Isolation Forest + GMM sampling
        if n_samples is None:
            raise ValueError("n_samples is required for iforest_gmm algorithm")

        n_samples = min(n_samples, n_total)

        selected_ids, _, _ = iforest_gmm_sampling(data, n_samples)
        return selected_ids, 0.0, len(selected_ids)  # No quality score for IForest+GMM

    raise ValueError(f"Unknown algorithm: {algorithm}")
