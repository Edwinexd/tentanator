from typing import Dict, List, Tuple, Optional, Literal

import numpy as np
import pandas as pd
from sklearn import pipeline
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

# Type for sampling algorithm selection
SamplingAlgorithm = Literal["kmeans_auto", "kmeans_fixed", "random", "maximin"]


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
        except Exception as e:
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


def maximin_sampling(
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


def get_samples(
    data: Dict[str, List[float]],
    algorithm: SamplingAlgorithm = "kmeans_auto",
    n_samples: Optional[int] = None,
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
        n_samples: Number of samples to select (required for some algorithms)

    Returns:
        Tuple of (list of selected sample IDs, quality score, number of samples selected)
    """
    n_total = len(data)

    if n_total == 0:
        return [], 0.0, 0

    if n_total == 1:
        return list(data.keys()), 0.0, 1

    if algorithm == "kmeans_auto":
        # Automatically find optimal k
        optimal_k, _ = find_optimal_k(data)
        _, medoid_ids, _, silhouette = cluster_with_kmean(data, optimal_k)
        print(f"✓ KMeans auto-optimization selected k={optimal_k}")
        return medoid_ids, silhouette, optimal_k

    if algorithm == "kmeans_fixed":
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

    if algorithm == "random":
        # Random sampling
        if n_samples is None:
            raise ValueError("n_samples is required for random algorithm")

        n_samples = min(n_samples, n_total)
        all_ids = list(data.keys())

        # Use numpy for reproducible random sampling
        rng = np.random.RandomState(42)
        selected_indices = rng.choice(len(all_ids), size=n_samples, replace=False)
        selected_ids = [all_ids[i] for i in selected_indices]

        print(f"✓ Random sampling selected {n_samples} samples")
        return selected_ids, 0.0, n_samples  # No quality score for random

    if algorithm == "maximin":
        # Maximin diversity sampling
        if n_samples is None:
            raise ValueError("n_samples is required for maximin algorithm")

        n_samples = min(n_samples, n_total)

        print(f"\n🎯 Maximin diversity sampling for {n_samples} samples...")
        selected_ids = maximin_sampling(data, n_samples)
        print(f"✓ Maximin sampling selected {n_samples} diverse samples")
        return selected_ids, 0.0, n_samples  # No quality score for maximin

    raise ValueError(f"Unknown algorithm: {algorithm}")
