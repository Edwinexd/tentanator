from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn import pipeline
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
from sklearn_extra.cluster import KMedoids

# Clusters data with KMedoids and returns a dict of 'id': cluster,
# medoid IDs, center of cluster, and the silhouette score of the clustering
def cluster_with_kmean(
    data: Dict[str, List[float]], k: int
) -> Tuple[Dict[str, int], List[str], np.ndarray, float]:
    """
    Cluster data using KMedoids algorithm.

    Args:
        data: Dictionary mapping IDs to feature vectors
        k: Number of clusters

    Returns:
        Tuple of (cluster assignments dict, medoid IDs list, cluster centers, silhouette score)
    """
    df = pd.DataFrame.from_dict(data, orient='index')

    pipe = pipeline.Pipeline(
        [('Scaler', StandardScaler()),
        ('Kmedoids', KMedoids(n_clusters=k, init='k-medoids++', random_state=42))]
    )
    pipe.fit(df)

    df['cluster'] = pipe.predict(df)
    scaled_data = pipe.named_steps['Scaler'].transform(
        df.drop(axis=1, columns='cluster')
    )
    score = silhouette_score(scaled_data, df['cluster'])

    # Get medoid indices and map back to original IDs
    medoid_indices = pipe.named_steps["Kmedoids"].medoid_indices_
    medoid_ids = [str(df.index[idx]) for idx in medoid_indices]

    centers_scaled = pipe.named_steps["Kmedoids"].cluster_centers_
    centers_orig = pipe.named_steps["Scaler"].inverse_transform(centers_scaled)

    return df['cluster'].to_dict(), medoid_ids, centers_orig, float(score)
