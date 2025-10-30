from typing import Tuple
import numpy as np
import pandas as pd
from sklearn import pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

#In theory clusters with Kmean and return a dict of 'id' : cluster, center of cluster and the silhouette score of the clustering
def cluster_with_kmean(data : dict, k : int) -> Tuple[dict, np.array, float]:
    
    df = pd.DataFrame.from_dict(data, orient = 'index')

    pipe = pipeline(
        [('Scaler', StandardScaler()),
        'Kmeans',KMeans(n_clusters = k, init = 'k-means++') ]
    )
    pipe.fit(df)
    
    df['cluster']= pipe.predict(df)
    score = silhouette_score(pipe.named_steps['Scaler'].transform(df.drop(axis = 1,columns='cluster')), df['cluster'] )

    centers_scaled = pipe.named_steps["Kmeans"].cluster_centers_
    centers_orig = pipe.named_steps["Scaler"].inverse_transform(centers_scaled)


    return df['cluster'].to_dict(), centers_orig, score
