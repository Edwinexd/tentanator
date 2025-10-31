"""
Test script to visualize embeddings using dimensionality reduction.
"""
import asyncio
import json
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

from sampling import (
    SamplingAlgorithm, get_samples, gptsort_sampling,
    cluster_with_kmean, iforest_gmm_sampling
)


def visualize_embeddings_2d(  # pylint: disable=too-many-arguments,too-many-positional-arguments,too-many-locals
    data: Dict[str, List[float]],
    clusters: Optional[Dict[str, int]] = None,
    representative_ids: Optional[List[str]] = None,
    title: str = "Embedding Visualization",
    save_path: Optional[str] = None,
    show_plot: bool = True
) -> pd.DataFrame:
    """
    Reduce embeddings to 2D using PCA and create matplotlib visualization.

    Args:
        data: Dictionary mapping IDs to feature vectors
        clusters: Optional cluster assignments for each ID
        representative_ids: Optional list of IDs to highlight as representatives
        title: Plot title
        save_path: Optional path to save the plot image
        show_plot: Whether to display the plot (default: True)

    Returns:
        DataFrame with ID, PC1, PC2, and optional cluster columns
    """
    if not data:
        print("No data to visualize")
        return pd.DataFrame()

    # Convert to DataFrame
    df = pd.DataFrame.from_dict(data, orient='index')
    original_dim = df.shape[1]

    # Apply PCA to reduce to 2D
    pca = PCA(n_components=2)
    reduced = pca.fit_transform(df)

    # Create results DataFrame
    result_df = pd.DataFrame({
        'ID': df.index,
        'PC1': reduced[:, 0],
        'PC2': reduced[:, 1]
    })

    # Add cluster assignments if provided
    if clusters:
        result_df['cluster'] = result_df['ID'].map(clusters)

    # Print summary
    print("\n📊 Dimensionality Reduction Results:")
    print(f"   Original dimensions: {original_dim}")
    print("   Reduced to: 2D")
    print(f"   Total samples: {len(data)}")
    print(f"   Variance explained: PC1={pca.explained_variance_ratio_[0]:.1%}, "
          f"PC2={pca.explained_variance_ratio_[1]:.1%}, "
          f"Total={sum(pca.explained_variance_ratio_):.1%}")

    # Create the plot
    _, ax = plt.subplots(figsize=(12, 8))

    if clusters:
        # Plot with different colors for each cluster
        unique_clusters = sorted(result_df['cluster'].unique())
        colors = plt.cm.tab10(np.linspace(0, 1, len(unique_clusters)))  # pylint: disable=no-member

        for i, cluster_id in enumerate(unique_clusters):
            cluster_data = result_df[result_df['cluster'] == cluster_id]
            ax.scatter(
                cluster_data['PC1'],
                cluster_data['PC2'],
                c=[colors[i]],
                label=f'Cluster {cluster_id} (n={len(cluster_data)})',
                alpha=0.6,
                s=100,
                edgecolors='black',
                linewidth=0.5
            )

        # Print cluster statistics
        print("\n📦 Cluster Statistics:")
        cluster_stats = result_df.groupby('cluster').agg({
            'PC1': ['mean', 'std'],
            'PC2': ['mean', 'std'],
            'ID': 'count'
        }).round(4)
        cluster_stats.columns = ['PC1_mean', 'PC1_std', 'PC2_mean', 'PC2_std', 'count']
        print(cluster_stats.to_string())
    else:
        # Plot all points with same color
        ax.scatter(
            result_df['PC1'],
            result_df['PC2'],
            alpha=0.6,
            s=100,
            c='steelblue',
            edgecolors='black',
            linewidth=0.5,
            label=f'All samples (n={len(result_df)})'
        )

    # Highlight representative samples if provided
    if representative_ids:
        rep_data = result_df[result_df['ID'].isin(representative_ids)]
        ax.scatter(
            rep_data['PC1'],
            rep_data['PC2'],
            c='red',
            marker='*',
            s=500,
            edgecolors='darkred',
            linewidth=2,
            label=f'Representatives (n={len(rep_data)})',
            zorder=10
        )

        # Annotate representative points
        for _, row in rep_data.iterrows():
            ax.annotate(
                row['ID'],
                (row['PC1'], row['PC2']),
                xytext=(5, 5),
                textcoords='offset points',
                fontsize=8,
                fontweight='bold',
                color='darkred'
            )

        print(f"\n⭐ Representative samples: {representative_ids}")

    # Formatting
    ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)", fontsize=12)
    ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)", fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend(loc='best', framealpha=0.9)
    ax.grid(True, alpha=0.3)
    ax.axhline(y=0, color='k', linestyle='--', alpha=0.3, linewidth=0.5)
    ax.axvline(x=0, color='k', linestyle='--', alpha=0.3, linewidth=0.5)

    plt.tight_layout()

    # Save if path provided
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"\n💾 Plot saved to: {save_path}")

    # Show plot
    if show_plot:
        plt.show()
    else:
        plt.close()

    return result_df


def load_embeddings_from_session(session_file: str) -> dict:
    """
    Load embeddings from a saved session file.
    Returns dict of {question_name: {id: embedding_vector}}
    """
    with open(session_file, 'r', encoding='utf-8') as f:
        session_data = json.load(f)
    return session_data.get('embeddings_cache', {})


def get_session_name(session_file: Path) -> str:
    """Extract clean session name from file path."""
    # Remove .json extension
    return session_file.stem


def sanitize_filename(name: str) -> str:
    """Sanitize a string to be safe for use as a filename."""
    # Replace spaces and problematic characters with underscores
    return name.replace(' ', '_').replace('/', '_').replace('\\', '_')


def load_session_data(session_file: Path) -> dict:
    """Load complete session data."""
    with open(session_file, 'r', encoding='utf-8') as f:
        return json.load(f)


def load_csv_data(csv_path: str, id_columns: list) -> pd.DataFrame:
    """Load CSV file with response data."""
    try:
        df = pd.read_csv(csv_path)

        # Drop rows with NaN in ID columns
        df = df.dropna(subset=id_columns)

        # Convert ID column to string to match embedding keys
        if len(id_columns) == 1:
            # Convert to int first (to remove .0) then to string
            df[id_columns[0]] = df[id_columns[0]].astype(int).astype(str)
            df = df.set_index(id_columns[0])
        else:
            # For multiple columns, convert each to int then string
            for col in id_columns:
                df[col] = df[col].astype(int).astype(str)
            df['_combined_id'] = df[id_columns].agg('_'.join, axis=1)
            df = df.set_index('_combined_id')
        return df
    except (OSError, KeyError, ValueError) as e:
        print(f"  ⚠️  Failed to load CSV {csv_path}: {e}")
        return None


def save_representative_texts(  # pylint: disable=too-many-arguments,too-many-positional-arguments
    viz_dir: Path,
    filename_base: str,
    question_name: str,
    sample_ids: list,
    df: pd.DataFrame,
    algorithm_name: str
):
    """Save the actual text of representative samples to a file."""
    txt_path = viz_dir / f"{filename_base}.txt"

    with open(txt_path, 'w', encoding='utf-8') as f:
        f.write(f"Representative Samples - {algorithm_name}\n")
        f.write(f"Question: {question_name}\n")
        f.write(f"Total representatives: {len(sample_ids)}\n")
        f.write("="* 80 + "\n\n")

        for i, sample_id in enumerate(sample_ids, 1):
            f.write(f"[{i}] ID: {sample_id}\n")
            f.write("-" * 80 + "\n")

            if df is not None and sample_id in df.index:
                response_text = df.loc[sample_id, question_name]
                # Handle NaN values
                if pd.isna(response_text):
                    f.write("(No response)\n")
                else:
                    f.write(f"{response_text}\n")
            else:
                f.write("(Response text not found in CSV)\n")

            f.write("\n")


def main():  # pylint: disable=too-many-locals,too-many-branches,too-many-statements,too-many-nested-blocks
    """Main visualization test."""
    # Find the most recent session file
    sessions_dir = Path('.tentanator_sessions')

    if not sessions_dir.exists():
        print("No sessions directory found. Run tentanator.py first to create sessions.")
        return

    session_files = list(sessions_dir.glob('*.json'))

    if not session_files:
        print("No session files found. Run tentanator.py first.")
        return

    print(f"Found {len(session_files)} session file(s)\n")

    # Process ALL sessions
    for session_file in sorted(session_files, key=lambda p: p.stat().st_mtime, reverse=True):  # pylint: disable=too-many-nested-blocks
        print("=" * 80)
        print(f"Processing session: {session_file.name}")
        print("=" * 80)

        # Create viz directory structure for this session
        session_name = get_session_name(session_file)
        viz_dir = Path('viz') / session_name
        viz_dir.mkdir(parents=True, exist_ok=True)
        print(f"Saving visualizations to: {viz_dir}/")

        # Load session data
        session_data = load_session_data(session_file)
        embeddings_by_question = session_data.get('embeddings_cache', {})

        if not embeddings_by_question:
            print("No embeddings found in this session. Skipping...\n")
            continue

        print(f"Found embeddings for {len(embeddings_by_question)} questions")

        # Load CSV data for retrieving actual response texts
        csv_file = session_data.get('csv_file')
        id_columns = session_data.get('id_columns', [])
        df = None

        if csv_file and id_columns:
            # Try to find the CSV in common locations
            csv_paths_to_try = [
                Path(csv_file),
                Path('exams') / csv_file,
                Path('graded_exams') / csv_file
            ]

            for csv_path in csv_paths_to_try:
                if csv_path.exists():
                    df = load_csv_data(str(csv_path), id_columns)
                    if df is not None:
                        print(f"Loaded response data from: {csv_path}")
                        break

            if df is None:
                print(f"  ⚠️  Could not find CSV file: {csv_file}")

        # Process each question's embeddings
        for question_name, embeddings in embeddings_by_question.items():
            print("\n" + "-"*80)
            print(f"QUESTION: {question_name}")
            print("-"*80)
            print(f"Number of student responses: {len(embeddings)}")

            if len(embeddings) < 2:
                print("Not enough embeddings to visualize (need at least 2)")
                continue

            # Sanitize question name for filename
            safe_question_name = sanitize_filename(question_name)

            # First, just visualize the raw embeddings
            print("\n" + "  "*2 + "Raw Embeddings")
            _ = visualize_embeddings_2d(
                embeddings,
                title=f"{question_name} - Raw Embeddings",
                save_path=str(viz_dir / f"{safe_question_name}_raw.png"),
                show_plot=False
            )

            # Now run all sampling algorithms and visualize
            if len(embeddings) >= 3:  # Need at least 3 for meaningful clustering
                # Determine number of samples for fixed algorithms (25 or all if less)
                n_samples = min(25, len(embeddings))

                algorithms_to_test = [
                    (SamplingAlgorithm.KMEANS_AUTO, None, "KMeans Auto"),
                    (SamplingAlgorithm.KMEANS_FIXED, n_samples, f"KMeans Fixed (k={n_samples})"),
                    (SamplingAlgorithm.RANDOM, n_samples, f"Random (n={n_samples})"),
                    (SamplingAlgorithm.MAXIMIN, n_samples, f"Maximin (n={n_samples})"),
                    (SamplingAlgorithm.GPTSORT, n_samples, f"GPTSort (n={n_samples})"),
                    (SamplingAlgorithm.IFOREST_GMM, n_samples, f"IForest+GMM (n={n_samples})")
                ]

                for algo_name, n_param, display_name in algorithms_to_test:
                    print("\n" + "  "*2 + f"{display_name}")

                    try:
                        # Special handling for gptsort (uses text, not embeddings)
                        if algo_name == SamplingAlgorithm.GPTSORT:
                            if df is None:
                                print("  ⚠️  Skipping gptsort: CSV data not available")
                                continue

                            # Build text_data dict from DataFrame
                            text_data = {}
                            for emb_id in embeddings.keys():
                                if emb_id in df.index:
                                    response_text = df.loc[emb_id, question_name]
                                    # Handle NaN values
                                    if not pd.isna(response_text):
                                        text_data[emb_id] = str(response_text)
                                    else:
                                        text_data[emb_id] = ""

                            if not text_data:
                                print("  ⚠️  No text data found for gptsort")
                                continue

                            # Run gptsort asynchronously
                            sample_ids = asyncio.run(gptsort_sampling(
                                text_data,
                                question_text=question_name,
                                n_samples=n_param
                            ))
                            k = len(sample_ids)
                            clusters = None

                        else:
                            # Run sampling algorithm for embedding-based methods
                            if n_param is None:
                                sample_ids, _, k = get_samples(
                                    embeddings,
                                    algorithm=algo_name
                                )
                            else:
                                sample_ids, _, k = get_samples(
                                    embeddings,
                                    algorithm=algo_name,
                                    n_samples=n_param
                                )

                            # Get cluster assignments if using kmeans or iforest_gmm
                            clusters = None
                            if algo_name in (
                                SamplingAlgorithm.KMEANS_AUTO,
                                SamplingAlgorithm.KMEANS_FIXED
                            ):
                                clusters, _, _, _ = cluster_with_kmean(embeddings, k)
                            elif algo_name == SamplingAlgorithm.IFOREST_GMM:
                                _, clusters, _ = iforest_gmm_sampling(embeddings, n_param)

                        # Visualize with representatives
                        _ = visualize_embeddings_2d(
                            embeddings,
                            clusters=clusters,
                            representative_ids=sample_ids,
                            title=f"{question_name} - {display_name}",
                            save_path=str(viz_dir / f"{safe_question_name}_{algo_name.value}.png"),
                            show_plot=False
                        )

                        # Save representative sample texts
                        save_representative_texts(
                            viz_dir=viz_dir,
                            filename_base=f"{safe_question_name}_{algo_name.value}",
                            question_name=question_name,
                            sample_ids=sample_ids,
                            df=df,
                            algorithm_name=display_name
                        )

                    except (ValueError, RuntimeError, OSError, KeyError, IndexError) as e:
                        print(f"  ⚠️  Failed to run {algo_name.value}: {e}")

        print()  # Blank line between sessions

    print("\n" + "="*80)
    print("Visualization complete!")
    print("="*80)


if __name__ == '__main__':
    main()
