"""
Test script to visualize embeddings using dimensionality reduction.
"""
import json
from pathlib import Path
import pandas as pd
from clustering import visualize_embeddings_2d, get_samples

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
    except Exception as e:
        print(f"  ⚠️  Failed to load CSV {csv_path}: {e}")
        return None


def save_representative_texts(
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


def main():
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
    for session_file in sorted(session_files, key=lambda p: p.stat().st_mtime, reverse=True):
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
            viz_df = visualize_embeddings_2d(
                embeddings,
                title=f"{question_name} - Raw Embeddings",
                save_path=str(viz_dir / f"{safe_question_name}_raw.png"),
                show_plot=False
            )

            # Now run all sampling algorithms and visualize
            if len(embeddings) >= 3:  # Need at least 3 for meaningful clustering
                from clustering import cluster_with_kmean

                # Determine number of samples for fixed algorithms (25 or all if less)
                n_samples = min(25, len(embeddings))

                algorithms_to_test = [
                    ("kmeans_auto", None, "KMeans Auto"),
                    ("kmeans_fixed", n_samples, f"KMeans Fixed (k={n_samples})"),
                    ("random", n_samples, f"Random (n={n_samples})"),
                    ("maximin", n_samples, f"Maximin (n={n_samples})")
                ]

                for algo_name, n_param, display_name in algorithms_to_test:
                    print("\n" + "  "*2 + f"{display_name}")

                    try:
                        # Run sampling algorithm
                        if n_param is None:
                            sample_ids, quality_score, k = get_samples(
                                embeddings,
                                algorithm=algo_name
                            )
                        else:
                            sample_ids, quality_score, k = get_samples(
                                embeddings,
                                algorithm=algo_name,
                                n_samples=n_param
                            )

                        # Get cluster assignments if using kmeans
                        clusters = None
                        if algo_name.startswith("kmeans"):
                            clusters, _, _, _ = cluster_with_kmean(embeddings, k)

                        # Visualize with representatives
                        viz_df_algo = visualize_embeddings_2d(
                            embeddings,
                            clusters=clusters,
                            representative_ids=sample_ids,
                            title=f"{question_name} - {display_name}",
                            save_path=str(viz_dir / f"{safe_question_name}_{algo_name}.png"),
                            show_plot=False
                        )

                        # Save representative sample texts
                        save_representative_texts(
                            viz_dir=viz_dir,
                            filename_base=f"{safe_question_name}_{algo_name}",
                            question_name=question_name,
                            sample_ids=sample_ids,
                            df=df,
                            algorithm_name=display_name
                        )

                    except Exception as e:
                        print(f"  ⚠️  Failed to run {algo_name}: {e}")

        print()  # Blank line between sessions

    print("\n" + "="*80)
    print("Visualization complete!")
    print("="*80)


if __name__ == '__main__':
    main()
