"""Analyze user similarity relationships for User-Based CF.

This module belongs to the analysis and monitoring layer. User-Based CF depends
on meaningful user-user relationships, so inspecting most/least similar users
and similarity distributions helps debug recommender behavior.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SIMILARITY_PATH = PROJECT_ROOT / "outputs" / "user_similarity.csv"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "outputs"


def configure_logging() -> None:
    """Configure console logging."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")


def load_similarity(similarity_path: Path) -> pd.DataFrame:
    """Load and validate user-user similarity matrix."""
    if not similarity_path.exists():
        raise FileNotFoundError(f"User similarity matrix not found: {similarity_path}")

    similarity = pd.read_csv(similarity_path, index_col=0)
    similarity.index = similarity.index.astype(str)
    similarity.columns = similarity.columns.astype(str)

    if similarity.empty:
        raise ValueError("Similarity matrix is empty.")
    if similarity.shape[0] != similarity.shape[1]:
        raise ValueError("Similarity matrix must be square.")
    if set(similarity.index) != set(similarity.columns):
        raise ValueError("Similarity matrix rows and columns must contain the same users.")

    return similarity.apply(pd.to_numeric, errors="coerce")


def analyze_user_similarity(
    user_id: str,
    similarity: pd.DataFrame,
    top_n: int = 10,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
    """Return most similar users, least similar users, and summary statistics."""
    if user_id not in similarity.index:
        raise ValueError(f"User {user_id} does not exist in similarity matrix.")
    if top_n < 1:
        raise ValueError("top_n must be at least 1.")

    scores = similarity.loc[user_id].drop(labels=user_id, errors="ignore").dropna()
    most_similar = scores.sort_values(ascending=False).head(top_n)
    least_similar = scores.sort_values(ascending=True).head(top_n)
    stats = scores.describe()

    most_frame = most_similar.rename("similarity").reset_index()
    most_frame = most_frame.rename(columns={"index": "userId"})
    least_frame = least_similar.rename("similarity").reset_index()
    least_frame = least_frame.rename(columns={"index": "userId"})

    return most_frame, least_frame, stats


def similarity_distribution(similarity: pd.DataFrame) -> pd.Series:
    """Return all off-diagonal similarity values for distribution analysis."""
    values = []
    users = list(similarity.index)
    for index, user_id in enumerate(users):
        values.extend(similarity.loc[user_id, users[index + 1 :]].dropna().tolist())
    return pd.Series(values, name="similarity")


def save_analysis(
    user_id: str,
    most_similar: pd.DataFrame,
    least_similar: pd.DataFrame,
    stats: pd.Series,
    distribution: pd.Series,
    output_dir: Path,
) -> None:
    """Save user analysis artifacts to CSV."""
    output_dir.mkdir(parents=True, exist_ok=True)
    most_similar.to_csv(output_dir / f"user_{user_id}_most_similar.csv", index=False)
    least_similar.to_csv(output_dir / f"user_{user_id}_least_similar.csv", index=False)
    stats.rename("value").to_csv(output_dir / f"user_{user_id}_similarity_stats.csv")
    distribution.to_csv(output_dir / "similarity_distribution.csv", index=False)
    logging.info("Saved user analysis CSV files to %s", output_dir)


def display_analysis(
    user_id: str,
    most_similar: pd.DataFrame,
    least_similar: pd.DataFrame,
    stats: pd.Series,
) -> None:
    """Print user similarity analysis."""
    print(f"\nUser {user_id}")
    print("\nMost Similar:")
    for row in most_similar.itertuples(index=False):
        print(f"User {row.userId} -> {row.similarity:.4f}")

    print("\nLeast Similar:")
    for row in least_similar.itertuples(index=False):
        print(f"User {row.userId} -> {row.similarity:.4f}")

    print("\nSimilarity Statistics:")
    print(stats.to_string())


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Analyze User-Based CF similarity relationships.")
    parser.add_argument("--user-id", type=str, default="1")
    parser.add_argument("--top-n", type=int, default=10)
    parser.add_argument("--similarity-path", type=Path, default=DEFAULT_SIMILARITY_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def main() -> None:
    """Run user similarity analysis."""
    configure_logging()
    args = parse_args()
    try:
        similarity = load_similarity(args.similarity_path)
        most_similar, least_similar, stats = analyze_user_similarity(
            str(args.user_id),
            similarity,
            top_n=args.top_n,
        )
        distribution = similarity_distribution(similarity)
        save_analysis(
            str(args.user_id),
            most_similar,
            least_similar,
            stats,
            distribution,
            args.output_dir,
        )
        display_analysis(str(args.user_id), most_similar, least_similar, stats)
    except Exception:
        logging.exception("User similarity analysis failed.")
        raise


if __name__ == "__main__":
    main()
