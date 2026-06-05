"""Visualize User-Based Collaborative Filtering behavior.

This module belongs to recommender-system analysis and communication. Plots help
explain sparsity, rating behavior, active users, popular movies, and the spread
of recommendation scores.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RATINGS_PATH = PROJECT_ROOT / "data" / "ml-latest-small" / "ratings.csv"
DEFAULT_MOVIES_PATH = PROJECT_ROOT / "data" / "ml-latest-small" / "movies.csv"
DEFAULT_SIMILARITY_PATH = PROJECT_ROOT / "outputs" / "user_similarity.csv"
DEFAULT_RECOMMENDATIONS_PATH = PROJECT_ROOT / "outputs" / "recommendations_user_1.csv"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "outputs" / "plots"


def configure_logging() -> None:
    """Configure console logging."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")


def load_csv(path: Path, label: str, **kwargs: object) -> pd.DataFrame:
    """Load a required CSV file."""
    if not path.exists():
        raise FileNotFoundError(f"{label} not found: {path}")
    return pd.read_csv(path, **kwargs)


def save_current_plot(output_path: Path) -> None:
    """Save and close the current matplotlib figure."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    logging.info("Saved plot: %s", output_path)


def plot_similarity_distribution(similarity: pd.DataFrame, output_dir: Path) -> None:
    """Plot off-diagonal user-user cosine similarity distribution."""
    similarity.index = similarity.index.astype(str)
    similarity.columns = similarity.columns.astype(str)
    values = []
    users = list(similarity.index)
    for index, user_id in enumerate(users):
        values.extend(pd.to_numeric(similarity.loc[user_id, users[index + 1 :]], errors="coerce"))

    plt.figure(figsize=(8, 5))
    plt.hist(pd.Series(values).dropna(), bins=40, color="#2f6f9f", edgecolor="white")
    plt.title("User-User Cosine Similarity Distribution")
    plt.xlabel("Cosine similarity")
    plt.ylabel("User pair count")
    save_current_plot(output_dir / "similarity_distribution.png")


def plot_rating_distribution(ratings: pd.DataFrame, output_dir: Path) -> None:
    """Plot rating value distribution."""
    plt.figure(figsize=(8, 5))
    ratings["rating"].hist(bins=10, color="#7a9e3f", edgecolor="white")
    plt.title("Rating Distribution")
    plt.xlabel("Rating")
    plt.ylabel("Count")
    save_current_plot(output_dir / "rating_distribution.png")


def plot_top_active_users(ratings: pd.DataFrame, output_dir: Path, top_n: int = 20) -> None:
    """Plot users with the most ratings."""
    active_users = ratings["userId"].value_counts().head(top_n).sort_values()
    plt.figure(figsize=(9, 6))
    active_users.plot(kind="barh", color="#6f5f90")
    plt.title(f"Top {top_n} Active Users")
    plt.xlabel("Number of ratings")
    plt.ylabel("User ID")
    save_current_plot(output_dir / "top_active_users.png")


def plot_top_rated_movies(
    ratings: pd.DataFrame,
    movies: pd.DataFrame,
    output_dir: Path,
    top_n: int = 20,
) -> None:
    """Plot movies with the largest number of ratings."""
    counts = ratings["movieId"].value_counts().head(top_n).rename("rating_count").reset_index()
    counts = counts.rename(columns={"index": "movieId"})
    counts["movieId"] = counts["movieId"].astype(str)
    movies["movieId"] = movies["movieId"].astype(str)
    counts = counts.merge(movies[["movieId", "title"]], on="movieId", how="left")
    counts = counts.sort_values("rating_count")

    plt.figure(figsize=(10, 7))
    plt.barh(counts["title"], counts["rating_count"], color="#b45f43")
    plt.title(f"Top {top_n} Most Rated Movies")
    plt.xlabel("Number of ratings")
    plt.ylabel("Movie")
    save_current_plot(output_dir / "top_rated_movies.png")


def plot_recommendation_score_distribution(recommendations: pd.DataFrame, output_dir: Path) -> None:
    """Plot recommendation score distribution for a generated recommendation file."""
    if "score" not in recommendations.columns:
        raise ValueError("Recommendations CSV must contain a score column.")

    plt.figure(figsize=(8, 5))
    recommendations["score"].hist(bins=15, color="#3f8f7f", edgecolor="white")
    plt.title("Recommendation Score Distribution")
    plt.xlabel("Recommendation score")
    plt.ylabel("Count")
    save_current_plot(output_dir / "recommendation_score_distribution.png")


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Create User-Based CF visualizations.")
    parser.add_argument("--ratings-path", type=Path, default=DEFAULT_RATINGS_PATH)
    parser.add_argument("--movies-path", type=Path, default=DEFAULT_MOVIES_PATH)
    parser.add_argument("--similarity-path", type=Path, default=DEFAULT_SIMILARITY_PATH)
    parser.add_argument("--recommendations-path", type=Path, default=DEFAULT_RECOMMENDATIONS_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def main() -> None:
    """Generate all visualization plots."""
    configure_logging()
    args = parse_args()
    try:
        ratings = load_csv(args.ratings_path, "ratings.csv")
        movies = load_csv(args.movies_path, "movies.csv")
        similarity = load_csv(args.similarity_path, "user similarity matrix", index_col=0)
        recommendations = load_csv(args.recommendations_path, "recommendations CSV")

        plot_similarity_distribution(similarity, args.output_dir)
        plot_rating_distribution(ratings, args.output_dir)
        plot_top_active_users(ratings, args.output_dir)
        plot_top_rated_movies(ratings, movies, args.output_dir)
        plot_recommendation_score_distribution(recommendations, args.output_dir)
    except Exception:
        logging.exception("Visualization generation failed.")
        raise


if __name__ == "__main__":
    main()
