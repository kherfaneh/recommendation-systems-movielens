"""Explain User-Based Collaborative Filtering recommendations.

This module belongs to the Explainability layer of the recommender system. It
answers a practical question: "Why did the model recommend this movie?"

For User-Based Collaborative Filtering, the answer is neighbor-based: similar
users rated the movie highly, and their ratings were weighted by cosine
similarity to the target user.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MATRIX_PATH = PROJECT_ROOT / "outputs" / "user_item_matrix.csv"
DEFAULT_SIMILARITY_PATH = PROJECT_ROOT / "outputs" / "user_similarity.csv"
DEFAULT_MOVIES_PATH = PROJECT_ROOT / "data" / "ml-latest-small" / "movies.csv"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "outputs"


def configure_logging() -> None:
    """Configure console logging."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")


def load_artifacts(
    matrix_path: Path,
    similarity_path: Path,
    movies_path: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load matrix, similarity, and movie metadata artifacts."""
    for path, label in (
        (matrix_path, "user-item matrix"),
        (similarity_path, "user similarity matrix"),
        (movies_path, "movies.csv"),
    ):
        if not path.exists():
            raise FileNotFoundError(f"Missing {label}: {path}")

    matrix = pd.read_csv(matrix_path, index_col=0)
    similarity = pd.read_csv(similarity_path, index_col=0)
    movies = pd.read_csv(movies_path)

    matrix.index = matrix.index.astype(str)
    matrix.columns = matrix.columns.astype(str)
    similarity.index = similarity.index.astype(str)
    similarity.columns = similarity.columns.astype(str)
    movies["movieId"] = movies["movieId"].astype(str)

    if "title" not in movies.columns:
        raise ValueError("movies.csv must contain a title column.")
    if set(matrix.index) != set(similarity.index):
        raise ValueError("Matrix users and similarity users do not match.")

    return matrix, similarity, movies


def get_movie_title(movie_id: str, movies: pd.DataFrame) -> str:
    """Return a movie title for a MovieLens movie ID."""
    match = movies.loc[movies["movieId"] == movie_id, "title"]
    return match.iloc[0] if not match.empty else f"Unknown title (movieId={movie_id})"


def get_top_contributors(
    user_id: str,
    movie_id: str,
    user_item_matrix: pd.DataFrame,
    user_similarity: pd.DataFrame,
    top_n: int = 5,
    min_similarity: float = 0.0,
) -> pd.DataFrame:
    """Return similar users who contributed most to one recommendation.

    Contribution is similarity * rating. This is the core explainable part of
    User-Based CF: each recommendation can be traced back to neighbor ratings.
    """
    if user_id not in user_item_matrix.index:
        raise ValueError(f"User {user_id} does not exist.")
    if movie_id not in user_item_matrix.columns:
        raise ValueError(f"Movie {movie_id} does not exist in the user-item matrix.")

    similarities = pd.to_numeric(user_similarity.loc[user_id], errors="coerce")
    similarities = similarities.drop(labels=user_id, errors="ignore").dropna()
    similarities = similarities[similarities > min_similarity]

    ratings = pd.to_numeric(user_item_matrix[movie_id], errors="coerce").dropna()
    neighbor_ids = similarities.index.intersection(ratings.index)
    if neighbor_ids.empty:
        raise ValueError(f"No similar users rated movie {movie_id}.")

    rows = []
    for neighbor_id in neighbor_ids:
        similarity = float(similarities.loc[neighbor_id])
        rating = float(ratings.loc[neighbor_id])
        rows.append(
            {
                "neighbor_user_id": neighbor_id,
                "similarity": similarity,
                "rating": rating,
                "weighted_contribution": similarity * rating,
            }
        )

    contributors = pd.DataFrame(rows)
    contributors = contributors.sort_values(
        by=["weighted_contribution", "similarity"],
        ascending=False,
    )
    return contributors.head(top_n)


def explain_recommendation(
    user_id: str,
    movie_id: str,
    user_item_matrix: pd.DataFrame,
    user_similarity: pd.DataFrame,
    movies: pd.DataFrame,
    top_contributors: int = 5,
) -> pd.DataFrame:
    """Explain one recommendation and calculate its weighted score."""
    contributors = get_top_contributors(
        user_id=user_id,
        movie_id=movie_id,
        user_item_matrix=user_item_matrix,
        user_similarity=user_similarity,
        top_n=top_contributors,
    )
    denominator = contributors["similarity"].sum()
    if denominator <= 0:
        raise ValueError("Cannot explain recommendation because similarity weights sum to zero.")

    score = contributors["weighted_contribution"].sum() / denominator
    contributors.insert(0, "user_id", user_id)
    contributors.insert(1, "movie_id", movie_id)
    contributors.insert(2, "title", get_movie_title(movie_id, movies))
    contributors["explained_score_from_top_contributors"] = score

    return contributors


def save_explanation(explanation: pd.DataFrame, output_dir: Path, user_id: str, movie_id: str) -> Path:
    """Save explanation rows to CSV."""
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"explanation_user_{user_id}_movie_{movie_id}.csv"
    explanation.to_csv(output_path, index=False)
    logging.info("Saved explanation to %s", output_path)
    return output_path


def display_explanation(explanation: pd.DataFrame) -> None:
    """Print a readable explanation for the recommendation."""
    first = explanation.iloc[0]
    print(f"\nMovie: {first['title']}")
    print(f"User: {first['user_id']}")
    print("\nRecommended because:")
    for row in explanation.itertuples(index=False):
        print(
            f"- User {row.neighbor_user_id} "
            f"(similarity={row.similarity:.4f}) rated {row.rating:.1f}; "
            f"weighted contribution={row.weighted_contribution:.4f}"
        )
    print(
        "\nScore formula: sum(similarity * neighbor rating) / sum(similarity) = "
        f"{first['explained_score_from_top_contributors']:.4f}"
    )


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Explain a User-Based CF recommendation.")
    parser.add_argument("--user-id", type=str, default="1")
    parser.add_argument("--movie-id", type=str, required=True)
    parser.add_argument("--top-contributors", type=int, default=5)
    parser.add_argument("--matrix-path", type=Path, default=DEFAULT_MATRIX_PATH)
    parser.add_argument("--similarity-path", type=Path, default=DEFAULT_SIMILARITY_PATH)
    parser.add_argument("--movies-path", type=Path, default=DEFAULT_MOVIES_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def main() -> None:
    """Run recommendation explanation for one user/movie pair."""
    configure_logging()
    args = parse_args()
    try:
        matrix, similarity, movies = load_artifacts(
            args.matrix_path,
            args.similarity_path,
            args.movies_path,
        )
        explanation = explain_recommendation(
            user_id=str(args.user_id),
            movie_id=str(args.movie_id),
            user_item_matrix=matrix,
            user_similarity=similarity,
            movies=movies,
            top_contributors=args.top_contributors,
        )
        save_explanation(explanation, args.output_dir, str(args.user_id), str(args.movie_id))
        display_explanation(explanation)
    except Exception:
        logging.exception("Failed to explain recommendation.")
        raise


if __name__ == "__main__":
    main()
