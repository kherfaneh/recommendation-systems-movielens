"""User-based collaborative filtering recommendation engine.

This script uses the artifacts created by the previous pipeline stages:
    outputs/user_item_matrix.csv
    outputs/user_similarity.csv
    data/ml-latest-small/movies.csv

It recommends movies for one target user by finding similar users, weighting
their ratings by cosine similarity, and excluding movies the target user has
already rated.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import NamedTuple

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MATRIX_PATH = PROJECT_ROOT / "outputs" / "user_item_matrix.csv"
DEFAULT_SIMILARITY_PATH = PROJECT_ROOT / "outputs" / "user_similarity.csv"
DEFAULT_MOVIES_PATH = PROJECT_ROOT / "data" / "ml-latest-small" / "movies.csv"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "outputs"


class RecommendationData(NamedTuple):
    """Container for the data required by the recommendation engine."""

    user_item_matrix: pd.DataFrame
    user_similarity: pd.DataFrame
    movies: pd.DataFrame


def configure_logging() -> None:
    """Configure readable console logging."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")


def validate_file_exists(file_path: Path, description: str) -> None:
    """Raise a clear error when a required input file is missing."""
    if not file_path.exists():
        raise FileNotFoundError(f"{description} not found: {file_path}")


def load_data(
    matrix_path: Path = DEFAULT_MATRIX_PATH,
    similarity_path: Path = DEFAULT_SIMILARITY_PATH,
    movies_path: Path = DEFAULT_MOVIES_PATH,
) -> RecommendationData:
    """Load and validate the user-item matrix, similarity matrix, and movie metadata.

    Args:
        matrix_path: Path to the saved user-item matrix CSV.
        similarity_path: Path to the saved user-user similarity CSV.
        movies_path: Path to MovieLens movies.csv.

    Returns:
        RecommendationData containing validated pandas DataFrames.
    """
    validate_file_exists(matrix_path, "User-item matrix")
    validate_file_exists(similarity_path, "User similarity matrix")
    validate_file_exists(movies_path, "Movie metadata file")

    logging.info("Loading user-item matrix from %s", matrix_path)
    user_item_matrix = pd.read_csv(matrix_path, index_col=0)

    logging.info("Loading user similarity matrix from %s", similarity_path)
    user_similarity = pd.read_csv(similarity_path, index_col=0)

    logging.info("Loading movie metadata from %s", movies_path)
    movies = pd.read_csv(movies_path)

    user_item_matrix.index = user_item_matrix.index.astype(str)
    user_item_matrix.columns = user_item_matrix.columns.astype(str)
    user_similarity.index = user_similarity.index.astype(str)
    user_similarity.columns = user_similarity.columns.astype(str)
    movies["movieId"] = movies["movieId"].astype(str)

    validate_input_data(user_item_matrix, user_similarity, movies)

    logging.info(
        "Loaded data: %s users, %s movies in matrix, %s movie metadata rows.",
        user_item_matrix.shape[0],
        user_item_matrix.shape[1],
        len(movies),
    )

    return RecommendationData(user_item_matrix, user_similarity, movies)


def validate_input_data(
    user_item_matrix: pd.DataFrame,
    user_similarity: pd.DataFrame,
    movies: pd.DataFrame,
) -> None:
    """Validate that recommendation inputs are complete and aligned."""
    if user_item_matrix.empty:
        raise ValueError("User-item matrix is empty.")

    if user_similarity.empty:
        raise ValueError("User similarity matrix is empty.")

    if user_similarity.shape[0] != user_similarity.shape[1]:
        raise ValueError("User similarity matrix must be square.")

    if set(user_item_matrix.index) != set(user_similarity.index):
        raise ValueError("User IDs in user-item matrix and similarity matrix do not match.")

    if set(user_similarity.index) != set(user_similarity.columns):
        raise ValueError("Similarity matrix row and column user IDs do not match.")

    required_movie_columns = {"movieId", "title"}
    missing_movie_columns = required_movie_columns - set(movies.columns)
    if missing_movie_columns:
        raise ValueError(f"movies.csv is missing columns: {sorted(missing_movie_columns)}")

    numeric_matrix = user_item_matrix.apply(pd.to_numeric, errors="coerce")
    introduced_missing = numeric_matrix.isna() & user_item_matrix.notna()
    if introduced_missing.any().any():
        bad_columns = introduced_missing.columns[introduced_missing.any()].tolist()
        raise TypeError(f"User-item matrix contains non-numeric values: {bad_columns}")


def validate_target_user(target_user_id: int, user_item_matrix: pd.DataFrame) -> str:
    """Validate that the target user exists and return its string user ID."""
    user_id = str(target_user_id)
    if user_id not in user_item_matrix.index:
        available_sample = ", ".join(user_item_matrix.index[:5].tolist())
        raise ValueError(
            f"User {target_user_id} does not exist. "
            f"Example available user IDs: {available_sample}"
        )

    rating_count = user_item_matrix.loc[user_id].notna().sum()
    if rating_count < 5:
        logging.warning(
            "User %s has only %s ratings. Recommendations may be weak.",
            user_id,
            rating_count,
        )
    else:
        logging.info("Target user %s has %s ratings.", user_id, rating_count)

    return user_id


def get_similar_users(
    target_user_id: str,
    user_similarity: pd.DataFrame,
    max_neighbors: int = 50,
    min_similarity: float = 0.0,
) -> pd.Series:
    """Return the most similar users to the target user.

    Cosine similarity is the ML signal here: it estimates how close two users'
    rating vectors are. Higher scores receive more influence during prediction.
    """
    if max_neighbors < 1:
        raise ValueError("max_neighbors must be at least 1.")

    similarities = user_similarity.loc[target_user_id].drop(labels=target_user_id)
    similarities = pd.to_numeric(similarities, errors="coerce").dropna()
    similarities = similarities[similarities > min_similarity]
    similarities = similarities.sort_values(ascending=False).head(max_neighbors)

    if similarities.empty:
        raise ValueError(f"No similar users found for user {target_user_id}.")

    logging.info(
        "Using %s similar users. Best similarity: %.4f",
        len(similarities),
        similarities.iloc[0],
    )
    return similarities


def calculate_recommendation_scores(
    target_user_id: str,
    user_item_matrix: pd.DataFrame,
    similar_users: pd.Series,
) -> pd.DataFrame:
    """Calculate weighted recommendation scores for unrated movies.

    The AI/ML step is the weighted aggregation: ratings from more similar users
    count more than ratings from less similar users.
    """
    target_ratings = pd.to_numeric(user_item_matrix.loc[target_user_id], errors="coerce")
    unrated_movies = target_ratings[target_ratings.isna()].index

    if len(unrated_movies) == 0:
        raise ValueError(f"User {target_user_id} has already rated every movie in the matrix.")

    neighbor_ratings = user_item_matrix.loc[similar_users.index, unrated_movies]
    neighbor_ratings = neighbor_ratings.apply(pd.to_numeric, errors="coerce")

    similar_user_count = neighbor_ratings.notna().sum(axis=0)
    candidate_movies = similar_user_count[similar_user_count > 0].index

    if len(candidate_movies) == 0:
        raise ValueError("Similar users have no ratings for movies the target user has not rated.")

    candidate_ratings = neighbor_ratings[candidate_movies]
    weights = similar_users.loc[candidate_ratings.index]
    weighted_sum = candidate_ratings.mul(weights, axis=0).sum(axis=0, skipna=True)

    # Normalize by only the weights from neighbors who actually rated each movie.
    rated_mask = candidate_ratings.notna()
    similarity_weight_sum = rated_mask.mul(weights, axis=0).sum(axis=0)
    scores = weighted_sum / similarity_weight_sum
    score_frame = pd.DataFrame(
        {
            "movieId": scores.index.astype(str),
            "score": scores.values,
            "similar_user_count": similar_user_count.loc[scores.index].values,
            "similarity_weight_sum": similarity_weight_sum.loc[scores.index].values,
        }
    )
    score_frame = score_frame.dropna(subset=["score"])
    score_frame = score_frame.sort_values(
        by=["score", "similar_user_count", "similarity_weight_sum"],
        ascending=[False, False, False],
    )

    logging.info("Candidate movies after filtering: %s", len(score_frame))
    return score_frame


def get_top_recommendations(
    scores: pd.DataFrame,
    movies: pd.DataFrame,
    top_n: int = 10,
    min_neighbor_ratings: int = 2,
) -> pd.DataFrame:
    """Join recommendation scores with movie titles and return the top N movies."""
    if top_n < 1:
        raise ValueError("top_n must be at least 1.")

    if scores.empty:
        raise ValueError("No recommendation scores were generated.")

    filtered_scores = scores[scores["similar_user_count"] >= min_neighbor_ratings]
    if filtered_scores.empty:
        logging.warning(
            "No movies met min_neighbor_ratings=%s. Falling back to all scored movies.",
            min_neighbor_ratings,
        )
        filtered_scores = scores

    score_frame = filtered_scores.head(top_n).copy()
    score_frame["movieId"] = score_frame["movieId"].astype(str)

    recommendations = score_frame.merge(
        movies[["movieId", "title"]],
        on="movieId",
        how="left",
    )

    recommendations["title"] = recommendations["title"].fillna(
        "Unknown title (movieId=" + recommendations["movieId"] + ")"
    )
    recommendations = recommendations[
        ["movieId", "title", "score", "similar_user_count", "similarity_weight_sum"]
    ]

    if recommendations.empty:
        raise ValueError("Recommendation results are empty after joining movie titles.")

    return recommendations


def save_recommendations(
    recommendations: pd.DataFrame,
    target_user_id: str,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
) -> Path:
    """Save recommendations for the target user to outputs/recommendations_user_<userId>.csv."""
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"recommendations_user_{target_user_id}.csv"
    recommendations.to_csv(output_path, index=False)
    logging.info("Saved recommendations to %s", output_path)
    return output_path


def display_recommendations(
    recommendations: pd.DataFrame,
    target_user_id: str,
    similar_users: pd.Series,
    scores: pd.DataFrame,
) -> None:
    """Print useful debugging information and the final recommendation table."""
    print(f"\nRecommendations for user {target_user_id}")
    print(f"Similar users used: {len(similar_users)}")
    print(f"Candidate movies scored: {len(scores)}")
    print("\nTop recommendation scores:")
    print(
        recommendations.to_string(
            index=False,
            formatters={
                "score": "{:.4f}".format,
                "similarity_weight_sum": "{:.4f}".format,
            },
        )
    )


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Generate user-based collaborative filtering recommendations."
    )
    parser.add_argument("--user-id", type=int, default=1, help="Target MovieLens user ID.")
    parser.add_argument("--top-n", type=int, default=10, help="Number of recommendations.")
    parser.add_argument(
        "--neighbors",
        type=int,
        default=50,
        help="Maximum number of similar users used for scoring.",
    )
    parser.add_argument(
        "--min-similarity",
        type=float,
        default=0.0,
        help="Minimum cosine similarity required for a neighbor.",
    )
    parser.add_argument(
        "--min-neighbor-ratings",
        type=int,
        default=2,
        help="Minimum number of similar users who rated a candidate movie.",
    )
    parser.add_argument(
        "--matrix-path",
        type=Path,
        default=DEFAULT_MATRIX_PATH,
        help="Path to outputs/user_item_matrix.csv.",
    )
    parser.add_argument(
        "--similarity-path",
        type=Path,
        default=DEFAULT_SIMILARITY_PATH,
        help="Path to outputs/user_similarity.csv.",
    )
    parser.add_argument(
        "--movies-path",
        type=Path,
        default=DEFAULT_MOVIES_PATH,
        help="Path to MovieLens movies.csv.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory where recommendations CSV files are saved.",
    )
    return parser.parse_args()


def main() -> None:
    """Run the recommendation pipeline for one target user."""
    configure_logging()
    args = parse_args()

    try:
        data = load_data(args.matrix_path, args.similarity_path, args.movies_path)
        target_user_id = validate_target_user(args.user_id, data.user_item_matrix)
        similar_users = get_similar_users(
            target_user_id,
            data.user_similarity,
            max_neighbors=args.neighbors,
            min_similarity=args.min_similarity,
        )
        scores = calculate_recommendation_scores(
            target_user_id,
            data.user_item_matrix,
            similar_users,
        )
        recommendations = get_top_recommendations(
            scores,
            data.movies,
            top_n=args.top_n,
            min_neighbor_ratings=args.min_neighbor_ratings,
        )
        save_recommendations(recommendations, target_user_id, args.output_dir)
        display_recommendations(recommendations, target_user_id, similar_users, scores)
    except Exception:
        logging.exception("Failed to generate recommendations.")
        raise


if __name__ == "__main__":
    main()
