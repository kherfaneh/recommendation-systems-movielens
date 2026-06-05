"""Evaluate a user-based collaborative filtering recommender.

Evaluation is critical because a recommender can look plausible while still
performing poorly on unseen user/movie interactions. This script uses a
train/test split, rebuilds the recommender from training data only, and then
measures prediction and ranking quality on held-out test ratings.

Machine Learning parts:
    - Train/test split
    - User similarity computation
    - Rating prediction with weighted neighbor ratings
    - Metric calculation

Recommendation-System Engineering parts:
    - Avoiding data leakage
    - Handling sparse user-item matrices
    - Generating Top-K recommendations
    - Managing cold-start and missing-neighbor cases
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RATINGS_PATH = PROJECT_ROOT / "data" / "ml-latest-small" / "ratings.csv"
DEFAULT_OUTPUT_PATH = PROJECT_ROOT / "outputs" / "evaluation_results.csv"
REQUIRED_COLUMNS = {"userId", "movieId", "rating"}


def configure_logging() -> None:
    """Configure readable console logging."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")


def load_ratings(ratings_path: Path = DEFAULT_RATINGS_PATH) -> pd.DataFrame:
    """Load MovieLens ratings.csv and validate required columns."""
    if not ratings_path.exists():
        raise FileNotFoundError(f"Ratings file not found: {ratings_path}")

    ratings = pd.read_csv(ratings_path)
    missing_columns = REQUIRED_COLUMNS - set(ratings.columns)
    if missing_columns:
        raise ValueError(f"ratings.csv is missing columns: {sorted(missing_columns)}")

    if ratings.empty:
        raise ValueError("ratings.csv is empty.")

    ratings = ratings.copy()
    ratings["userId"] = ratings["userId"].astype(str)
    ratings["movieId"] = ratings["movieId"].astype(str)
    ratings["rating"] = pd.to_numeric(ratings["rating"], errors="coerce")

    if ratings[["userId", "movieId", "rating"]].isna().any().any():
        raise ValueError("ratings.csv contains missing values in required columns.")

    invalid_ratings = ratings[(ratings["rating"] < 0) | (ratings["rating"] > 5)]
    if not invalid_ratings.empty:
        raise ValueError("ratings.csv contains ratings outside the expected 0-5 range.")

    logging.info(
        "Loaded %s ratings from %s users and %s movies.",
        len(ratings),
        ratings["userId"].nunique(),
        ratings["movieId"].nunique(),
    )
    return ratings


def split_train_test(
    ratings: pd.DataFrame,
    test_size: float = 0.2,
    random_seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split ratings into train and test sets with a reproducible random seed.

    Train/test splitting is necessary because evaluation should measure how well
    the recommender predicts interactions it did not see during model building.
    Test ratings must not be used to build the matrix or similarity scores.
    """
    if not 0 < test_size < 1:
        raise ValueError("test_size must be between 0 and 1.")

    train, test = train_test_split(
        ratings,
        test_size=test_size,
        random_state=random_seed,
        shuffle=True,
    )

    logging.info("Split ratings into %s train rows and %s test rows.", len(train), len(test))
    return train.copy(), test.copy()


def build_train_matrix(ratings: pd.DataFrame) -> pd.DataFrame:
    """Build a user-item matrix from training ratings only."""
    matrix = ratings.pivot_table(
        index="userId",
        columns="movieId",
        values="rating",
        aggfunc="mean",
    )

    if matrix.empty:
        raise ValueError("Training user-item matrix is empty.")

    matrix = matrix.sort_index().sort_index(axis=1)
    logging.info("Built training matrix with shape %s.", matrix.shape)
    return matrix


def build_user_item_matrix(ratings: pd.DataFrame) -> pd.DataFrame:
    """Backward-compatible alias for build_train_matrix()."""
    return build_train_matrix(ratings)


def compute_similarity(user_item_matrix: pd.DataFrame) -> pd.DataFrame:
    """Compute user-user cosine similarity from training data only."""
    if user_item_matrix.shape[0] < 2:
        raise ValueError("At least two users are required to compute similarity.")

    filled_matrix = user_item_matrix.fillna(0)
    similarity_values = cosine_similarity(filled_matrix)
    similarity = pd.DataFrame(
        similarity_values,
        index=user_item_matrix.index,
        columns=user_item_matrix.index,
    )

    logging.info("Computed training similarity matrix with shape %s.", similarity.shape)
    return similarity


def predict_rating(
    user_id: str,
    movie_id: str,
    user_item_matrix: pd.DataFrame,
    user_similarity: pd.DataFrame,
    max_neighbors: int = 50,
    min_similarity: float = 0.0,
    global_mean: float | None = None,
) -> float | None:
    """Predict one user's rating for one movie using weighted neighbor ratings.

    The prediction is a weighted average where cosine similarity is the weight.
    If the user or movie is unavailable in training data, return a fallback mean
    when available. This avoids crashing on cold-start test examples.
    """
    if global_mean is None:
        global_mean = float(user_item_matrix.stack().mean())

    if user_id not in user_item_matrix.index:
        return global_mean

    if movie_id not in user_item_matrix.columns:
        return global_mean

    similarities = user_similarity.loc[user_id].drop(labels=user_id, errors="ignore")
    similarities = pd.to_numeric(similarities, errors="coerce").dropna()
    similarities = similarities[similarities > min_similarity].sort_values(ascending=False)

    movie_ratings = user_item_matrix[movie_id].dropna()
    available_neighbors = similarities.index.intersection(movie_ratings.index)
    if available_neighbors.empty:
        return global_mean

    available_neighbors = available_neighbors[:max_neighbors]
    weights = similarities.loc[available_neighbors]
    ratings = movie_ratings.loc[available_neighbors]

    denominator = weights.sum()
    if denominator <= 0:
        return global_mean

    return float(np.dot(weights, ratings) / denominator)


def predict_test_ratings(
    test: pd.DataFrame,
    user_item_matrix: pd.DataFrame,
    user_similarity: pd.DataFrame,
    max_neighbors: int = 50,
) -> pd.DataFrame:
    """Generate predictions for every user/movie pair in the test set."""
    global_mean = float(user_item_matrix.stack().mean())
    predictions: list[dict[str, float | str]] = []

    for row in test.itertuples(index=False):
        prediction = predict_rating(
            user_id=str(row.userId),
            movie_id=str(row.movieId),
            user_item_matrix=user_item_matrix,
            user_similarity=user_similarity,
            max_neighbors=max_neighbors,
            global_mean=global_mean,
        )
        if prediction is not None:
            predictions.append(
                {
                    "userId": str(row.userId),
                    "movieId": str(row.movieId),
                    "actual_rating": float(row.rating),
                    "predicted_rating": float(prediction),
                }
            )

    prediction_frame = pd.DataFrame(predictions)
    if prediction_frame.empty:
        raise ValueError("No test predictions could be generated.")

    logging.info("Generated %s test rating predictions.", len(prediction_frame))
    return prediction_frame


def generate_recommendations(
    user_id: str,
    user_item_matrix: pd.DataFrame,
    user_similarity: pd.DataFrame,
    top_k: int,
    max_neighbors: int = 50,
) -> list[str]:
    """Generate Top-K movie recommendations for one user from training data.

    This ranking step is where recommendation-system engineering becomes
    important: only unseen movies are scored, then the highest predicted scores
    are returned.
    """
    if user_id not in user_item_matrix.index:
        return []

    user_ratings = user_item_matrix.loc[user_id]
    candidate_movies = user_ratings[user_ratings.isna()].index
    if len(candidate_movies) == 0:
        return []

    similarities = user_similarity.loc[user_id].drop(labels=user_id, errors="ignore")
    similarities = pd.to_numeric(similarities, errors="coerce").dropna()
    similarities = similarities[similarities > 0].sort_values(ascending=False).head(max_neighbors)
    if similarities.empty:
        return []

    neighbor_ratings = user_item_matrix.loc[similarities.index, candidate_movies]
    neighbor_ratings = neighbor_ratings.apply(pd.to_numeric, errors="coerce")
    rated_mask = neighbor_ratings.notna()

    weighted_sum = neighbor_ratings.mul(similarities, axis=0).sum(axis=0, skipna=True)
    denominator = rated_mask.mul(similarities, axis=0).sum(axis=0)
    scores = (weighted_sum / denominator).replace([np.inf, -np.inf], np.nan).dropna()

    if scores.empty:
        return []

    return scores.sort_values(ascending=False).head(top_k).index.astype(str).tolist()


def generate_recommendations_fast(
    user_id: str,
    user_ids: list[str],
    movie_ids: list[str],
    rating_values: np.ndarray,
    similarity_values: np.ndarray,
    user_position: dict[str, int],
    top_k: int,
    max_neighbors: int = 50,
) -> list[str]:
    """Generate Top-K recommendations using NumPy for faster ranking evaluation."""
    if user_id not in user_position:
        return []

    user_idx = user_position[user_id]
    user_ratings = rating_values[user_idx]
    candidate_mask = np.isnan(user_ratings)
    if not np.any(candidate_mask):
        return []

    similarities = similarity_values[user_idx].copy()
    similarities[user_idx] = 0.0
    similarities = np.nan_to_num(similarities, nan=0.0)
    similarities[similarities <= 0] = 0.0

    positive_neighbors = np.flatnonzero(similarities > 0)
    if positive_neighbors.size == 0:
        return []

    if positive_neighbors.size > max_neighbors:
        top_neighbor_order = np.argpartition(
            similarities[positive_neighbors],
            -max_neighbors,
        )[-max_neighbors:]
        neighbor_idx = positive_neighbors[top_neighbor_order]
    else:
        neighbor_idx = positive_neighbors

    neighbor_idx = neighbor_idx[np.argsort(similarities[neighbor_idx])[::-1]]
    weights = similarities[neighbor_idx]
    neighbor_ratings = rating_values[neighbor_idx][:, candidate_mask]
    rated_mask = ~np.isnan(neighbor_ratings)

    weighted_ratings = np.nan_to_num(neighbor_ratings, nan=0.0) * weights[:, np.newaxis]
    weighted_sum = weighted_ratings.sum(axis=0)
    denominator = rated_mask.astype(float).T @ weights

    valid = denominator > 0
    if not np.any(valid):
        return []

    candidate_indices = np.flatnonzero(candidate_mask)[valid]
    scores = weighted_sum[valid] / denominator[valid]
    if scores.size == 0:
        return []

    limit = min(top_k, scores.size)
    if scores.size > limit:
        top_indices = np.argpartition(scores, -limit)[-limit:]
        top_indices = top_indices[np.argsort(scores[top_indices])[::-1]]
    else:
        top_indices = np.argsort(scores)[::-1]

    del user_ids  # Kept for explicit alignment with the arrays.
    return [movie_ids[candidate_indices[index]] for index in top_indices]


def evaluate_rmse(predictions: pd.DataFrame) -> float:
    """Calculate RMSE.

    RMSE measures average prediction error with larger mistakes penalized more.
    Lower is better.
    """
    mse = mean_squared_error(
        predictions["actual_rating"],
        predictions["predicted_rating"],
    )
    return float(np.sqrt(mse))


def evaluate_mae(predictions: pd.DataFrame) -> float:
    """Calculate MAE.

    MAE measures the average absolute rating error. Lower is better.
    """
    return float(
        mean_absolute_error(
            predictions["actual_rating"],
            predictions["predicted_rating"],
        )
    )


def evaluate_precision_recall(
    train: pd.DataFrame,
    test: pd.DataFrame,
    user_item_matrix: pd.DataFrame,
    user_similarity: pd.DataFrame,
    k_values: Iterable[int],
    relevance_threshold: float = 4.0,
    max_neighbors: int = 50,
) -> dict[int, tuple[float, float]]:
    """Evaluate Precision@K and Recall@K against held-out relevant movies.

    Precision@K asks: of the K recommended movies, how many were relevant?
    Recall@K asks: of the user's relevant held-out movies, how many did we find?
    Higher is better for both.
    """
    del train  # Kept in the signature to make the train/test contract explicit.

    relevant_by_user = (
        test[test["rating"] >= relevance_threshold]
        .groupby("userId")["movieId"]
        .apply(lambda values: set(values.astype(str)))
        .to_dict()
    )

    if not relevant_by_user:
        raise ValueError("No relevant test ratings found for Precision@K/Recall@K evaluation.")

    k_values = sorted(k_values)
    max_k = max(k_values)
    per_user_recommendations: dict[str, list[str]] = {}
    user_ids = user_item_matrix.index.astype(str).tolist()
    movie_ids = user_item_matrix.columns.astype(str).tolist()
    rating_values = user_item_matrix.to_numpy(dtype=float)
    similarity_values = user_similarity.loc[user_ids, user_ids].to_numpy(dtype=float)
    user_position = {user_id: index for index, user_id in enumerate(user_ids)}

    for user_id in relevant_by_user:
        per_user_recommendations[str(user_id)] = generate_recommendations_fast(
            user_id=str(user_id),
            user_ids=user_ids,
            movie_ids=movie_ids,
            rating_values=rating_values,
            similarity_values=similarity_values,
            user_position=user_position,
            top_k=max_k,
            max_neighbors=max_neighbors,
        )

    results: dict[int, tuple[float, float]] = {}
    for k in k_values:
        if k < 1:
            raise ValueError("All K values must be at least 1.")

        precision_values: list[float] = []
        recall_values: list[float] = []

        for user_id, relevant_movies in relevant_by_user.items():
            recommendations = per_user_recommendations.get(str(user_id), [])[:k]
            if not recommendations:
                continue

            recommended_set = set(recommendations)
            hits = len(recommended_set.intersection(relevant_movies))
            precision_values.append(hits / k)
            recall_values.append(hits / len(relevant_movies))

        precision = float(np.mean(precision_values)) if precision_values else 0.0
        recall = float(np.mean(recall_values)) if recall_values else 0.0
        results[k] = (precision, recall)

        logging.info(
            "Precision@%s: %.4f, Recall@%s: %.4f over %s users.",
            k,
            precision,
            k,
            recall,
            len(precision_values),
        )

    return results


def save_results(
    output_path: Path,
    rmse: float,
    mae: float,
    precision_recall: dict[int, tuple[float, float]],
) -> None:
    """Save all evaluation metrics to outputs/evaluation_results.csv."""
    rows = [
        {"metric": "RMSE", "k": "", "value": rmse, "better": "lower"},
        {"metric": "MAE", "k": "", "value": mae, "better": "lower"},
    ]

    for k, (precision, recall) in precision_recall.items():
        rows.append({"metric": "Precision@K", "k": k, "value": precision, "better": "higher"})
        rows.append({"metric": "Recall@K", "k": k, "value": recall, "better": "higher"})

    output_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(output_path, index=False)
    logging.info("Saved evaluation results to %s", output_path)


def print_summary(rmse: float, mae: float, precision_recall: dict[int, tuple[float, float]]) -> None:
    """Print a concise evaluation summary."""
    print("\n---")
    print("## Evaluation Results")
    print()
    print(f"RMSE: {rmse:.4f}")
    print(f"MAE: {mae:.4f}")
    for k, (precision, recall) in precision_recall.items():
        print(f"Precision@{k}: {precision:.4f}")
        print(f"Recall@{k}: {recall:.4f}")
    print("---------------")


def parse_k_values(raw_values: str) -> list[int]:
    """Parse comma-separated K values such as '5,10,20'."""
    try:
        values = [int(value.strip()) for value in raw_values.split(",") if value.strip()]
    except ValueError as exc:
        raise ValueError("--k-values must be comma-separated integers.") from exc

    if not values:
        raise ValueError("At least one K value is required.")

    return values


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Evaluate user-based collaborative filtering on MovieLens."
    )
    parser.add_argument("--ratings-path", type=Path, default=DEFAULT_RATINGS_PATH)
    parser.add_argument("--output-path", type=Path, default=DEFAULT_OUTPUT_PATH)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--random-seed", type=int, default=42)
    parser.add_argument("--neighbors", type=int, default=50)
    parser.add_argument("--k-values", type=str, default="5,10,20")
    parser.add_argument(
        "--relevance-threshold",
        type=float,
        default=4.0,
        help="Ratings at or above this value are relevant for Precision@K/Recall@K.",
    )
    return parser.parse_args()


def main() -> None:
    """Run the full recommender evaluation pipeline."""
    configure_logging()
    args = parse_args()

    try:
        k_values = parse_k_values(args.k_values)
        ratings = load_ratings(args.ratings_path)
        train, test = split_train_test(
            ratings,
            test_size=args.test_size,
            random_seed=args.random_seed,
        )

        # Data leakage pitfall: build these artifacts from train only.
        user_item_matrix = build_train_matrix(train)
        user_similarity = compute_similarity(user_item_matrix)

        predictions = predict_test_ratings(
            test,
            user_item_matrix,
            user_similarity,
            max_neighbors=args.neighbors,
        )
        rmse = evaluate_rmse(predictions)
        mae = evaluate_mae(predictions)
        precision_recall = evaluate_precision_recall(
            train,
            test,
            user_item_matrix,
            user_similarity,
            k_values=k_values,
            relevance_threshold=args.relevance_threshold,
            max_neighbors=args.neighbors,
        )

        save_results(args.output_path, rmse, mae, precision_recall)
        print_summary(rmse, mae, precision_recall)
    except Exception:
        logging.exception("Evaluation failed.")
        raise


if __name__ == "__main__":
    main()
