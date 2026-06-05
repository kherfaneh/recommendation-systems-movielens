"""Build a MovieLens user-item ratings matrix.

Expected input:
    data/ml-latest-small/ratings.csv

Default output:
    outputs/user_item_matrix.csv
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import pandas as pd


REQUIRED_COLUMNS = {"userId", "movieId", "rating"}
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RATINGS_PATH = PROJECT_ROOT / "data" / "ml-latest-small" / "ratings.csv"
DEFAULT_OUTPUT_PATH = PROJECT_ROOT / "outputs" / "user_item_matrix.csv"


def configure_logging() -> None:
    """Configure readable console logging for this script."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s: %(message)s",
    )


def load_ratings(ratings_path: Path) -> pd.DataFrame:
    """Load ratings data from CSV and fail with a clear error if unavailable."""
    if not ratings_path.exists():
        raise FileNotFoundError(
            f"Ratings file not found: {ratings_path}. "
            "Expected MovieLens ml-latest-small ratings.csv."
        )

    logging.info("Loading ratings from %s", ratings_path)
    ratings = pd.read_csv(ratings_path)
    logging.info("Loaded %s rating rows.", len(ratings))
    return ratings


def validate_ratings(ratings: pd.DataFrame) -> None:
    """Validate that ratings data has the required MovieLens columns and values."""
    missing_columns = REQUIRED_COLUMNS - set(ratings.columns)
    if missing_columns:
        raise ValueError(f"ratings.csv is missing required columns: {sorted(missing_columns)}")

    if ratings.empty:
        raise ValueError("ratings.csv is empty.")

    required_frame = ratings[list(REQUIRED_COLUMNS)]
    null_counts = required_frame.isna().sum()
    invalid_nulls = null_counts[null_counts > 0]
    if not invalid_nulls.empty:
        raise ValueError(f"Required columns contain missing values: {invalid_nulls.to_dict()}")

    for column in REQUIRED_COLUMNS:
        if not pd.api.types.is_numeric_dtype(ratings[column]):
            raise TypeError(f"Column '{column}' must be numeric.")

    invalid_ratings = ratings[(ratings["rating"] < 0) | (ratings["rating"] > 5)]
    if not invalid_ratings.empty:
        raise ValueError(
            "Column 'rating' contains values outside the expected MovieLens range 0-5."
        )

    logging.info(
        "Validated ratings: %s users, %s movies.",
        ratings["userId"].nunique(),
        ratings["movieId"].nunique(),
    )


def create_user_item_matrix(ratings: pd.DataFrame) -> pd.DataFrame:
    """Create a user-item matrix, averaging duplicate user/movie ratings if present."""
    duplicate_count = ratings.duplicated(subset=["userId", "movieId"]).sum()
    if duplicate_count:
        logging.warning(
            "Found %s duplicate userId/movieId ratings. Averaging duplicates.",
            duplicate_count,
        )

    matrix = ratings.pivot_table(
        index="userId",
        columns="movieId",
        values="rating",
        aggfunc="mean",
    )

    if matrix.empty:
        raise ValueError("User-item matrix is empty after pivoting ratings data.")

    matrix = matrix.sort_index().sort_index(axis=1)
    logging.info("Created user-item matrix with shape %s.", matrix.shape)
    return matrix


def save_matrix(matrix: pd.DataFrame, output_path: Path) -> None:
    """Save the user-item matrix as CSV, creating parent folders when needed."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    matrix.to_csv(output_path)
    logging.info("Saved user-item matrix to %s", output_path)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Build a MovieLens user-item matrix.")
    parser.add_argument(
        "--ratings-path",
        type=Path,
        default=DEFAULT_RATINGS_PATH,
        help="Path to MovieLens ratings.csv.",
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
        help="Where to save the user-item matrix CSV.",
    )
    return parser.parse_args()


def main() -> None:
    """Load ratings, validate them, build the matrix, and save it."""
    configure_logging()
    args = parse_args()

    try:
        ratings = load_ratings(args.ratings_path)
        validate_ratings(ratings)
        matrix = create_user_item_matrix(ratings)
        save_matrix(matrix, args.output_path)

        print("\nUser-item matrix shape:", matrix.shape)
        print("\nSample rows:")
        print(matrix.head())
    except Exception:
        logging.exception("Failed to create user-item matrix.")
        raise


if __name__ == "__main__":
    main()
