"""Compute user-user cosine similarity from a user-item matrix.

Expected input:
    outputs/user_item_matrix.csv

Default output:
    outputs/user_similarity.csv
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MATRIX_PATH = PROJECT_ROOT / "outputs" / "user_item_matrix.csv"
DEFAULT_OUTPUT_PATH = PROJECT_ROOT / "outputs" / "user_similarity.csv"


def configure_logging() -> None:
    """Configure readable console logging for this script."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s: %(message)s",
    )


def load_user_item_matrix(matrix_path: Path) -> pd.DataFrame:
    """Load a saved user-item matrix from CSV."""
    if not matrix_path.exists():
        raise FileNotFoundError(
            f"User-item matrix file not found: {matrix_path}. "
            "Run src/user-item-matrix.py first."
        )

    logging.info("Loading user-item matrix from %s", matrix_path)
    matrix = pd.read_csv(matrix_path, index_col=0)
    logging.info("Loaded matrix with shape %s.", matrix.shape)
    return matrix


def validate_matrix(matrix: pd.DataFrame) -> None:
    """Validate matrix dimensions and numeric values before similarity calculation."""
    if matrix.empty:
        raise ValueError("User-item matrix is empty.")

    if matrix.shape[0] < 2:
        raise ValueError("At least two users are required to compute user-user similarity.")

    if matrix.shape[1] < 1:
        raise ValueError("At least one movie column is required.")

    numeric_matrix = matrix.apply(pd.to_numeric, errors="coerce")
    introduced_missing = numeric_matrix.isna() & matrix.notna()
    if introduced_missing.any().any():
        bad_columns = introduced_missing.columns[introduced_missing.any()].tolist()
        raise TypeError(f"Matrix contains non-numeric values in columns: {bad_columns}")

    all_missing_rows = numeric_matrix.isna().all(axis=1).sum()
    if all_missing_rows:
        raise ValueError(f"{all_missing_rows} users have no ratings in the matrix.")

    logging.info(
        "Validated matrix: %s users, %s movies, %.2f%% missing values.",
        matrix.shape[0],
        matrix.shape[1],
        100 * numeric_matrix.isna().sum().sum() / numeric_matrix.size,
    )


def compute_user_similarity(matrix: pd.DataFrame) -> pd.DataFrame:
    """Compute user-user cosine similarity after replacing missing ratings with zero."""
    numeric_matrix = matrix.apply(pd.to_numeric, errors="coerce")

    # Missing values mean "user did not rate this movie"; use 0 for vector similarity.
    matrix_filled = numeric_matrix.fillna(0)
    similarity_values = cosine_similarity(matrix_filled)

    similarity = pd.DataFrame(
        similarity_values,
        index=matrix.index,
        columns=matrix.index,
    )

    logging.info("Computed user-user similarity matrix with shape %s.", similarity.shape)
    return similarity


def save_similarity(similarity: pd.DataFrame, output_path: Path) -> None:
    """Save the similarity matrix as CSV, creating parent folders when needed."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    similarity.to_csv(output_path)
    logging.info("Saved user similarity matrix to %s", output_path)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Compute MovieLens user cosine similarity.")
    parser.add_argument(
        "--matrix-path",
        type=Path,
        default=DEFAULT_MATRIX_PATH,
        help="Path to the user-item matrix CSV.",
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
        help="Where to save the user-user similarity CSV.",
    )
    return parser.parse_args()


def main() -> None:
    """Load a user-item matrix, compute cosine similarity, and save it."""
    configure_logging()
    args = parse_args()

    try:
        matrix = load_user_item_matrix(args.matrix_path)
        validate_matrix(matrix)
        similarity = compute_user_similarity(matrix)
        save_similarity(similarity, args.output_path)

        print("\nSimilarity matrix shape:", similarity.shape)
        print("\nSample values:")
        print(similarity.iloc[:5, :5])
    except Exception:
        logging.exception("Failed to compute user-user cosine similarity.")
        raise


if __name__ == "__main__":
    main()
