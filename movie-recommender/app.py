"""Educational Streamlit UI for User-Based Collaborative Filtering.

Run locally:
    streamlit run app.py

The app is intentionally simple and explanatory. It loads the existing project
artifacts, generates recommendations for a selected user, and shows how similar
users contribute to each recommendation.
"""

from __future__ import annotations

from pathlib import Path
from typing import NamedTuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st


PROJECT_ROOT = Path(__file__).resolve().parent
MATRIX_PATH = PROJECT_ROOT / "outputs" / "user_item_matrix.csv"
SIMILARITY_PATH = PROJECT_ROOT / "outputs" / "user_similarity.csv"
MOVIES_PATH = PROJECT_ROOT / "data" / "ml-latest-small" / "movies.csv"


class AppData(NamedTuple):
    """Container for the CSV artifacts used by the app."""

    user_item_matrix: pd.DataFrame
    user_similarity: pd.DataFrame
    movies: pd.DataFrame


def require_file(path: Path, label: str) -> None:
    """Show a friendly Streamlit error if a required project artifact is missing."""
    if not path.exists():
        st.error(f"Missing {label}: `{path}`")
        st.info("Run the pipeline scripts first, then restart this app.")
        st.stop()


@st.cache_data(show_spinner="Loading recommender artifacts...")
def load_data() -> AppData:
    """Load user-item matrix, user similarity matrix, and MovieLens movie metadata."""
    require_file(MATRIX_PATH, "user-item matrix")
    require_file(SIMILARITY_PATH, "user similarity matrix")
    require_file(MOVIES_PATH, "MovieLens movies.csv")

    user_item_matrix = pd.read_csv(MATRIX_PATH, index_col=0)
    user_similarity = pd.read_csv(SIMILARITY_PATH, index_col=0)
    movies = pd.read_csv(MOVIES_PATH)

    user_item_matrix.index = user_item_matrix.index.astype(str)
    user_item_matrix.columns = user_item_matrix.columns.astype(str)
    user_similarity.index = user_similarity.index.astype(str)
    user_similarity.columns = user_similarity.columns.astype(str)
    movies["movieId"] = movies["movieId"].astype(str)

    if user_item_matrix.empty or user_similarity.empty:
        st.error("The recommender artifacts are empty. Rebuild the pipeline outputs.")
        st.stop()
    if set(user_item_matrix.index) != set(user_similarity.index):
        st.error("User IDs in the matrix and similarity files do not match.")
        st.stop()
    if "title" not in movies.columns:
        st.error("movies.csv must contain a `title` column.")
        st.stop()
    if "genres" not in movies.columns:
        movies["genres"] = "(no genres listed)"

    return AppData(user_item_matrix, user_similarity, movies)


def matrix_sparsity(user_item_matrix: pd.DataFrame) -> float:
    """Return the percentage of missing values in the user-item matrix."""
    total_cells = user_item_matrix.shape[0] * user_item_matrix.shape[1]
    missing_cells = user_item_matrix.isna().sum().sum()
    return float(100 * missing_cells / total_cells)


def get_seen_movies(
    user_id: str,
    user_item_matrix: pd.DataFrame,
    movies: pd.DataFrame,
) -> pd.DataFrame:
    """Return movies already rated by the selected user, sorted by rating."""
    ratings = pd.to_numeric(user_item_matrix.loc[user_id], errors="coerce").dropna()
    seen = ratings.rename("rating").reset_index()
    seen = seen.rename(columns={seen.columns[0]: "movieId"})
    seen["movieId"] = seen["movieId"].astype(str)
    seen = seen.merge(movies[["movieId", "title", "genres"]], on="movieId", how="left")
    seen["title"] = seen["title"].fillna("Unknown title")
    seen["genres"] = seen["genres"].fillna("(no genres listed)")
    return seen[["movieId", "title", "genres", "rating"]].sort_values(
        by=["rating", "title"],
        ascending=[False, True],
    )


def favorite_genres(seen_movies: pd.DataFrame, top_n: int = 5) -> pd.DataFrame:
    """Estimate favorite genres from the user's highest-rated movies."""
    if seen_movies.empty or "genres" not in seen_movies.columns:
        return pd.DataFrame(columns=["genre", "rating_count", "average_rating"])

    rows: list[dict[str, float | str]] = []
    for row in seen_movies.itertuples(index=False):
        for genre in str(row.genres).split("|"):
            if genre and genre != "(no genres listed)":
                rows.append({"genre": genre, "rating": float(row.rating)})

    if not rows:
        return pd.DataFrame(columns=["genre", "rating_count", "average_rating"])

    genre_frame = pd.DataFrame(rows)
    summary = (
        genre_frame.groupby("genre")["rating"]
        .agg(rating_count="count", average_rating="mean")
        .reset_index()
        .sort_values(by=["average_rating", "rating_count"], ascending=[False, False])
        .head(top_n)
    )
    return summary


def get_similar_users(
    user_id: str,
    user_similarity: pd.DataFrame,
    max_neighbors: int = 50,
    min_similarity: float = 0.0,
) -> pd.Series:
    """Return the most similar users to a selected user."""
    similarities = pd.to_numeric(user_similarity.loc[user_id], errors="coerce")
    similarities = similarities.drop(labels=user_id, errors="ignore").dropna()
    similarities = similarities[similarities > min_similarity]
    return similarities.sort_values(ascending=False).head(max_neighbors)


def calculate_recommendation_scores(
    user_id: str,
    user_item_matrix: pd.DataFrame,
    similar_users: pd.Series,
) -> pd.DataFrame:
    """Score unrated movies using similarity-weighted neighbor ratings.

    This is the AI inference step: similar users vote on movies the target user
    has not seen, and each vote is weighted by cosine similarity.
    """
    target_ratings = pd.to_numeric(user_item_matrix.loc[user_id], errors="coerce")
    candidate_movies = target_ratings[target_ratings.isna()].index
    neighbor_ratings = user_item_matrix.loc[similar_users.index, candidate_movies]
    neighbor_ratings = neighbor_ratings.apply(pd.to_numeric, errors="coerce")

    support = neighbor_ratings.notna().sum(axis=0)
    rated_mask = neighbor_ratings.notna()
    weighted_sum = neighbor_ratings.mul(similar_users, axis=0).sum(axis=0, skipna=True)
    weight_sum = rated_mask.mul(similar_users, axis=0).sum(axis=0)
    scores = (weighted_sum / weight_sum).replace([np.inf, -np.inf], np.nan).dropna()

    score_frame = pd.DataFrame(
        {
            "movieId": scores.index.astype(str),
            "score": scores.values,
            "similar_user_count": support.loc[scores.index].values,
            "similarity_weight_sum": weight_sum.loc[scores.index].values,
        }
    )
    return score_frame.sort_values(
        by=["score", "similar_user_count", "similarity_weight_sum"],
        ascending=[False, False, False],
    )


def attach_movie_titles(scores: pd.DataFrame, movies: pd.DataFrame, top_n: int) -> pd.DataFrame:
    """Attach human-readable movie titles to recommendation scores."""
    recommendations = scores.head(top_n).merge(
        movies[["movieId", "title"]],
        on="movieId",
        how="left",
    )
    recommendations["title"] = recommendations["title"].fillna(
        "Unknown title (movieId=" + recommendations["movieId"] + ")"
    )
    return recommendations[
        ["movieId", "title", "score", "similar_user_count", "similarity_weight_sum"]
    ]


def get_top_contributors(
    user_id: str,
    movie_id: str,
    user_item_matrix: pd.DataFrame,
    similar_users: pd.Series,
    top_n: int = 5,
) -> pd.DataFrame:
    """Show which similar users contributed most to one movie recommendation."""
    movie_ratings = pd.to_numeric(user_item_matrix[movie_id], errors="coerce").dropna()
    neighbor_ids = similar_users.index.intersection(movie_ratings.index)

    rows: list[dict[str, float | str]] = []
    for neighbor_id in neighbor_ids:
        similarity = float(similar_users.loc[neighbor_id])
        rating = float(movie_ratings.loc[neighbor_id])
        rows.append(
            {
                "user_id": str(neighbor_id),
                "similarity": similarity,
                "rating": rating,
                "weighted_contribution": similarity * rating,
            }
        )

    del user_id
    contributors = pd.DataFrame(rows)
    if contributors.empty:
        return contributors
    return contributors.sort_values(
        by=["weighted_contribution", "similarity"],
        ascending=False,
    ).head(top_n)


def score_formula(contributors: pd.DataFrame) -> str:
    """Create a readable weighted-average formula from contributor rows."""
    if contributors.empty:
        return "No contributors available."

    numerator = contributors["weighted_contribution"].sum()
    denominator = contributors["similarity"].sum()
    score = numerator / denominator if denominator > 0 else np.nan
    return (
        "final score = "
        f"sum(weighted contributions) / sum(similarities) = "
        f"{numerator:.4f} / {denominator:.4f} = {score:.4f}"
    )


def plot_top_similar_users(similar_users: pd.Series, top_n: int = 10) -> plt.Figure:
    """Create a matplotlib bar chart for the top similar users."""
    top_users = similar_users.head(top_n).sort_values()
    fig, ax = plt.subplots(figsize=(8, 4.8))
    ax.barh(top_users.index.astype(str), top_users.values, color="#2f6f9f")
    ax.set_title(f"Top {top_n} Most Similar Users")
    ax.set_xlabel("Cosine similarity")
    ax.set_ylabel("User ID")
    ax.set_xlim(0, max(0.01, float(top_users.max()) * 1.15))
    fig.tight_layout()
    return fig


def render_selected_user_profile(
    user_id: str,
    user_item_matrix: pd.DataFrame,
    movies: pd.DataFrame,
) -> pd.DataFrame:
    """Render rating behavior and favorite genres for the selected user."""
    st.subheader("Selected User Profile")
    seen_movies = get_seen_movies(user_id, user_item_matrix, movies)

    col1, col2, col3 = st.columns(3)
    col1.metric("Rated movies", len(seen_movies))
    col2.metric(
        "Average rating",
        f"{seen_movies['rating'].mean():.2f}" if not seen_movies.empty else "n/a",
    )
    col3.metric("Highest rating", f"{seen_movies['rating'].max():.1f}" if not seen_movies.empty else "n/a")

    with st.expander("Favorite genres", expanded=True):
        genres = favorite_genres(seen_movies)
        if genres.empty:
            st.info("No genre information is available for this user.")
        else:
            st.dataframe(genres, width="stretch", hide_index=True)

    with st.expander("Top rated movies by this user", expanded=True):
        st.dataframe(seen_movies.head(10), width="stretch", hide_index=True)

    return seen_movies


def render_movies_already_seen(seen_movies: pd.DataFrame) -> None:
    """Render all movies rated by the selected user."""
    st.subheader("Movies Already Seen")
    st.caption("These movies are removed from recommendation candidates.")
    st.dataframe(seen_movies, width="stretch", hide_index=True)


def render_matrix_explorer(user_item_matrix: pd.DataFrame) -> None:
    """Render matrix size, sparsity, and a real sample of the matrix."""
    st.subheader("Matrix Explorer and Sparsity Explanation")
    users, movies = user_item_matrix.shape
    total_cells = users * movies
    observed_ratings = int(user_item_matrix.notna().sum().sum())
    missing_ratings = int(user_item_matrix.isna().sum().sum())

    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Users", users)
    col2.metric("Movies", movies)
    col3.metric("Known ratings", observed_ratings)
    col4.metric("Missing ratings", missing_ratings)
    col5.metric("Sparsity", f"{100 * missing_ratings / total_cells:.2f}%")

    st.caption(
        "Sparsity means most user/movie pairs have no rating. This is why the "
        "system needs similar users to infer preferences."
    )
    st.dataframe(user_item_matrix.iloc[:8, :8], width="stretch")


def render_recommendation_journey(
    user_id: str,
    seen_movies: pd.DataFrame,
    similar_users: pd.Series,
    scores: pd.DataFrame,
    recommendations: pd.DataFrame,
    user_item_matrix: pd.DataFrame,
) -> None:
    """Render the step-by-step journey from selected user to final recommendations."""
    st.subheader("Recommendation Journey")
    st.markdown("**User -> Find Similar Users -> Collect Their Highly Rated Movies -> Remove Already Seen Movies -> Compute Weighted Scores -> Generate Final Recommendations**")

    with st.expander("1. User", expanded=True):
        st.write(f"Selected user: `{user_id}`")
        st.write(f"This user has already rated `{len(seen_movies)}` movies.")

    with st.expander("2. Find Similar Users", expanded=True):
        st.dataframe(
            similar_users.head(10).rename("cosine_similarity").reset_index().rename(columns={"index": "user_id"}),
            width="stretch",
            hide_index=True,
        )

    with st.expander("3. Collect Highly Rated Movies From Similar Users"):
        top_neighbor_ids = similar_users.head(10).index
        neighbor_ratings = user_item_matrix.loc[top_neighbor_ids].apply(pd.to_numeric, errors="coerce")
        highly_rated = (
            neighbor_ratings.stack()
            .rename("rating")
            .reset_index()
        )
        highly_rated = highly_rated.rename(
            columns={
                highly_rated.columns[0]: "neighbor_user_id",
                highly_rated.columns[1]: "movieId",
            }
        )
        highly_rated = highly_rated[highly_rated["rating"] >= 4.0].head(20)
        st.dataframe(highly_rated, width="stretch", hide_index=True)

    with st.expander("4. Remove Already Seen Movies"):
        seen_ids = set(seen_movies["movieId"].astype(str))
        candidate_count_before = user_item_matrix.shape[1]
        st.write(f"All movies in matrix: `{candidate_count_before}`")
        st.write(f"Movies already seen by user {user_id}: `{len(seen_ids)}`")
        st.write(f"Scored candidate movies after filtering: `{len(scores)}`")

    with st.expander("5. Compute Weighted Scores"):
        st.dataframe(
            scores.head(10)[["movieId", "score", "similar_user_count", "similarity_weight_sum"]],
            width="stretch",
            hide_index=True,
        )

    with st.expander("6. Generate Final Recommendations", expanded=True):
        st.dataframe(recommendations, width="stretch", hide_index=True)


def render_user_comparison(
    user_id: str,
    similar_users: pd.Series,
    user_item_matrix: pd.DataFrame,
    movies: pd.DataFrame,
) -> None:
    """Compare selected user with their most similar user."""
    st.subheader("Compare Two Users")
    if similar_users.empty:
        st.info("No similar user available for comparison.")
        return

    neighbor_id = str(similar_users.index[0])
    similarity_score = float(similar_users.iloc[0])
    user_ratings = pd.to_numeric(user_item_matrix.loc[user_id], errors="coerce")
    neighbor_ratings = pd.to_numeric(user_item_matrix.loc[neighbor_id], errors="coerce")
    overlap = user_ratings.dropna().index.intersection(neighbor_ratings.dropna().index)

    st.write(
        f"Comparing selected user `{user_id}` with most similar user `{neighbor_id}` "
        f"(cosine similarity = `{similarity_score:.4f}`)."
    )

    if len(overlap) == 0:
        st.info("These users have no overlapping rated movies in the matrix.")
        return

    comparison = pd.DataFrame(
        {
            "movieId": overlap.astype(str),
            f"user_{user_id}_rating": user_ratings.loc[overlap].values,
            f"user_{neighbor_id}_rating": neighbor_ratings.loc[overlap].values,
        }
    )
    comparison = comparison.merge(movies[["movieId", "title"]], on="movieId", how="left")
    comparison = comparison[["movieId", "title", f"user_{user_id}_rating", f"user_{neighbor_id}_rating"]]
    st.dataframe(comparison.sort_values("title").head(30), width="stretch", hide_index=True)


def render_educational_intro() -> None:
    """Render the beginner-friendly explanation at the top of the app."""
    st.title("MovieLens User-Based Collaborative Filtering")
    st.subheader("How User-Based Collaborative Filtering Works")

    st.markdown(
        """
        1. **Users are represented as vectors**: every user is a row of movie
           ratings in the user-item matrix.
        2. **Cosine similarity measures similarity**: users with similar rating
           patterns get higher similarity scores.
        3. **Similar users vote on unseen movies**: for the selected user, the
           system looks at movies rated by similar users but not yet rated by
           the selected user.
        4. **A weighted average produces the recommendation score**: ratings
           from more similar users count more.
        """
    )

    st.markdown(
        """
        **Data Engineering**: loading ratings and building the user-item matrix.

        **Machine Learning**: computing user-user cosine similarity.

        **AI Inference**: scoring unseen movies for a selected user.

        **Explainability Layer**: showing which similar users influenced each
        recommendation.
        """
    )


def render_debug_panel(
    similar_users: pd.Series,
    scores: pd.DataFrame,
    user_item_matrix: pd.DataFrame,
) -> None:
    """Render useful recommender debugging information."""
    st.subheader("Debug Panel")
    col1, col2, col3 = st.columns(3)
    col1.metric("Similar users used", len(similar_users))
    col2.metric("Candidate movies", len(scores))
    col3.metric("Matrix sparsity", f"{matrix_sparsity(user_item_matrix):.2f}%")


def render_recommendations(
    recommendations: pd.DataFrame,
    user_id: str,
    user_item_matrix: pd.DataFrame,
    similar_users: pd.Series,
) -> None:
    """Render recommendations and per-movie explanation panels."""
    st.subheader("Top-N Recommendations")
    st.dataframe(
        recommendations,
        width="stretch",
        hide_index=True,
        column_config={
            "score": st.column_config.NumberColumn("score", format="%.4f"),
            "similar_user_count": "similar user count",
        },
    )

    st.subheader("Explanation Panel")
    for row in recommendations.itertuples(index=False):
        with st.expander(f"{row.title} | score={row.score:.4f}"):
            contributors = get_top_contributors(
                user_id=user_id,
                movie_id=str(row.movieId),
                user_item_matrix=user_item_matrix,
                similar_users=similar_users,
            )
            if contributors.empty:
                st.warning("No contributor details available for this movie.")
                continue

            st.markdown(f"**Movie:** {row.title}")
            st.markdown("---")
            st.dataframe(
                contributors,
                width="stretch",
                hide_index=True,
                column_config={
                    "similarity": st.column_config.NumberColumn("similarity", format="%.4f"),
                    "rating": st.column_config.NumberColumn("rating", format="%.1f"),
                    "weighted_contribution": st.column_config.NumberColumn(
                        "weighted contribution",
                        format="%.4f",
                    ),
                },
            )
            st.code(score_formula(contributors), language="text")
            st.caption("Each contribution is similarity * rating.")


def main() -> None:
    """Run the Streamlit application."""
    st.set_page_config(page_title="User-Based CF Explorer", layout="wide")
    render_educational_intro()

    data = load_data()
    user_ids = sorted(data.user_item_matrix.index.tolist(), key=lambda value: int(value))

    st.sidebar.header("Controls")
    selected_user = st.sidebar.selectbox("Select user_id", user_ids, index=0)
    top_n = st.sidebar.select_slider("Top-N recommendations", options=[5, 10, 20], value=10)
    neighbor_count = st.sidebar.slider("Similar users to use", 5, 100, 50, step=5)

    similar_users = get_similar_users(
        selected_user,
        data.user_similarity,
        max_neighbors=neighbor_count,
    )

    if similar_users.empty:
        st.error("No similar users found for the selected user.")
        st.stop()

    scores = calculate_recommendation_scores(
        selected_user,
        data.user_item_matrix,
        similar_users,
    )
    if scores.empty:
        st.error("No candidate movies could be scored for this user.")
        st.stop()

    recommendations = attach_movie_titles(scores, data.movies, top_n)

    seen_movies = render_selected_user_profile(
        selected_user,
        data.user_item_matrix,
        data.movies,
    )
    render_matrix_explorer(data.user_item_matrix)
    render_debug_panel(similar_users, scores, data.user_item_matrix)
    render_recommendation_journey(
        selected_user,
        seen_movies,
        similar_users,
        scores,
        recommendations,
        data.user_item_matrix,
    )
    render_movies_already_seen(seen_movies)
    render_user_comparison(
        selected_user,
        similar_users,
        data.user_item_matrix,
        data.movies,
    )
    st.subheader("User Similarity Visualization")
    st.pyplot(plot_top_similar_users(similar_users, top_n=10))
    render_recommendations(
        recommendations,
        selected_user,
        data.user_item_matrix,
        similar_users,
    )


if __name__ == "__main__":
    main()
