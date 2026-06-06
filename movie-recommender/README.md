# Movie Recommender

An educational movie recommendation system built with **User-Based Collaborative
Filtering** on the MovieLens `ml-latest-small` dataset.

This project is designed for learning. It shows the full path from raw ratings
to recommendations, explanations, evaluation metrics, visualizations, and an
interactive Streamlit app.

The project intentionally focuses on User-Based Collaborative Filtering only.
It does not use item-based filtering, SVD, matrix factorization, deep learning,
or hybrid recommenders.

## What This Project Does

The system answers this question:

> If users with similar taste liked certain movies, which of those movies should
> we recommend to a selected user?

For example, if user `1` is similar to users `57`, `469`, and `156`, and those
users rated a movie highly, the system can recommend that movie to user `1`.

## Project Structure

```text
movie-recommender/
|-- app.py
|-- requirements.txt
|-- README.md
|-- data/
|   `-- ml-latest-small/
|       |-- ratings.csv
|       `-- movies.csv
|-- outputs/
|   |-- user_item_matrix.csv
|   |-- user_similarity.csv
|   |-- recommendations_user_<userId>.csv
|   |-- evaluation_results.csv
|   `-- plots/
|-- src/
|   |-- dataset-connection/
|   |   `-- dataset-connection.py
|   |-- user-item-matrix.py
|   |-- similarity-cosine.py
|   |-- recommendation-engine.py
|   |-- recommendation-explainer.py
|   |-- user-analysis.py
|   |-- visualization.py
|   `-- evaluation.py
`-- tests/
    `-- verify_environment.py
```

## Main Workflow

The project works in stages:

1. **Data loading**
   Load MovieLens `ratings.csv` and `movies.csv`.

2. **User-item matrix**
   Create a table where rows are users, columns are movies, and values are
   ratings.

3. **User-user similarity**
   Calculate how similar users are using cosine similarity.

4. **Recommendation engine**
   Recommend movies that the selected user has not already rated.

5. **Explainability**
   Show which similar users caused a recommendation.

6. **Evaluation**
   Use a train/test split to measure recommender quality.

7. **Visualization and Streamlit**
   Make the system easier to understand visually.

## How User-Based Collaborative Filtering Works

### 1. Users Are Vectors

Each user is represented as a vector of movie ratings.

Example:

```text
User 1 = [4.0, missing, 5.0, 3.5, missing]
User 2 = [4.5, 2.0, missing, 3.0, 5.0]
```

### 2. Cosine Similarity Measures Taste Similarity

Cosine similarity compares the direction of two user rating vectors:

```text
similarity(u, v) = dot(u, v) / (||u|| * ||v||)
```

Higher similarity means two users have more similar rating behavior.

### 3. Similar Users Vote on Unseen Movies

For a selected user, the system looks at movies the user has not rated yet.
Then it checks whether similar users rated those movies.

### 4. Weighted Average Produces the Score

The recommendation score is calculated as:

```text
score = sum(similarity(target_user, neighbor) * neighbor_rating)
        / sum(similarity(target_user, neighbor))
```

Users with higher similarity have more influence.

## Module Guide

| File | Role | What it does |
| --- | --- | --- |
| `src/user-item-matrix.py` | Data Engineering | Builds `outputs/user_item_matrix.csv` |
| `src/similarity-cosine.py` | Machine Learning | Builds `outputs/user_similarity.csv` |
| `src/recommendation-engine.py` | AI Inference | Generates recommendations for a user |
| `src/recommendation-explainer.py` | Explainability | Explains why a movie was recommended |
| `src/user-analysis.py` | Analysis | Shows most and least similar users |
| `src/visualization.py` | Visualization | Creates plots in `outputs/plots/` |
| `src/evaluation.py` | Evaluation | Computes RMSE, MAE, Precision@K, Recall@K |
| `app.py` | Interactive UI | Streamlit app for visual exploration |

## Setup on Windows

Open PowerShell and go to the project:

```powershell
cd D:\CF-project\movie-recommender
```

Activate the virtual environment:

```powershell
.\.venv\Scripts\Activate.ps1
```

If PowerShell blocks activation, run this once:

```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

Then activate again:

```powershell
.\.venv\Scripts\Activate.ps1
```

Install dependencies:

```powershell
python -m pip install -r requirements.txt
```

Verify the environment:

```powershell
python tests\verify_environment.py
```

## Dataset

This project expects the MovieLens `ml-latest-small` dataset here:

```text
movie-recommender/data/ml-latest-small/ratings.csv
movie-recommender/data/ml-latest-small/movies.csv
```

Required columns:

```text
ratings.csv: userId, movieId, rating
movies.csv: movieId, title
```

## Run the Full Pipeline Manually

Run these commands from the project root:

```powershell
cd D:\CF-project\movie-recommender
.\.venv\Scripts\Activate.ps1
```

Build the user-item matrix:

```powershell
python src\user-item-matrix.py
```

Compute user-user cosine similarity:

```powershell
python src\similarity-cosine.py
```

Generate recommendations for user `1`:

```powershell
python src\recommendation-engine.py --user-id 1 --top-n 10
```

Explain one recommended movie:

```powershell
python src\recommendation-explainer.py --user-id 1 --movie-id 3022
```

Analyze similar users:

```powershell
python src\user-analysis.py --user-id 1 --top-n 10
```

Create plots:

```powershell
python src\visualization.py
```

Evaluate the recommender:

```powershell
python src\evaluation.py --k-values 5,10,20
```

## Run the Streamlit App

The Streamlit app is the easiest way to understand the system visually.

From the project root:

```powershell
cd D:\CF-project\movie-recommender
.\.venv\Scripts\Activate.ps1
streamlit run app.py
```

If `streamlit` is not recognized, use:

```powershell
python -m streamlit run app.py
```

Then open:

```text
http://localhost:8501
```

## How to Use the Streamlit App

1. Use the sidebar to select a `user_id`.
2. Choose the number of recommendations: `5`, `10`, or `20`.
3. Adjust how many similar users should influence the result.
4. Read the debug panel:
   - similar users used
   - candidate movies scored
   - matrix sparsity
5. View the top movie recommendations.
6. Open each movie in the Explanation Panel to see which users influenced it.
7. Look at the similarity chart to understand the selected user's closest
   neighbors.

The app is educational, not a production deployment. Its purpose is to make the
recommendation pipeline visible and understandable.

## Important Outputs

After running the pipeline, these files are created:

```text
outputs/user_item_matrix.csv
outputs/user_similarity.csv
outputs/recommendations_user_1.csv
outputs/explanation_user_1_movie_3022.csv
outputs/evaluation_results.csv
outputs/plots/
```

## Example Recommendation Output

```text
movieId,title,score,similar_user_count,similarity_weight_sum
3022,"General, The (1926)",5.0000,2,0.6757
475,"In the Name of the Father (1993)",4.8423,3,0.8454
```

## Example Explanation Output

```text
Movie: General, The (1926)

Recommended because:
- User 57 similarity=0.3450 rating=5.0 contribution=1.7252
- User 469 similarity=0.3307 rating=5.0 contribution=1.6533
- User 156 similarity=0.2092 rating=5.0 contribution=1.0459
```

## Example Evaluation Output

```text
RMSE: 0.9749
MAE: 0.7532
Precision@5: 0.0013
Recall@5: 0.0006
Precision@10: 0.0017
Recall@10: 0.0018
Precision@20: 0.0026
Recall@20: 0.0066
```

## How to Interpret Evaluation Metrics

`RMSE`: Rating prediction error. Lower is better. Penalizes large errors more
strongly.

`MAE`: Average absolute rating error. Lower is better and easier to interpret.

`Precision@K`: Of the top K recommended movies, how many were actually relevant
in the held-out test data. Higher is better.

`Recall@K`: Of all relevant held-out movies, how many appeared in the top K.
Higher is better.

For recommendation systems, ranking metrics like `Precision@K` and `Recall@K`
often matter more than RMSE because users usually see only the top few
recommendations.

## Common Pitfalls

**Sparsity**: Most users rate only a small number of movies. This makes user
similarity noisy.

**Cold start**: New users and new movies have no rating history, so User-Based
CF cannot understand them well.

**Popularity bias**: Movies rated by many users are easier to recommend.

**Data leakage**: Evaluation must build the matrix and similarity from training
data only. Test data must stay unseen until evaluation.

**Weak neighbors**: If similar users are not truly similar, recommendations may
look random or generic.

## Current Project Focus

This project is complete as a User-Based Collaborative Filtering learning
system. Good future work within the same scope would be:

```text
- better neighbor filtering
- user mean-centering
- minimum support thresholds
- cleaner UI controls
- more educational visualizations
```

The goal is to understand User-Based Collaborative Filtering deeply before
moving to other recommender approaches.
