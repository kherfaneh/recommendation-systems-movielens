# Movie Recommender

A complete educational User-Based Collaborative Filtering recommender system
using the MovieLens `ml-latest-small` dataset.

This project intentionally focuses on User-Based Collaborative Filtering only.
It does not implement item-based filtering, SVD, matrix factorization, deep
learning, or hybrid recommenders.

## Project Architecture

```text
movie-recommender/
├── data/ml-latest-small/
│   ├── ratings.csv
│   └── movies.csv
├── outputs/
│   ├── user_item_matrix.csv
│   ├── user_similarity.csv
│   ├── recommendations_user_<userId>.csv
│   ├── evaluation_results.csv
│   └── plots/
├── src/
│   ├── user-item-matrix.py
│   ├── similarity-cosine.py
│   ├── recommendation-engine.py
│   ├── recommendation-explainer.py
│   ├── user-analysis.py
│   ├── visualization.py
│   └── evaluation.py
└── requirements.txt
```

## Environment

Activate the local Windows virtual environment:

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

## User-Based CF Workflow

1. Load and validate MovieLens ratings.
2. Build a user-item matrix where rows are users and columns are movies.
3. Fill missing ratings with zero only for similarity calculation.
4. Compute user-user cosine similarity.
5. For a target user, find similar users.
6. Score unrated movies using similar users' weighted ratings.
7. Export recommendations and explanations.
8. Evaluate on held-out test ratings without data leakage.

## Cosine Similarity

Each user is represented as a rating vector. Cosine similarity measures whether
two users point in a similar direction:

```text
similarity(u, v) = dot(u, v) / (||u|| * ||v||)
```

In this project, higher cosine similarity means two users have more similar
rating behavior. Similar users are more influential during recommendation.

## Recommendation Score

For a target user and candidate movie:

```text
score = sum(similarity(target, neighbor) * neighbor_rating)
        / sum(similarity(target, neighbor))
```

Only movies the target user has not already rated are recommended.

## Evaluation Metrics

`RMSE`: Measures rating prediction error and penalizes large mistakes. Lower is
better.

`MAE`: Measures average absolute rating prediction error. Lower is better.

`Precision@K`: Of the top K recommended movies, how many were relevant in the
held-out test set. Higher is better.

`Recall@K`: Of the relevant held-out movies, how many appeared in the top K.
Higher is better.

Evaluation rebuilds the matrix and similarity from training data only. This is
important because using test ratings during model building would cause data
leakage and make results unrealistically optimistic.

## Run Modules

Build the user-item matrix:

```powershell
python src\user-item-matrix.py
```

Compute user-user cosine similarity:

```powershell
python src\similarity-cosine.py
```

Generate recommendations:

```powershell
python src\recommendation-engine.py --user-id 1 --top-n 10
```

Explain a recommendation:

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

## Example Outputs

Recommendation output:

```text
movieId,title,score,similar_user_count,similarity_weight_sum
3022,"General, The (1926)",5.0,2,0.6757
```

Explanation output:

```text
Movie: General, The (1926)
Recommended because:
- User 266 (similarity=0.3574) rated 5.0
- User 313 (similarity=0.3183) rated 5.0
```

Evaluation output:

```text
RMSE: 0.9749
MAE: 0.7532
Precision@10: 0.0022
Recall@10: 0.0022
```

## Module Roles

`user-item-matrix.py`: Data Engineering. Converts raw ratings into a model-ready
matrix.

`similarity-cosine.py`: Machine Learning. Computes user-user similarity from
rating vectors.

`recommendation-engine.py`: AI Inference. Produces recommendations for a target
user.

`recommendation-explainer.py`: Explainability. Shows which neighbors caused a
recommendation.

`user-analysis.py`: Recommendation-System Analysis. Inspects similarity
relationships.

`visualization.py`: Analysis and Communication. Visualizes sparsity, ratings,
similarities, and recommendation score behavior.

`evaluation.py`: Evaluation. Measures quality on unseen ratings and prevents
data leakage.

## Common Pitfalls

Sparsity: Most users rate only a small fraction of movies, so similarity can be
weak or noisy.

Cold start: New users or new movies have no rating history, so User-Based CF has
little information.

Popularity bias: Popular movies are more likely to be recommended because many
neighbors rated them.

Data leakage: Evaluation must build the matrix and similarity from training data
only, never from the full dataset.
