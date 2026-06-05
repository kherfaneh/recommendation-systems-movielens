import pandas as pd

# Load datasets
ratings = pd.read_csv("data/ml-latest-small/ratings.csv")
movies = pd.read_csv("data/ml-latest-small/movies.csv")

# View basic info
print(ratings.head())
print(movies.head())

print(ratings.info())
print(movies.info())