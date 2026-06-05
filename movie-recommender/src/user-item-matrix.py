user_item_matrix = ratings.pivot(
    index='userId',
    columns='movieId',
    values='rating'
)

print(user_item_matrix.shape)
print(user_item_matrix.head())