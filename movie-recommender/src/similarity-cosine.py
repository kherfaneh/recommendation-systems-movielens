from sklearn.metrics.pairwise import cosine_similarity

# fill NaN with 0 for similarity calculation
matrix_filled = user_item_matrix.fillna(0)

user_similarity = cosine_similarity(matrix_filled)

print(user_similarity.shape)