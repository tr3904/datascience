
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import pickle

# Load data
movies = pd.read_csv('C:/Users/ACER/Downloads/movies.csv')
ratings = pd.read_csv('C:/Users/ACER/Downloads/ratings.csv')

# Create user-item matrix
user_item_matrix = ratings.pivot_table(index='userId', columns='movieId', values='rating').fillna(0)

# Compute similarity
user_similarity = cosine_similarity(user_item_matrix)

# Save model
with open('models/model.pkl', 'wb') as f:
    pickle.dump((user_item_matrix, user_similarity), f)

print("Model trained and saved!")