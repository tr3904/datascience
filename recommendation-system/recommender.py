import pickle
import pandas as pd

# Load model
with open('models/model.pkl', 'rb') as f:
    user_item_matrix, user_similarity = pickle.load(f)

movies = pd.read_csv('C:/Users/ACER/Downloads/movies.csv')

def recommend_movies(user_id, top_n=5):
    user_index = user_id - 1

    similarity_scores = list(enumerate(user_similarity[user_index]))
    similarity_scores = sorted(similarity_scores, key=lambda x: x[1], reverse=True)

    similar_users = [i[0] for i in similarity_scores[1:6]]

    recommended_movies = {}

    for sim_user in similar_users:
        for movie_id, rating in user_item_matrix.iloc[sim_user].items():
            if user_item_matrix.iloc[user_index][movie_id] == 0 and rating > 3:
                recommended_movies[movie_id] = rating

    recommended_movies = sorted(recommended_movies.items(), key=lambda x: x[1], reverse=True)

    movie_ids = [movie[0] for movie in recommended_movies[:top_n]]

    return movies[movies['movieId'].isin(movie_ids)]