from fastapi import FastAPI
from recommender import recommend_movies

app = FastAPI()

@app.get("/")
def home():
    return {"message": "Recommendation System API"}

@app.get("/recommend/{user_id}")
def recommend(user_id: int):
    result = recommend_movies(user_id)
    return result.to_dict(orient='records')