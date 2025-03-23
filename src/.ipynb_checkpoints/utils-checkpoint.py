import pickle
import pandas as pd
from typing import List, Tuple
import torch

MODELS_PATH = "models/"

def save_model(model, model_name) -> None:
    """
    This function will save a model to the models folder.
    Args:
        model: The model to save
        model_name (str): The name of the model
    Returns:
        None
    """
    with open(f"{MODELS_PATH}{model_name}.pkl", "wb") as f:
        pickle.dump(model, f)
    
def load_model(model_name) -> object:
    """
    This function will load a model from the models folder.
    Args:
        model_name (str): The name of the model to load
    Returns:
        model: The model loaded
    """
    with open(f"models/ncf_model.pkl", "rb") as f:
        return pickle.load(f)
    

def get_top_n_movies(data, n) -> List[Tuple[str, float]]:
    """
    This function will give the predictions for unseen users.
    It will return the highest ranked movies from all reviews.
    Args:
        data (pd.DataFrame): The data containing the reviews
        n (int): The number of movies to return
    Returns:
        list: A list of tuples with the movie_id and the rating
    """

    # Group data by movie_id and calculate the mean rating
    movie_ratings = data.groupby('movie_id')['rating'].mean().reset_index() 
    # Sort the movies by rating
    top_movies = movie_ratings.sort_values(by='rating', ascending=False).head(n)
    
    # Return with the movie_id and the rating as a list of tuples
    return list(zip(top_movies['movie_id'], top_movies['rating']))


if __name__ == "__main__":

    # Load the data
    data = pd.read_csv('data/recommendation_data.csv')

    # Get the top 10 movies
    top_movies = get_top_n_movies(data, 10)

    print(top_movies)

    # Load a model with torch

    # model = load_model("ncf_model")

    # # Get the size of the model
    # model_size = get_model_size(model)

    # print(model_size)

    pass