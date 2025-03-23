import pandas as pd
from threading import Lock

COLUMNS = {
    "ratelog": ["movie_id", "rating", "time", "user_id"],
    "movies": ["movie_id", "tmdb_id", "imdb_id", "title", "original_title", "adult", "belongs_to_collection", "budget", "genres", "homepage", "original_language", "overview", "popularity", "poster_path", "production_companies", "production_countries", "release_date", "revenue", "runtime", "spoken_languages", "status", "vote_average", "vote_count"],
    "users": ["user_id", "age", "occupation", "gender"], 
    # "watch_movies": [],
}

DATASETS = {
    "ratelog": pd.DataFrame(columns=COLUMNS["ratelog"]),
    "movies": pd.DataFrame(columns=COLUMNS["movies"]),
    "users": pd.DataFrame(columns=COLUMNS["users"]),
    # "watch_movies": pd.DataFrame(columns=COLUMNS["watch_movies"]),
}

DATA_LOCKS = {
    "ratelog": Lock(),
    "movies": Lock(),
    "users": Lock(),
    # "watch_movies": Lock(),
}

DATA_FILENAMES = {
    "ratelog": "ratelog.csv",
    "movies": "movies.csv",
    "users": "users.csv",
    # "watch_movies": "watch_movies.csv",
}

DATA_PATH = "data/full_data"


def get_dataset(name):
    """
    Get a dataset by name.
    """
    
    with DATA_LOCKS[name]:
        data = DATASETS[name].copy()
    
    return data

# General update requires getting the dataset -> see for new data and update

def update_dataset(name, data):
    """
    Update a dataset by name.
    """
    
    with DATA_LOCKS[name]:
        DATASETS[name] = data.copy()
    
    return


def download_data():
    """
    Download the data from the server.
    """

    # Download the data
    for name, filename in DATA_FILENAMES.items():
        data = pd.read_csv(f"{DATA_PATH}/{filename}")
        update_dataset(name, data)
    
    return


def save_dataset(name):
    """
    Save a dataset to disk.
    """
    
    with DATA_LOCKS[name]:
        DATASETS[name].to_csv(f"{DATA_PATH}/{DATA_FILENAMES[name]}", index=False)
    
    return

def save_datasets():
    """
    Save all datasets to disk.
    """
    
    for name in DATASETS.keys():
        save_dataset(name)
    
    return




