import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from api_get import get_movie_data, get_user_data
import os
from ast import literal_eval
import data_manager as dm

DATA_PATH = r"data"
DATA_FILE = "recommendation_data.csv"
RATELOG_FILE = "ratelog_2025-02.csv"
MOVIES_FILE = "movies.csv"
USERS_FILE = "users.csv"
COLUMNS = ["movie_id", "user_id", "rating"]
EXTRA = COLUMNS + ["popularity", "vote_average", "age", "gender"]     # Genres should be added when they appear
TH = 4

TRAINING_DATA_PATH = "data/training_data"


def load_data(file_path):
    """
    Load data from a csv file.

    Args:
        file_path: str, path to the csv file.

    Returns:
        pandas.DataFrame, the loaded data.
    """
    data = pd.read_csv(file_path)
    return data

def create_data_grafo(data):
    """
    Create a graph of the data.

    Args:
        data: pandas.DataFrame, the data to be plotted.

    Returns:
        None
    """

    # Create nodes with "movie_id" and "user_id"
    G = nx.Graph()
    G.add_nodes_from(data["movie_id"], bipartite=0)
    G.add_nodes_from(data["user_id"], bipartite=1)

    # Create edges between "movie_id" and "user_id" and add "rating" as an attribute
    for i in range(len(data)):
        G.add_edge(data["movie_id"][i], data["user_id"][i], rating=data["rating"][i])
    
    # Plot the graph
    pos = nx.spring_layout(G)
    nx.draw(G, pos, with_labels=True)
    labels = nx.get_edge_attributes(G, 'rating')
    nx.draw_networkx_edge_labels(G, pos, edge_labels=labels)
    plt.show()
    
def print_stats(data):
    """
    Print statistics of the data.

    Args:
        data: pandas.DataFrame, the data.
    
    Returns:
        None

    """
    print("\nStatistics of the data: \n")
    print(f"Number of reviews: {len(data)}")
    # Number of unique movies and users
    n_movies = len(set(data["movie_id"]))
    n_users = len(set(data["user_id"]))
    print(f"\nNumber of unique movies: {n_movies}")
    print(f"Number of unique users: {n_users}")

    # Average movies watched per user and average users watched a movie
    movies = data.groupby("user_id")["movie_id"].count()
    users = data.groupby("movie_id")["user_id"].count()
    avg_movies_per_user = np.mean(movies)
    avg_users_per_movie = np.mean(users)
    print(f"\nAverage movies watched per user: {avg_movies_per_user}")
    print(f"Average users watched a movie: {avg_users_per_movie}")

    # Print Nan per column
    print("\nNumber of NaN values per column: ")
    print(data.isnull().sum())

def parse_list_dict(value):
    """
    Parse the columns that are lists or dictionaries.

    Args:
        data: pandas.DataFrame, the data.

    Returns:
        pandas.DataFrame, the data with parsed columns.
    """
    try:
        return literal_eval(value) if isinstance(value, str) else value
    except:
        return value


class Dataset:

    def __init__(self, extra=False, relevant=False):
        if not os.path.exists(DATA_PATH):
            os.makedirs(DATA_PATH)

        self.extra = extra
        self.relevant = relevant

        # Load datasets
        self.ratelog = self.load_csv(RATELOG_FILE)
        self.movies = self.load_csv(MOVIES_FILE)
        self.users = self.load_csv(USERS_FILE)
        self.data = self.load_csv(DATA_FILE, generate_if_missing=True)

    def load_csv(self, filename, generate_if_missing=False):
        path = os.path.join(DATA_PATH, filename)
        if os.path.exists(path):
            return pd.read_csv(path)
        elif generate_if_missing:
            return self.get_data_file()
        
        return pd.DataFrame()
        

    def insert_single_data(self, ratelog_row):
        cols = EXTRA if self.extra else COLUMNS
        entry = {col: None for col in cols}
        entry.update({
            "movie_id": ratelog_row["movie_id"],
            "user_id": ratelog_row["user_id"],
            "rating": ratelog_row["rating"]
        })

        return entry

    def get_data_file(self):
        cols = EXTRA if self.extra else COLUMNS
        if self.ratelog is None or self.ratelog.empty:
            return pd.DataFrame(columns=cols)
        
        entries = [self.insert_single_data(self.ratelog.iloc[i]) for i in range(len(self.ratelog))]
        self.data = pd.DataFrame(entries)
        
        if self.extra:
            movie_ids = self.data["movie_id"].unique()
            user_ids = self.data["user_id"].unique()
            
            # movie_data = pd.DataFrame([get_movie_data(mid) for mid in movie_ids])
            # user_data = pd.DataFrame([get_user_data(uid) for uid in user_ids])

            movie_data = pd.read_csv("data/movie_data_full.csv", index_col=0)
            user_data = pd.read_csv("data/user_data_full.csv", index_col=0)

            movie_data['genres'] = movie_data['genres'].apply(parse_list_dict)
            
            self.movies = pd.concat([self.movies, movie_data], ignore_index=True)
            self.users = pd.concat([self.users, user_data], ignore_index=True)
            
            for i, row in movie_data.iterrows():
                self.data.loc[self.data["movie_id"] == row["id"], ["popularity", "vote_average"]] = [row["popularity"], row["vote_average"]]
                for genre in row["genres"]:
                    self.data[genre['name']] = self.data["movie_id"].map(lambda x: 1 if x == row["id"] else 0)
            
            for i, row in user_data.iterrows():
                self.data.loc[self.data["user_id"] == row["user_id"], ["age", row["gender"]]] = [row["age"], 1]
        
        self.data.fillna(0, inplace=True)

        if self.relevant:
            self.data["relevant"] = self.data["rating"].map(lambda x: 1 if x >= 4 else 0)
        return self.data

    def save_data(self):
        # Check if not empty
        if not self.data.empty:
            self.data.to_csv(os.path.join(DATA_PATH, DATA_FILE), index=False)

        if not self.movies.empty:
            self.movies.to_csv(os.path.join(DATA_PATH, MOVIES_FILE), index=False)

        if not self.users.empty:
            self.users.to_csv(os.path.join(DATA_PATH, USERS_FILE), index=False)


def add_data(filename):
    """
    Add new data from ratelog file to the datasets.
    """

    # We read the data from the "ratelog_2025-02.csv" file
    ratelog = pd.read_csv(filename)

    # We update the dm.DATASETS["ratelog"] with the new data
    dm.update_dataset("ratelog", ratelog)

    # Get unique movie_ids and user_ids and add them to the dm.DATASETS["movies"] and dm.DATASETS["users"] if they are not already there
    movie_ids = ratelog["movie_id"].unique()
    user_ids = ratelog["user_id"].unique()

    movies = dm.get_dataset("movies")
    users = dm.get_dataset("users")

    for movie_id in movie_ids:
        if movie_id not in movies["movie_id"]:
            movie_data = get_movie_data(movie_id)
            movie_data["movie_id"] = movie_data["id"]
            movie_data.pop("id")
            movies = movies._append(movie_data, ignore_index=True)
    
    for user_id in user_ids:
        if user_id not in users["user_id"]:
            user_data = get_user_data(user_id)
            users = users._append(user_data, ignore_index=True)

    # Save the updated datasets
    # Update
    dm.update_dataset("movies", movies)
    dm.update_dataset("users", users)

    # Save
    dm.save_datasets()

def get_training_data(extra, relevant, save, filename = None):

    # We get a copy of the dm.DATASETS["ratelog"] dataset
    ratelog = dm.get_dataset("ratelog")

    # We drop the time column
    data = ratelog.drop("time", axis=1)

    if relevant:
        ratelog["relevant"] = ratelog["rating"].map(lambda x: 1 if x >= TH else 0)

    if extra:
        
        pass
    
    if save:
        data.to_csv(os.path.join(TRAINING_DATA_PATH, filename), index=False)

    return data
        

    pass


if __name__ == "__main__":

    add_data("data/ratelog_2025-02.csv")
    
