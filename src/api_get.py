import requests

IP = "128.2.204.215"

def get_movie_data(movie_id):
    url = f"http://{IP}:8080/movie/{movie_id}"
    
    try:
        response = requests.get(url)
        response.raise_for_status()  # Raise an exception for 4xx/5xx errors
        return response.json()  # Return the data as JSON
    except requests.exceptions.RequestException as e:
        print(f"Error fetching movie data: {e}")
        return None
    

def get_user_data(user_id):
    url = f"http://{IP}:8080/user/{user_id}"

    try:
        response = requests.get(url)
        response.raise_for_status()  # Raise an exception for 4xx/5xx errors
        return response.json()  # Return the data as JSON
    except requests.exceptions.RequestException as e:
        print(f"Error fetching user data: {e}")
        return None