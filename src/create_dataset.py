import pandas as pd
import requests
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

headers = {
    "accept": "application/json",
    "Authorization": "Bearer eyJhbGciOiJIUzI1NiJ9.eyJhdWQiOiI5ODMyNDZlOTI4ZWI1NzJmMTA5ZDNiMDNhZGYwZmJiOSIsIm5iZiI6MTc1ODc1NDM4OC45NTUwMDAyLCJzdWIiOiI2OGQ0NzY1NDI1YzRmN2EyMjg5MzYzOGQiLCJzY29wZXMiOlsiYXBpX3JlYWQiXSwidmVyc2lvbiI6MX0.esdJ4znV3D3yBzu6tPbLA9kq0OR0SZVoNQU3Dq5rW4U",
}


def fetch_credits(row) -> tuple[int, str, str]:
    movie_id = row["movieId"]
    tmdb_id = row["tmdbId"]

    if pd.isna(tmdb_id):
        return movie_id, "", ""

    url_credits = f"https://api.themoviedb.org/3/movie/{int(tmdb_id)}/credits"
    url_details = f"https://api.themoviedb.org/3/movie/{int(tmdb_id)}"
    
    try:
        r_credits = requests.get(url_credits, headers=headers, timeout=10)
        r_details = requests.get(url_details, headers=headers, timeout=10)

        if r_credits.status_code == 200 and r_details.status_code == 200:
            credits = r_credits.json()
            details = r_details.json()

            cast = credits.get("cast", [])
            crew = credits.get("crew", [])
            genres = details.get("genres", [])

            actors = [
                a["name"] for a in cast if a.get("known_for_department") == "Acting"
            ]
            directors = [m["name"] for m in crew if m.get("job") == "Director"]
            genres_list = [g["name"] for g in genres]

            return movie_id, "|".join(actors), "|".join(directors), "|".join(genres_list)
        else:
            return movie_id, "", "", ""
    except Exception as e:
        return movie_id, "", "", ""


actors_map = {}
directors_map = {}

movies_data = pd.read_csv("data/movies.csv")
links_data = pd.read_csv("data/links.csv")

with ThreadPoolExecutor(max_workers=10) as executor:
    futures = {
        executor.submit(fetch_credits, row): row["movieId"]
        for _, row in links_data.iterrows()
    }

    for future in tqdm(as_completed(futures), total=len(futures)):
        movie_id, actors_str, directors_str = future.result()
        actors_map[movie_id] = actors_str
        directors_map[movie_id] = directors_str

movies_data["actors"] = (
    movies_data["movieId"].map(actors_map).fillna(movies_data.get("actors", ""))
)
movies_data["directors"] = (
    movies_data["movieId"].map(directors_map).fillna(movies_data.get("directors", ""))
)

missing_actors = movies_data[movies_data["actors"] == ""]
missing_directors = movies_data[movies_data["directors"] == ""]

missing_info = pd.concat([missing_actors, missing_directors]).drop_duplicates()

movies_data.to_csv("data/movies_enriched.csv", index=False)
missing_info.to_csv("data/movies_missing_info.csv", index=False)
