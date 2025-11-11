import os
import pandas as pd
import torch
from torch_geometric.data import HeteroData
from torch_geometric.transforms import ToUndirected


def get_dataset(data_dir: str, device: torch.device) -> HeteroData:
    if os.path.exists(f"{data_dir}/graph.pt"):
        dataset = torch.load(f"{data_dir}/graph.pt", map_location=device)
        return dataset
    else:
        print("Loading csv files...")

        ratings = pd.read_csv("data/ratings_cleaned.csv")
        movies = pd.read_csv("data/movies_enriched_cleaned.csv")

        unique_users = ratings["userId"].unique()
        unique_movies = ratings["movieId"].unique()

        user2idx = {uid: i for i, uid in enumerate(unique_users)}
        movie2idx = {mid: i for i, mid in enumerate(unique_movies)}

        ratings["user_idx"] = ratings["userId"].map(user2idx)
        ratings["movie_idx"] = ratings["movieId"].map(movie2idx)

        genre_features = movies["genres"].str.get_dummies("|").values
        genre_features = torch.tensor(genre_features, dtype=torch.float, device=device)

        print("Loading overview embeddings...")

        overview_embeds = torch.load("data/overview_embeddings.pt", map_location=device)

        movie_features = torch.cat([genre_features, overview_embeds], dim=1)

        print("Constructing HeteroData object...")

        data = HeteroData()

        num_users = len(user2idx)
        data["user"].x = torch.zeros((num_users, 1), dtype=torch.float, device=device)

        data["movie"].x = movie_features.float().to(device)

        edge_index_user_to_movie = torch.stack(
            [
                torch.tensor(ratings["user_idx"].values, dtype=torch.long),
                torch.tensor(ratings["movie_idx"].values, dtype=torch.long),
            ]
        )

        data["user", "rates", "movie"].edge_index = edge_index_user_to_movie

        ratings_tensor = (
            torch.tensor(ratings["rating"].values, dtype=torch.float16)
            .unsqueeze(1)
            .to(device)
        )
        data["user", "rates", "movie"].edge_attr = ratings_tensor

        data = ToUndirected()(data)
