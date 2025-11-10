import torch
from torch_geometric.data import HeteroData, Dataset
import pandas as pd
from tqdm import tqdm

class UserMovieGraphDataset(Dataset):
    def __init__(self, config, transform=None, pre_transform=None):
        super().__init__(None, transform, pre_transform)

        self.ratings = pd.read_csv(config["ratings_path"])
        self.movies = pd.read_csv(config["movies_path"])

        self.user_id_map = {
            uid: i for i, uid in enumerate(self.ratings["userId"].unique())
        }
        self.movie_id_map = {
            mid: i for i, mid in enumerate(self.movies["movieId"].unique())
        }

        genres = self.movies['genres'].str.get_dummies('|').values
        overview_embeds = torch.load(config["overview_embeddings_path"], map_location='cpu')
        movie_features = torch.cat([torch.tensor(genres, dtype=torch.float), overview_embeds], dim=1)

        self.movie_features = movie_features
        self.user_groups = self.ratings.groupby("userId")

    def len(self):
        return len(self.user_groups)

    def get(self, idx):
        user_id = list(self.user_groups.groups.keys())[idx]
        user_mapped = self.user_id_map[user_id]

        user_data = self.user_groups.get_group(user_id)

        movie_ids = user_data["movieId"].map(self.movie_id_map).values
        movie_ids = torch.tensor(movie_ids, dtype=torch.long)

        data = HeteroData()

        data["user"].x = torch.ones(1, 1)

        data["movie"].x = self.movie_features[movie_ids]

        num_movies = len(movie_ids)
        edge_index = torch.stack([
            torch.zeros(num_movies, dtype=torch.long),
            torch.arange(num_movies, dtype=torch.long)
        ], dim=0)

        data["user", "rates", "movie"].edge_index = edge_index

        ratings = torch.tensor(user_data["rating"].values, dtype=torch.float)
        data["user", "rates", "movie"].edge_attr = ratings.unsqueeze(1)

        return data
