import os
import pandas as pd
import torch
from torch_geometric.data import HeteroData
from torch_geometric.transforms import ToUndirected, RandomLinkSplit
from torch_geometric.loader import LinkNeighborLoader


class RecommenderDataModule:
    def __init__(self, config: dict):
        self.config = config
        self.device = torch.device("cpu")
        self.data_dir = config["data_directory"]

        self.dataset = None
        self.train_data = None
        self.val_data = None
        self.test_data = None

    def _create_dataset(self) -> HeteroData:
        print("Loading csv files...")

        ratings = pd.read_csv(f"{self.data_dir}/ratings_cleaned.csv")
        movies = pd.read_csv(f"{self.data_dir}/movies_enriched_cleaned.csv")

        unique_users = ratings["userId"].unique()
        unique_movies = ratings["movieId"].unique()

        user2idx = {uid: i for i, uid in enumerate(unique_users)}
        movie2idx = {mid: i for i, mid in enumerate(unique_movies)}

        ratings["user_idx"] = ratings["userId"].map(user2idx)
        ratings["movie_idx"] = ratings["movieId"].map(movie2idx)

        genre_features = movies["genres"].str.get_dummies("|").values
        genre_features = torch.tensor(
            genre_features, dtype=torch.float, device=self.device
        )

        print("Loading overview embeddings...")

        overview_embeds = torch.load(
            f"{self.data_dir}/overview_embeddings.pt", map_location=self.device
        )

        movie_features = torch.cat([genre_features, overview_embeds], dim=1)

        print("Constructing HeteroData object...")

        data = HeteroData()

        num_users = len(user2idx)
        data["user"].x = torch.zeros(
            (num_users, 1), dtype=torch.float, device=self.device
        )

        data["movie"].x = movie_features.float().to(self.device)

        edge_index_user_to_movie = torch.stack(
            [
                torch.tensor(ratings["user_idx"].values, dtype=torch.long),
                torch.tensor(ratings["movie_idx"].values, dtype=torch.long),
            ]
        )

        data["user", "rates", "movie"].edge_index = edge_index_user_to_movie

        ratings_tensor = (
            torch.tensor(ratings["rating"].values, dtype=torch.float32)
            .unsqueeze(1)
            .to(self.device)
        )
        data["user", "rates", "movie"].edge_attr = ratings_tensor

        data = ToUndirected()(data)

        print("Saving processed graph data...")
        torch.save(data, f"{self.data_dir}/graph.pt")

        return data

    def _load_dataset(self) -> HeteroData:
        print("Loading processed graph data...")

        dataset = torch.load(
            f"{self.data_dir}/graph.pt",
            map_location=self.device,
            weights_only=False,
        )

        return dataset

    def _split_dataset(self) -> tuple[HeteroData, HeteroData, HeteroData]:
        print("Splitting dataset...")

        transform = RandomLinkSplit(
            **self.config["dataset"],
            edge_types=("user", "rates", "movie"),
            rev_edge_types=("movie", "rev_rates", "user"),
        )

        train_data, val_data, test_data = transform(self.dataset)
        return train_data, val_data, test_data

    def setup(self):
        self.dataset = (
            self._load_dataset()
            if os.path.exists(f"{self.data_dir}/graph.pt")
            else self._create_dataset()
        )

        self.train_data, self.val_data, self.test_data = self._split_dataset()

    def get_train_loader(self) -> LinkNeighborLoader:
        print("Preparing train data loader...")

        train_loader = LinkNeighborLoader(
            **self.config["loader"]["train"],
            data=self.train_data,
            edge_label_index=(
                ("user", "rates", "movie"),
                self.train_data["user", "rates", "movie"].edge_label_index,
            ),
            edge_label=self.train_data["user", "rates", "movie"].edge_label,
        )

        return train_loader

    def get_val_loader(self) -> LinkNeighborLoader:
        print("Preparing validation data loader...")

        val_loader = LinkNeighborLoader(
            **self.config["loader"]["val_test"],
            data=self.val_data,
            edge_label_index=(
                ("user", "rates", "movie"),
                self.val_data["user", "rates", "movie"].edge_label_index,
            ),
            edge_label=self.val_data["user", "rates", "movie"].edge_label,
        )

        return val_loader

    def get_test_loader(self) -> LinkNeighborLoader:
        print("Preparing test data loader...")

        test_loader = LinkNeighborLoader(
            **self.config["loader"]["val_test"],
            data=self.test_data,
            edge_label_index=(
                ("user", "rates", "movie"),
                self.test_data["user", "rates", "movie"].edge_label_index,
            ),
            edge_label=self.test_data["user", "rates", "movie"].edge_label,
        )

        return test_loader
