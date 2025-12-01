import torch
from torch.nn import Module, Linear, Embedding, Dropout, Parameter
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch_geometric.nn import SAGEConv, to_hetero
from torch_geometric.data import HeteroData


class NovaGNNEncoder(Module):
    def __init__(self, hidden_channels: int, dropout: float):
        super().__init__()
        self.conv1 = SAGEConv(hidden_channels, hidden_channels)
        self.conv2 = SAGEConv(hidden_channels, hidden_channels)
        self.dropout = Dropout(dropout)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = self.dropout(x.relu())
        x = self.conv2(x, edge_index)
        x = self.dropout(x)
        return x


class NovaLinkPredictor(Module):
    def __init__(
        self,
        hidden_channels: int,
        dropout: float,
        emb_dropout: float,
        movie_feat_dim: int,
        metadata: list,
        num_movies: int,
        use_movie_emb: bool,
    ):
        super().__init__()

        self.global_user_feature = Parameter(torch.randn(hidden_channels))

        self.movie_lin = Linear(movie_feat_dim, hidden_channels)
        self.use_movie_emb = use_movie_emb
        self.movie_emb = Embedding(num_movies, hidden_channels)

        self.emb_dropout = Dropout(emb_dropout)

        self.gnn = NovaGNNEncoder(hidden_channels, dropout)
        self.gnn = to_hetero(self.gnn, metadata=metadata)

    def forward(self, data: HeteroData):
        user_emb = self.global_user_feature.unsqueeze(0).repeat(
            data["user"].node_id.size(0), 1
        )

        movie_feat = self.movie_lin(data["movie"].x)

        if self.use_movie_emb:
            movie_emb = self.emb_dropout(self.movie_emb(data["movie"].node_id))
            if movie_feat.size(0) != movie_emb.size(0):
                fixed = torch.zeros_like(movie_emb)
                fixed[: movie_feat.size(0)] = movie_feat
                movie_feat = fixed
            movie_x = movie_emb + movie_feat
        else:
            movie_x = movie_feat

        x_dict = {
            "user": user_emb,
            "movie": movie_x,
        }

        x_dict = self.gnn(x_dict, data.edge_index_dict)

        user_edges = data["user", "rates", "movie"].edge_label_index[0]
        movie_edges = data["user", "rates", "movie"].edge_label_index[1]

        edge_feat_user = x_dict["user"][user_edges]
        edge_feat_movie = x_dict["movie"][movie_edges]

        return (edge_feat_user * edge_feat_movie).sum(dim=-1)


class NovaModelModule:
    def __init__(
        self,
        model_config: dict,
        optimizer_config: dict,
        scheduler_config: dict,
        device: torch.device,
        num_movies: int,
        metadata: list,
    ):
        self.model = NovaLinkPredictor(
            **model_config,
            num_movies=num_movies,
            metadata=metadata
        ).to(device)

        self.optimizer = Adam(self.model.parameters(), **optimizer_config)
        self.scheduler = ReduceLROnPlateau(self.optimizer, **scheduler_config)

    def parameters(self):
        return self.model.parameters()
