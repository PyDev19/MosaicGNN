import torch
from torch.nn import (
    Module,
    Linear,
    Embedding,
    Dropout,
    Sequential,
    ReLU,
)
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch_geometric.nn import SAGEConv, HeteroConv


class NodeEmbeddings(Module):
    def __init__(self, num_users, embed_dim, movie_feat_dim, dropout=0.3):
        super().__init__()
        self.user_emb = Embedding(num_users, embed_dim)
        self.proj_movie = Sequential(
            Linear(movie_feat_dim, embed_dim), ReLU(), Dropout(dropout)
        )

    def forward(self, data):
        x_dict = {
            "user": self.user_emb.weight,
            "movie": self.proj_movie(data["movie"].x),
        }

        return x_dict


class SAGE_Encoder(Module):
    def __init__(self, hidden_channels, dropout=0.3):
        super().__init__()
        self.conv1 = HeteroConv(
            {
                ("user", "rates", "movie"): SAGEConv((-1, -1), hidden_channels),
                ("movie", "rev_rates", "user"): SAGEConv((-1, -1), hidden_channels),
            },
        )
        self.conv2 = HeteroConv(
            {
                ("user", "rates", "movie"): SAGEConv((-1, -1), hidden_channels),
                ("movie", "rev_rates", "user"): SAGEConv((-1, -1), hidden_channels),
            }
        )

        self.dropout = Dropout(dropout)

    def forward(self, x_dict, edge_index_dict):
        h0 = x_dict

        x_dict = self.conv1(x_dict, edge_index_dict)
        x_dict = {k: self.dropout(x.relu()) for k, x in x_dict.items()}

        res = {k: h0[k] + x_dict[k] for k in x_dict}

        x_dict = self.conv2(res, edge_index_dict)
        x_dict = {k: self.dropout(x) for k, x in x_dict.items()}

        return x_dict


class EdgeDecoder(Module):
    def __init__(self, hidden_channels):
        super().__init__()
        self.linear_1 = Linear(2 * hidden_channels, hidden_channels)
        self.linear_2 = Linear(hidden_channels, 1)

    def forward(self, user_emb, movie_emb, edge_label_index):
        u = user_emb[edge_label_index[0]]
        m = movie_emb[edge_label_index[1]]
        x = torch.cat([u, m], dim=-1)

        x = self.linear_1(x).relu()
        x = self.linear_2(x)

        return x


class NOVA_GNN(Module):
    def __init__(
        self,
        num_users=100,
        movie_feat_dim=128,
        hidden_channels=64,
        dropout=0.3,
    ):
        super().__init__()
        self.embed = NodeEmbeddings(num_users, hidden_channels, movie_feat_dim, dropout)
        self.gnn = SAGE_Encoder(hidden_channels, dropout)
        self.decoder = EdgeDecoder(hidden_channels)

    def forward(self, data):
        x_dict = self.embed(data)

        x_dict = self.gnn(x_dict, data.edge_index_dict)
        edge_label_index = data["user", "rates", "movie"].edge_label_index

        for k in x_dict:
            x_dict[k] = torch.nn.functional.normalize(x_dict[k], p=2, dim=-1)

        out = self.decoder(x_dict["user"], x_dict["movie"], edge_label_index)
        return out


class RecommenderModelModule:
    def __init__(
        self,
        model_config: dict,
        optimizer_config: dict,
        scheduler_config: dict,
        device: torch.device,
    ):
        self.model = NOVA_GNN(**model_config).to(device)
        self.optimizer = AdamW(self.model.parameters(), **optimizer_config)
        self.scheduler = ReduceLROnPlateau(self.optimizer, **scheduler_config)

    def parameters(self):
        return self.model.parameters()
