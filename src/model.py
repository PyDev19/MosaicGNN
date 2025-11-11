import torch
from torch.nn import Module, Linear, Embedding
from torch_geometric.nn import SAGEConv, HeteroConv

class NodeEmbeddings(Module):
    def __init__(self, num_users, num_movies, embed_dim, movie_feat_dim):
        super().__init__()
        self.user_emb = Embedding(num_users, embed_dim)
        self.proj_movie = Linear(movie_feat_dim, embed_dim)

    def forward(self, data):
        x_dict = {
            "user": self.user_emb.weight,
            "movie": self.proj_movie(data["movie"].x)
        }
        
        return x_dict

class GNNEncoder(Module):
    def __init__(self, hidden_channels):
        super().__init__()
        self.conv1 = HeteroConv({
            ('user', 'rates', 'movie'): SAGEConv((-1, -1), hidden_channels),
            ('movie', 'rev_rates', 'user'): SAGEConv((-1, -1), hidden_channels),
        })
        
        self.conv2 = HeteroConv({
            ('user', 'rates', 'movie'): SAGEConv((-1, -1), hidden_channels),
            ('movie', 'rev_rates', 'user'): SAGEConv((-1, -1), hidden_channels),
        })

    def forward(self, x_dict, edge_index_dict):
        x_dict = self.conv1(x_dict, edge_index_dict)
        
        x_dict = {k: x.relu() for k, x in x_dict.items()}
        
        x_dict = self.conv2(x_dict, edge_index_dict)
        
        return x_dict

class EdgeDecoder(Module):
    def __init__(self, hidden_channels):
        super().__init__()
        self.linear_1 = Linear(2 * hidden_channels, hidden_channels)
        self.linear_2 = Linear(hidden_channels, 1)

    def forward(self, user_emb, movie_emb, edge_label_index):
        user_x = user_emb[edge_label_index[0]]
        movie_x = movie_emb[edge_label_index[1]]
        
        x = torch.cat([user_x, movie_x], dim=-1)
        
        x = torch.relu(self.linear_1(x))
        x = self.linear_2(x)
        
        return x

class MosaicGNN(Module):
    def __init__(self, num_users=100, num_movies=100, movie_feat_dim=128, hidden_channels=64):
        super().__init__()
        self.embed = NodeEmbeddings(num_users, num_movies, hidden_channels, movie_feat_dim)
        self.gnn = GNNEncoder(hidden_channels)
        self.decoder = EdgeDecoder(hidden_channels)

    def forward(self, data):
        x_dict = self.embed(data)
        
        x_dict = self.gnn(x_dict, data.edge_index_dict)
        
        edge_label_index = data["user", "rates", "movie"].edge_label_index
        
        out = self.decoder(x_dict["user"], x_dict["movie"], edge_label_index)
        
        return out

