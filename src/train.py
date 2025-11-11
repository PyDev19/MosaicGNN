import torch
from dataset import get_dataset
from model import MosaicGNN
from torch_geometric.loader import LinkNeighborLoader
from torch.nn import MSELoss
from torch.optim import Adam
from torch_geometric.transforms import RandomLinkSplit
from tqdm import tqdm

data_dir = "data/"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

config = {
    "data_directory": "data/",
    "dataset": {
        "val_ratio": 0.1,
        "test_ratio": 0.1,
        "disjoint_train_ratio": 0.3,
        "neg_sampling_ratio": 2.0,
        "add_negative_train_samples": False,
        "is_undirected": True,
    }
}

dataset = get_dataset(config["data_directory"], device)

print("Splitting data into train, val, and test sets...")

transform = RandomLinkSplit(
    **config["dataset"],
    edge_types=("user", "rates", "movie"),
    rev_edge_types=("movie", "rev_rates", "user"),
)

train_data, val_data, test_data = transform(dataset)

print("Setting up data loaders...")
train_loader = LinkNeighborLoader(
    data=train_data,
    num_neighbors=[20, 10],
    neg_sampling_ratio=2.0,
    edge_label_index=(
        ("user", "rates", "movie"),
        train_data["user", "rates", "movie"].edge_label_index,
    ),
    edge_label=train_data["user", "rates", "movie"].edge_label,
    batch_size=1024,
    shuffle=True,
)

val_loader = LinkNeighborLoader(
    data=val_data,
    num_neighbors=[-1],
    edge_label_index=(
        ("user", "rates", "movie"),
        val_data["user", "rates", "movie"].edge_label_index,
    ),
    edge_label=val_data["user", "rates", "movie"].edge_label,
    batch_size=2048,
    shuffle=False,
)

test_loader = LinkNeighborLoader(
    data=test_data,
    num_neighbors=[-1],
    edge_label_index=(
        ("user", "rates", "movie"),
        test_data["user", "rates", "movie"].edge_label_index,
    ),
    edge_label=test_data["user", "rates", "movie"].edge_label,
    batch_size=2048,
    shuffle=False,
)

print("Initializing model, optimizer, and loss function...")

model = MosaicGNN(200948, 85198, 403, hidden_channels=64).to(device)
optimizer = Adam(model.parameters(), lr=1e-3)
criterion = MSELoss()

print("Starting training loop...")

for batch in tqdm(train_loader, desc="Training", leave=True):
    batch = batch.to(device)
    print(batch)
    pred = model(batch)

    loss = criterion(pred.view(-1), batch["user", "rates", "movie"].edge_label.float())
    optimizer.zero_grad()

    loss.backward()
    optimizer.step()

    tqdm.write(f"Loss: {loss.item():.4f}")
