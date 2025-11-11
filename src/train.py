import torch
import yaml
from dataset import get_dataset
from model import MosaicGNN
from torch_geometric.loader import LinkNeighborLoader
from torch.nn import MSELoss
from torch.optim import Adam
from torch_geometric.transforms import RandomLinkSplit
from tqdm import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)

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
    **config["loader"]["train"],
    data=train_data,
    edge_label_index=(
        ("user", "rates", "movie"),
        train_data["user", "rates", "movie"].edge_label_index,
    ),
    edge_label=train_data["user", "rates", "movie"].edge_label,
)

val_loader = LinkNeighborLoader(
    **config["loader"]["val_test"],
    data=val_data,
    edge_label_index=(
        ("user", "rates", "movie"),
        val_data["user", "rates", "movie"].edge_label_index,
    ),
    edge_label=val_data["user", "rates", "movie"].edge_label,
)

test_loader = LinkNeighborLoader(
    **config["loader"]["val_test"],
    data=test_data,
    edge_label_index=(
        ("user", "rates", "movie"),
        test_data["user", "rates", "movie"].edge_label_index,
    ),
    edge_label=test_data["user", "rates", "movie"].edge_label,
)

print("Initializing model, optimizer, and loss function...")

model = MosaicGNN(**config["model"]).to(device)
optimizer = Adam(model.parameters(), lr=1e-3)
criterion = MSELoss()

print("Starting training loop...")

num_epochs = config['num_epochs']
for epoch in range(1, num_epochs + 1):
    model.train()
    total_loss = 0.0

    with tqdm(train_loader, desc=f"Epoch {epoch}/{num_epochs}", leave=False) as pbar:
        for batch in pbar:
            batch = batch.to(device)

            pred = model(batch)
            loss = criterion(
                pred.view(-1),
                batch["user", "rates", "movie"].edge_label.float()
            )

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            pbar.set_postfix({"batch_loss": f"{loss.item():.4f}"})

    avg_loss = total_loss / len(train_loader)
    print(f"Epoch {epoch:03d} | Avg Train Loss: {avg_loss:.4f}")

torch.save(model.state_dict(), "mosaic_gnn_model.pth")